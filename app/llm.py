"""Cerebras의 OpenAI 호환 Chat Completions API 어댑터.

이 모듈은 제공자 통신·재시도·관대한 응답 파싱만 담당한다. 입력 문맥 조립은
prompts.py, Spring 응답 변환은 service.py에서 담당한다.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx
from fastapi import HTTPException
from pydantic import ValidationError

from app.config import Settings
from app.models import ExcuseResult, LLM_RESULT_SCHEMA
from app.reply_quality import REPLY_JUDGE_SCHEMA, ReplyQualityVerdict

logger = logging.getLogger("tongchoo.ai")

# 이 상태 코드는 요청을 다시 보내도 성공할 가능성이 있는 일시적 네트워크·제공자
# 장애다. 인증, 결제, 안전성 거절처럼 설정 또는 사용자 입력을 바꿔야 하는 오류는
# 재시도 대상에 넣지 않는다.
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


def api_error(status_code: int, code: str, message: str) -> HTTPException:
    """FastAPI와 Spring이 함께 사용할 수 있는 일관된 오류 본문을 만든다.

    제공자별 예외 메시지를 그대로 노출하지 않고 안정적인 ``code``를 반환하면 Spring은
    HTTP 상태와 무관하게 재시도·사용자 안내 정책을 구현할 수 있다.
    """
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


class _ResponseParseError(Exception):
    """제공자 응답이 기대한 Chat Completions JSON 구조가 아닐 때 사용하는 내부 예외.

    이 예외는 외부 API로 노출하지 않고 ``generate``에서 제한 재시도 또는
    응답 본문이 있으면 가능한 범위에서 보완하고, 빈 응답만 오류로 처리한다.
    """


class _TruncatedResponse(Exception):
    """응답이 max_completion_tokens에 걸려 잘렸을 때 사용하는 내부 예외.

    일반 형식 오류와 달리 토큰 예산을 늘려 한 번 더 요청할 수 있으므로 별도 타입으로
    구분한다.
    """


class ReplyJudgeParseError(Exception):
    """Judge 응답이 품질 판정 계약으로 읽히지 않을 때 사용하는 내부 예외.

    생성 결과 본문은 관대한 파서로 보존할 수 있지만, Judge는 승인 근거이므로 형식을
    해석하지 못하면 품질 실패로 처리한다.
    """


class CerebrasClient:
    """Cerebras OpenAI 호환 Chat Completions API의 비동기 클라이언트.

    이 클래스는 HTTP 통신, 제공자 재시도, Structured Outputs 파싱만 담당한다. 어떤
    문맥을 넣을지와 결과가 이전 답변을 반복하는지는 각각 prompts.py와 service.py의
    제품 규칙으로 분리한다.
    """

    def __init__(self, settings: Settings):
        """API 키·엔드포인트·타임아웃을 한 번만 검증하고 준비한다.

        키가 없을 때 서버 시작을 막지 않는 대신, 생성 호출 시 명확한 설정 오류를 낸다.
        이 방식은 /health와 로컬 API 문서 확인을 가능하게 한다.
        """
        if not settings.cerebras_api_key:
            raise api_error(
                503,
                "AI_CONFIGURATION_ERROR",
                "Cerebras API 키가 설정되지 않았습니다.",
            )

        self.settings = settings
        # OpenAI 호환 base URL 끝의 슬래시 유무와 관계없이 정확한 endpoint가 되도록
        # rstrip 후 Chat Completions 경로를 붙인다.
        self.endpoint = f"{settings.cerebras_base_url.rstrip('/')}/chat/completions"
        # 연결 수립과 응답 생성 시간은 성격이 다르므로 별도 설정을 적용한다. pool도
        # connect 시간에 맞춰 대기 중인 연결 확보 때문에 요청이 오래 멈추지 않게 한다.
        self.timeout = httpx.Timeout(
            connect=settings.cerebras_connect_timeout_seconds,
            read=settings.cerebras_read_timeout_seconds,
            write=settings.cerebras_read_timeout_seconds,
            pool=settings.cerebras_connect_timeout_seconds,
        )

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        request_id: str,
    ) -> ExcuseResult:
        """Structured Outputs로 생성하고, 재시도 가능한 오류만 제한적으로 재시도한다.

        네트워크·일시적 제공자 오류·잘린 응답·형식 오류만 최대 `max_attempts`까지
        재시도한다. 안전성 차단이나 인증·결제 오류는 재시도하지 않는다.
        """
        # perf_counter는 시스템 시각 변경 영향을 받지 않아 요청 지연시간 측정에 적합하다.
        started = time.perf_counter()
        max_attempts = max(1, self.settings.max_attempts)
        completion_tokens = self.settings.max_completion_tokens
        headers = {
            "Authorization": f"Bearer {self.settings.cerebras_api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            "llm_request request_id=%s provider=cerebras model=%s max_attempts=%s",
            request_id,
            self.settings.cerebras_model,
            max_attempts,
        )

        # 요청 하나 안에서만 HTTP 클라이언트를 사용해 연결과 타임아웃 설정을 명확히 한다.
        # 재시도는 같은 client를 재사용해 불필요한 연결 생성을 줄인다.
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(max_attempts):
                try:
                    payload = self._build_payload(
                        system_prompt,
                        user_prompt,
                        completion_tokens,
                    )
                    response = await client.post(
                        self.endpoint,
                        headers=headers,
                        json=payload,
                    )

                    if response.status_code >= 400:
                        # 제공자가 오류 JSON을 보냈더라도 성공 응답 스키마로 파싱하지 않는다.
                        # 일시적 상태 코드만 다음 반복으로 넘기고, 나머지는 즉시 안정적 오류로
                        # 변환해 Spring이 잘못된 요청을 불필요하게 재시도하지 않게 한다.
                        if (
                            self._can_retry(attempt, max_attempts)
                            and response.status_code in RETRYABLE_STATUS_CODES
                        ):
                            logger.warning(
                                "llm_retry request_id=%s reason=http_%s attempt=%s",
                                request_id,
                                response.status_code,
                                attempt + 1,
                            )
                            continue
                        raise self._provider_error(response.status_code)

                    try:
                        # strict schema가 JSON 형식을 보장해도, 제공자 응답 구조 자체가
                        # 비정상인 경우를 대비해 HTTP 성공 이후 한 번 더 정규화한다.
                        body = response.json()
                        result = self._parse_result(body)
                    except _TruncatedResponse:
                        if self._can_retry(attempt, max_attempts):
                            # 잘린 응답은 같은 프롬프트를 더 큰 토큰 예산으로 한 번 재시도한다.
                            # 무조건 큰 기본값을 쓰는 것보다 평소 지연시간·비용을 낮출 수 있다.
                            completion_tokens = max(
                                completion_tokens,
                                self.settings.length_retry_completion_tokens,
                            )
                            logger.warning(
                                "llm_retry request_id=%s reason=length attempt=%s max_tokens=%s",
                                request_id,
                                attempt + 1,
                                completion_tokens,
                            )
                            continue
                        raise api_error(
                            502,
                            "LLM_TRUNCATED",
                            "Cerebras 응답이 토큰 제한으로 잘렸습니다.",
                        )
                    except (
                        ValidationError,
                        json.JSONDecodeError,
                        _ResponseParseError,
                    ) as exc:
                        if self._can_retry(attempt, max_attempts):
                            logger.warning(
                                "llm_retry request_id=%s reason=parse attempt=%s",
                                request_id,
                                attempt + 1,
                            )
                            continue
                        # 부가 필드 검증이 실패해도 응답 본문이 있으면 버리지 않는다.
                        # 최종적으로 excuse만 보존하고 나머지는 기본값으로 채운다.
                        logger.warning(
                            "llm_best_effort_parse request_id=%s error=%s",
                            request_id,
                            type(exc).__name__,
                        )
                        result = self._fallback_result(body)

                    # usage는 제공자 버전에 따라 생략될 수 있으므로, 로깅 실패가 생성
                    # 성공을 뒤집지 않게 아래 helper로 안전하게 꺼낸다.
                    elapsed_ms = round((time.perf_counter() - started) * 1000)
                    usage = body.get("usage") if isinstance(body, dict) else None
                    logger.info(
                        "llm_success request_id=%s provider=cerebras model=%s elapsed_ms=%s "
                        "prompt_tokens=%s completion_tokens=%s",
                        request_id,
                        self.settings.cerebras_model,
                        elapsed_ms,
                        self._usage_value(usage, "prompt_tokens"),
                        self._usage_value(usage, "completion_tokens"),
                    )
                    return result
                except HTTPException:
                    # 이미 의미 있는 HTTP 오류로 변환된 경우에는 상위 계층으로 그대로 전달한다.
                    raise
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    # DNS 실패·소켓 연결 실패·읽기 타임아웃은 같은 복구 정책을 적용한다.
                    if self._can_retry(attempt, max_attempts):
                        logger.warning(
                            "llm_retry request_id=%s reason=network attempt=%s",
                            request_id,
                            attempt + 1,
                        )
                        continue
                    logger.warning("llm_network_error request_id=%s", request_id)
                    raise api_error(
                        503,
                        "CEREBRAS_CONNECTION_ERROR",
                        "Cerebras에 연결할 수 없습니다.",
                    ) from exc
                except httpx.HTTPError as exc:
                    logger.warning("llm_http_error request_id=%s", request_id)
                    raise api_error(
                        503,
                        "CEREBRAS_CONNECTION_ERROR",
                        "Cerebras 요청을 완료하지 못했습니다.",
                    ) from exc
                except Exception as exc:
                    logger.exception("llm_unexpected_error request_id=%s", request_id)
                    raise api_error(
                        502,
                        "CEREBRAS_UNAVAILABLE",
                        "Cerebras 요청을 완료하지 못했습니다.",
                    ) from exc

        # 반복문에서 항상 반환·예외가 발생하지만, 타입 검사와 방어적 처리를 위해 남겨 둔다.
        raise api_error(
            502, "CEREBRAS_UNAVAILABLE", "Cerebras 요청을 완료하지 못했습니다."
        )

    async def judge_reply(
        self,
        system_prompt: str,
        user_prompt: str,
        request_id: str,
    ) -> ReplyQualityVerdict:
        """한 번의 독립 Judge 호출로 REPLY 후보 품질을 판정한다.

        Judge 자체를 재시도하면 후보 한 세트당 판정이 여러 번 달라질 수 있다.
        따라서 JSON 파싱 실패만 서비스의 품질 재생성 경로에 맡기고, 연결·제공자
        오류는 정상 운영 오류로 그대로 반환한다.
        """
        headers = {
            "Authorization": f"Bearer {self.settings.cerebras_api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.endpoint,
                    headers=headers,
                    json=self._build_judge_payload(system_prompt, user_prompt),
                )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            logger.warning("reply_judge_network_error request_id=%s", request_id)
            raise api_error(
                503,
                "CEREBRAS_CONNECTION_ERROR",
                "Cerebras에 연결할 수 없습니다.",
            ) from exc
        except httpx.HTTPError as exc:
            logger.warning("reply_judge_http_error request_id=%s", request_id)
            raise api_error(
                503,
                "CEREBRAS_CONNECTION_ERROR",
                "Cerebras 요청을 완료하지 못했습니다.",
            ) from exc

        if response.status_code >= 400:
            raise self._provider_error(response.status_code)

        try:
            verdict = self._parse_reply_quality(response.json())
        except (ValidationError, json.JSONDecodeError, _ResponseParseError) as exc:
            logger.warning(
                "reply_judge_parse_error request_id=%s error=%s",
                request_id,
                type(exc).__name__,
            )
            raise ReplyJudgeParseError() from exc

        logger.info(
            "reply_judge_success request_id=%s diversity=%s duplicate=%s",
            request_id,
            verdict.diversityScore,
            verdict.semanticDuplicate,
        )
        return verdict

    def _build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        completion_tokens: int,
    ) -> dict[str, Any]:
        """Cerebras OpenAI 호환 API가 요구하는 비스트리밍 JSON 요청 본문을 만든다.

        ``stream=False``로 두는 이유는 Spring이 부분 토큰이 아닌 하나의 결과를 저장하기
        때문이다. 최소 스키마는 제공하되 strict mode는 끄고, 모델이 자연스럽게 응답할
        여지를 둔다. 누락·추가 필드는 응답 파서가 방어적으로 보완한다.
        """
        return {
            "model": self.settings.cerebras_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.settings.temperature,
            "max_completion_tokens": completion_tokens,
            "stream": False,
            "reasoning_effort": self.settings.reasoning_effort,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "tongchoo_excuse_v1",
                    "strict": False,
                    "schema": LLM_RESULT_SCHEMA,
                },
            },
        }

    def _build_judge_payload(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Judge 전용의 느슨한 JSON Schema 요청 본문을 만든다."""
        return {
            "model": self.settings.cerebras_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_completion_tokens": self.settings.reply_judge_max_completion_tokens,
            "stream": False,
            "reasoning_effort": self.settings.reasoning_effort,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "tongchoo_reply_quality_v1",
                    "strict": False,
                    "schema": REPLY_JUDGE_SCHEMA,
                },
            },
        }

    @staticmethod
    def _parse_result(body: Any) -> ExcuseResult:
        """응답 본문을 읽고 답변이 있으면 부가 필드를 기본값으로 보완한다.

        제공자 JSON은 신뢰 경계 밖의 데이터다. choices/message/content의 타입을 모두
        확인한 뒤에만 JSON 문자열을 파싱하고, 최종 결과를 ``ExcuseResult``로 검증한다.
        """
        if not isinstance(body, dict):
            raise _ResponseParseError("provider response is not an object")

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise _ResponseParseError("provider response has no choices")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise _ResponseParseError("provider choice is not an object")
        if choice.get("finish_reason") == "length":
            # content가 우연히 유효한 JSON처럼 보여도 완전한 응답이 아닐 수 있으므로
            # 우선 토큰 예산 재시도 경로로 보낸다.
            raise _TruncatedResponse()

        message = choice.get("message")
        if not isinstance(message, dict):
            raise _ResponseParseError("provider message is not an object")
        if message.get("refusal"):
            # 제공자의 안전성 거절은 동일한 요청 재시도로 해결되지 않으므로 즉시 422로
            # 변환한다. 원본 refusal 내용은 외부에 그대로 전달하지 않는다.
            raise api_error(
                422, "CEREBRAS_REFUSAL", "Cerebras가 요청을 처리하지 않았습니다."
            )

        content = message.get("content")
        if isinstance(content, dict):
            payload = content
            raw_content = json.dumps(content, ensure_ascii=False)
        elif isinstance(content, str) and content.strip():
            raw_content = content.strip()
            try:
                payload = json.loads(raw_content)
            except json.JSONDecodeError:
                # JSON이 아닌 일반 답변도 버리지 않고 그대로 저장한다.
                payload = {"excuse": raw_content}
        else:
            raise _ResponseParseError("provider content is empty")

        if not isinstance(payload, dict):
            payload = {"excuse": raw_content}

        excuse = (
            CerebrasClient._first_text(
                payload, "excuse", "answer", "text", "message"
            )
            or raw_content
        )[:1000]
        options = payload.get("replyOptions") or payload.get("reply_options")
        if not isinstance(options, list):
            options = [excuse, excuse]
        options = [str(item).strip()[:200] for item in options if str(item).strip()]
        while len(options) < 2:
            options.append(excuse)

        normalized = {
            "excuse": excuse,
            "recommendedAction": (
                CerebrasClient._first_text(
                    payload, "recommendedAction", "recommended_action"
                )
                or "상대에게 현재 상황을 짧게 설명하고 바로 확인한다."
            )[:300],
            "likelyFollowUp": (
                CerebrasClient._first_text(
                    payload, "likelyFollowUp", "likely_follow_up"
                )
                or "그래서 지금 어떻게 할 건데?"
            )[:300],
            "replyOptions": options[:3],
            "successRate": CerebrasClient._bounded_int(payload.get("successRate", payload.get("success_rate", 50)), 0, 100, 50),
            "realism": CerebrasClient._bounded_int(payload.get("realism", 3), 1, 5, 3),
            "persuasion": CerebrasClient._bounded_int(payload.get("persuasion", 3), 1, 5, 3),
            "suspicionLevel": payload.get("suspicionLevel", payload.get("suspicion_level", "MEDIUM")) if payload.get("suspicionLevel", payload.get("suspicion_level", "MEDIUM")) in {"LOW", "MEDIUM", "HIGH"} else "MEDIUM",
            "riskFactors": CerebrasClient._string_list(payload.get("riskFactors", payload.get("risk_factors")), ["추가 확인이 필요함"]),
            "aftermath": CerebrasClient._aftermath_list(payload.get("aftermath", payload.get("aftermaths"))),
            "remember": CerebrasClient._string_list(payload.get("remember"), []),
        }
        return ExcuseResult.model_validate(normalized)

    @staticmethod
    def _parse_reply_quality(body: Any) -> ReplyQualityVerdict:
        """Judge의 JSON 본문을 내부 품질 판정 모델로 읽는다."""
        if not isinstance(body, dict):
            raise _ResponseParseError("judge response is not an object")
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise _ResponseParseError("judge response has no choice")
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            raise _ResponseParseError("judge response was truncated")
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("refusal"):
            raise _ResponseParseError("judge message is unavailable")
        content = message.get("content")
        if isinstance(content, dict):
            payload = content
        elif isinstance(content, str) and content.strip():
            payload = json.loads(content.strip())
        else:
            raise _ResponseParseError("judge content is empty")
        if not isinstance(payload, dict):
            raise _ResponseParseError("judge content is not an object")
        return ReplyQualityVerdict.model_validate(payload)

    @staticmethod
    def _fallback_result(body: Any) -> ExcuseResult:
        """구조가 깨진 응답에서도 본문을 저장할 수 있도록 최소 결과를 만든다."""
        content = "응답을 확인했습니다."
        try:
            content = body["choices"][0]["message"].get("content") or content
        except (KeyError, IndexError, AttributeError, TypeError):
            pass
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        excuse = content.strip()[:1000] or "응답을 확인했습니다."
        return ExcuseResult(
            excuse=excuse,
            recommendedAction="상대에게 현재 상황을 짧게 설명하고 바로 확인한다.",
            likelyFollowUp="그래서 지금 어떻게 할 건데?",
            replyOptions=[excuse, excuse],
            successRate=50,
            realism=3,
            persuasion=3,
            suspicionLevel="MEDIUM",
            riskFactors=["추가 확인이 필요함"],
            aftermath=[{
                "when": "오늘",
                "dayOffset": 0,
                "question": "상대가 추가로 확인할 수 있음",
                "collapseRate": 50,
            }],
            remember=[],
        )

    @staticmethod
    def _first_text(payload: dict[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _bounded_int(value: Any, minimum: int, maximum: int, fallback: int) -> int:
        try:
            return max(minimum, min(maximum, int(value)))
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _string_list(value: Any, fallback: list[str]) -> list[str]:
        if not isinstance(value, list):
            return fallback
        values = [item.strip()[:200] for item in value if isinstance(item, str) and item.strip()]
        return values[:8] or fallback

    @staticmethod
    def _aftermath_list(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list) or not value:
            return [{
                "when": "오늘",
                "dayOffset": 0,
                "question": "상대가 추가로 확인할 수 있음",
                "collapseRate": 50,
            }]
        results = []
        for item in value[:4]:
            if not isinstance(item, dict):
                continue
            results.append({
                "when": str(item.get("when", item.get("whenLabel", "오늘")))[:100],
                "dayOffset": CerebrasClient._bounded_int(item.get("dayOffset", item.get("day_offset", 0)), 0, 365, 0),
                "question": str(item.get("question", "상대가 추가로 확인할 수 있음"))[:300],
                "collapseRate": CerebrasClient._bounded_int(item.get("collapseRate", item.get("collapse_rate", 50)), 0, 100, 50),
            })
        return results or CerebrasClient._aftermath_list(None)

    @staticmethod
    def _usage_value(usage: Any, key: str) -> int | None:
        """제공자가 usage를 누락해도 로깅 때문에 요청이 실패하지 않도록 처리한다."""
        if isinstance(usage, dict):
            value = usage.get(key)
            return value if isinstance(value, int) else None
        return None

    @staticmethod
    def _provider_error(status_code: int) -> HTTPException:
        """Cerebras 상태 코드를 Spring이 처리하기 쉬운 안정적인 오류 코드로 변환한다.

        Spring은 공급자 HTTP 세부 상태에 의존하지 않고 ``code`` 기준으로 재시도와
        사용자 안내를 선택할 수 있다. 키·결제 오류는 운영자가 조치해야 하므로 각각
        구분해 반환한다.
        """
        if status_code in {401, 403}:
            return api_error(
                503,
                "CEREBRAS_AUTH_ERROR",
                "Cerebras API 인증에 실패했습니다.",
            )
        if status_code == 402:
            return api_error(
                402,
                "CEREBRAS_BILLING_REQUIRED",
                "Cerebras API 결제 또는 quota를 확인해주세요.",
            )
        if status_code == 429:
            return api_error(
                429,
                "CEREBRAS_RATE_LIMITED",
                "Cerebras 사용량 제한에 도달했습니다.",
            )
        if status_code in RETRYABLE_STATUS_CODES:
            return api_error(
                503,
                "CEREBRAS_UNAVAILABLE",
                "Cerebras 서비스를 일시적으로 사용할 수 없습니다.",
            )
        return api_error(
            502,
            "CEREBRAS_PROVIDER_ERROR",
            "Cerebras 요청이 처리되지 않았습니다.",
        )

    @staticmethod
    def _can_retry(attempt: int, max_attempts: int) -> bool:
        """현재 시도가 마지막 시도가 아닌지 읽기 쉬운 이름으로 표현한다.

        attempt는 0부터 시작하므로 ``attempt + 1 < max_attempts``일 때만 다음 요청을
        보낼 수 있다.
        """
        return attempt + 1 < max_attempts
