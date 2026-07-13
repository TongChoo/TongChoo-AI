"""Cerebras의 OpenAI 호환 Chat Completions API 어댑터.

이 모듈은 제공자 통신·재시도·Structured Outputs 파싱만 담당한다. 입력 문맥 조립은
prompts.py, Spring 응답 변환은 service.py에서 담당한다.
"""

import json
import logging
import time
from typing import Any

import httpx
from fastapi import HTTPException
from pydantic import ValidationError

from app.config import Settings
from app.models import ExcuseResult, LLM_RESULT_SCHEMA

logger = logging.getLogger("tongchoo.ai")

RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


def api_error(status_code: int, code: str, message: str) -> HTTPException:
    """FastAPI와 Spring이 함께 사용할 수 있는 일관된 오류 본문을 만든다."""
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


class _ResponseParseError(Exception):
    """제공자 응답이 기대한 Chat Completions JSON 구조가 아닐 때 사용한다."""


class _TruncatedResponse(Exception):
    """응답이 max_completion_tokens에 걸려 잘렸을 때 사용한다."""


class CerebrasClient:
    """Cerebras OpenAI 호환 Chat Completions API의 비동기 클라이언트."""

    def __init__(self, settings: Settings):
        """API 키·엔드포인트·타임아웃을 한 번만 검증하고 준비한다."""
        if not settings.cerebras_api_key:
            raise api_error(
                503,
                "AI_CONFIGURATION_ERROR",
                "Cerebras API 키가 설정되지 않았습니다.",
            )

        self.settings = settings
        self.endpoint = f"{settings.cerebras_base_url.rstrip('/')}/chat/completions"
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
                        if self._can_retry(attempt, max_attempts) and response.status_code in RETRYABLE_STATUS_CODES:
                            logger.warning(
                                "llm_retry request_id=%s reason=http_%s attempt=%s",
                                request_id,
                                response.status_code,
                                attempt + 1,
                            )
                            continue
                        raise self._provider_error(response.status_code)

                    try:
                        body = response.json()
                        result = self._parse_result(body)
                    except _TruncatedResponse:
                        if self._can_retry(attempt, max_attempts):
                            # 잘린 응답은 한 번 더 넉넉한 토큰 예산으로 재시도한다.
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
                    except (ValidationError, json.JSONDecodeError, _ResponseParseError) as exc:
                        if self._can_retry(attempt, max_attempts):
                            logger.warning(
                                "llm_retry request_id=%s reason=parse attempt=%s",
                                request_id,
                                attempt + 1,
                            )
                            continue
                        raise api_error(
                            422,
                            "LLM_PARSE_ERROR",
                            "Cerebras 응답이 출력 형식과 일치하지 않습니다.",
                        ) from exc

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
        raise api_error(502, "CEREBRAS_UNAVAILABLE", "Cerebras 요청을 완료하지 못했습니다.")

    def _build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        completion_tokens: int,
    ) -> dict[str, Any]:
        """Cerebras OpenAI 호환 API가 요구하는 비스트리밍 JSON 요청 본문을 만든다."""
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
                    "strict": True,
                    "schema": LLM_RESULT_SCHEMA,
                },
            },
        }

    @staticmethod
    def _parse_result(body: Any) -> ExcuseResult:
        """Chat Completions 응답의 첫 선택지를 꺼내 Pydantic 모델로 최종 검증한다."""
        if not isinstance(body, dict):
            raise _ResponseParseError("provider response is not an object")

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise _ResponseParseError("provider response has no choices")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise _ResponseParseError("provider choice is not an object")
        if choice.get("finish_reason") == "length":
            raise _TruncatedResponse()

        message = choice.get("message")
        if not isinstance(message, dict):
            raise _ResponseParseError("provider message is not an object")
        if message.get("refusal"):
            raise api_error(422, "CEREBRAS_REFUSAL", "Cerebras가 요청을 처리하지 않았습니다.")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise _ResponseParseError("provider content is empty")

        return ExcuseResult.model_validate(json.loads(content))

    @staticmethod
    def _usage_value(usage: Any, key: str) -> int | None:
        """제공자가 usage를 누락해도 로깅 때문에 요청이 실패하지 않도록 처리한다."""
        if isinstance(usage, dict):
            value = usage.get(key)
            return value if isinstance(value, int) else None
        return None

    @staticmethod
    def _provider_error(status_code: int) -> HTTPException:
        """Cerebras 상태 코드를 Spring이 처리하기 쉬운 안정적인 오류 코드로 변환한다."""
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
        """현재 시도가 마지막 시도가 아닌지 읽기 쉬운 이름으로 표현한다."""
        return attempt + 1 < max_attempts
