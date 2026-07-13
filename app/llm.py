import json
import logging
import time

from fastapi import HTTPException
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from pydantic import ValidationError

from app.config import Settings
from app.models import ExcuseResult, LLM_RESULT_SCHEMA
from app.safety import validate_output_safety

logger = logging.getLogger("tongchoo.ai")


def api_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


class OpenAIClient:
    """Small adapter around the official OpenAI Python SDK."""

    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise api_error(503, "AI_CONFIGURATION_ERROR", "OpenAI API key is not configured")

        self.settings = settings
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        request_id: str,
    ) -> ExcuseResult:
        started = time.perf_counter()
        logger.info(
            "llm_request request_id=%s provider=openai model=%s",
            request_id,
            self.settings.openai_model,
        )

        try:
            completion = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=self.settings.max_completion_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "tongchoo_excuse_v1",
                        "strict": True,
                        "schema": LLM_RESULT_SCHEMA,
                    },
                },
            )
            result = self._parse_result(completion)
            try:
                validate_output_safety(result)
            except ValueError as exc:
                raise api_error(
                    422,
                    "SAFETY_BLOCKED",
                    "생성 결과가 안전 정책을 통과하지 못했습니다.",
                ) from exc

            elapsed_ms = round((time.perf_counter() - started) * 1000)
            usage = completion.usage
            logger.info(
                "llm_success request_id=%s provider=openai model=%s elapsed_ms=%s "
                "prompt_tokens=%s completion_tokens=%s",
                request_id,
                self.settings.openai_model,
                elapsed_ms,
                getattr(usage, "prompt_tokens", None),
                getattr(usage, "completion_tokens", None),
            )
            return result
        except HTTPException:
            raise
        except AuthenticationError as exc:
            logger.warning("llm_auth_error request_id=%s", request_id)
            raise api_error(503, "OPENAI_AUTH_ERROR", "OpenAI API 인증에 실패했습니다.") from exc
        except RateLimitError as exc:
            logger.warning("llm_rate_limited request_id=%s", request_id)
            raise api_error(429, "OPENAI_RATE_LIMITED", "OpenAI 사용량 제한에 도달했습니다.") from exc
        except APIConnectionError as exc:
            logger.warning("llm_connection_error request_id=%s", request_id)
            raise api_error(503, "OPENAI_CONNECTION_ERROR", "OpenAI에 연결할 수 없습니다.") from exc
        except APIStatusError as exc:
            logger.error(
                "llm_provider_error request_id=%s status_code=%s",
                request_id,
                exc.status_code,
            )
            if exc.status_code == 402:
                raise api_error(
                    402,
                    "OPENAI_BILLING_REQUIRED",
                    "OpenAI API 결제 또는 quota를 확인해주세요.",
                ) from exc
            raise api_error(502, "OPENAI_PROVIDER_ERROR", "OpenAI 요청이 처리되지 않았습니다.") from exc
        except (ValidationError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            logger.warning("llm_parse_error request_id=%s", request_id)
            raise api_error(
                422,
                "LLM_PARSE_ERROR",
                "OpenAI 응답이 출력 형식과 일치하지 않습니다.",
            ) from exc
        except Exception as exc:
            logger.exception("llm_unexpected_error request_id=%s", request_id)
            raise api_error(502, "OPENAI_UNAVAILABLE", "OpenAI 요청을 완료하지 못했습니다.") from exc

    @staticmethod
    def _parse_result(completion) -> ExcuseResult:
        choice = completion.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise api_error(502, "LLM_TRUNCATED", "OpenAI 응답이 토큰 제한으로 잘렸습니다.")

        message = choice.message
        if getattr(message, "refusal", None):
            raise api_error(422, "OPENAI_REFUSAL", "OpenAI가 요청을 처리하지 않았습니다.")

        content = message.content
        if not content:
            raise api_error(502, "LLM_EMPTY_RESPONSE", "OpenAI가 빈 응답을 반환했습니다.")

        return ExcuseResult.model_validate(json.loads(content))
