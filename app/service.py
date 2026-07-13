"""AI 생성 흐름과 REPLY 품질 검증을 담당한다."""

from __future__ import annotations

import logging
from app.config import Settings
from app.llm import CerebrasClient
from app.models import ExcuseResult, GenerateRequest, GenerationMode, SpringExcuseResponse
from app.prompts import (
    build_reply_system_prompt,
    build_reply_user_prompt,
    build_system_prompt,
    build_user_prompt,
)

logger = logging.getLogger("tongchoo.service")

class ExcuseGenerationService:
    """프롬프트 구성, Cerebras 호출, REPLY 품질 재생성을 조율한다."""

    def __init__(self, settings: Settings):
        self.client = CerebrasClient(settings)

    async def generate(self, request: GenerateRequest, request_id: str) -> ExcuseResult:
        if request.mode == GenerationMode.REPLY:
            return await self.client.generate(
                build_reply_system_prompt(),
                build_reply_user_prompt(
                    request,
                    max_memory_chars=self.client.settings.max_memory_chars,
                ),
                request_id,
            )

        return await self._generate_once(request, request_id, build_system_prompt())

    async def generate_for_spring(
        self,
        request: GenerateRequest,
        request_id: str,
    ) -> SpringExcuseResponse:
        result = await self.generate(request, request_id)
        logger.info(
            "ai_generation_result request_id=%s mode=%s round=%s excuse=%s",
            request_id,
            request.mode.value,
            request.roundNumber,
            _excerpt(result.excuse),
        )
        return SpringExcuseResponse.from_result(result)

    # CREATE/EVOLVE 생성
    async def _generate_once(
        self,
        request: GenerateRequest,
        request_id: str,
        system_prompt: str,
    ) -> ExcuseResult:
        user_prompt = build_user_prompt(
            request,
            max_memory_chars=self.client.settings.max_memory_chars,
        )
        return await self.client.generate(system_prompt, user_prompt, request_id)


def _excerpt(value: str, limit: int = 120) -> str:
    return value.replace("\n", " ").strip()[:limit]
