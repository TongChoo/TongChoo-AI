"""AI 생성 흐름을 조율하는 서비스 계층.

HTTP 엔드포인트는 요청을 표준 모델로 바꾸고, 이 서비스는 프롬프트 구성과 제공자
호출을 담당한다. DB 저장은 Spring의 책임이므로 이 모듈에는 저장 코드가 없다.
"""

from app.config import Settings
from app.llm import CerebrasClient
from app.models import ExcuseResult, GenerateRequest, SpringExcuseResponse
from app.prompts import build_system_prompt, build_user_prompt


class ExcuseGenerationService:
    """프롬프트 생성 → Cerebras 호출 순서를 담당한다."""

    def __init__(self, settings: Settings):
        self.client = CerebrasClient(settings)

    async def generate(self, request: GenerateRequest, request_id: str) -> ExcuseResult:
        """제공자 응답과 같은 평면 구조의 생성 결과를 반환한다."""
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(
            request,
            max_memory_chars=self.client.settings.max_memory_chars,
        )
        return await self.client.generate(system_prompt, user_prompt, request_id)

    async def generate_for_spring(
        self,
        request: GenerateRequest,
        request_id: str,
    ) -> SpringExcuseResponse:
        """제공자 결과를 Spring의 `ExcuseResponse` 핵심 구조로 변환한다."""
        result = await self.generate(request, request_id)
        return SpringExcuseResponse.from_result(request, result)
