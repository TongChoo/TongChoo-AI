from app.config import Settings
from app.llm import OpenAIClient, api_error
from app.models import ExcuseResult, GenerateRequest, SpringExcuseResponse
from app.prompts import build_system_prompt, build_user_prompt
from app.safety import SafetyAction, classify_input


class ExcuseGenerationService:
    def __init__(self, settings: Settings):
        self.client = OpenAIClient(settings)

    def generate(self, request: GenerateRequest, request_id: str) -> ExcuseResult:
        action = classify_input(
            request.situation,
            request.currentExcuse,
            request.incomingMessage,
            request.rootExcuse,
            request.memory,
            *(turn.content for turn in request.conversation),
        )
        if action == SafetyAction.BLOCK:
            raise api_error(
                422,
                "SAFETY_BLOCKED",
                "요청 내용을 안전한 범위에서 처리할 수 없습니다.",
            )

        return self.client.generate(
            build_system_prompt(),
            build_user_prompt(request, action),
            request_id,
        )

    def generate_for_spring(
        self,
        request: GenerateRequest,
        request_id: str,
    ) -> SpringExcuseResponse:
        result = self.generate(request, request_id)
        return SpringExcuseResponse.from_result(request, result)
