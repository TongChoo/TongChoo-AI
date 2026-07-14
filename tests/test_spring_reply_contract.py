"""Spring → FastAPI REPLY 요청 계약의 회귀 테스트.

실제 Cerebras를 호출하지 않는다. Spring의 Java record가 직렬화하는 camelCase 필드와
conversation의 ``content`` 키를 FastAPI가 모두 받고, 내부 ``GenerateRequest``까지
손실 없이 전달하는지 검증한다.
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import app, get_service
from app.models import SpringExcuseResponse


class _FakeGenerationService:
    """Cerebras 대신 수신한 요청을 저장하는 테스트 전용 서비스.

    HTTP 엔드포인트의 Pydantic 검증과 DTO 변환을 함께 통과시키되, 네트워크 호출·API
    키·비용과 무관하게 계약만 검사할 수 있게 한다.
    """

    def __init__(self) -> None:
        self.request = None
        self.request_id = None

    async def generate_for_spring(self, request, request_id):
        """실제 생성 대신 유효한 Spring 응답을 돌려주고 입력을 보관한다."""
        self.request = request
        self.request_id = request_id
        return SpringExcuseResponse(
            excuse="사전 공유를 놓친 제 책임입니다. 오늘 일정 영향부터 정리해 바로 공유드리겠습니다.",
            successRate=50,
            realism=3,
            persuasion=3,
            suspicionLevel="MEDIUM",
            replyOptions=[
                "지금 일정부터 확인하겠습니다.",
                "미리 공유하지 못해 죄송합니다. 오늘 일정 영향부터 정리해 바로 공유드리겠습니다.",
                "이번엔 제 일정 관리가 졌네요. 바로 정리해서 만회하겠습니다.",
            ],
            riskFactors=[{"content": "일정 지연", "sortOrder": 0}],
            rememberItems=[{"content": "사전 공유", "sortOrder": 0}],
            aftermaths=[
                {
                    "whenLabel": "오늘",
                    "dayOffset": 0,
                    "question": "일정은 정리됐나요?",
                    "collapseRate": 20,
                    "sortOrder": 0,
                }
            ],
        )


class SpringReplyContractTests(unittest.TestCase):
    """Spring이 보내는 2차 REPLY 문맥이 FastAPI에서 보존되는지 확인한다."""

    def setUp(self) -> None:
        """엔드포인트 의존성을 가짜 서비스로 교체한 독립 HTTP 클라이언트를 만든다."""
        self.service = _FakeGenerationService()
        app.dependency_overrides[get_service] = lambda: self.service
        self.client = TestClient(app)

    def tearDown(self) -> None:
        """다른 테스트가 실제 서비스 의존성을 보지 않도록 교체 값을 제거한다."""
        app.dependency_overrides.clear()

    def test_reply_context_from_spring_is_preserved(self) -> None:
        """currentExcuse·conversation·roundNumber가 2차 답장 생성까지 전달되는지 검증한다."""
        response = self.client.post(
            "/internal/v1/excuses/reply",
            headers={"X-Request-ID": "reply-contract-test"},
            json={
                "situation": "팀 회의에 20분 늦었다",
                "target": "TEAM_LEAD",
                "tone": "MILD",
                "rootExcuse": "회의 시작 시간을 잘못 봤습니다.",
                "currentExcuse": "미리 공유하지 못해 죄송합니다. 지금 일정부터 확인하겠습니다.",
                "conversation": [
                    {
                        "role": "assistant",
                        "content": "회의 시작 시간을 잘못 봤습니다.",
                    },
                    {"role": "user", "content": "왜 미리 공유하지 않았나요?"},
                    {
                        "role": "assistant",
                        "content": "미리 공유하지 못해 죄송합니다. 지금 일정부터 확인하겠습니다.",
                    },
                    {
                        "role": "user",
                        "content": "그래서 오늘 일정은 어떻게 할 건가요?",
                    },
                ],
                "roundNumber": 3,
                "incomingMessage": "그래서 오늘 일정은 어떻게 할 건가요?",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()["replyOptions"]), 3)
        self.assertEqual(self.service.request_id, "reply-contract-test")
        self.assertEqual(
            self.service.request.currentExcuse,
            "미리 공유하지 못해 죄송합니다. 지금 일정부터 확인하겠습니다.",
        )
        self.assertEqual(
            self.service.request.incomingMessage,
            "그래서 오늘 일정은 어떻게 할 건가요?",
        )
        self.assertEqual(self.service.request.roundNumber, 3)
        self.assertEqual(len(self.service.request.conversation), 4)
        self.assertEqual(self.service.request.conversation[-1].role.value, "user")
        self.assertEqual(
            self.service.request.conversation[-1].message,
            "그래서 오늘 일정은 어떻게 할 건가요?",
        )

    def test_evolve_context_keeps_custom_target_description(self) -> None:
        """진화 요청도 최초 자연어 관계를 그대로 이어받는지 검증한다."""
        response = self.client.post(
            "/internal/v1/excuses/evolve",
            json={
                "situation": "회식 참석이 어렵습니다.",
                "target": "CUSTOM",
                "targetDescription": "회사 부장님",
                "tone": "MILD",
                "rootExcuse": "개인 사정이 있어 회식 참석이 어렵습니다.",
                "currentExcuse": "개인 사정이 있어 회식 참석이 어렵습니다.",
                "roundNumber": 1,
                "direction": "더 짧게",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.service.request.target.value, "CUSTOM")
        self.assertEqual(self.service.request.targetDescription, "회사 부장님")
        self.assertEqual(self.service.request.evolveDirection, "더 짧게")

    def test_reply_context_keeps_custom_target_description(self) -> None:
        response = self.client.post(
            "/internal/v1/excuses/reply",
            json={
                "situation": "회식 참석이 어렵습니다.",
                "target": "CUSTOM",
                "targetDescription": "회사 부장님",
                "tone": "MILD",
                "rootExcuse": "개인 사정이 있어 회식 참석이 어렵습니다.",
                "currentExcuse": "개인 사정이 있어 회식 참석이 어렵습니다.",
                "roundNumber": 2,
                "incomingMessage": "개인 사정이 뭔가요?",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.service.request.target.value, "CUSTOM")
        self.assertEqual(self.service.request.targetDescription, "회사 부장님")
        self.assertEqual(self.service.request.incomingMessage, "개인 사정이 뭔가요?")

    def test_custom_target_description_is_preserved(self) -> None:
        response = self.client.post(
            "/internal/v1/excuses/create",
            json={
                "situation": "약속 시간에 늦었다",
                "target": "CUSTOM",
                "targetDescription": "같은 프로젝트를 진행하는 친한 선배",
                "tone": "MILD",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.service.request.target.value, "CUSTOM")
        self.assertEqual(
            self.service.request.targetDescription,
            "같은 프로젝트를 진행하는 친한 선배",
        )

    def test_custom_target_without_description_is_rejected(self) -> None:
        response = self.client.post(
            "/internal/v1/excuses/create",
            json={
                "situation": "약속 시간에 늦었다",
                "target": "CUSTOM",
                "tone": "MILD",
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_non_custom_target_with_description_is_rejected(self) -> None:
        response = self.client.post(
            "/internal/v1/excuses/create",
            json={
                "situation": "팀 회의에 늦었다",
                "target": "TEAM_LEAD",
                "targetDescription": "회사 부장님",
                "tone": "MILD",
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_non_custom_target_with_blank_description_is_rejected(self) -> None:
        response = self.client.post(
            "/internal/v1/excuses/create",
            json={
                "situation": "팀 회의에 늦었다",
                "target": "TEAM_LEAD",
                "targetDescription": "",
                "tone": "MILD",
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_custom_target_over_100_characters_is_rejected(self) -> None:
        response = self.client.post(
            "/internal/v1/excuses/create",
            json={
                "situation": "약속 시간에 늦었다",
                "target": "CUSTOM",
                "targetDescription": "가" * 101,
                "tone": "MILD",
            },
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
