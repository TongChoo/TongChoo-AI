"""REPLY 프롬프트와 품질 게이트의 회귀 테스트.

이 테스트들은 실제 Cerebras를 호출하지 않는다. 고정된 provider 결과를 넣어 프롬프트
우선순위, 중복 감지, 품질 실패 후 정확히 한 번의 재생성이 결정적으로 동작하는지만
검증한다.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import HTTPException

from app.models import ExcuseResult, GenerateRequest, GenerationMode, Target, Tone
from app.prompts import build_user_prompt
from app.service import (
    ExcuseGenerationService,
    _max_similarity,
    _reply_options_quality_failures,
    _reply_quality_failures,
)

# 해커톤 데모에서 자주 보이는 대상·상황·추궁 조합이다. 각 answers는 3라운드 동안
# conversation에 쌓일 이전 assistant 답변을 뜻하며, 최신 user 메시지가 우선되는지
# 확인하는 데 사용한다.
REPLY_CASES = (
    {
        "name": "teacher_topic_mismatch",
        "target": Target.TEACHER,
        "tone": Tone.MILD,
        "situation": "과제 주제를 잘못 이해해 제출했다",
        "incoming": "이건 수업에서 다룬 주제와 전혀 다르잖아. 왜 확인하지 않았니?",
        "answers": (
            "주제를 잘못 이해해서 제출했어요. 다시 확인하고 수정하겠습니다.",
            "맞아요, 주제 확인을 제대로 못 했습니다. 오늘 수업 자료 기준으로 다시 작성해 제출하겠습니다.",
            "제가 요구사항을 놓쳤습니다. 먼저 질문드렸어야 했고, 수정본을 내일까지 보내겠습니다.",
        ),
    },
    {
        "name": "team_lead_schedule_pressure",
        "target": Target.TEAM_LEAD,
        "tone": Tone.MILD,
        "situation": "팀 회의에 20분 늦었다",
        "incoming": "지금 일정이 밀리고 있는데 왜 미리 공유하지 않았나요?",
        "answers": (
            "회의에 늦은 건 제 잘못입니다. 지금 바로 합류하고 놓친 내용은 정리해서 공유하겠습니다.",
            "미리 알리지 못해 죄송합니다. 우선 회의 자료를 확인하고 오늘 일정에 영향이 없도록 처리하겠습니다.",
            "일정에 영향을 준 점 인정합니다. 회의록을 먼저 확인한 뒤 제 작업 일정을 다시 맞춰 공유드리겠습니다.",
        ),
    },
    {
        "name": "parent_location_check",
        "target": Target.PARENT,
        "tone": Tone.MILD,
        "situation": "늦은 귀가로 연락을 못 했다",
        "incoming": "지금 어디야? 왜 연락도 안 하고 늦었어?",
        "answers": (
            "늦게 연락해서 미안해요. 지금 바로 위치를 보내고 집에 도착하면 다시 연락할게요.",
            "걱정하게 해서 죄송해요. 지금 있는 곳을 알려드리고 귀가 시간을 정확히 말씀드릴게요.",
            "연락을 놓친 건 제 잘못이에요. 현재 위치와 집에 도착할 시간을 바로 보내드릴게요.",
        ),
    },
    {
        "name": "lover_trust_problem",
        "target": Target.LOVER,
        "tone": Tone.DESPERATE,
        "situation": "연락을 늦게 해서 상대가 서운해했다",
        "incoming": "맨날 바쁘다고만 하고, 이제 어떻게 믿어야 해?",
        "answers": (
            "믿기 어렵게 만든 건 내 잘못이야. 지금부터는 늦어질 때 먼저 말하고 오늘은 제대로 이야기하고 싶어.",
            "네가 그렇게 느끼게 한 걸 알아. 말로만 미안하다고 하지 않고 연락이 늦어질 때 먼저 알려줄게.",
            "신뢰를 잃게 만든 게 너무 미안해. 변명하지 않고 오늘 네 이야기를 끝까지 듣고 바꾸겠다고 약속할게.",
        ),
    },
    {
        "name": "friend_cancelled_plan",
        "target": Target.FRIEND,
        "tone": Tone.SLICK,
        "situation": "친구와의 약속을 갑자기 취소했다",
        "incoming": "또 파토야? 이번엔 무슨 핑계인데?",
        "answers": (
            "이번엔 내가 약속 관리에 완전히 졌다. 핑계 말고 내가 다음 약속 장소와 시간을 잡을게.",
            "들켰네, 약속 지키기 최하위권이다. 이번 건 내가 예약까지 해두고 제대로 만회할게.",
            "핑계 대회 출전은 취소할게. 이번 약속은 내가 다시 잡고 네가 고른 메뉴로 살게.",
        ),
    },
)

# 정상 결과로 간주하는 세 가지 후보 역할: 짧은 직접 답장, 책임·행동 답장, 가벼운
# 관계 수습 답장. 개별 테스트가 후보 품질과 무관할 때 이 기본값을 재사용한다.
DEFAULT_REPLY_OPTIONS = [
    "미리 알리지 못했습니다. 지금 일정부터 확인하겠습니다.",
    "미리 공유하지 못해 죄송합니다. 회의 자료를 확인하고 일정 영향부터 바로 정리해 공유드리겠습니다.",
    "이번엔 제 일정 관리가 졌네요. 다음 회의는 제가 먼저 확인해서 바로 알려드리겠습니다.",
]


def _request(case, round_number, conversation):
    """시나리오 한 라운드를 service.py가 받는 표준 REPLY 요청으로 변환한다."""
    return GenerateRequest(
        mode=GenerationMode.REPLY,
        situation=case["situation"],
        target=case["target"],
        tone=case["tone"],
        currentExcuse=(
            conversation[-2]["message"]
            if len(conversation) >= 2
            else case["answers"][0]
        ),
        incomingMessage=case["incoming"],
        conversation=[
            {"role": item["role"], "message": item["message"]} for item in conversation
        ],
        roundNumber=round_number,
    )


class ReplyQualityTests(unittest.TestCase):
    def test_fifteen_rounds_prioritize_latest_user_message_and_tone(self):
        """5개 시나리오 × 3라운드의 프롬프트에 최신 질문·톤·후보 역할이 모두 들어가는지 확인한다."""
        for case in REPLY_CASES:
            conversation = []
            for round_number, answer in enumerate(case["answers"], start=1):
                conversation.extend(
                    [
                        {"role": "assistant", "message": answer},
                        {"role": "user", "message": case["incoming"]},
                    ]
                )
                request = _request(case, round_number, conversation)
                prompt = build_user_prompt(request)
                self.assertIn(case["incoming"], prompt, case["name"])
                self.assertIn("마지막 user 메시지", prompt, case["name"])
                self.assertIn(case["tone"].value, prompt, case["name"])
                self.assertIn("정확히 3개", prompt, case["name"])
                self.assertIn("1. 짧고 현실적인 직접 답장", prompt, case["name"])
                self.assertIn("2. 정중하고 책임감 있는 답장", prompt, case["name"])
                self.assertIn("3. 가벼운 관계 수습 답장", prompt, case["name"])

    def test_high_similarity_is_detected(self):
        """완전히 같은 문장은 설정한 70% 유사도 기준을 넘는지 확인한다."""
        self.assertGreaterEqual(
            _max_similarity(
                "회의에 늦은 건 제 잘못입니다. 지금 바로 합류하고 놓친 내용은 정리해서 공유하겠습니다.",
                [
                    "회의에 늦은 건 제 잘못입니다. 지금 바로 합류하고 놓친 내용은 정리해서 공유하겠습니다."
                ],
            ),
            0.70,
        )

    def test_reply_is_regenerated_once_when_too_similar(self):
        """첫 본문이 currentExcuse와 같으면 한 번 재생성하고 새 결과를 반환하는지 확인한다."""
        service = _service(
            _result(
                "회의에 늦은 건 제 잘못입니다. 지금 바로 합류하고 놓친 내용은 정리해서 공유하겠습니다."
            ),
            _result(
                "사전 연락을 놓친 제 책임입니다. 지금 일정표를 다시 맞추고 지연된 부분을 끝나는 대로 보고드리겠습니다."
            ),
        )

        result = asyncio.run(service.generate(_team_lead_request(), "test-request"))

        self.assertIn("사전 연락을 놓친", result.excuse)
        self.assertEqual(service.client.generate.await_count, 2)

    def test_similar_reply_options_fail_quality_validation(self):
        """어순만 조금 다른 후보 세 개는 독립적인 선택지로 인정하지 않는지 확인한다."""
        failures = _reply_options_quality_failures(
            [
                "미리 공유하지 못해 죄송합니다.",
                "미리 공유하지 못해 정말 죄송합니다.",
                "미리 공유하지 못해 죄송해요.",
            ]
        )

        self.assertIn("REPLY_OPTIONS_TOO_SIMILAR", failures)

    def test_similar_reply_options_are_regenerated_once(self):
        """후보 간 중복도 본문 중복과 동일하게 한 번 재생성하는지 확인한다."""
        service = _service(
            _result(
                "사전 연락을 놓친 제 책임입니다. 지금 일정표를 다시 맞추고 지연된 부분을 끝나는 대로 보고드리겠습니다.",
                [
                    "미리 공유하지 못해 죄송합니다.",
                    "미리 공유하지 못해 정말 죄송합니다.",
                    "미리 공유하지 못해 죄송해요.",
                ],
            ),
            _result(
                "미리 말씀드리지 못한 점은 제 실수입니다. 회의가 끝나기 전에 일정 지연분을 정리해서 먼저 공유드리겠습니다."
            ),
        )

        result = asyncio.run(
            service.generate(_team_lead_request(), "options-retry-test")
        )

        self.assertEqual(len(result.replyOptions), 3)
        self.assertEqual(service.client.generate.await_count, 2)

    def test_too_short_reply_fails_quality_validation(self):
        """Pydantic 최소 길이를 통과해도 제품 품질 최소 길이에 못 미치면 실패하는지 확인한다."""
        failures = _reply_quality_failures(
            _team_lead_request(),
            _result("미리 공유하지 못해 죄송합니다. 바로 할게요."),
            [],
        )

        self.assertIn("REPLY_TOO_SHORT", failures)

    def test_unrelated_reply_is_regenerated_once(self):
        """incomingMessage와 무관한 범용 답장은 재생성 대상으로 분류되는지 확인한다."""
        service = _service(
            _result("오늘은 조용히 기다리겠습니다. 내일 다시 말씀드리겠습니다."),
            _result(
                "사전 연락을 놓친 제 책임입니다. 지금 일정표를 다시 맞추고 지연된 부분을 끝나는 대로 보고드리겠습니다."
            ),
        )
        request = _team_lead_request()

        result = asyncio.run(service.generate(request, "directness-test"))

        self.assertIn("사전 연락을 놓친", result.excuse)
        self.assertEqual(service.client.generate.await_count, 2)

    def test_second_quality_failure_returns_422(self):
        """두 번째 결과도 반복되면 무한 재시도하지 않고 422를 반환하는지 확인한다."""
        repeated = _result(
            "회의에 늦은 건 제 잘못입니다. 지금 바로 합류하고 놓친 내용은 정리해서 공유하겠습니다."
        )
        service = _service(repeated, repeated)

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(service.generate(_team_lead_request(), "failed-retry-test"))

        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(raised.exception.detail["code"], "REPLY_QUALITY_REPEATED")


def _team_lead_request():
    """중복·질문 반응성 테스트가 공유하는 팀장 추궁 상황의 REPLY 요청을 만든다."""
    return GenerateRequest(
        mode=GenerationMode.REPLY,
        situation="팀 회의에 20분 늦었다",
        target=Target.TEAM_LEAD,
        tone=Tone.MILD,
        currentExcuse="회의에 늦은 건 제 잘못입니다. 지금 바로 합류하고 놓친 내용은 정리해서 공유하겠습니다.",
        incomingMessage="왜 미리 공유하지 않았나요?",
        conversation=[
            {
                "role": "assistant",
                "message": "회의에 늦은 건 제 잘못입니다. 지금 바로 합류하고 놓친 내용은 정리해서 공유하겠습니다.",
            },
            {"role": "user", "message": "왜 미리 공유하지 않았나요?"},
        ],
        roundNumber=2,
    )


def _service(*results):
    """Cerebras 호출 대신 전달된 결과를 순서대로 반환하는 가짜 서비스 인스턴스를 만든다.

    실제 ``__init__``은 API 키를 요구하므로, 단위 테스트에서는 ``__new__``로 인스턴스를
    만든 뒤 필요한 client 속성만 주입한다. AsyncMock의 호출 횟수로 재생성 횟수도 검증한다.
    """
    service = ExcuseGenerationService.__new__(ExcuseGenerationService)
    service.client = SimpleNamespace(
        settings=SimpleNamespace(max_memory_chars=12000),
        generate=AsyncMock(side_effect=results),
    )
    return service


def _result(excuse, reply_options=None):
    """품질 검증에 필요한 최소한의 유효한 ``ExcuseResult`` fixture를 만든다.

    테스트가 본문·후보 품질만 바꾸기 쉽도록 나머지 Structured Output 필드는 정상적인
    기본값으로 고정한다.
    """
    return ExcuseResult(
        excuse=excuse,
        recommendedAction="바로 상황을 공유한다.",
        likelyFollowUp="언제 처리할 수 있어?",
        replyOptions=DEFAULT_REPLY_OPTIONS if reply_options is None else reply_options,
        successRate=50,
        realism=3,
        persuasion=3,
        suspicionLevel="MEDIUM",
        riskFactors=["반복 지각"],
        aftermath=[
            {
                "when": "오늘",
                "dayOffset": 0,
                "question": "처리했어?",
                "collapseRate": 20,
            }
        ],
        remember=["사실을 추가하지 않기"],
    )


if __name__ == "__main__":
    unittest.main()
