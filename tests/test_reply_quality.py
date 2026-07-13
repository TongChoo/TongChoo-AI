import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.config import Settings
from app.models import (
    Aftermath,
    ConversationRole,
    ConversationTurn,
    ExcuseResult,
    GenerateRequest,
    GenerationMode,
    SuspicionLevel,
    SituationProfile,
    SituationSeverity,
    Target,
    Tone,
)
from app.service import ExcuseGenerationService, _latest_message_relevance_issue


def result(excuse: str, options: list[str]) -> ExcuseResult:
    return ExcuseResult(
        excuse=excuse,
        recommendedAction="바로 수정한다.",
        likelyFollowUp="언제 올릴 수 있어?",
        replyOptions=options,
        successRate=60,
        realism=4,
        persuasion=4,
        suspicionLevel=SuspicionLevel.MEDIUM,
        riskFactors=["시간 약속을 지켜야 한다."],
        aftermath=[
            {
                "when": "오늘",
                "dayOffset": 0,
                "question": "언제 완료돼?",
                "collapseRate": 30,
            }
        ],
        remember=["10분 안에 공유하기"],
    )


def request() -> GenerateRequest:
    return GenerateRequest(
        mode=GenerationMode.REPLY,
        situation="PPT 제출이 늦었다",
        target=Target.TEAM_LEAD,
        tone=Tone.MILD,
        situationSeverity=SituationSeverity.LIGHT,
        currentExcuse="곧 올리겠습니다.",
        incomingMessage="그래서 지금 어떻게 할 건데?",
        roundNumber=2,
        conversation=[
            ConversationTurn(
                role=ConversationRole.ASSISTANT,
                message="곧 올리겠습니다.",
            )
        ],
    )


def light_profile() -> SituationProfile:
    return SituationProfile(
        severity=SituationSeverity.LIGHT,
        formality="CASUAL",
        hasImpact=False,
        needsAccountability=False,
        needsNextAction=False,
        humorAllowed=True,
        minSentences=1,
        maxSentences=2,
        minLength=20,
        maxLength=100,
    )


def test_reply_quality_gate_regenerates_repeated_answer():
    service = ExcuseGenerationService(
        Settings(
            CEREBRAS_API_KEY="test-key",
            REPLY_QUALITY_MAX_ATTEMPTS=2,
        )
    )
    service.client.generate = AsyncMock(
        side_effect=[
            result("곧 올리겠습니다.", ["곧 올리겠습니다.", "곧 올리겠습니다."]),
            result(
                "지금 자료 상태부터 확인해서 바로 올리겠습니다.",
                [
                    "지금 PPT 상태를 확인한 뒤 바로 공유하겠습니다.",
                    "늦어진 자료부터 확인해서 이어서 올리겠습니다.",
                ],
            ),
        ]
    )
    service.client.classify_situation = AsyncMock(return_value=light_profile())

    generated = asyncio.run(service.generate(request(), "request-1"))

    assert generated.excuse.startswith("지금 자료 상태")
    assert service.client.generate.await_count == 2
    service.client.classify_situation.assert_not_awaited()


def test_reply_quality_gate_returns_422_after_final_rejection():
    service = ExcuseGenerationService(
        Settings(
            CEREBRAS_API_KEY="test-key",
            REPLY_QUALITY_MAX_ATTEMPTS=2,
        )
    )
    repeated = result("곧 올리겠습니다.", ["곧 올리겠습니다.", "곧 올리겠습니다."])
    service.client.generate = AsyncMock(side_effect=[repeated, repeated])
    service.client.classify_situation = AsyncMock(return_value=light_profile())

    with pytest.raises(HTTPException) as error:
        asyncio.run(service.generate(request(), "request-2"))

    assert error.value.status_code == 422
    assert error.value.detail["code"] == "REPLY_QUALITY_REJECTED"
    service.client.classify_situation.assert_not_awaited()


def test_reason_question_accepts_natural_korean_cause_expressions():
    assert _latest_message_relevance_issue(
        "왜 안 지켰어 시간을",
        "어젯밤에 잠들어 제출 마감 시간을 놓쳤습니다.",
    ) is None
    assert _latest_message_relevance_issue(
        "왜 제출하지 않았나요?",
        "제 시간 관리가 부족했습니다.",
    ) is None


def test_reply_returns_best_safe_candidate_after_relevance_false_positive():
    service = ExcuseGenerationService(
        Settings(
            CEREBRAS_API_KEY="test-key",
            REPLY_QUALITY_MAX_ATTEMPTS=2,
        )
    )
    first = result(
        "일정 관리가 미흡했습니다. 제 책임입니다.",
        [
            "일정을 제대로 챙기지 못한 제 책임입니다.",
            "제 판단이 부족했습니다. 변명하지 않겠습니다.",
        ],
    )
    second = result(
        "시간 관리가 미흡했습니다. 제 책임입니다.",
        [
            "일정 확인이 미흡했던 제 책임입니다.",
            "제 판단이 부족했습니다. 핑계 대지 않겠습니다.",
        ],
    )
    service.client.generate = AsyncMock(side_effect=[first, second])

    generated = asyncio.run(service.generate(request(), "request-best-effort"))

    assert generated.excuse in {first.excuse, second.excuse}
    assert service.client.generate.await_count == 2


def test_aftermath_rejects_generic_work_follow_up_questions():
    service = ExcuseGenerationService(Settings(CEREBRAS_API_KEY="test-key"))
    generated = result(
        "집에 정전이 나서 알람이 꺼졌습니다.",
        ["정전 때문에 늦었습니다.", "알람이 꺼져 늦었습니다."],
    ).model_copy(
        update={
            "aftermath": [
                Aftermath(
                    when="당일",
                    dayOffset=0,
                    question="회의 핵심 내용은 무엇인가요?",
                    collapseRate=20,
                ),
                Aftermath(
                    when="다음 날",
                    dayOffset=1,
                    question="다음 회의 일정을 재조정해야 할까요?",
                    collapseRate=15,
                ),
            ]
        }
    )

    issues = service._aftermath_quality_issues(generated)

    assert "후폭풍 질문이 현재 변명의 주장이나 허점과 연결되지 않음" in issues


def test_aftermath_accepts_questions_that_probe_the_excuse_claim():
    service = ExcuseGenerationService(Settings(CEREBRAS_API_KEY="test-key"))
    generated = result(
        "집에 정전이 나서 알람이 꺼졌습니다.",
        ["정전 때문에 늦었습니다.", "알람이 꺼져 늦었습니다."],
    ).model_copy(
        update={
            "aftermath": [
                Aftermath(
                    when="당일",
                    dayOffset=0,
                    question="정전은 몇 시에 복구됐어?",
                    collapseRate=55,
                ),
                Aftermath(
                    when="다음 날",
                    dayOffset=1,
                    question="정전 안내 문자나 관리실 공지 있어?",
                    collapseRate=70,
                ),
            ]
        }
    )

    assert service._aftermath_quality_issues(generated) == []


def test_light_aftermath_allows_simple_direct_question_without_claim_anchor():
    service = ExcuseGenerationService(Settings(CEREBRAS_API_KEY="test-key"))
    generated = result(
        "커피 사 오는 걸 깜빡했어. 지금 바로 사올게!",
        ["미안, 지금 사올게!", "커피 깜빡했다, 바로 갈게!"],
    ).model_copy(
        update={
            "aftermath": [
                Aftermath(
                    when="지금",
                    dayOffset=0,
                    question="그래서 어떻게 할 거야?",
                    collapseRate=10,
                )
            ]
        }
    )

    assert service._aftermath_quality_issues(
        generated, SituationSeverity.LIGHT
    ) == []
