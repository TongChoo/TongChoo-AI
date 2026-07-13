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
    Target,
    Tone,
)
from app.service import ExcuseGenerationService


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
                "지금 최종 검수 중이고 10분 안에 올리겠습니다.",
                [
                    "파일 오류를 수정해서 바로 공유하겠습니다.",
                    "현재 수정 중이며 완료되는 즉시 링크를 보내겠습니다.",
                ],
            ),
        ]
    )

    generated = asyncio.run(service.generate(request(), "request-1"))

    assert generated.excuse.startswith("지금 최종 검수")
    assert service.client.generate.await_count == 2


def test_reply_quality_gate_returns_422_after_final_rejection():
    service = ExcuseGenerationService(
        Settings(
            CEREBRAS_API_KEY="test-key",
            REPLY_QUALITY_MAX_ATTEMPTS=2,
        )
    )
    repeated = result("곧 올리겠습니다.", ["곧 올리겠습니다.", "곧 올리겠습니다."])
    service.client.generate = AsyncMock(side_effect=[repeated, repeated])

    with pytest.raises(HTTPException) as error:
        asyncio.run(service.generate(request(), "request-2"))

    assert error.value.status_code == 422
    assert error.value.detail["code"] == "REPLY_QUALITY_REJECTED"


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
