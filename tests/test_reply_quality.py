import asyncio
from unittest.mock import AsyncMock

import pytest

from app.config import Settings
from app.llm import ReplyJudgeParseError
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
from app.reply_quality import ReplyCandidateJudge, ReplyQualityVerdict
from app.service import ExcuseGenerationService


def result(excuse: str, options: list[str]) -> ExcuseResult:
    return ExcuseResult(
        excuse=excuse,
        recommendedAction="바로 확인한다.",
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


def valid_verdict() -> ReplyQualityVerdict:
    return ReplyQualityVerdict(
        candidateScores=[
            ReplyCandidateJudge(
                directness=36,
                factuality=28,
                registerScore=14,
                fluency=14,
            )
            for _ in range(3)
        ],
        diversityScore=88,
        semanticDuplicate=False,
    )


def method_request() -> GenerateRequest:
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


def privacy_request() -> GenerateRequest:
    return GenerateRequest(
        mode=GenerationMode.REPLY,
        situation="회식 참석이 어렵습니다.",
        target=Target.CUSTOM,
        targetDescription="회사 부장님",
        tone=Tone.MILD,
        rootExcuse="개인 사정이 있어 회식 참석이 어렵습니다.",
        currentExcuse="개인 사정이 있어 회식 참석이 어렵습니다.",
        incomingMessage="개인 사정이 뭔가요?",
        roundNumber=2,
    )


def test_reply_quality_gate_regenerates_repeated_answer() -> None:
    service = ExcuseGenerationService(
        Settings(CEREBRAS_API_KEY="test-key", REPLY_QUALITY_MAX_ATTEMPTS=2)
    )
    service.client.generate = AsyncMock(
        side_effect=[
            result(
                "곧 올리겠습니다.",
                ["곧 올리겠습니다.", "곧 올리겠습니다.", "곧 올리겠습니다."],
            ),
            result(
                "기본 문장과 달라도 첫 후보가 기준이 됩니다.",
                [
                    "지금 일정 영향부터 확인하겠습니다.",
                    "미리 공유하지 못해 죄송합니다. 확인되는 대로 정리해 말씀드리겠습니다.",
                    "필요한 부분부터 정리해서 바로 공유드리겠습니다.",
                ],
            ),
        ]
    )
    service.client.judge_reply = AsyncMock(return_value=valid_verdict())

    generated = asyncio.run(service.generate(method_request(), "request-1"))

    assert generated.excuse == "지금 일정 영향부터 확인하겠습니다."
    assert generated.replyOptions[0] == generated.excuse
    assert service.client.generate.await_count == 2
    assert service.client.judge_reply.await_count == 1


def test_reply_quality_gate_returns_safe_fallback_after_final_rejection() -> None:
    service = ExcuseGenerationService(
        Settings(CEREBRAS_API_KEY="test-key", REPLY_QUALITY_MAX_ATTEMPTS=2)
    )
    repeated = result(
        "곧 올리겠습니다.",
        ["곧 올리겠습니다.", "곧 올리겠습니다.", "곧 올리겠습니다."],
    )
    service.client.generate = AsyncMock(side_effect=[repeated, repeated])
    service.client.judge_reply = AsyncMock()

    generated = asyncio.run(service.generate(method_request(), "request-2"))

    assert len(generated.replyOptions) == 3
    assert len(set(generated.replyOptions)) == 3
    assert generated.excuse == generated.replyOptions[0]
    assert service.client.judge_reply.await_count == 0


def test_detail_question_without_fact_requires_polite_privacy_boundary() -> None:
    service = ExcuseGenerationService(Settings(CEREBRAS_API_KEY="test-key"))
    generated = result(
        "개인 사정이라서요.",
        [
            "개인 사정이라서요.",
            "비밀이라서요 😉",
            "개인적인 이유가 있습니다.",
        ],
    )

    issues = service.validator.reply_issues(generated, privacy_request())

    assert "근거 없는 상세 요구에 정중한 공개 거절로 직접 답하지 않음" in issues
    assert "공식 관계 답장에 농담 또는 가벼운 회피가 포함됨" in issues
    assert "공식 관계 답장에 이모지가 포함됨" in issues


def test_formal_privacy_reply_passes_with_three_direct_options() -> None:
    service = ExcuseGenerationService(
        Settings(CEREBRAS_API_KEY="test-key", REPLY_QUALITY_MAX_ATTEMPTS=2)
    )
    service.client.generate = AsyncMock(
        return_value=result(
            "원래 기본 문장",
            [
                "개인적인 부분이라 자세히 말씀드리기 어렵습니다. 양해 부탁드립니다.",
                "사적인 사유라 구체적으로 말씀드리기 어려운 점 이해 부탁드립니다.",
                "이번 회식 참석은 어렵습니다. 개인적인 부분은 자세히 말씀드리기 어려운 점 양해 부탁드립니다.",
            ],
        )
    )
    service.client.judge_reply = AsyncMock(return_value=valid_verdict())

    generated = asyncio.run(service.generate(privacy_request(), "request-3"))

    assert generated.excuse == generated.replyOptions[0]
    assert "자세히 말씀드리기 어렵습니다" in generated.excuse
    assert service.client.judge_reply.await_count == 1


def test_privacy_reply_uses_safe_fallback_instead_of_returning_422() -> None:
    """Cerebras가 상세 질문을 두 번 무시해도 사용자에게 서버 오류를 내지 않는다."""
    service = ExcuseGenerationService(
        Settings(CEREBRAS_API_KEY="test-key", REPLY_QUALITY_MAX_ATTEMPTS=2)
    )
    ignored_question = result(
        "개인 사정으로 오늘 회식 참석이 어렵습니다.",
        [
            "개인 사정으로 오늘 회식 참석이 어렵습니다.",
            "개인 사정이 있어 오늘 회식에 참여하지 못하겠습니다.",
            "개인 사정이라 참석이 힘듭니다.",
        ],
    )
    service.client.generate = AsyncMock(
        side_effect=[ignored_question, ignored_question]
    )
    service.client.judge_reply = AsyncMock()

    generated = asyncio.run(
        service.generate(privacy_request(), "privacy-fallback")
    )

    assert generated.excuse == generated.replyOptions[0]
    assert "자세히 말씀드리기 어렵습니다" in generated.excuse
    assert len(generated.replyOptions) == 3
    assert len(set(generated.replyOptions)) == 3
    assert all("😉" not in option for option in generated.replyOptions)
    assert service.client.generate.await_count == 2
    assert service.client.judge_reply.await_count == 0


def test_casual_privacy_fallback_also_passes_deterministic_checks() -> None:
    casual_request = privacy_request().model_copy(
        update={
            "target": Target.FRIEND,
            "targetDescription": None,
        }
    )
    service = ExcuseGenerationService(
        Settings(CEREBRAS_API_KEY="test-key", REPLY_QUALITY_MAX_ATTEMPTS=1)
    )
    ignored_question = result(
        "개인 사정이야.",
        ["개인 사정이야.", "비밀이야.", "그냥 참석이 어려워."],
    )
    service.client.generate = AsyncMock(return_value=ignored_question)
    service.client.judge_reply = AsyncMock()

    generated = asyncio.run(
        service.generate(casual_request, "casual-privacy-fallback")
    )

    assert generated.excuse == generated.replyOptions[0]
    assert service.validator.reply_issues(generated, casual_request) == []


def test_judge_semantic_duplicate_retries_then_returns_distinct_candidates() -> None:
    service = ExcuseGenerationService(
        Settings(CEREBRAS_API_KEY="test-key", REPLY_QUALITY_MAX_ATTEMPTS=2)
    )
    valid = result(
        "기본",
        [
            "지금 일정 영향부터 확인하겠습니다.",
            "미리 공유하지 못해 죄송합니다. 확인되는 대로 정리해 말씀드리겠습니다.",
            "필요한 부분부터 정리해서 바로 공유드리겠습니다.",
        ],
    )
    duplicate_verdict = valid_verdict().model_copy(
        update={"diversityScore": 35, "semanticDuplicate": True}
    )
    service.client.generate = AsyncMock(side_effect=[valid, valid])
    service.client.judge_reply = AsyncMock(
        side_effect=[duplicate_verdict, valid_verdict()]
    )

    generated = asyncio.run(service.generate(method_request(), "request-4"))

    assert generated.replyOptions[0] == generated.excuse
    assert service.client.generate.await_count == 2
    assert service.client.judge_reply.await_count == 2


def test_privacy_boundary_candidates_allow_constrained_diversity() -> None:
    """공개 거절은 결론이 같아도 문장 역할이 다르면 정상 후보로 취급한다."""
    service = ExcuseGenerationService(Settings(CEREBRAS_API_KEY="test-key"))
    constrained_verdict = ReplyQualityVerdict(
        candidateScores=[
            ReplyCandidateJudge(
                directness=28,
                factuality=27,
                registerScore=12,
                fluency=10,
            )
            for _ in range(3)
        ],
        diversityScore=45,
        semanticDuplicate=True,
        issues=["같은 공개 거절 결론을 공유함"],
    )

    assert service.validator.reply_judge_issues(constrained_verdict, privacy_request()) == []


def test_privacy_boundary_reply_is_not_rejected_for_judge_duplicate_flag() -> None:
    """2라운드 공개 거절은 같은 결론이라는 이유만으로 422가 되지 않는다."""
    service = ExcuseGenerationService(Settings(CEREBRAS_API_KEY="test-key"))
    service.client.generate = AsyncMock(
        return_value=result(
            "기본",
            [
                "개인적인 부분이라 자세히 말씀드리기 어렵습니다. 양해 부탁드립니다.",
                "사적인 사유라 구체적으로 말씀드리기 어려운 점 이해 부탁드립니다.",
                "이번 회식 참석은 어렵습니다. 개인적인 부분은 자세히 말씀드리기 어렵습니다.",
            ],
        )
    )
    service.client.judge_reply = AsyncMock(
        return_value=ReplyQualityVerdict(
            candidateScores=[
                ReplyCandidateJudge(
                    directness=28,
                    factuality=27,
                    registerScore=12,
                    fluency=10,
                )
                for _ in range(3)
            ],
            diversityScore=45,
            semanticDuplicate=True,
        )
    )

    generated = asyncio.run(service.generate(privacy_request(), "request-privacy"))

    assert generated.excuse == generated.replyOptions[0]
    assert service.client.generate.await_count == 1


def test_incomplete_judge_verdict_does_not_block_safe_reply() -> None:
    """느슨한 구조화 출력에서 Judge 필드가 빠져도 안전한 본문은 반환한다."""
    service = ExcuseGenerationService(Settings(CEREBRAS_API_KEY="test-key"))
    generated_result = result(
        "기본",
        [
            "지금 일정 영향부터 확인하겠습니다.",
            "미리 공유하지 못해 죄송합니다. 확인되는 대로 정리해 말씀드리겠습니다.",
            "필요한 부분부터 정리해서 바로 공유드리겠습니다.",
        ],
    )
    service.client.generate = AsyncMock(return_value=generated_result)
    service.client.judge_reply = AsyncMock(
        return_value=ReplyQualityVerdict(candidateScores=[])
    )

    generated = asyncio.run(service.generate(method_request(), "request-incomplete"))

    assert generated.excuse == generated.replyOptions[0]
    assert service.client.generate.await_count == 1


def test_judge_parse_failure_falls_open_after_deterministic_checks() -> None:
    service = ExcuseGenerationService(Settings(CEREBRAS_API_KEY="test-key"))
    generated_result = result(
        "기본",
        [
            "지금 일정 영향부터 확인하겠습니다.",
            "미리 공유하지 못해 죄송합니다. 확인되는 대로 정리해 말씀드리겠습니다.",
            "필요한 부분부터 정리해서 바로 공유드리겠습니다.",
        ],
    )
    service.client.generate = AsyncMock(return_value=generated_result)
    service.client.judge_reply = AsyncMock(side_effect=ReplyJudgeParseError())

    generated = asyncio.run(service.generate(method_request(), "request-parse"))

    assert generated.excuse == generated.replyOptions[0]
    assert service.client.generate.await_count == 1


def test_complete_judge_hard_violation_still_rejects() -> None:
    service = ExcuseGenerationService(Settings(CEREBRAS_API_KEY="test-key"))
    verdict = valid_verdict().model_copy(
        update={
            "candidateScores": [
                ReplyCandidateJudge(
                    directness=36,
                    factuality=28,
                    registerScore=14,
                    fluency=14,
                    hardViolation=True,
                ),
                ReplyCandidateJudge(
                    directness=36,
                    factuality=28,
                    registerScore=14,
                    fluency=14,
                ),
                ReplyCandidateJudge(
                    directness=36,
                    factuality=28,
                    registerScore=14,
                    fluency=14,
                ),
            ]
        }
    )

    assert "1번 후보에 금지된 품질 문제가 있음" in service.validator.reply_judge_issues(
        verdict,
        method_request(),
    )


def test_invented_fact_and_broken_particle_are_rejected() -> None:
    service = ExcuseGenerationService(Settings(CEREBRAS_API_KEY="test-key"))
    generated = result(
        "은 회식 참석이 어렵습니다.",
        [
            "은 회식 참석이 어렵습니다.",
            "몸살이 나서 참석이 어렵습니다.",
            "개인적인 부분이라 자세히 말씀드리기 어렵습니다. 양해 부탁드립니다.",
        ],
    )

    issues = service.validator.reply_issues(generated, privacy_request())

    assert "조사만 남은 깨진 문장" in issues
    assert "입력에 없는 구체적 사실을 새로 만듦" in issues


def test_aftermath_rejects_generic_work_follow_up_questions() -> None:
    service = ExcuseGenerationService(Settings(CEREBRAS_API_KEY="test-key"))
    generated = result(
        "집에 정전이 나서 알람이 꺼졌습니다.",
        ["정전 때문에 늦었습니다.", "알람이 꺼져 늦었습니다.", "정전 상황을 확인 중입니다."],
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

    issues = service.validator.aftermath_issues(generated)

    assert "후폭풍 질문이 현재 변명의 주장이나 허점과 연결되지 않음" in issues


def test_aftermath_accepts_questions_that_probe_the_excuse_claim() -> None:
    service = ExcuseGenerationService(Settings(CEREBRAS_API_KEY="test-key"))
    generated = result(
        "집에 정전이 나서 알람이 꺼졌습니다.",
        ["정전 때문에 늦었습니다.", "알람이 꺼져 늦었습니다.", "정전 상황을 확인 중입니다."],
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

    assert service.validator.aftermath_issues(generated) == []


def test_create_rejects_generic_risk_and_empty_memory() -> None:
    service = ExcuseGenerationService(Settings(CEREBRAS_API_KEY="test-key"))
    generated = result(
        "해당 부분은 아직 숙지하지 못했습니다. 확인 후 답변드리겠습니다.",
        [
            "해당 부분은 아직 숙지하지 못했습니다. 확인 후 답변드리겠습니다.",
            "정확히 알지 못하는 내용이라 확인하겠습니다.",
            "모르는 부분은 확인한 뒤 말씀드리겠습니다.",
        ],
    ).model_copy(update={"riskFactors": ["정보 부족"], "remember": []})

    assert service.validator.create_auxiliary_issues(generated) == [
        "위험 요소가 지나치게 포괄적임",
        "기억해야 할 설정이 비어 있음",
    ]


def test_aftermath_rejects_question_linked_only_by_generic_word() -> None:
    service = ExcuseGenerationService(Settings(CEREBRAS_API_KEY="test-key"))
    generated = result(
        "해당 부분은 아직 숙지하지 못했습니다. 확인 후 답변드리겠습니다.",
        [
            "해당 부분은 아직 숙지하지 못했습니다. 확인 후 답변드리겠습니다.",
            "정확히 알지 못하는 내용이라 확인하겠습니다.",
            "모르는 부분은 확인한 뒤 말씀드리겠습니다.",
        ],
    ).model_copy(update={
        "aftermath": [Aftermath(
            when="당일",
            dayOffset=0,
            question="어떤 부분이 정확히 궁금하신가요?",
            collapseRate=20,
        )]
    })

    assert "후폭풍 질문이 현재 변명의 주장이나 허점과 연결되지 않음" in (
        service.validator.aftermath_issues(generated)
    )
