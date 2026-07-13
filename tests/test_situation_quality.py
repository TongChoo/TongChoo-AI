"""상황 심각도별 생성 규칙과 품질 재생성 테스트."""

import asyncio
from unittest.mock import AsyncMock

from app.config import Settings
from app.models import (
    ExcuseResult,
    GenerateRequest,
    GenerationMode,
    SituationProfile,
    SituationSeverity,
    SuspicionLevel,
    Target,
    Tone,
)
from app.service import (
    ExcuseGenerationService,
    apply_severity_guardrails,
    count_sentences,
    sanitize_unsupported_time_promises,
    validate_grounding,
    validate_situation_fit,
)
from app.prompts import build_user_prompt


def profile(severity: SituationSeverity) -> SituationProfile:
    ranges = {
        SituationSeverity.LIGHT: (1, 2, 20, 100),
        SituationSeverity.NORMAL: (2, 3, 60, 180),
        SituationSeverity.SERIOUS: (3, 5, 120, 350),
    }
    minimum, maximum, min_length, max_length = ranges[severity]
    return SituationProfile(
        severity=severity,
        formality="CASUAL" if severity == SituationSeverity.LIGHT else "FORMAL",
        hasImpact=severity != SituationSeverity.LIGHT,
        needsAccountability=severity != SituationSeverity.LIGHT,
        needsNextAction=severity != SituationSeverity.LIGHT,
        humorAllowed=severity == SituationSeverity.LIGHT,
        minSentences=minimum,
        maxSentences=maximum,
        minLength=min_length,
        maxLength=max_length,
    )


def result(excuse: str) -> ExcuseResult:
    return ExcuseResult(
        excuse=excuse,
        recommendedAction="바로 확인한다.",
        likelyFollowUp="그래서 지금 어떻게 할 건데?",
        replyOptions=["지금 확인하겠습니다.", "내용을 정리해 공유하겠습니다."],
        successRate=60,
        realism=4,
        persuasion=4,
        suspicionLevel=SuspicionLevel.MEDIUM,
        riskFactors=["일정 영향"],
        aftermath=[
            {
                "when": "오늘",
                "dayOffset": 0,
                "question": "그럼 언제 확인할 수 있어?",
                "collapseRate": 30,
            }
        ],
        remember=["다음 행동 공유"],
    )


def test_sentence_counter_handles_korean_endings_and_line_breaks() -> None:
    assert count_sentences("지금 확인할게요\n정리해서 바로 공유하겠습니다") == 2
    assert count_sentences("시간을 잘못 봤다ㅋㅋ 지금 뛰어가는 중") == 1


def test_serious_result_rejects_short_answer_without_next_action() -> None:
    issues = validate_situation_fit(
        result("제가 마감을 놓쳤습니다. 죄송합니다."),
        profile(SituationSeverity.SERIOUS),
    )

    assert "상황 심각도에 비해 답변이 너무 짧습니다." in issues
    assert "필요한 설명과 수습 행동이 부족합니다." in issues
    assert "상대가 확인할 수 있는 다음 행동이 없습니다." in issues


def test_serious_bullshit_prompt_removes_jokes_but_keeps_tone_trace() -> None:
    request = GenerateRequest(
        situation="고객사 최종 발표자료 제출을 놓쳤다",
        target=Target.TEAM_LEAD,
        tone=Tone.BULLSHIT,
    )

    prompt_text = build_user_prompt(request, profile=profile(SituationSeverity.SERIOUS))

    assert "심각도: SERIOUS" in prompt_text
    assert "유머 허용: False" in prompt_text
    assert "BULLSHIT 톤이어도 유머 대신 표현만 조금 부드럽게" in prompt_text


def test_grounding_rejects_invented_cause_and_time_promise() -> None:
    request = GenerateRequest(
        situation="팀장님과 점심 약속 장소에 5분 늦게 도착했다",
        target=Target.TEAM_LEAD,
        tone=Tone.MILD,
    )

    issues = validate_grounding(
        result("교통이 막혀서 늦었습니다. 오늘 안으로 다시 연락하겠습니다."),
        request,
    )

    assert "입력에 없는 원인이나 사건이 포함되어 있습니다." in issues
    assert "입력에 없는 구체적인 시간 약속이 포함되어 있습니다." in issues


def test_grounding_allows_time_already_present_in_input() -> None:
    request = GenerateRequest(
        situation="친구와 게임 약속에 10분 늦었다",
        target=Target.FRIEND,
        tone=Tone.MILD,
    )

    assert validate_grounding(result("미안, 10분 늦었어. 지금 들어갈게."), request) == []


def test_length_gate_allows_approximately_ten_percent_tolerance() -> None:
    normal = profile(SituationSeverity.NORMAL).model_copy(
        update={"needsNextAction": False}
    )
    text = "가" * 54 + ". 나왔던 상황을 인정합니다."

    assert "상황 심각도에 비해 답변이 너무 짧습니다." not in validate_situation_fit(
        result(text), normal
    )


def test_next_action_accepts_honorific_messenger_expression() -> None:
    normal = profile(SituationSeverity.NORMAL)
    text = (
        "자료 공유가 늦어진 점 죄송합니다. 현재 상태를 확인하고 "
        "완성된 부분부터 바로 공유드리겠습니다."
    )

    assert "상대가 확인할 수 있는 다음 행동이 없습니다." not in validate_situation_fit(
        result(text), normal
    )


def test_next_action_accepts_immediate_casual_expression() -> None:
    light = profile(SituationSeverity.LIGHT)

    assert "상대가 확인할 수 있는 다음 행동이 없습니다." not in validate_situation_fit(
        result("커피 깜빡해서 미안, 지금 바로 사 와서 줄게!"), light
    )


def test_grounding_rejects_unstated_concrete_impact() -> None:
    request = GenerateRequest(
        situation="안전 장비 점검을 누락한 채 행사를 진행했다",
        target=Target.TEAM_LEAD,
        tone=Tone.MILD,
    )

    issues = validate_grounding(
        result("점검을 누락해 현장에 불안감이 발생했습니다. 지금 확인하겠습니다."),
        request,
    )

    assert "입력에 없는 구체적인 영향이 실제 발생한 것처럼 표현되었습니다." in issues


def test_time_sanitizer_removes_only_unsupported_time_promise() -> None:
    request = GenerateRequest(
        situation="내 작업을 마감하지 못해서 다른 팀원의 작업까지 막혔다",
        target=Target.TEAM_MEMBER,
        tone=Tone.MILD,
    )
    generated = result("오늘 작업을 확인하고 내일까지 공유하겠습니다.")

    sanitized = sanitize_unsupported_time_promises(generated, request)

    assert sanitized.excuse == "작업을 확인하고 공유하겠습니다."
    assert validate_grounding(sanitized, request) == []


def test_time_sanitizer_keeps_time_from_original_situation() -> None:
    request = GenerateRequest(
        situation="친구와 게임 약속에 10분 늦었다",
        target=Target.FRIEND,
        tone=Tone.MILD,
    )

    sanitized = sanitize_unsupported_time_promises(
        result("10분 늦어서 미안해. 지금 들어갈게."), request
    )

    assert "10분" in sanitized.excuse


def test_guardrail_promotes_missed_final_interview_to_serious() -> None:
    normal = profile(SituationSeverity.NORMAL)
    request = GenerateRequest(
        situation="중요한 최종 면접 일정을 잊어서 참석하지 못했다",
        target=Target.CUSTOM,
        targetDescription="채용 담당자",
        tone=Tone.MILD,
    )

    guarded = apply_severity_guardrails(normal, request)

    assert guarded.severity == SituationSeverity.SERIOUS
    assert guarded.humorAllowed is False
    assert (guarded.minSentences, guarded.maxSentences) == (3, 5)


def test_guardrail_does_not_promote_short_team_lead_lunch_delay() -> None:
    light = profile(SituationSeverity.LIGHT)
    request = GenerateRequest(
        situation="팀장님과 점심 약속 장소에 5분 늦었다",
        target=Target.TEAM_LEAD,
        tone=Tone.MILD,
    )

    assert apply_severity_guardrails(light, request).severity == SituationSeverity.LIGHT


def test_guardrail_promotes_missed_assignment_deadline_to_normal() -> None:
    light = profile(SituationSeverity.LIGHT)
    request = GenerateRequest(
        situation="영어 수행평가 제출 시간이 어젯밤 12시까지였는데 잠들어서 제출을 못했어",
        target=Target.TEACHER,
        tone=Tone.SLICK,
    )

    guarded = apply_severity_guardrails(light, request)

    assert guarded.severity == SituationSeverity.NORMAL
    assert guarded.humorAllowed is False
    assert (guarded.minSentences, guarded.maxSentences) == (2, 3)


def test_create_regenerates_with_specific_situation_fit_feedback() -> None:
    service = ExcuseGenerationService(
        Settings(CEREBRAS_API_KEY="test-key", SITUATION_QUALITY_MAX_ATTEMPTS=2)
    )
    normal = profile(SituationSeverity.NORMAL)
    service.client.classify_situation = AsyncMock(return_value=normal)
    service.client.generate = AsyncMock(
        side_effect=[
            result("늦어서 죄송합니다."),
            result(
                "회의 시간을 잘못 확인해서 늦었습니다. 미리 알리지 못한 점 죄송하고, "
                "지금 바로 들어가 놓친 내용부터 확인하겠습니다."
            ),
        ]
    )
    request = GenerateRequest(
        situation="동아리 회의 시간을 착각해서 30분 늦었다",
        target=Target.TEAM_MEMBER,
        tone=Tone.MILD,
    )

    generated = asyncio.run(service.generate(request, "situation-test"))

    assert generated.excuse.startswith("회의 시간을 잘못 확인")
    assert service.client.generate.await_count == 2
    retry_prompt = service.client.generate.await_args_list[1].args[1]
    assert "상황 심각도에 비해 답변이 너무 짧습니다." in retry_prompt
    assert "상대가 확인할 수 있는 다음 행동이 없습니다." in retry_prompt


def test_spring_create_response_contains_classified_severity() -> None:
    service = ExcuseGenerationService(Settings(CEREBRAS_API_KEY="test-key"))
    serious = profile(SituationSeverity.SERIOUS)
    service.client.classify_situation = AsyncMock(return_value=serious)
    service.client.generate = AsyncMock(
        return_value=result(
            "최종 발표자료를 마감까지 전달하지 못한 점 죄송합니다. 고객사 검토에 "
            "영향을 줄 수 있는 상황을 만든 제 책임입니다. 현재 자료 상태를 확인하고 "
            "완성된 부분부터 공유하겠습니다. 같은 누락이 없도록 검토 절차를 점검하겠습니다."
        )
    )
    request = GenerateRequest(
        situation="고객사 최종 발표자료 제출을 놓쳤다",
        target=Target.TEAM_LEAD,
        tone=Tone.MILD,
    )

    response = asyncio.run(service.generate_for_spring(request, "create-contract"))

    assert response.situationSeverity == SituationSeverity.SERIOUS


def test_reply_profile_keeps_severity_but_uses_shorter_range() -> None:
    reply_profile = profile(SituationSeverity.SERIOUS).for_mode(GenerationMode.REPLY)

    assert reply_profile.severity == SituationSeverity.SERIOUS
    assert (reply_profile.minSentences, reply_profile.maxSentences) == (2, 4)
    assert (reply_profile.minLength, reply_profile.maxLength) == (60, 260)
