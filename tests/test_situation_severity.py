"""심각도별 길이·사과와 장례식 회귀 품질 검사."""

from app.config import Settings
from app.models import (
    ExcuseResult,
    GenerateRequest,
    SituationProfile,
    SituationSeverity,
    SuspicionLevel,
    Target,
    Tone,
)
from app.reply_quality import fluency_issues, register_consistency_issues
from app.service import (
    ExcuseGenerationService,
    _apply_situation_guardrails,
    _accountability_aftermath_fallback,
    validate_grounding_and_register,
    validate_situation_fit,
)


def test_simple_five_minute_delay_is_corrected_to_light() -> None:
    source_request = GenerateRequest(
        situation="친구와 약속에 5분 지각했어",
        target=Target.FRIEND,
        tone=Tone.MILD,
    )
    corrected = _apply_situation_guardrails(normal_profile(), source_request)

    assert corrected.severity == SituationSeverity.LIGHT
    assert corrected.hasImpact is False
    assert corrected.needsAccountability is False
    assert corrected.needsNextAction is False


def test_high_stakes_short_delay_is_not_corrected_to_light() -> None:
    source_request = GenerateRequest(
        situation="고객 발표에 5분 지각했어",
        target=Target.TEAM_LEAD,
        tone=Tone.MILD,
    )
    corrected = _apply_situation_guardrails(normal_profile(), source_request)

    assert corrected.severity == SituationSeverity.NORMAL


def result(excuse: str, question: str = "주변 분들께도 사과했니?") -> ExcuseResult:
    return ExcuseResult(
        excuse=excuse,
        recommendedAction="행동을 멈추고 사과한다.",
        likelyFollowUp=question,
        replyOptions=[
            excuse,
            "엄마, 내가 예의 없이 행동해서 정말 미안해. 지금부터 조용히 하고 주변 분들께도 사과할게.",
            "장례식장에서 소란을 피운 건 내 잘못이야. 미안해. 바로 행동을 멈추고 예의를 지킬게.",
        ],
        successRate=60,
        realism=4,
        persuasion=4,
        suspicionLevel=SuspicionLevel.MEDIUM,
        riskFactors=["장례식 예절 위반"],
        aftermath=[{
            "when": "즉시",
            "dayOffset": 0,
            "question": question,
            "collapseRate": 30,
        }],
        remember=["장례식에서는 조용히 행동하기"],
    )


def request() -> GenerateRequest:
    return GenerateRequest(
        situation="할머니 장례식장에서 뛰어다니고 소리를 질러서 엄마가 화가 났어",
        target=Target.PARENT,
        tone=Tone.MILD,
    )


def normal_profile() -> SituationProfile:
    return SituationProfile(
        severity=SituationSeverity.NORMAL,
        formality="CASUAL",
        hasImpact=True,
        needsAccountability=True,
        needsNextAction=True,
        humorAllowed=False,
        minSentences=2,
        maxSentences=3,
        minLength=60,
        maxLength=180,
    )


def test_mixed_casual_and_polite_endings_are_rejected() -> None:
    issues = register_consistency_issues(
        "엄마, 미안해. 장례식에서 소리를 질렀어. 지금은 조용히 하고 있어요."
    )

    assert "한 답장 안에서 반말과 존댓말이 섞임" in issues


def test_invented_surprise_reason_is_rejected() -> None:
    generated = result(
        "엄마, 미안해. 장례식에서 갑자기 놀라서 무심코 소리를 질렀어. 지금은 조용히 할게."
    )

    assert "입력에 없는 원인이나 사건을 새로 만들었습니다." in (
        validate_grounding_and_register(generated, request())
    )


def test_forensic_aftermath_question_is_rejected() -> None:
    service = ExcuseGenerationService(Settings(CEREBRAS_API_KEY="test-key"))
    generated = result(
        "엄마, 장례식장에서 소란을 피운 건 내 잘못이야. 정말 미안해. 지금부터 조용히 하고 주변 분들께도 사과할게.",
        "뛰고 소리 낸 순간이 정확히 몇 초였는지 알려줄 수 있어?",
    )

    assert "후폭풍 질문이 실제 대화보다 지나치게 수사식이거나 부자연스러움" in (
        service.validator.aftermath_issues(generated)
    )


def test_normal_wrongdoing_rejects_one_line_without_apology_and_repair() -> None:
    issues = validate_situation_fit(
        result("장례식장에서 소리를 질렀어."),
        normal_profile(),
    )

    assert "상황 심각도에 비해 답변이 너무 짧습니다." in issues
    assert "잘못을 한 상황인데 상대에게 직접 사과하지 않았습니다." in issues
    assert "잘못을 수습하기 위한 다음 행동이 없습니다." in issues


def test_normal_wrongdoing_accepts_apology_and_immediate_repair() -> None:
    generated = result(
        "엄마, 장례식장에서 소란을 피운 건 내 잘못이야. 정말 미안해. 지금부터 조용히 하고 주변 분들께도 사과할게."
    )

    assert validate_situation_fit(generated, normal_profile()) == []
    assert validate_grounding_and_register(generated, request()) == []


def test_accountability_aftermath_fallback_keeps_valid_model_type() -> None:
    service = ExcuseGenerationService(Settings(CEREBRAS_API_KEY="test-key"))
    generated = result(
        "엄마, 장례식장에서 소란을 피운 건 내 잘못이야. 정말 미안해. 지금부터 조용히 하고 주변 분들께도 사과할게.",
        "뛰고 소리 낸 순간이 정확히 몇 초였어?",
    )

    fallback = _accountability_aftermath_fallback(generated, request())

    assert fallback.aftermath[0].question == (
        "그럼 지금 말한 대로 바로 행동을 고치고 수습할 수 있겠어?"
    )
    assert fallback.excuse == generated.excuse
    assert service.validator.aftermath_issues(fallback, request()) == []


def test_double_negative_broken_sentence_is_rejected() -> None:
    assert "부정 표현이 중복된 깨진 문장" in fluency_issues(
        "다른 사람들 방해 안 않게 할게."
    )


def test_unstated_time_and_work_state_are_rejected_without_text_deletion() -> None:
    serious_request = GenerateRequest(
        situation="고객사 최종 발표자료 제출을 놓쳐 고객 일정에 차질이 생겼다",
        target=Target.TEAM_LEAD,
        tone=Tone.MILD,
    )
    generated = result(
        "제출을 놓쳐 죄송합니다. 수정된 자료를 내일 오전 9시까지 전달하겠습니다. 재발 방지 절차를 점검하겠습니다."
    )

    issues = validate_grounding_and_register(generated, serious_request)

    assert "입력에 없는 작업 상태를 완료했거나 확정한 것처럼 만들었습니다." in issues
    assert "입력에 없는 구체적인 시간 약속을 만들었습니다." in issues


def test_accountability_fallback_never_rewrites_situation_specific_body() -> None:
    serious_request = GenerateRequest(
        situation="고객사 최종 발표자료 제출을 놓쳐 고객 일정에 큰 차질이 생겼고 팀원 작업까지 막혔다",
        target=Target.TEAM_LEAD,
        tone=Tone.MILD,
    )
    original = result(
        "자료 제출을 놓쳐 고객 일정과 팀 작업에 차질을 드린 점 죄송합니다. 제 책임입니다. 현재 상태부터 확인해 필요한 대응을 정리하겠습니다. 같은 누락이 없도록 확인 절차를 점검하겠습니다."
    )
    fallback = _accountability_aftermath_fallback(
        original,
        serious_request,
    )

    assert fallback.excuse == original.excuse
    assert fallback.replyOptions == original.replyOptions
    assert validate_grounding_and_register(
        fallback, serious_request, include_options=False
    ) == []


def test_unstated_cause_progress_and_completion_are_rejected_generically() -> None:
    source_request = GenerateRequest(
        situation="약속 장소에 5분 늦었다",
        target=Target.FRIEND,
        tone=Tone.MILD,
    )
    generated = result(
        "일정 관리 부실이 원인이야. 지금 자료를 전달하고 있어. 이미 도착했어."
    )

    issues = validate_grounding_and_register(
        generated, source_request, include_options=False
    )

    assert "입력에 없는 원인을 사실처럼 단정했습니다." in issues
    assert "입력에 없는 진행 상태를 이미 수행 중인 것처럼 만들었습니다." in issues
    assert "입력에 없는 완료·도착 상태를 새로 만들었습니다." in issues


def test_new_artifact_and_system_commitment_are_rejected_generically() -> None:
    source_request = GenerateRequest(
        situation="자료 제출을 놓쳐 일정에 차질이 생겼다",
        target=Target.TEAM_LEAD,
        tone=Tone.MILD,
    )
    generated = result(
        "일정 관리 실수로 제출을 놓쳤습니다. 현재 자료를 재정비하고 있습니다. 업데이트된 파일을 전달하겠습니다. 공유 시스템을 도입하겠습니다."
    )

    issues = validate_grounding_and_register(
        generated, source_request, include_options=False
    )

    assert "입력에 없는 구체적인 인과관계를 새로 만들었습니다." in issues
    assert "입력에 없는 진행 상태를 이미 수행 중인 것처럼 만들었습니다." in issues
    assert "입력에 없는 수정·완성 산출물을 새로 만들었습니다." in issues
    assert "입력에 없는 시스템·도구 도입을 약속했습니다." in issues


def test_unstated_live_location_is_rejected_in_every_create_option() -> None:
    source_request = GenerateRequest(
        situation="약속 장소에 5분 늦었다",
        target=Target.FRIEND,
        tone=Tone.MILD,
    )
    generated = result("5분 늦어서 미안해.").model_copy(update={
        "replyOptions": [
            "5분 늦어서 미안해.",
            "지금 가고 있어. 미안해.",
            "곧 도착할게. 조금만 기다려줘.",
        ]
    })

    issues = validate_grounding_and_register(generated, source_request)

    assert "입력에 없는 이동·도착 상태를 단정했습니다." in issues


def test_create_option_register_is_checked_even_when_main_excuse_is_valid() -> None:
    generated = result(
        "엄마, 장례식장에서 소란을 피운 건 내 잘못이야. 정말 미안해. 지금부터 조용히 할게."
    ).model_copy(update={
        "replyOptions": [
            "엄마, 정말 미안해. 지금부터 조용히 할게.",
            "엄마, 죄송합니다. 지금부터 조용히 할게.",
            "내가 잘못했어. 바로 사과할게.",
        ]
    })

    issues = validate_grounding_and_register(generated, request())

    assert "한 답장 안에서 반말과 존댓말이 섞임" in issues
