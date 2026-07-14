"""생성 재시도 문구와 안전한 부분 복구 정책."""

from __future__ import annotations

from app.models import ExcuseResult, GenerateRequest, SituationProfile, SituationSeverity
from app.reply_quality import is_formal_relationship


def create_retry_instruction(issues: list[str]) -> str:
    if not issues:
        return ""
    return (
        "\n\n[직전 결과 거절 사유]\n- "
        + "\n- ".join(issues)
        + "\n입력에 없는 원인·현재 진행 상태·완료 상태·시간 약속을 만들지 마세요. "
        "세 replyOptions 모두 같은 사실과 같은 말투 기준을 지키세요. "
        "잘못 인정·직접적인 사과·앞으로 할 수습 행동을 excuse에 넣으세요. "
        "riskFactors에는 '정보 부족' 같은 포괄적 표현 대신 실제 대화 위험을 쓰고, "
        "remember에는 이후 답장에서 유지할 입력 사실을 한 개 이상 쓰세요."
    )


def repair_aftermath_only(result: ExcuseResult, request: GenerateRequest) -> ExcuseResult:
    """본문과 후보는 유지하고 실패한 후폭풍만 안전한 질문으로 교체한다."""
    if request.target.value == "PARENT":
        question = "그럼 지금 말한 대로 바로 행동을 고치고 수습할 수 있겠어?"
    elif is_formal_relationship(request):
        question = "그럼 말씀하신 수습 행동을 바로 진행할 수 있나요?"
    else:
        question = "그럼 지금 말한 수습 행동을 바로 할 수 있어?"
    repaired = result.aftermath[0].model_copy(update={
        "when": "즉시",
        "dayOffset": 0,
        "question": question,
        "collapseRate": 35,
    })
    return result.model_copy(update={"aftermath": [repaired]})


def safe_create_body(
    result: ExcuseResult,
    request: GenerateRequest,
    profile: SituationProfile,
) -> ExcuseResult:
    """모델이 계속 사실을 발명할 때 입력 밖 상태가 없는 최종 안전 본문을 만든다."""
    formal = is_formal_relationship(request)
    if profile.severity == SituationSeverity.SERIOUS:
        if formal:
            options = [(
                "문제를 사전에 확인하고 대비하지 못한 제 잘못입니다. "
                "이로 인해 업무에 차질을 드린 점 정말 죄송합니다. "
                "우선 필요한 수습 범위를 확인하고 가능한 대응부터 정리하겠습니다. "
                "같은 문제가 반복되지 않도록 사전 확인 절차를 점검하겠습니다."
            ), (
                "필요한 준비를 사전에 확인하지 못한 책임은 제게 있습니다. 죄송합니다. "
                "업무에 생긴 차질을 먼저 확인하고 가능한 수습 방안을 정리하겠습니다. "
                "앞으로는 같은 누락이 없도록 사전 점검 절차를 확인하겠습니다."
            ), (
                "사전에 대비하지 못해 업무에 차질을 드린 점 사과드립니다. 제 잘못입니다. "
                "필요한 대응 범위를 확인한 뒤 할 수 있는 조치부터 정리하겠습니다. "
                "재발하지 않도록 준비 단계의 확인 절차를 점검하겠습니다."
            )]
        else:
            options = [(
                "미리 확인하고 대비하지 못한 내 잘못이야. 피해를 줘서 정말 미안해. "
                "우선 필요한 수습 범위를 확인하고 할 수 있는 대응부터 정리할게. "
                "같은 문제가 반복되지 않도록 다음에는 미리 점검할게."
            ), "미리 준비하지 못한 건 내 책임이야. 정말 미안해. 필요한 대응부터 확인하고 정리할게.",
                "사전에 확인하지 못해서 피해를 줬어. 미안해. 수습할 수 있는 일부터 확인할게."]
    elif formal:
        options = [(
            "미리 확인하고 준비하지 못한 제 잘못입니다. 죄송합니다. "
            "필요한 대응을 확인하고 가능한 일부터 정리하겠습니다."
        ), "사전에 준비하지 못해 죄송합니다. 필요한 대응부터 확인하겠습니다.",
            "준비가 부족했던 제 책임입니다. 가능한 수습 방안을 먼저 정리하겠습니다."]
    else:
        options = ["미리 확인하지 못한 내 잘못이야. 미안해. 필요한 대응부터 확인할게.",
            "내가 미리 준비하지 못했어. 미안해. 할 수 있는 일부터 정리할게.",
            "준비가 부족했던 건 내 책임이야. 미안해. 필요한 대응을 먼저 확인할게."]
    risk_factors = (
        ["답변의 구체성 부족", "상대의 추가 질문 가능성"]
        if formal
        else ["설명의 구체성 부족", "추가 질문 가능성"]
    )
    repaired = result.model_copy(update={
        "excuse": options[0],
        "replyOptions": options,
        "riskFactors": risk_factors,
        "remember": [request.situation],
    })
    return repair_aftermath_only(repaired, request)


def safe_reply_body(result: ExcuseResult, request: GenerateRequest) -> ExcuseResult:
    """후속 답장 재생성까지 실패했을 때 사실을 추가하지 않는 짧은 책임 답장."""
    if is_formal_relationship(request):
        options = [
            "미리 준비하지 못한 제 잘못입니다. 죄송합니다. 필요한 대응부터 확인하겠습니다.",
            "사전에 확인하지 못한 책임은 제게 있습니다. 죄송합니다. 수습할 일을 먼저 정리하겠습니다.",
            "준비가 부족했던 제 책임입니다. 죄송합니다. 가능한 대응부터 확인하겠습니다.",
        ]
    else:
        options = [
            "미리 준비하지 못한 내 잘못이야. 미안해. 필요한 대응부터 확인할게.",
            "사전에 확인하지 못한 건 내 책임이야. 미안해. 수습할 일부터 정리할게.",
            "준비가 부족했던 건 내 잘못이야. 미안해. 할 수 있는 대응부터 확인할게.",
        ]
    return result.model_copy(update={"excuse": options[0], "replyOptions": options})
