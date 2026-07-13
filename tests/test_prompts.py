"""프롬프트 모드 분리와 핵심 지시의 회귀를 막는 테스트."""

from app.models import GenerateRequest, GenerationMode, Target, Tone
from app.prompts import (
    build_classification_prompt,
    build_reply_system_prompt,
    build_reply_user_prompt,
    build_system_prompt,
    build_user_prompt,
)


def test_create_prompt_excludes_reply_only_instructions() -> None:
    request = GenerateRequest(
        mode=GenerationMode.CREATE,
        situation="팀 회의에 20분 늦었다",
        target=Target.TEAM_LEAD,
        tone=Tone.MILD,
    )

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(request)

    assert "이번 작업은 새 변명을 만드는 일이 아니다" not in system_prompt
    assert "최신 메시지" not in user_prompt
    assert "팀 회의에 20분 늦었다" in user_prompt
    assert "replyOptions는 정확히 3개의 문자열" in system_prompt
    assert "변명의 진위나 일관성" in system_prompt
    assert "회의 핵심 내용은 무엇인가요?" in system_prompt


def test_reply_prompt_prioritizes_latest_message_and_human_tone() -> None:
    request = GenerateRequest(
        mode=GenerationMode.REPLY,
        situation="팀 회의에 20분 늦었다",
        target=Target.TEAM_LEAD,
        tone=Tone.MILD,
        situationSeverity="NORMAL",
        rootExcuse="회의 시작 시간을 잘못 봤습니다.",
        currentExcuse="미리 공유하지 못했습니다.",
        incomingMessage="왜 미리 연락하지 않았어요?",
        conversation=[
            {"role": "assistant", "message": "회의 시작 시간을 잘못 봤습니다."},
            {"role": "user", "message": "왜 미리 연락하지 않았어요?"},
        ],
        roundNumber=2,
    )

    system_prompt = build_reply_system_prompt()
    user_prompt = build_reply_user_prompt(request)

    assert "상대가 방금 보낸 메시지" in system_prompt
    assert "고객센터 답변" in system_prompt
    assert "왜 미리 연락하지 않았어요?" in user_prompt
    assert "직접 답장, 수습 답장, 긴장 완화 답장" in user_prompt
    assert "변명의 진위나 일관성" in system_prompt


def test_custom_target_description_is_included_as_relationship_context() -> None:
    request = GenerateRequest(
        mode=GenerationMode.CREATE,
        situation="약속 시간에 늦었다",
        target=Target.CUSTOM,
        targetDescription="같은 프로젝트를 진행하는 친한 선배",
        tone=Tone.MILD,
    )

    user_prompt = build_user_prompt(request)

    assert "CUSTOM" in user_prompt
    assert "같은 프로젝트를 진행하는 친한 선배" in user_prompt


def test_classification_prompt_uses_impact_and_safe_normal_default() -> None:
    request = GenerateRequest(
        situation="고객사 최종 발표자료 제출을 놓쳤다",
        target=Target.TEAM_LEAD,
        tone=Tone.BULLSHIT,
    )

    prompt = build_classification_prompt(request)

    assert "마감, 금전, 안전, 고객" in prompt
    assert "팀장이나 선생님이라는 이유만으로 SERIOUS" in prompt
    assert "판단이 애매하면 NORMAL" in prompt
    assert "팀장과 점심 약속에 5분 늦었고" in prompt
    assert "친구의 부탁을 잊어 실제 금전 손해" in prompt
    assert "SERIOUS: 3~5문장, 120~350자" in prompt
