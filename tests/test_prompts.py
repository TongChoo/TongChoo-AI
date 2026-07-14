"""프롬프트 모드 분리와 핵심 지시의 회귀를 막는 테스트."""

from app.models import GenerateRequest, GenerationMode, Target, Tone
from app.prompts import (
    build_reply_judge_system_prompt,
    build_reply_judge_user_prompt,
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


def test_formal_custom_relationship_is_passed_to_reply_and_judge_prompts() -> None:
    request = GenerateRequest(
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

    reply_prompt = build_reply_system_prompt()
    judge_system_prompt = build_reply_judge_system_prompt()
    judge_user_prompt = build_reply_judge_user_prompt(
        request,
        [
            "개인적인 부분이라 자세히 말씀드리기 어렵습니다. 양해 부탁드립니다.",
            "사적인 사유라 구체적으로 말씀드리기 어려운 점 이해 부탁드립니다.",
            "이번 회식 참석은 어렵습니다. 개인적인 부분은 자세히 말씀드리기 어렵습니다.",
        ],
    )

    assert "회사 부장님" in judge_user_prompt
    assert "공식 관계" in judge_user_prompt
    assert "상세 요구" in judge_user_prompt
    assert "이모지·농담·가벼운 회피" in reply_prompt
    assert "candidateScores" in judge_system_prompt
