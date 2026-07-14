"""REPLY 후보의 결정적 검사와 Judge 계약을 모은다.

이 모듈은 모델이 매번 같은 규칙을 안정적으로 따르도록, 명확하게 판별 가능한
오류(이모지, 깨진 조사, 입력 밖의 대표적인 사실 발명)를 먼저 차단한다. 질문의
실질적인 대응성·사실성·다양성은 별도 Cerebras Judge가 판정한다.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.models import GenerateRequest, Target


class QuestionIntent(str, Enum):
    """최신 상대 메시지가 요구하는 답변 종류."""

    DETAIL = "상세 요구"
    REASON = "이유"
    TIME = "시간"
    METHOD = "방법"
    PLACE = "장소"
    PERSON = "인물"
    YES_NO = "예/아니오"
    OTHER = "기타"


class ReplyCandidateJudge(BaseModel):
    """Judge가 한 후보에 매긴 가중치별 점수.

    Judge 응답은 내부 전용의 느슨한 계약이다. 일부 부가 필드가 빠져도 파싱은 하되,
    빠진 점수는 0점으로 간주해서 사용자에게 낮은 품질 결과가 전달되지 않게 한다.
    """

    model_config = ConfigDict(extra="ignore")

    directness: int = Field(default=0, ge=0, le=40)
    factuality: int = Field(default=0, ge=0, le=30)
    registerScore: int = Field(default=0, ge=0, le=15)
    fluency: int = Field(default=0, ge=0, le=15)
    hardViolation: bool = False
    issues: list[str] = Field(default_factory=list, max_length=8)

    @property
    def score(self) -> int:
        return self.directness + self.factuality + self.registerScore + self.fluency


class ReplyQualityVerdict(BaseModel):
    """후보 세 개와 후보 집합에 대한 Judge 판정."""

    model_config = ConfigDict(extra="ignore")

    candidateScores: list[ReplyCandidateJudge] = Field(default_factory=list)
    diversityScore: int = Field(default=0, ge=0, le=100)
    semanticDuplicate: bool = True
    issues: list[str] = Field(default_factory=list, max_length=8)


FORMAL_TARGETS = frozenset({Target.TEACHER, Target.TEAM_LEAD, Target.TEAM_MEMBER})
_CASUAL_RELATION_MARKERS = (
    "친구",
    "친한",
    "절친",
    "동기",
    "연인",
    "애인",
    "남자친구",
    "여자친구",
    "누나",
    "언니",
    "오빠",
)
_EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
_LIGHT_EVASION_PATTERN = re.compile(r"(?:비밀(?:이에요|입니다|이라서요)?|ㅋㅋ+|ㅎㅎ+|농담(?:이에요|입니다)?)")
_BROKEN_PARTICLE_PATTERN = re.compile(
    r"(?:^|[\s,.!?\"'“”])(?:은|는|이|가|을|를|에|에게|에서|으로|와|과)(?=$|[\s,.!?\"'“”])"
)
_CASUAL_ENDING_PATTERN = re.compile(
    r"(?:미안해|했어|였어|질렀어|할게|할거야|하고 있어)(?=$|[\s,.!?])"
)
_POLITE_ENDING_PATTERN = re.compile(
    r"(?:요|니다|까요|나요)(?=$|[\s.!?])"
)
_BROKEN_FLUENCY_PATTERN = re.compile(r"(?:안\s*않|못\s*못|않\s*않)")
_PRIVACY_BOUNDARY_MARKERS = (
    "자세히 말씀드리기 어렵",
    "구체적으로 말씀드리기 어렵",
    "말씀드리기 어렵",
    "개인적인 부분",
    "사적인 부분",
    "공개하기 어렵",
    "양해 부탁",
    "이해 부탁",
)
_INVENTED_FACT_MARKERS = (
    "병원",
    "감기",
    "몸살",
    "진단서",
    "증명서",
    "교통사고",
    "사고가",
    "장례",
    "출장",
)
_GROUNDED_DETAIL_MARKERS = _INVENTED_FACT_MARKERS + (
    "가족",
    "행사",
    "진료",
    "치료",
    "시험",
    "수업",
    "면접",
    "상담",
    "예약",
    "개인 일정",
)


def classify_question_intent(message: str) -> QuestionIntent:
    """질문 의도를 우선순위에 따라 하나로 분류한다.

    장소·인물처럼 ``무엇``과 함께 쓰일 수 있는 구체 유형을 먼저 잡아, 단순 키워드
    매칭이 ``어디서``를 상세 요구로 잘못 분류하지 않게 한다.
    """

    compact = re.sub(r"\s+", "", message)
    if any(marker in compact for marker in ("언제", "몇시", "몇시간", "몇분")):
        return QuestionIntent.TIME
    if any(marker in compact for marker in ("어디", "장소")):
        return QuestionIntent.PLACE
    if any(marker in compact for marker in ("누구", "누가", "어떤사람")):
        return QuestionIntent.PERSON
    if any(marker in compact for marker in ("어떻게", "어쩔", "방법")):
        return QuestionIntent.METHOD
    if any(marker in compact for marker in ("왜", "이유")):
        return QuestionIntent.REASON
    if any(marker in compact for marker in ("뭐", "무엇", "뭔가", "어떤")):
        return QuestionIntent.DETAIL
    if re.search(r"(?:인가요|인가|맞나요|맞아|할거야|할까요|됐어|되나요|했어)\?*$", compact):
        return QuestionIntent.YES_NO
    return QuestionIntent.OTHER


def is_formal_relationship(request: GenerateRequest) -> bool:
    """입력 관계에서 격식을 보수적으로 추론한다.

    CUSTOM은 친밀함이 명시된 경우에만 편한 말투를 허용한다. 그 밖의 자연어 관계와
    애매한 설명은 공식 관계로 처리한다.
    """

    if request.target in FORMAL_TARGETS:
        return True
    if request.target != Target.CUSTOM:
        return False
    description = (request.targetDescription or "").replace(" ", "")
    return not any(marker in description for marker in _CASUAL_RELATION_MARKERS)


def relationship_register_label(request: GenerateRequest) -> str:
    """프롬프트에 넣을 관계·말투 안내 문장."""

    if is_formal_relationship(request):
        return "공식 관계: 존댓말을 쓰고 이모지·농담·가벼운 회피를 금지한다."
    return "친밀한 관계: 자연스러운 구어체는 허용하지만 질문 회피와 사실 발명은 금지한다."


def deterministic_candidate_issues(
    candidate: str,
    request: GenerateRequest,
    intent: QuestionIntent,
) -> list[str]:
    """LLM 호출 없이 확실히 판별 가능한 후보 오류를 반환한다."""

    text = candidate.strip()
    issues: list[str] = []
    if not text:
        return ["비어 있는 답장 후보"]
    if _BROKEN_PARTICLE_PATTERN.search(text):
        issues.append("조사만 남은 깨진 문장")
    issues.extend(fluency_issues(text))
    issues.extend(register_consistency_issues(text))
    issues.extend(relationship_register_issues(text, request))

    if is_formal_relationship(request):
        if _EMOJI_PATTERN.search(text):
            issues.append("공식 관계 답장에 이모지가 포함됨")
        if _LIGHT_EVASION_PATTERN.search(text):
            issues.append("공식 관계 답장에 농담 또는 가벼운 회피가 포함됨")

    if requires_privacy_boundary(request, intent):
        if not any(marker in text for marker in _PRIVACY_BOUNDARY_MARKERS):
            issues.append("근거 없는 상세 요구에 정중한 공개 거절로 직접 답하지 않음")

    context = _request_facts(request)
    for marker in _INVENTED_FACT_MARKERS:
        if marker in text and marker not in context:
            issues.append("입력에 없는 구체적 사실을 새로 만듦")
            break

    return issues


def register_consistency_issues(text: str) -> list[str]:
    """한 후보 안에서 반말 종결과 존댓말 종결이 섞였는지 검사한다."""
    if _CASUAL_ENDING_PATTERN.search(text) and _POLITE_ENDING_PATTERN.search(text):
        return ["한 답장 안에서 반말과 존댓말이 섞임"]
    return []


def fluency_issues(text: str) -> list[str]:
    if _BROKEN_FLUENCY_PATTERN.search(text):
        return ["부정 표현이 중복된 깨진 문장"]
    return []


def relationship_register_issues(
    text: str,
    request: GenerateRequest,
) -> list[str]:
    """관계가 요구하는 말투와 후보의 종결 말투가 어긋나는지 검사한다."""
    source = request.situation.replace(" ", "")
    if (
        request.target == Target.PARENT
        and any(marker in source for marker in ("엄마", "아빠"))
        and _POLITE_ENDING_PATTERN.search(text)
    ):
        return ["엄마·아빠에게 보내는 답장이 갑자기 존댓말로 바뀜"]
    if is_formal_relationship(request) and _CASUAL_ENDING_PATTERN.search(text):
        return ["공식 관계 답장에 반말이 포함됨"]
    return []


def _has_grounded_detail(request: GenerateRequest) -> bool:
    """현재 문맥에 실제로 공개 가능한 구체 사실이 있는지 보수적으로 판단한다."""

    facts = _request_facts(request)
    return bool(facts) and any(marker in facts for marker in _GROUNDED_DETAIL_MARKERS)


def requires_privacy_boundary(
    request: GenerateRequest,
    intent: QuestionIntent,
) -> bool:
    """사실을 만들지 않고 정중한 공개 거절로 답해야 하는지 판단한다.

    상세 이유를 묻더라도 대화 안에 공개 가능한 근거가 있으면 그 사실을 답할 수 있다.
    반대로 근거가 없을 때 세 후보가 모두 같은 경계 결론을 공유하는 것은 정상적인
    제약이므로, 서비스 계층의 다양성 기준도 이 상태를 함께 고려한다.
    """

    return intent in {QuestionIntent.DETAIL, QuestionIntent.REASON} and not _has_grounded_detail(request)


def _request_facts(request: GenerateRequest) -> str:
    return " ".join(
        value
        for value in (
            request.situation,
            request.rootExcuse or "",
            request.currentExcuse or "",
            *(turn.message for turn in request.conversation),
        )
        if value
    )


# Judge에는 필드 이름·숫자 범위만 남긴 느슨한 JSON Schema를 사용한다. strict를 끄고
# Pydantic의 안전한 기본값으로 보완한 뒤, 누락된 점수는 품질 실패로 처리한다.
REPLY_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidateScores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "directness": {"type": "integer"},
                    "factuality": {"type": "integer"},
                    "registerScore": {"type": "integer"},
                    "fluency": {"type": "integer"},
                    "hardViolation": {"type": "boolean"},
                    "issues": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "diversityScore": {"type": "integer"},
        "semanticDuplicate": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
}
