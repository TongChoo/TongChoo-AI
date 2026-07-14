"""AI 생성 흐름과 REPLY 품질 검증을 담당한다."""

from __future__ import annotations

import logging
import math
import re
from difflib import SequenceMatcher
from app.config import Settings
from app.llm import CerebrasClient, api_error
from app.models import (
    ExcuseResult,
    GenerateRequest,
    GenerationMode,
    SituationProfile,
    SituationSeverity,
    SpringExcuseResponse,
)
from app.prompts import (
    build_classification_prompt,
    build_classification_system_prompt,
    build_reply_system_prompt,
    build_reply_user_prompt,
    build_system_prompt,
    build_user_prompt,
)

logger = logging.getLogger("tongchoo.service")

class ExcuseGenerationService:
    """프롬프트 구성, Cerebras 호출, REPLY 품질 재생성을 조율한다."""

    def __init__(self, settings: Settings):
        self.client = CerebrasClient(settings)

    async def generate(self, request: GenerateRequest, request_id: str) -> ExcuseResult:
        result, _ = await self._generate_with_profile(request, request_id)
        return result

    async def _generate_with_profile(
        self,
        request: GenerateRequest,
        request_id: str,
    ) -> tuple[ExcuseResult, SituationProfile]:
        profile = await self._resolve_profile(request, request_id)
        if request.mode == GenerationMode.REPLY:
            result = await self._generate_reply_with_quality_gate(
                request,
                profile,
                request_id,
            )
            return result, profile

        result = await self._generate_create_with_quality_gate(
            request,
            profile,
            request_id,
        )
        return result, profile

    async def _resolve_profile(
        self,
        request: GenerateRequest,
        request_id: str,
    ) -> SituationProfile:
        if request.mode == GenerationMode.REPLY:
            profile = profile_from_persisted_severity(
                request.situationSeverity,
                request,
            ).for_mode(request.mode)
            logger.info(
                "situation_reused request_id=%s mode=REPLY severity=%s",
                request_id,
                profile.severity.value,
            )
            return profile

        classified = await self.client.classify_situation(
            build_classification_system_prompt(),
            build_classification_prompt(request),
            request_id,
        )
        profile = apply_severity_guardrails(classified, request)
        logger.info(
            "situation_classified request_id=%s mode=CREATE severity=%s",
            request_id,
            profile.severity.value,
        )
        return profile

    async def _generate_create_with_quality_gate(
        self,
        request: GenerateRequest,
        profile: SituationProfile,
        request_id: str,
    ) -> ExcuseResult:
        """상황 적합성과 후폭풍을 검사하고 실패 후보 중 최선도 보존한다."""
        base_prompt = build_user_prompt(
            request,
            profile=profile,
            max_memory_chars=self.client.settings.max_memory_chars,
        )
        rejected: list[str] = []
        candidates: list[tuple[ExcuseResult, list[str]]] = []
        max_attempts = max(
            self.client.settings.situation_quality_max_attempts,
            self.client.settings.aftermath_quality_max_attempts,
        )

        for attempt in range(max_attempts):
            correction = ""
            if rejected:
                correction = (
                    "\n\n[직전 결과의 상황 적합성 문제]\n- "
                    + "\n- ".join(rejected)
                    + "\n입력에 없는 사실을 만들지 마세요. 위 문제를 직접 고치고, "
                    "excuse는 상황에 맞는 구조로 새로 작성하세요. aftermath는 새 excuse의 "
                    "주장이나 허점을 상대방이 직접 확인하는 질문으로 작성하세요."
                )
            result = await self.client.generate(
                build_system_prompt(),
                base_prompt + correction,
                request_id,
            )
            result = sanitize_unsupported_time_promises(result, request)
            rejected = [
                *validate_situation_fit(result, profile),
                *validate_grounding(result, request),
                *self._aftermath_quality_issues(result, profile.severity),
            ]
            rejected = list(dict.fromkeys(rejected))
            candidates.append((result, rejected))
            if not rejected:
                return result
            logger.warning(
                "create_quality_retry request_id=%s attempt=%s issues=%s",
                request_id,
                attempt + 1,
                "; ".join(rejected),
            )

        safe_candidates = [
            candidate
            for candidate in candidates
            if not (_GROUNDING_ISSUES & set(candidate[1]))
        ]
        if not safe_candidates:
            raise api_error(
                422,
                "GROUNDING_QUALITY_REJECTED",
                "입력에 없는 사실이 없는 변명을 만들지 못했습니다. 다시 시도해주세요.",
            )
        best_result, best_issues = max(
            safe_candidates,
            key=lambda item: _situation_fit_score(
                item[0], profile, item[1], request
            ),
        )
        logger.warning(
            "create_quality_best_effort request_id=%s severity=%s issues=%s",
            request_id,
            profile.severity.value,
            "; ".join(best_issues),
        )
        return best_result

    async def _generate_reply_with_quality_gate(
        self,
        request: GenerateRequest,
        profile: SituationProfile,
        request_id: str,
    ) -> ExcuseResult:
        """답장 결과의 반복·후보 중복·최신 질문 누락을 검사하고 제한 재생성한다."""
        base_prompt = build_reply_user_prompt(
            request,
            profile=profile,
            max_memory_chars=self.client.settings.max_memory_chars,
        )
        rejected: list[str] = []
        candidates: list[tuple[ExcuseResult, list[str]]] = []

        for attempt in range(self.client.settings.reply_quality_max_attempts):
            correction = ""
            if rejected:
                correction = (
                    "\n\n[직전 결과 거절 사유]\n- "
                    + "\n- ".join(rejected)
                    + "\n직전 문장을 반복하지 말고 최신 상대 메시지에 직접 답하는 "
                    "서로 다른 후보를 새로 작성하세요."
                )
            result = await self.client.generate(
                build_reply_system_prompt(),
                base_prompt + correction,
                request_id,
            )
            result = sanitize_unsupported_time_promises(result, request)
            rejected = self._reply_quality_issues(result, request)
            rejected.extend(validate_situation_fit(result, profile))
            rejected.extend(validate_grounding(result, request))
            rejected = list(dict.fromkeys(rejected))
            candidates.append((result, rejected))
            if not rejected:
                return result
            logger.warning(
                "reply_quality_retry request_id=%s attempt=%s issues=%s",
                request_id,
                attempt + 1,
                "; ".join(rejected),
            )

        safe_candidates = [
            candidate
            for candidate in candidates
            if not (_REPLY_HARD_REJECTION_MESSAGES & set(candidate[1]))
        ]
        if safe_candidates:
            best_result, best_issues = max(
                safe_candidates,
                key=lambda item: _situation_fit_score(
                    item[0], profile, item[1], request
                ),
            )
            logger.warning(
                "reply_quality_best_effort request_id=%s severity=%s issues=%s",
                request_id,
                profile.severity.value,
                "; ".join(best_issues),
            )
            return best_result

        raise api_error(
            422,
            "REPLY_QUALITY_REJECTED",
            "상황에 맞는 답장 후보를 만들지 못했습니다. 다시 시도해주세요.",
        )

    def _reply_quality_issues(
        self,
        result: ExcuseResult,
        request: GenerateRequest,
    ) -> list[str]:
        issues: list[str] = []
        candidates = result.replyOptions
        normalized_candidates = [_normalize_text(candidate) for candidate in candidates]

        for left_index, left in enumerate(normalized_candidates):
            for right in normalized_candidates[left_index + 1 :]:
                if _similarity(left, right) >= self.client.settings.reply_similarity_threshold:
                    issues.append("기본 답변과 후보 답변이 서로 지나치게 비슷함")
                    break
            if issues:
                break

        previous_answers = [
            turn.message
            for turn in request.conversation
            if turn.role.value == "assistant"
        ]
        if request.currentExcuse:
            previous_answers.append(request.currentExcuse)
        for candidate in candidates:
            if any(
                _similarity(_normalize_text(candidate), _normalize_text(previous))
                >= self.client.settings.reply_similarity_threshold
                for previous in previous_answers
            ):
                issues.append("이전 라운드 답변을 거의 그대로 반복함")
                break

        relevance_issue = _latest_message_relevance_issue(
            request.incomingMessage or "",
            " ".join(candidates),
        )
        if relevance_issue:
            issues.append(relevance_issue)

        issues.extend(self._aftermath_quality_issues(result))

        return list(dict.fromkeys(issues))

    def _aftermath_quality_issues(
        self,
        result: ExcuseResult,
        severity: SituationSeverity | None = None,
    ) -> list[str]:
        """후폭풍이 일반 업무 질문이 아니라 현재 변명의 검증 질문인지 검사한다."""
        issues: list[str] = []
        questions = [item.question.strip() for item in result.aftermath]
        excuse_anchors = _meaningful_tokens(result.excuse)

        for question in questions:
            if not question.endswith("?"):
                issues.append("후폭풍 질문이 상대방의 직접적인 의문문이 아님")

            question_tokens = _meaningful_tokens(question)
            has_excuse_anchor = bool(excuse_anchors & question_tokens)
            has_verification_signal = any(
                marker in question
                for marker in _AFTERMATH_VERIFICATION_SIGNALS
            )
            if (
                severity != SituationSeverity.LIGHT
                and not has_excuse_anchor
                and not has_verification_signal
            ):
                issues.append("후폭풍 질문이 현재 변명의 주장이나 허점과 연결되지 않음")

        for left_index, left in enumerate(questions):
            for right in questions[left_index + 1 :]:
                if _similarity(_normalize_text(left), _normalize_text(right)) >= 0.85:
                    issues.append("후폭풍 질문이 서로 지나치게 비슷함")
                    break

        return list(dict.fromkeys(issues))

    async def generate_for_spring(
        self,
        request: GenerateRequest,
        request_id: str,
    ) -> SpringExcuseResponse:
        result, profile = await self._generate_with_profile(request, request_id)
        logger.info(
            "ai_generation_result request_id=%s mode=%s round=%s excuse=%s",
            request_id,
            request.mode.value,
            request.roundNumber,
            _excerpt(result.excuse),
        )
        return SpringExcuseResponse.from_result(result, profile.severity)

def _excerpt(value: str, limit: int = 120) -> str:
    return value.replace("\n", " ").strip()[:limit]


_NEXT_ACTION_MARKERS = (
    "확인하겠습니다",
    "확인할게",
    "공유하겠습니다",
    "공유할게",
    "수정하겠습니다",
    "수정할게",
    "전달하겠습니다",
    "전달할게",
    "정리하겠습니다",
    "정리할게",
    "점검하겠습니다",
    "점검할게",
    "보내겠습니다",
    "보낼게",
    "올리겠습니다",
    "올릴게",
    "제출하겠습니다",
    "제출할게",
    "들어가겠습니다",
    "들어갈게",
    "연락하겠습니다",
    "연락할게",
    "처리하겠습니다",
    "처리할게",
    "가는 중",
    "진행 중",
    "이동 중",
    "도착하겠습니다",
    "도착할게",
    "들어오겠습니다",
    "들어올게",
    "예정",
    "바로 하겠습니다",
    "바로 할게",
)

_NEXT_ACTION_PATTERN = re.compile(
    r"(?:확인|공유|수정|전달|정리|점검|보내|올리|제출|도착|들어오|들어가|"
    r"연락|처리|복구|검토|보고|추가|이동|수행|강화).{0,18}"
    r"(?:하겠습니다|하겠|할게|드리겠습니다|드릴게|겠습니다|예정|진행\s*중|이동\s*중)"
)

_IMMEDIATE_ACTION_CONTEXT = ("지금", "현재", "바로", "즉시", "우선")
_ACTION_ENDING_PATTERN = re.compile(
    r"(?:하겠습니다|하겠|할게|할게요|드리겠습니다|드릴게|줄게|살게|"
    r"올게|가겠습니다|가서|겠습니다|올리니|보내니|공유하니|"
    r"고\s*있습니다|중입니다|예정입니다)"
)

_SITUATION_FIT_MESSAGES = frozenset(
    {
        "변명 본문이 비어 있습니다.",
        "상황 심각도에 비해 답변이 너무 짧습니다.",
        "상황에 비해 답변이 지나치게 깁니다.",
        "필요한 설명과 수습 행동이 부족합니다.",
        "답변이 지나치게 장황합니다.",
        "상대가 확인할 수 있는 다음 행동이 없습니다.",
    }
)

_GROUNDING_ISSUES = frozenset(
    {
        "입력에 없는 구체적인 시간 약속이 포함되어 있습니다.",
        "입력에 없는 원인이나 사건이 포함되어 있습니다.",
        "제출 누락을 입력에 없는 미완성 상태로 바꿨습니다.",
        "입력에 없는 구체적인 영향이 실제 발생한 것처럼 표현되었습니다.",
    }
)

_REPLY_HARD_REJECTION_MESSAGES = frozenset(
    {
        "변명 본문이 비어 있습니다.",
        "이전 라운드 답변을 거의 그대로 반복함",
        *_GROUNDING_ISSUES,
    }
)


def count_sentences(text: str) -> int:
    """문장부호·줄바꿈·한국어 종결 표현으로 의미 단위 개수를 근사한다."""
    stripped = text.strip()
    if not stripped:
        return 0
    chunks = [
        chunk.strip()
        for chunk in re.split(r"[.!?]+|\n+", stripped)
        if chunk.strip()
    ]
    count = 0
    for chunk in chunks:
        korean_endings = re.findall(
            r"(?:요|니다|한다|했다|된다|됐다|할게|할게요)(?=\s|$)",
            chunk,
        )
        count += max(1, len(korean_endings))
    return max(1, count)


def validate_situation_fit(
    result: ExcuseResult,
    profile: SituationProfile,
) -> list[str]:
    """명백하게 짧거나 장황하고, 필요한 행동이 없는 결과를 걸러낸다."""
    problems: list[str] = []
    excuse = result.excuse.strip()
    length = len(excuse)
    sentence_count = count_sentences(excuse)
    # 기획 기준의 '약 60자'를 정확히 60자로 잘라 버리지 않도록 15% 오차를 허용한다.
    minimum_length = math.ceil(profile.minLength * 0.85)
    maximum_length = math.floor(profile.maxLength * 1.1)

    if not excuse:
        problems.append("변명 본문이 비어 있습니다.")
    if length < minimum_length:
        problems.append("상황 심각도에 비해 답변이 너무 짧습니다.")
    if length > maximum_length:
        problems.append("상황에 비해 답변이 지나치게 깁니다.")
    if sentence_count < profile.minSentences:
        problems.append("필요한 설명과 수습 행동이 부족합니다.")
    if sentence_count > profile.maxSentences:
        problems.append("답변이 지나치게 장황합니다.")
    if profile.needsNextAction and not _has_next_action(excuse):
        problems.append("상대가 확인할 수 있는 다음 행동이 없습니다.")

    return problems


def _has_next_action(text: str) -> bool:
    if any(marker in text for marker in _NEXT_ACTION_MARKERS):
        return True
    if _NEXT_ACTION_PATTERN.search(text):
        return True
    return any(marker in text for marker in _IMMEDIATE_ACTION_CONTEXT) and bool(
        _ACTION_ENDING_PATTERN.search(text)
    )


_UNSUPPORTED_CAUSE_MARKERS = (
    "교통",
    "차가 막",
    "정전",
    "알람",
    "병원",
    "아파",
    "질병",
    "고장",
    "오류",
    "인터넷",
    "배터리",
    "서버 장애",
    "파일 손상",
    "눈이 안 보",
)

_TIME_TOKEN_PATTERN = re.compile(
    r"오늘(?:\s*(?:안으로|중으로|까지))?|"
    r"내일(?:\s*(?:안으로|중으로|까지))?|"
    r"오전|오후|"
    r"\d+\s*(?:분|시간|시)(?:\s*(?:안으로|중으로|뒤|까지))?"
)

_UNSUPPORTED_IMPACT_MARKERS = (
    "혼란이 발생",
    "불안감이 발생",
    "차질이 생",
    "지연되었습니다",
    "지연되고 있습니다",
    "데이터 손실 위험이 발생",
    "서비스 중단",
)

_SOURCE_IMPACT_MARKERS = (
    "영향",
    "피해",
    "손해",
    "막힌",
    "지연",
    "차질",
    "위험",
    "문제가 발생",
)


def validate_grounding(
    result: ExcuseResult,
    request: GenerateRequest,
) -> list[str]:
    """입력에 없던 시간 약속과 흔한 가짜 원인을 명백한 범위에서 탐지한다."""
    issues: list[str] = []
    source = request.situation.strip()
    candidates = result.replyOptions
    for candidate in candidates:
        output_times = set(_TIME_TOKEN_PATTERN.findall(candidate))
        if any(time_token not in source for time_token in output_times):
            issues.append("입력에 없는 구체적인 시간 약속이 포함되어 있습니다.")

        if any(
            marker in candidate and marker not in source
            for marker in _UNSUPPORTED_CAUSE_MARKERS
        ):
            issues.append("입력에 없는 원인이나 사건이 포함되어 있습니다.")

        if (
            "제출하지 못" in source
            and "완성" not in source
            and any(marker in candidate for marker in ("완성하지 못", "미완성"))
        ):
            issues.append("제출 누락을 입력에 없는 미완성 상태로 바꿨습니다.")

        if (
            not any(marker in source for marker in _SOURCE_IMPACT_MARKERS)
            and any(marker in candidate for marker in _UNSUPPORTED_IMPACT_MARKERS)
        ):
            issues.append(
                "입력에 없는 구체적인 영향이 실제 발생한 것처럼 표현되었습니다."
            )

    return list(dict.fromkeys(issues))


def sanitize_unsupported_time_promises(
    result: ExcuseResult,
    request: GenerateRequest,
) -> ExcuseResult:
    """입력에 없던 시간 표현만 제거해 사실을 추가하지 않는 문장으로 정규화한다."""
    source = request.situation.strip()

    def sanitize(text: str) -> str:
        cleaned = _TIME_TOKEN_PATTERN.sub(
            lambda match: match.group(0) if match.group(0) in source else "",
            text,
        )
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)
        return cleaned.strip()

    return result.model_copy(
        update={
            "excuse": sanitize(result.excuse),
            "replyOptions": [sanitize(option) for option in result.replyOptions],
        }
    )


def apply_severity_guardrails(
    profile: SituationProfile,
    request: GenerateRequest,
) -> SituationProfile:
    """명백한 복합 위험 신호가 AI 분류 변동으로 낮아지는 것을 막는다."""
    situation = request.situation.replace(" ", "")
    serious_combinations = (
        "면접" in situation
        and any(marker in situation for marker in ("잊", "불참", "참석하지못")),
        "손해" in situation and any(marker in situation for marker in ("금전", "결제", "발생")),
        "안전" in situation and "점검" in situation and "누락" in situation,
        "내부정보" in situation and "전달" in situation,
        "고객" in situation and "마감" in situation and "제출하지못" in situation,
        "배포" in situation and "백업" in situation and "누락" in situation,
        any(marker in situation for marker in ("다른팀원", "팀원들"))
        and any(marker in situation for marker in ("막힌", "막혔", "모두막")),
    )
    normal_combinations = (
        "준비물" in situation
        and "일정" in situation
        and any(marker in situation for marker in ("밀렸", "늦어졌", "지연")),
        any(marker in situation for marker in ("과제", "수행평가"))
        and "제출" in situation
        and any(marker in situation for marker in ("마감", "기한", "제출시간"))
        and any(
            marker in situation
            for marker in ("못했", "못함", "놓쳤", "늦었", "지났", "초과")
        ),
    )

    severity = profile.severity
    if any(serious_combinations):
        severity = SituationSeverity.SERIOUS
    elif severity == SituationSeverity.LIGHT and any(normal_combinations):
        severity = SituationSeverity.NORMAL
    if severity == profile.severity:
        return profile

    ranges = {
        SituationSeverity.NORMAL: (2, 3, 60, 180),
        SituationSeverity.SERIOUS: (3, 5, 120, 350),
    }
    min_sentences, max_sentences, min_length, max_length = ranges[severity]
    return profile.model_copy(
        update={
            "severity": severity,
            "hasImpact": True,
            "needsAccountability": True,
            "needsNextAction": True,
            "humorAllowed": False,
            "minSentences": min_sentences,
            "maxSentences": max_sentences,
            "minLength": min_length,
            "maxLength": max_length,
        }
    )


def profile_from_persisted_severity(
    severity: SituationSeverity | None,
    request: GenerateRequest,
) -> SituationProfile:
    """Spring이 저장한 심각도로 REPLY용 프로필을 복원한다.

    DB 값이 REPLY 검증 전에 필수로 확인되지만, 직접 호출에서도 안전한 NORMAL 기본값을
    사용한다. 관계의 격식만 대상에서 복원하고 심각도는 다시 분류하지 않는다.
    """
    resolved = severity or SituationSeverity.NORMAL
    ranges = {
        SituationSeverity.LIGHT: (1, 2, 20, 100),
        SituationSeverity.NORMAL: (2, 3, 60, 180),
        SituationSeverity.SERIOUS: (3, 5, 120, 350),
    }
    min_sentences, max_sentences, min_length, max_length = ranges[resolved]
    if request.target.value in {"FRIEND", "LOVER"}:
        formality = "CASUAL"
    elif request.target.value in {"TEACHER", "TEAM_LEAD"}:
        formality = "FORMAL"
    else:
        formality = "POLITE"

    return SituationProfile(
        severity=resolved,
        formality=formality,
        hasImpact=resolved != SituationSeverity.LIGHT,
        needsAccountability=resolved != SituationSeverity.LIGHT,
        needsNextAction=resolved != SituationSeverity.LIGHT,
        humorAllowed=resolved == SituationSeverity.LIGHT,
        minSentences=min_sentences,
        maxSentences=max_sentences,
        minLength=min_length,
        maxLength=max_length,
    )


def _situation_fit_score(
    result: ExcuseResult,
    profile: SituationProfile,
    issues: list[str],
    request: GenerateRequest | None = None,
) -> int:
    """모든 재생성이 미달일 때 사용자에게 돌려줄 더 나은 후보를 고른다."""
    excuse = result.excuse.strip()
    length = len(excuse)
    sentences = count_sentences(excuse)
    score = 0
    if math.ceil(profile.minLength * 0.85) <= length <= math.floor(
        profile.maxLength * 1.1
    ):
        score += 2
    if profile.minSentences <= sentences <= profile.maxSentences:
        score += 2
    if not profile.needsNextAction or _has_next_action(excuse):
        score += 2
    if profile.severity != SituationSeverity.SERIOUS or not any(
        marker in excuse for marker in ("ㅋㅋ", "ㅎㅎ", "우주의 기운", "외계인")
    ):
        score += 2
    if request is None or not validate_grounding(result, request):
        score += 2
    if not any("지나치게 비슷" in issue for issue in issues):
        score += 1
    return score


def _normalize_text(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", value).lower()


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _latest_message_relevance_issue(incoming: str, answer: str) -> str | None:
    """명확한 질문 유형만 검사해 과도한 오탐 없이 최신 질문 무시를 잡는다."""
    incoming_compact = _normalize_text(incoming)
    answer_compact = _normalize_text(answer)
    rules = (
        (
            ("왜", "이유"),
            (
                "때문",
                "해서",
                "라서",
                "므로",
                "바람에",
                "탓",
                "제가",
                "사실",
                "잠들",
                "깜빡",
                "잊",
                "착각",
                "실수",
                "놓쳤",
                "놓쳐",
                "확인하지못",
                "관리하지못",
                "관리를못",
                "제대로못",
                "부족했",
                "부족해서",
            ),
            "이유 질문에 대한 설명이 없음",
        ),
        (("언제", "몇시"), ("오늘", "내일", "지금", "분", "시", "까지"), "시간 질문에 대한 답이 없음"),
        (("어떻게", "어쩔"), ("바로", "지금", "확인", "수정", "보내", "올리", "하겠", "할게"), "해결 방법 질문에 대한 행동 답변이 없음"),
    )
    for incoming_markers, answer_markers, message in rules:
        if any(marker in incoming_compact for marker in incoming_markers) and not any(
            marker in answer_compact for marker in answer_markers
        ):
            return message
    return None


_GENERIC_AFTERMATH_WORDS = {
    "회의",
    "업무",
    "일정",
    "자료",
    "과제",
    "약속",
    "상황",
    "내용",
    "핵심",
    "다음",
    "질문",
    "답변",
    "죄송",
    "바로",
    "지금",
    "정말",
    "제가",
    "저도",
}

_AFTERMATH_VERIFICATION_SIGNALS = (
    "왜",
    "언제",
    "몇 시",
    "몇시",
    "어디",
    "누가",
    "진짜",
    "증거",
    "내역",
    "기록",
    "사진",
    "공지",
    "맞아",
    "맞나요",
    "확인",
    "그럼",
    "그러면",
)


def _meaningful_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for raw_token in re.findall(r"[0-9a-zA-Z가-힣]{2,}", value.lower()):
        token = raw_token
        for suffix in ("으로", "에서", "에게", "부터", "까지", "처럼", "은", "는", "이", "가", "을", "를", "에", "도"):
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                token = token[: -len(suffix)]
                break
        if len(token) >= 2 and token not in _GENERIC_AFTERMATH_WORDS:
            tokens.add(token)
    return tokens
