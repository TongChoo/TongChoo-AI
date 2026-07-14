"""AI 생성 흐름과 REPLY 품질 검증을 담당한다."""

from __future__ import annotations

import logging
import math
import re
from difflib import SequenceMatcher
from app.config import Settings
from app.llm import CerebrasClient, ReplyJudgeParseError, api_error
from app.generation_policy import (
    create_retry_instruction,
    repair_aftermath_only,
    safe_create_body,
    safe_reply_body,
)
from app.quality_validator import QualityValidator, validate_create_result
from app.situation_classifier import (
    SituationClassifier,
    apply_guardrails as _apply_situation_guardrails,
    profile_from_persisted_severity as _profile_from_persisted_severity,
)
from app.models import (
    Aftermath,
    ExcuseResult,
    GenerateRequest,
    GenerationMode,
    SituationProfile,
    SituationSeverity,
    SpringExcuseResponse,
    SuspicionLevel,
)
from app.prompts import (
    build_evolve_system_prompt,
    build_evolve_user_prompt,
    build_classification_prompt,
    build_classification_system_prompt,
    build_create_judge_system_prompt,
    build_create_judge_user_prompt,
    build_reply_judge_system_prompt,
    build_reply_judge_user_prompt,
    build_reply_system_prompt,
    build_reply_user_prompt,
    build_system_prompt,
    build_user_prompt,
)
from app.reply_quality import (
    ReplyQualityVerdict,
    classify_question_intent,
    deterministic_candidate_issues,
    fluency_issues,
    is_formal_relationship,
    relationship_register_issues,
    register_consistency_issues,
    requires_privacy_boundary,
)

logger = logging.getLogger("tongchoo.service")

class ExcuseGenerationService:
    """프롬프트 구성, Cerebras 호출, REPLY 품질 재생성을 조율한다."""

    def __init__(self, settings: Settings):
        self.client = CerebrasClient(settings)
        self.situation_classifier = SituationClassifier(self.client)
        self.validator = QualityValidator(settings)

    async def generate(self, request: GenerateRequest, request_id: str) -> ExcuseResult:
        result, _ = await self._generate_with_profile(request, request_id)
        return result

    async def _generate_with_profile(
        self,
        request: GenerateRequest,
        request_id: str,
    ) -> tuple[ExcuseResult, SituationProfile]:
        demo_result = _demo_safe_result(request) if self.client.settings.demo_safe_mode else None
        if demo_result is not None:
            profile = _profile_from_persisted_severity(request).model_copy(
                update={"severity": SituationSeverity.NORMAL}
            )
            logger.warning("demo_safe_result request_id=%s mode=%s", request_id, request.mode.value)
            return demo_result, profile

        if request.mode == GenerationMode.REPLY:
            profile = self.situation_classifier.persisted(request)
            result = await self._generate_reply_with_quality_gate(
                request, profile, request_id
            )
            return result, profile

        profile = await self.situation_classifier.classify(request, request_id)
        result = await self._generate_create_with_quality_gate(
            request, profile, request_id
        )
        return result, profile

    async def _generate_create_with_quality_gate(
        self,
        request: GenerateRequest,
        profile: SituationProfile,
        request_id: str,
    ) -> ExcuseResult:
        """최초 변명과 직접 연결되지 않은 후폭풍 질문을 거절하고 재생성한다."""
        if request.mode == GenerationMode.EVOLVE:
            system_prompt = build_evolve_system_prompt()
            base_prompt = build_evolve_user_prompt(
                request,
                max_memory_chars=self.client.settings.max_memory_chars,
            )
        else:
            system_prompt = build_system_prompt()
            base_prompt = build_user_prompt(
                request,
                profile=profile,
                max_memory_chars=self.client.settings.max_memory_chars,
            )
        rejected: list[str] = []
        last_result: ExcuseResult | None = None
        candidates: list[ExcuseResult] = []

        max_attempts = max(
            self.client.settings.aftermath_quality_max_attempts,
            self.client.settings.situation_quality_max_attempts,
        )
        for attempt in range(max_attempts):
            correction = ""
            if rejected:
                correction = create_retry_instruction(rejected)
            result = await self.client.generate(
                system_prompt,
                base_prompt + correction,
                request_id,
                # CREATE는 새 사실을 상상하는 다양성보다 입력 사실 보존이 우선이다.
                # REPLY의 표현 다양성 온도와 분리해 모든 상황에 같은 정책을 적용한다.
                temperature=0.2,
            )
            last_result = result
            candidates.append(result)
            report = validate_create_result(
                result,
                lambda value: [
                    *validate_situation_fit(value, profile),
                    *validate_grounding_and_register(value, request, include_options=True),
                ],
                lambda value: self.validator.aftermath_issues(value, request),
            )
            # 본문·후보가 모두 안전하면 후폭풍 하나 때문에 전체 생성을 다시 하지 않는다.
            if report.body_is_safe and report.aftermath_issues:
                repaired = repair_aftermath_only(result, request)
                repaired_aftermath_issues = self.validator.aftermath_issues(repaired, request)
                if not repaired_aftermath_issues:
                    result = repaired
                    last_result = repaired
                    candidates[-1] = repaired
                    report = validate_create_result(
                        repaired,
                        lambda value: [
                            *validate_situation_fit(value, profile),
                            *validate_grounding_and_register(value, request, include_options=True),
                        ],
                        lambda value: self.validator.aftermath_issues(value, request),
                    )
            rejected = report.issues
            if not rejected:
                try:
                    verdict = await self.client.judge_reply(
                        build_create_judge_system_prompt(),
                        build_create_judge_user_prompt(request, profile, result),
                        request_id,
                    )
                except ReplyJudgeParseError:
                    if not self.client.settings.reply_judge_fail_open:
                        rejected = ["CREATE 품질 판정을 해석하지 못함"]
                else:
                    rejected = self.validator.create_judge_issues(verdict)
            if not rejected:
                return result
            logger.warning(
                "aftermath_quality_retry request_id=%s attempt=%s issues=%s",
                request_id,
                attempt + 1,
                "; ".join(rejected),
            )

        if last_result is not None:
            safe_body = next(
                (
                    candidate
                    for candidate in reversed(candidates)
                    if not validate_situation_fit(candidate, profile)
                    and not validate_grounding_and_register(
                        candidate, request, include_options=False
                    )
                ),
                None,
            )
            fallback = (
                safe_create_body(safe_body, request, profile)
                if safe_body is not None
                else safe_create_body(last_result, request, profile)
            )
        else:
            fallback = None
        if fallback is not None:
            fallback_issues = [
                *validate_situation_fit(fallback, profile),
                *validate_grounding_and_register(
                    fallback, request, include_options=True
                ),
                *self.validator.aftermath_issues(fallback, request),
            ]
            if not fallback_issues:
                logger.warning(
                    "create_quality_accountability_fallback request_id=%s",
                    request_id,
                )
                return fallback

        raise api_error(
            422,
            "AFTERMATH_QUALITY_REJECTED",
            "변명과 연결된 후폭풍 질문을 만들지 못했습니다. 다시 시도해주세요.",
        )

    async def _generate_reply_with_quality_gate(
        self,
        request: GenerateRequest,
        profile: SituationProfile,
        request_id: str,
    ) -> ExcuseResult:
        """답장 후보를 결정적 규칙과 독립 Judge로 검사하고 한 번만 재생성한다."""
        base_prompt = build_reply_user_prompt(
            request,
            profile=profile,
            max_memory_chars=self.client.settings.max_memory_chars,
        )
        rejected: list[str] = []
        last_result: ExcuseResult | None = None
        intent = classify_question_intent(request.incomingMessage or "")
        privacy_boundary_required = requires_privacy_boundary(request, intent)

        for attempt in range(self.client.settings.reply_quality_max_attempts):
            correction = ""
            if rejected:
                correction = (
                    "\n\n[직전 결과 거절 사유]\n- "
                    + "\n- ".join(rejected)
                    + "\n직전 문장을 반복하지 말고 최신 상대 메시지에 직접 답하는 "
                    "서로 다른 후보를 새로 작성하세요."
                )
                if privacy_boundary_required:
                    correction += (
                        "\n이번 질문은 공개 가능한 근거가 없는 상세 요구입니다. "
                        "세 후보 모두 새 이유를 만들거나 '개인 사정'을 되풀이하지 말고, "
                        "자세한 내용은 말씀드리기 어렵다는 경계와 양해 요청을 직접 "
                        "포함하세요. 예: '개인적인 부분이라 자세히 말씀드리기 "
                        "어렵습니다. 양해 부탁드립니다.'"
                    )
            result = await self.client.generate(
                build_reply_system_prompt(),
                base_prompt + correction,
                request_id,
            )
            result = _canonicalize_reply_result(result)
            last_result = result
            rejected = self.validator.reply_issues(result, request)
            rejected.extend(
                validate_situation_fit(
                    result,
                    profile,
                    require_apology=False,
                    enforce_minimum=profile.severity == SituationSeverity.SERIOUS,
                )
            )
            rejected = list(dict.fromkeys(rejected))
            if not rejected and len(result.replyOptions) == 3:
                try:
                    verdict = await self.client.judge_reply(
                        build_reply_judge_system_prompt(),
                        build_reply_judge_user_prompt(request, result.replyOptions),
                        request_id,
                    )
                except ReplyJudgeParseError:
                    logger.warning(
                        "reply_judge_parse_error request_id=%s fail_open=%s",
                        request_id,
                        self.client.settings.reply_judge_fail_open,
                    )
                    if not self.client.settings.reply_judge_fail_open:
                        rejected = ["답장 품질 판정을 해석하지 못함"]
                else:
                    rejected = self.validator.reply_judge_issues(verdict, request)
            if not rejected:
                return result
            logger.warning(
                "reply_quality_retry request_id=%s attempt=%s issues=%s",
                request_id,
                attempt + 1,
                "; ".join(rejected),
            )

        if privacy_boundary_required and last_result is not None:
            fallback = _privacy_boundary_fallback(last_result, request)
            fallback_issues = self.validator.reply_issues(fallback, request)
            fallback_issues.extend(
                validate_situation_fit(
                    fallback,
                    profile,
                    require_apology=False,
                    enforce_minimum=profile.severity == SituationSeverity.SERIOUS,
                )
            )
            if not fallback_issues:
                logger.warning(
                    "reply_quality_privacy_fallback request_id=%s attempts=%s",
                    request_id,
                    self.client.settings.reply_quality_max_attempts,
                )
                return fallback
            logger.error(
                "reply_quality_privacy_fallback_rejected request_id=%s issues=%s",
                request_id,
                "; ".join(fallback_issues),
            )

        if last_result is not None and profile.severity == SituationSeverity.SERIOUS:
            fallback = safe_reply_body(last_result, request)
            fallback_issues = validate_grounding_and_register(fallback, request)
            fallback_issues.extend(validate_situation_fit(
                fallback, profile, require_apology=False, enforce_minimum=False
            ))
            if not fallback_issues:
                logger.warning("reply_quality_accountability_fallback request_id=%s", request_id)
                return fallback

        raise api_error(
            422,
            "REPLY_QUALITY_REJECTED",
            "상황에 맞는 답장 후보를 만들지 못했습니다. 다시 시도해주세요.",
        )

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


_DEMO_SITUATION = "팀 프로젝트 발표 자료를 아직 하나도 못 만들었는데 30분 뒤가 발표야"


def _demo_safe_result(request: GenerateRequest) -> ExcuseResult | None:
    """발표 영상용으로 합의한 한 시나리오만 결정적인 안전 결과를 반환한다."""
    if request.situation.strip() != _DEMO_SITUATION:
        return None

    incoming = (request.incomingMessage or "").strip()
    if (
        request.mode == GenerationMode.REPLY
        and "왜" in incoming
        and any(marker in incoming for marker in ("늦", "준비"))
    ):
        options = [
            "제가 일정을 제대로 관리하지 못했습니다. 미리 준비하지 못해 죄송합니다.",
            "자료 준비 일정을 놓친 제 잘못입니다. 변명할 여지가 없습니다. 죄송합니다.",
            "제가 준비 시간을 제대로 관리하지 못했습니다. 늦어진 점 정말 죄송합니다.",
        ]
        follow_up = "그럼 지금부터 어떻게 진행할 건가요?"
    elif (
        request.mode == GenerationMode.REPLY
        and "다음" in incoming
        and any(marker in incoming for marker in ("미리", "준비"))
    ):
        options = [
            "네, 죄송합니다. 다음부터는 마감 전에 준비 상황을 점검하고 미리 공유하겠습니다.",
            "네, 알겠습니다. 같은 일이 반복되지 않도록 준비 일정을 미리 확인하겠습니다.",
            "말씀해주신 부분 명심하겠습니다. 다음에는 준비 상황을 사전에 공유드리겠습니다.",
        ]
        follow_up = "다음에는 준비 상황을 언제 공유할 건가요?"
    elif request.mode == GenerationMode.CREATE:
        options = [
            "발표 자료를 미리 준비하지 못한 제 잘못입니다. 정말 죄송합니다. 지금부터 핵심 내용과 발표 순서를 우선 정리하겠습니다.",
            "발표 준비가 늦어진 점 죄송합니다. 제 일정 관리가 부족했습니다. 지금부터 발표에 필요한 핵심 내용을 먼저 정리하겠습니다.",
            "자료를 준비하지 못해 팀에 부담을 드린 점 죄송합니다. 우선 발표 흐름과 핵심 내용을 정리하겠습니다.",
        ]
        follow_up = "발표 자료는 지금부터 어떻게 준비할 건가요?"
    else:
        return None

    return ExcuseResult(
        excuse=options[0],
        recommendedAction="잘못을 인정하고 발표에 필요한 핵심 내용부터 정리한다.",
        likelyFollowUp=follow_up,
        replyOptions=options,
        successRate=72,
        realism=4,
        persuasion=4,
        suspicionLevel=SuspicionLevel.LOW,
        riskFactors=["발표 준비 시간이 부족함"],
        aftermath=[Aftermath(
            when="즉시",
            dayOffset=0,
            question=follow_up,
            collapseRate=35,
        )],
        remember=["발표 자료를 미리 준비하지 못함", "30분 뒤 발표 예정"],
    )


_APOLOGY_MARKERS = ("죄송", "미안", "사과", "송구")
_NEXT_ACTION_MARKERS = (
    "지금", "바로", "즉시", "그만", "멈추", "조용", "자제", "확인",
    "수정", "공유", "전달", "제출", "정리", "처리", "복구", "사과드리",
    "재발", "다시는", "하겠습니다", "할게",
)
_UNSUPPORTED_CAUSE_MARKERS = (
    "교통", "정전", "알람", "병원", "몸살", "고장", "오류", "배터리",
    "놀라서", "무심코", "흥분해서", "깜빡해서", "정신이 없어서", "급한 전화",
    "저의 착오", "제 착오",
)
_UNSUPPORTED_STATE_MARKERS = (
    "수정된 자료", "완성된 자료", "자료를 완성", "작업을 완료",
)
_TIME_PROMISE_PATTERN = re.compile(
    r"(?:오늘|내일|모레)(?:\s*(?:오전|오후))?(?:\s*\d+\s*시)?(?:까지|안으로)?|"
    r"(?:오전|오후)\s*\d+\s*시(?:까지)?|"
    r"\d+\s*(?:분|시간)\s*(?:뒤|안에|까지)"
)
_NEW_CAUSE_ASSERTION_PATTERN = re.compile(r"(?:원인은|원인이|원인으로|원인입니다)")
_CAUSAL_DETAIL_PATTERN = re.compile(
    r"[가-힣A-Za-z0-9 ]{2,30}(?:때문에|탓에|실수로|착오로)"
)
_ASSERTED_PROGRESS_PATTERN = re.compile(
    r"(?:현재.{0,45}(?:하고|진행\s*중|작업\s*중|처리\s*중)|"
    r"지금(?!부터).{0,45}(?:하고\s*있|진행\s*중|작업\s*중|처리\s*중))"
)
_UNSUPPORTED_LIVE_STATE_PATTERN = re.compile(
    r"(?:(?:지금\s*)?(?:가고|이동하고|출발하고)\s*있|가는\s*중|이동\s*중|"
    r"곧\s*도착|도착할게|도착하겠습니다)"
)
_ASSERTED_COMPLETION_MARKERS = (
    "도착했", "완료했", "끝냈", "제출했", "전달했", "보냈", "수정했",
)
_NEW_ARTIFACT_STATE_PATTERN = re.compile(
    r"(?:수정|업데이트|완성|보완)(?:된|한)\s*(?:자료|파일|문서|결과물)"
)
_NEW_SYSTEM_COMMITMENT_PATTERN = re.compile(
    r"(?:시스템|도구|프로세스|절차).{0,20}(?:도입|구축|설치|구매)하겠습니다"
)


def count_sentences(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(
        1,
        len([chunk for chunk in re.split(r"[.!?]+|\n+", stripped) if chunk.strip()]),
    )


def validate_situation_fit(
    result: ExcuseResult,
    profile: SituationProfile,
    *,
    require_apology: bool = True,
    enforce_minimum: bool = True,
) -> list[str]:
    """심각도에 맞는 길이와 잘못 인정·사과·수습 행동을 강제한다."""
    excuse = result.excuse.strip()
    issues: list[str] = []
    minimum_length = math.ceil(profile.minLength * 0.9)
    maximum_length = math.floor(profile.maxLength * 1.1)
    sentence_count = count_sentences(excuse)
    if enforce_minimum and len(excuse) < minimum_length:
        issues.append("상황 심각도에 비해 답변이 너무 짧습니다.")
    if len(excuse) > maximum_length:
        issues.append("상황에 비해 답변이 지나치게 깁니다.")
    if enforce_minimum and sentence_count < profile.minSentences:
        issues.append("상황 심각도에 필요한 문장 수가 부족합니다.")
    if sentence_count > profile.maxSentences:
        issues.append("답변 문장 수가 지나치게 많습니다.")
    if (
        require_apology
        and profile.needsAccountability
        and not any(marker in excuse for marker in _APOLOGY_MARKERS)
    ):
        issues.append("잘못을 한 상황인데 상대에게 직접 사과하지 않았습니다.")
    if profile.needsNextAction and not any(
        marker in excuse for marker in _NEXT_ACTION_MARKERS
    ):
        issues.append("잘못을 수습하기 위한 다음 행동이 없습니다.")
    return issues


def validate_grounding_and_register(
    result: ExcuseResult,
    request: GenerateRequest,
    *,
    include_options: bool = True,
) -> list[str]:
    """입력에 없던 흔한 원인과 한 문장 안의 말투 혼용을 차단한다."""
    source = request.situation
    issues: list[str] = []
    candidates = [result.excuse]
    if include_options:
        candidates.extend(result.replyOptions)
    candidates = list(dict.fromkeys(candidates))
    for candidate in candidates:
        if any(
            marker in candidate and marker not in source
            for marker in _UNSUPPORTED_CAUSE_MARKERS
        ):
            issues.append("입력에 없는 원인이나 사건을 새로 만들었습니다.")
        if any(
            marker in candidate and marker not in source
            for marker in _UNSUPPORTED_STATE_MARKERS
        ):
            issues.append("입력에 없는 작업 상태를 완료했거나 확정한 것처럼 만들었습니다.")
        if any(
            match.group(0) not in source
            for match in _TIME_PROMISE_PATTERN.finditer(candidate)
        ):
            issues.append("입력에 없는 구체적인 시간 약속을 만들었습니다.")
        if (
            _NEW_CAUSE_ASSERTION_PATTERN.search(candidate)
            and not _NEW_CAUSE_ASSERTION_PATTERN.search(source)
        ):
            issues.append("입력에 없는 원인을 사실처럼 단정했습니다.")
        for cause_match in _CAUSAL_DETAIL_PATTERN.finditer(candidate):
            if cause_match.group(0).strip() not in source:
                issues.append("입력에 없는 구체적인 인과관계를 새로 만들었습니다.")
                break
        if (
            _ASSERTED_PROGRESS_PATTERN.search(candidate)
            and not re.search(r"(?:현재|지금|진행\s*중|하고\s*있)", source)
        ):
            issues.append("입력에 없는 진행 상태를 이미 수행 중인 것처럼 만들었습니다.")
        if (
            _UNSUPPORTED_LIVE_STATE_PATTERN.search(candidate)
            and not _UNSUPPORTED_LIVE_STATE_PATTERN.search(source)
        ):
            issues.append("입력에 없는 이동·도착 상태를 단정했습니다.")
        if any(
            marker in candidate and marker not in source
            for marker in _ASSERTED_COMPLETION_MARKERS
        ):
            issues.append("입력에 없는 완료·도착 상태를 새로 만들었습니다.")
        if (
            _NEW_ARTIFACT_STATE_PATTERN.search(candidate)
            and not _NEW_ARTIFACT_STATE_PATTERN.search(source)
        ):
            issues.append("입력에 없는 수정·완성 산출물을 새로 만들었습니다.")
        if (
            _NEW_SYSTEM_COMMITMENT_PATTERN.search(candidate)
            and not _NEW_SYSTEM_COMMITMENT_PATTERN.search(source)
        ):
            issues.append("입력에 없는 시스템·도구 도입을 약속했습니다.")
        issues.extend(register_consistency_issues(candidate))
        issues.extend(relationship_register_issues(candidate, request))
        issues.extend(fluency_issues(candidate))
    return list(dict.fromkeys(issues))


def _has_unrequested_precision(question: str, source: str) -> bool:
    """원문에 없는 초·분·횟수·정확 시점을 캐묻는 질문을 일반 규칙으로 잡는다."""
    asks_for_measurement = bool(
        re.search(r"(?:몇|\d+)\s*(?:초|분|시간|번|회)", question)
    )
    asks_for_exact_point = "정확" in question and any(
        marker in question for marker in ("순간", "시점", "시간", "횟수")
    )
    source_has_measurement = bool(
        re.search(r"\d+\s*(?:초|분|시간|번|회|시)", source)
    )
    return (asks_for_measurement or asks_for_exact_point) and not source_has_measurement


def _canonicalize_reply_result(result: ExcuseResult) -> ExcuseResult:
    """UI가 보여 줄 세 후보를 REPLY 결과의 유일한 기준으로 맞춘다."""
    options = [option.strip() for option in result.replyOptions if option.strip()]
    if len(options) != 3:
        return result.model_copy(update={"replyOptions": options})
    return result.model_copy(
        update={
            "excuse": options[0],
            "replyOptions": options,
        }
    )


def _privacy_boundary_fallback(
    result: ExcuseResult,
    request: GenerateRequest,
) -> ExcuseResult:
    """공개할 사실이 없을 때 검증된 문장으로 422 대신 안전한 답장을 제공한다.

    제공자가 재생성 후에도 최신 상세 질문을 되풀이하는 경우에만 사용한다. 새 이유를
    만들지 않고 공개 거절·양해·기존 참석 결론이라는 서로 다른 역할로 세 후보를 만든다.
    """

    if is_formal_relationship(request):
        options = [
            "개인적인 부분이라 자세히 말씀드리기 어렵습니다. 양해 부탁드립니다.",
            "죄송하지만 구체적인 내용은 말씀드리기 어려운 점 이해 부탁드립니다.",
            "참석이 어렵다는 말씀은 그대로입니다. 사적인 부분은 공개하기 어려운 점 양해 부탁드립니다.",
        ]
    else:
        options = [
            "개인적인 부분이라 자세히 말하기는 어려워. 이해해 줘.",
            "미안하지만 개인적인 부분이라 구체적으로 말하기 어려워.",
            "참석하기 어렵다는 건 그대로야. 사적인 부분은 공개하지 않을게.",
        ]

    return result.model_copy(
        update={
            "excuse": options[0],
            "replyOptions": options,
        }
    )


def _accountability_aftermath_fallback(
    result: ExcuseResult,
    request: GenerateRequest,
) -> ExcuseResult:
    """본문은 건드리지 않고 관계에 맞는 일반 수습 확인 질문만 보정한다."""
    if request.target.value == "PARENT":
        question = "그럼 지금 말한 대로 바로 행동을 고치고 수습할 수 있겠어?"
    elif is_formal_relationship(request):
        question = "그럼 말씀하신 수습 행동을 지금 바로 진행할 수 있나요?"
    else:
        question = "그럼 지금 말한 수습 행동을 바로 할 수 있어?"
    fallback_aftermath = result.aftermath[0].model_copy(update={
        "when": "즉시",
        "dayOffset": 0,
        "question": question,
        "collapseRate": 35,
    })
    return result.model_copy(update={"aftermath": [fallback_aftermath]})


def _normalize_text(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", value).lower()


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _latest_message_relevance_issue(incoming: str, answer: str) -> str | None:
    """시간·방법 질문처럼 형식적으로 확인 가능한 직접 대응을 검사한다."""
    incoming_compact = _normalize_text(incoming)
    answer_compact = _normalize_text(answer)
    rules = (
        (("언제", "몇시"), ("오늘", "내일", "지금", "분", "시", "까지"), "시간 질문에 대한 답이 없음"),
        (("어떻게", "어쩔"), ("바로", "지금", "확인", "수정", "보내", "올리", "하겠", "할게"), "방법 질문에 대한 행동 답변이 없음"),
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
