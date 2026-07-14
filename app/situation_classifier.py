"""상황 심각도·관계 분류와 결정적 보정 규칙."""

from __future__ import annotations

import re

from app.models import GenerateRequest, GenerationMode, SituationProfile, SituationSeverity
from app.prompts import build_classification_prompt, build_classification_system_prompt


_RANGES = {
    SituationSeverity.LIGHT: (1, 2, 20, 100),
    SituationSeverity.NORMAL: (2, 3, 60, 180),
    SituationSeverity.SERIOUS: (3, 5, 120, 350),
}
_HIGH_STAKES = ("면접", "시험", "발표", "고객", "사고", "수술", "장례", "마감")


class SituationClassifier:
    def __init__(self, client):
        self.client = client

    async def classify(self, request: GenerateRequest, request_id: str) -> SituationProfile:
        classified = await self.client.classify_situation(
            build_classification_system_prompt(),
            build_classification_prompt(request),
            request_id,
        )
        return apply_guardrails(classified, request)

    @staticmethod
    def persisted(request: GenerateRequest) -> SituationProfile:
        return profile_from_persisted_severity(request).for_mode(request.mode)


def profile_from_persisted_severity(request: GenerateRequest) -> SituationProfile:
    severity = request.situationSeverity or SituationSeverity.NORMAL
    minimum, maximum, min_length, max_length = _RANGES[severity]
    if request.target.value in {"FRIEND", "LOVER"}:
        formality = "CASUAL"
    elif request.target.value in {"TEACHER", "TEAM_LEAD"}:
        formality = "FORMAL"
    else:
        formality = "POLITE"
    accountable = severity == SituationSeverity.SERIOUS
    return SituationProfile(
        severity=severity,
        formality=formality,
        hasImpact=severity != SituationSeverity.LIGHT,
        needsAccountability=accountable,
        needsNextAction=accountable,
        humorAllowed=severity == SituationSeverity.LIGHT,
        minSentences=minimum,
        maxSentences=maximum,
        minLength=min_length,
        maxLength=max_length,
    )


def apply_guardrails(profile: SituationProfile, request: GenerateRequest) -> SituationProfile:
    severity = profile.severity
    delay_match = re.search(r"(\d+)\s*분.{0,12}(?:지각|늦)", request.situation)
    if (
        delay_match
        and int(delay_match.group(1)) <= 10
        and not any(marker in request.situation for marker in _HIGH_STAKES)
    ):
        severity = SituationSeverity.LIGHT
    minimum, maximum, min_length, max_length = _RANGES[severity]
    downgraded_to_light = (
        severity == SituationSeverity.LIGHT
        and profile.severity != SituationSeverity.LIGHT
    )
    return profile.model_copy(update={
        "severity": severity,
        "hasImpact": False if downgraded_to_light else profile.hasImpact,
        "needsAccountability": (
            False if downgraded_to_light else profile.needsAccountability
        ),
        "needsNextAction": False if downgraded_to_light else profile.needsNextAction,
        "humorAllowed": profile.humorAllowed and severity == SituationSeverity.LIGHT,
        "minSentences": minimum,
        "maxSentences": maximum,
        "minLength": min_length,
        "maxLength": max_length,
    })
