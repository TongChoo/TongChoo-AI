"""본문·후보와 후폭풍 검증 결과를 서로 분리해 표현한다."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import logging
import re
from typing import Callable

from app.models import ExcuseResult, GenerateRequest
from app.reply_quality import (
    ReplyQualityVerdict,
    classify_question_intent,
    deterministic_candidate_issues,
    relationship_register_issues,
    requires_privacy_boundary,
)

logger = logging.getLogger("tongchoo.quality_validator")


@dataclass(frozen=True)
class CreateQualityReport:
    body_issues: list[str]
    aftermath_issues: list[str]

    @property
    def issues(self) -> list[str]:
        return list(dict.fromkeys([*self.body_issues, *self.aftermath_issues]))

    @property
    def body_is_safe(self) -> bool:
        return not self.body_issues


def validate_create_result(
    result: ExcuseResult,
    body_checks: Callable[[ExcuseResult], list[str]],
    aftermath_checks: Callable[[ExcuseResult], list[str]],
) -> CreateQualityReport:
    return CreateQualityReport(
        body_issues=list(dict.fromkeys(body_checks(result))),
        aftermath_issues=list(dict.fromkeys(aftermath_checks(result))),
    )


class QualityValidator:
    """후보 집합·Judge·후폭풍 품질 검사를 한곳에서 담당한다."""

    def __init__(self, settings):
        self.settings = settings

    def reply_issues(self, result: ExcuseResult, request: GenerateRequest) -> list[str]:
        issues: list[str] = []
        candidates = list(result.replyOptions)
        if len(candidates) != 3:
            return ["replyOptions가 정확히 3개가 아님"]
        if result.excuse != candidates[0]:
            issues.append("기본 excuse가 첫 번째 답장 후보와 일치하지 않음")
        intent = classify_question_intent(request.incomingMessage or "")
        for candidate in candidates:
            issues.extend(deterministic_candidate_issues(candidate, request, intent))
        normalized = [_normalize_text(candidate) for candidate in candidates]
        for left_index, left in enumerate(normalized):
            if any(
                _similarity(left, right) >= self.settings.reply_similarity_threshold
                for right in normalized[left_index + 1 :]
            ):
                issues.append("답장 후보가 서로 지나치게 비슷함")
                break
        previous = [
            turn.message for turn in request.conversation if turn.role.value == "assistant"
        ]
        if request.currentExcuse:
            previous.append(request.currentExcuse)
        if any(
            any(
                _similarity(_normalize_text(candidate), _normalize_text(old))
                >= self.settings.reply_similarity_threshold
                for old in previous
            )
            for candidate in candidates
        ):
            issues.append("이전 라운드 답변을 거의 그대로 반복함")
        for candidate in candidates:
            relevance = _latest_message_relevance_issue(
                request.incomingMessage or "", candidate
            )
            if relevance:
                issues.append(relevance)
                break
        return list(dict.fromkeys(issues))

    def reply_judge_issues(
        self, verdict: ReplyQualityVerdict, request: GenerateRequest
    ) -> list[str]:
        issues = [
            f"{index}번 후보에 금지된 품질 문제가 있음"
            for index, candidate in enumerate(verdict.candidateScores, start=1)
            if candidate.hardViolation
        ]
        required = {"directness", "factuality", "registerScore", "fluency"}
        if len(verdict.candidateScores) != 3 or any(
            not required.issubset(candidate.model_fields_set)
            for candidate in verdict.candidateScores
        ):
            logger.warning("reply_judge_incomplete_verdict")
            return list(dict.fromkeys(issues))
        for index, candidate in enumerate(verdict.candidateScores, start=1):
            if candidate.score < self.settings.reply_candidate_min_score:
                issues.append(
                    f"{index}번 후보 Judge 점수가 "
                    f"{self.settings.reply_candidate_min_score}점 미만"
                )
        intent = classify_question_intent(request.incomingMessage or "")
        threshold = (
            self.settings.reply_privacy_diversity_min_score
            if requires_privacy_boundary(request, intent)
            else self.settings.reply_diversity_min_score
        )
        if verdict.diversityScore < threshold:
            issues.append(f"후보 집합의 의미 다양성이 {threshold}점 미만")
        return list(dict.fromkeys(issues))

    def create_judge_issues(self, verdict: ReplyQualityVerdict) -> list[str]:
        if len(verdict.candidateScores) != 3:
            return []
        issues: list[str] = []
        for index, candidate in enumerate(verdict.candidateScores, start=1):
            details = ", ".join(candidate.issues[:3])
            if candidate.hardViolation:
                message = f"{index}번 후보에 입력 밖 사실·말투·후폭풍 문제가 있음"
                issues.append(f"{message}: {details}" if details else message)
            elif candidate.score < self.settings.reply_candidate_min_score:
                issues.append(
                    f"{index}번 후보 품질 점수가 "
                    f"{self.settings.reply_candidate_min_score}점 미만"
                )
        issues.extend(verdict.issues[:3])
        return list(dict.fromkeys(issues))

    def aftermath_issues(
        self, result: ExcuseResult, request: GenerateRequest | None = None
    ) -> list[str]:
        issues: list[str] = []
        questions = [item.question.strip() for item in result.aftermath]
        anchors = _meaningful_tokens(result.excuse)
        for question in questions:
            if not question.endswith("?"):
                issues.append("후폭풍 질문이 상대방의 직접적인 의문문이 아님")
            has_anchor = bool(anchors & _meaningful_tokens(question))
            has_signal = any(marker in question for marker in _AFTERMATH_SIGNALS)
            if not has_anchor and not has_signal:
                issues.append("후폭풍 질문이 현재 변명의 주장이나 허점과 연결되지 않음")
            source = f"{result.excuse} {request.situation if request else ''}"
            if _has_unrequested_precision(question, source):
                issues.append("후폭풍 질문이 실제 대화보다 지나치게 수사식이거나 부자연스러움")
            if (
                request is not None
                and request.target.value == "PARENT"
                and relationship_register_issues(question, request)
            ):
                issues.append("후폭풍 질문 말투가 부모와 자녀 관계에 맞지 않음")
        for left_index, left in enumerate(questions):
            if any(
                _similarity(_normalize_text(left), _normalize_text(right)) >= 0.85
                for right in questions[left_index + 1 :]
            ):
                issues.append("후폭풍 질문이 서로 지나치게 비슷함")
                break
        return list(dict.fromkeys(issues))


_GENERIC_WORDS = {
    "회의", "업무", "일정", "자료", "과제", "약속", "상황", "내용", "핵심",
    "다음", "질문", "답변", "죄송", "바로", "지금", "정말", "제가", "저도",
}
_AFTERMATH_SIGNALS = (
    "왜", "언제", "몇 시", "몇시", "어디", "누가", "진짜", "증거", "내역",
    "기록", "사진", "공지", "맞아", "맞나요", "확인", "그럼", "그러면",
)


def _normalize_text(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", value).lower()


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _latest_message_relevance_issue(incoming: str, answer: str) -> str | None:
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


def _has_unrequested_precision(question: str, source: str) -> bool:
    asks_measurement = bool(re.search(r"(?:몇|\d+)\s*(?:초|분|시간|번|회)", question))
    asks_exact = "정확" in question and any(
        marker in question for marker in ("순간", "시점", "시간", "횟수")
    )
    source_has_measurement = bool(re.search(r"\d+\s*(?:초|분|시간|번|회|시)", source))
    return (asks_measurement or asks_exact) and not source_has_measurement


def _meaningful_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for raw in re.findall(r"[0-9a-zA-Z가-힣]{2,}", value.lower()):
        token = raw
        for suffix in ("으로", "에서", "에게", "부터", "까지", "처럼", "은", "는", "이", "가", "을", "를", "에", "도"):
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                token = token[: -len(suffix)]
                break
        if len(token) >= 2 and token not in _GENERIC_WORDS:
            tokens.add(token)
    return tokens
