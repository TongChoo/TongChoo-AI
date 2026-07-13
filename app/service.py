"""AI 생성 흐름과 REPLY 품질 검증을 담당한다."""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from app.config import Settings
from app.llm import CerebrasClient, api_error
from app.models import ExcuseResult, GenerateRequest, GenerationMode, SpringExcuseResponse
from app.prompts import (
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
        if request.mode == GenerationMode.REPLY:
            return await self._generate_reply_with_quality_gate(request, request_id)

        return await self._generate_create_with_quality_gate(request, request_id)

    async def _generate_create_with_quality_gate(
        self,
        request: GenerateRequest,
        request_id: str,
    ) -> ExcuseResult:
        """최초 변명과 직접 연결되지 않은 후폭풍 질문을 거절하고 재생성한다."""
        base_prompt = build_user_prompt(
            request,
            max_memory_chars=self.client.settings.max_memory_chars,
        )
        rejected: list[str] = []

        for attempt in range(self.client.settings.aftermath_quality_max_attempts):
            correction = ""
            if rejected:
                correction = (
                    "\n\n[직전 후폭풍 결과 거절 사유]\n- "
                    + "\n- ".join(rejected)
                    + "\n일반 업무 질문을 제거하고, 새로 작성한 excuse의 주장이나 "
                    "허점을 상대방이 직접 확인하는 질문으로 aftermath를 다시 작성하세요."
                )
            result = await self.client.generate(
                build_system_prompt(),
                base_prompt + correction,
                request_id,
            )
            rejected = self._aftermath_quality_issues(result)
            if not rejected:
                return result
            logger.warning(
                "aftermath_quality_retry request_id=%s attempt=%s issues=%s",
                request_id,
                attempt + 1,
                "; ".join(rejected),
            )

        raise api_error(
            422,
            "AFTERMATH_QUALITY_REJECTED",
            "변명과 연결된 후폭풍 질문을 만들지 못했습니다. 다시 시도해주세요.",
        )

    async def _generate_reply_with_quality_gate(
        self,
        request: GenerateRequest,
        request_id: str,
    ) -> ExcuseResult:
        """답장 결과의 반복·후보 중복·최신 질문 누락을 검사하고 제한 재생성한다."""
        base_prompt = build_reply_user_prompt(
            request,
            max_memory_chars=self.client.settings.max_memory_chars,
        )
        rejected: list[str] = []

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
            rejected = self._reply_quality_issues(result, request)
            if not rejected:
                return result
            logger.warning(
                "reply_quality_retry request_id=%s attempt=%s issues=%s",
                request_id,
                attempt + 1,
                "; ".join(rejected),
            )

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
        candidates = [result.excuse, *result.replyOptions]
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

    def _aftermath_quality_issues(self, result: ExcuseResult) -> list[str]:
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
            if not has_excuse_anchor and not has_verification_signal:
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
        result = await self.generate(request, request_id)
        logger.info(
            "ai_generation_result request_id=%s mode=%s round=%s excuse=%s",
            request_id,
            request.mode.value,
            request.roundNumber,
            _excerpt(result.excuse),
        )
        return SpringExcuseResponse.from_result(result)

def _excerpt(value: str, limit: int = 120) -> str:
    return value.replace("\n", " ").strip()[:limit]


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
        (("왜", "이유"), ("때문", "해서", "라서", "제가", "사실"), "이유 질문에 대한 설명이 없음"),
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
