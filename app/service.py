"""AI 생성 흐름과 REPLY 품질 검증을 담당한다."""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from itertools import combinations

from fastapi import HTTPException

from app.config import Settings
from app.llm import CerebrasClient
from app.models import ExcuseResult, GenerateRequest, GenerationMode, SpringExcuseResponse
from app.prompts import build_system_prompt, build_user_prompt

logger = logging.getLogger("tongchoo.service")

REPLY_SIMILARITY_THRESHOLD = 0.70
REPLY_MIN_EXCUSE_CHARS = 25
REPLY_MIN_OPTION_CHARS = 8
REPLY_OPTION_COUNT = 3

REPLY_INTENT_CUES = (
    (("어디", "위치"), ("위치", "여기", "장소", "도착", "집")),
    (("언제", "시간", "몇"), ("시간", "오늘", "지금", "바로", "까지", "분")),
    (("믿", "신뢰"), ("믿", "신뢰", "약속", "행동", "바꾸")),
    (("왜", "이유"), ("미리", "확인", "연락", "공유", "놓쳤", "잘못")),
)
STOPWORDS = frozenset({"왜", "어떻게", "무슨", "이번", "그냥", "정말", "너무", "이제", "지금", "하고", "해서", "안", "또"})
RESPONSIBILITY_MARKERS = ("죄송", "미안", "잘못", "인정", "책임", "하겠", "드리겠", "바로", "확인", "공유", "보내", "연락", "정리")
QUALITY_MESSAGES = {
    "REPLY_TOO_SHORT": "답장 본문이 너무 짧습니다.",
    "REPLY_TOO_SIMILAR": "이전 답변과 너무 유사합니다.",
    "REPLY_INCOMING_IGNORED": "incomingMessage에 직접 반응하지 않았습니다.",
    "REPLY_OPTIONS_COUNT": "REPLY 모드의 replyOptions는 정확히 3개여야 합니다.",
    "REPLY_OPTION_TOO_SHORT": "replyOptions 중 너무 짧은 후보가 있습니다.",
    "REPLY_OPTIONS_TOO_SIMILAR": "replyOptions가 서로 너무 유사합니다.",
    "REPLY_OPTIONS_STRATEGY_WEAK": "replyOptions의 역할 구분이 약합니다.",
}


class ExcuseGenerationService:
    """프롬프트 구성, Cerebras 호출, REPLY 품질 재생성을 조율한다."""

    def __init__(self, settings: Settings):
        self.client = CerebrasClient(settings)

    async def generate(self, request: GenerateRequest, request_id: str) -> ExcuseResult:
        system_prompt = build_system_prompt()
        if request.mode != GenerationMode.REPLY:
            return await self._generate_once(request, request_id, system_prompt)

        return await self._generate_reply(request, request_id, system_prompt)

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

    # 두번째 생성
    async def _generate_reply(
        self,
        request: GenerateRequest,
        request_id: str,
        system_prompt: str,
    ) -> ExcuseResult:
        previous_answers = _previous_reply_answers(request)
        blocked_texts: list[str] | None = None
        failure_messages: list[str] | None = None

        for attempt in range(2):
            result = await self._generate_once(
                request,
                request_id,
                system_prompt,
                avoid_texts=blocked_texts,
                quality_failures=failure_messages,
            )
            failures = _reply_quality_failures(request, result, previous_answers)
            if not failures:
                if attempt:
                    logger.info("reply_quality_recovered request_id=%s", request_id)
                return result

            if attempt:
                logger.warning(
                    "reply_quality_failed request_id=%s failures=%s",
                    request_id,
                    _failure_codes(failures),
                )
                _raise_quality_error(failures)

            logger.warning(
                "reply_quality_retry request_id=%s failures=%s",
                request_id,
                _failure_codes(failures),
            )
            previous_answers.append(result.excuse)
            blocked_texts = _unique_texts(previous_answers + result.replyOptions)
            failure_messages = [QUALITY_MESSAGES[code] for code in failures]

        raise AssertionError("REPLY quality loop should return or raise")

    # 첫 번째 생성
    async def _generate_once(
        self,
        request: GenerateRequest,
        request_id: str,
        system_prompt: str,
        *,
        avoid_texts: list[str] | None = None,
        quality_failures: list[str] | None = None,
    ) -> ExcuseResult:
        user_prompt = build_user_prompt(
            request,
            max_memory_chars=self.client.settings.max_memory_chars,
            avoid_texts=avoid_texts,
            quality_failures=quality_failures,
        )
        return await self.client.generate(system_prompt, user_prompt, request_id)

# REPLY 품질 검증
def _reply_quality_failures(
    request: GenerateRequest,
    result: ExcuseResult,
    previous_answers: list[str],
) -> list[str]:
    failures: list[str] = []
    if len(_normalize(result.excuse)) < REPLY_MIN_EXCUSE_CHARS:
        failures.append("REPLY_TOO_SHORT")

    if _max_similarity(result.excuse, previous_answers) >= REPLY_SIMILARITY_THRESHOLD:
        failures.append("REPLY_TOO_SIMILAR")

    if not _addresses_incoming_message(request.incomingMessage or "", result.excuse):
        failures.append("REPLY_INCOMING_IGNORED")

    return failures + _reply_options_quality_failures(result.replyOptions)

# REPLY 품질 검증 실패시 HTTPException 발생
def _reply_options_quality_failures(options: list[str]) -> list[str]:
    if len(options) != REPLY_OPTION_COUNT:
        return ["REPLY_OPTIONS_COUNT"]

    normalized_options = [_normalize(option) for option in options]
    failures: list[str] = []
    if any(len(option) < REPLY_MIN_OPTION_CHARS for option in normalized_options):
        failures.append("REPLY_OPTION_TOO_SHORT")

    if any(
        _similarity(left, right) >= REPLY_SIMILARITY_THRESHOLD
        for left, right in combinations(options, 2)
    ):
        failures.append("REPLY_OPTIONS_TOO_SIMILAR")

    if len(normalized_options[0]) >= len(normalized_options[1]):
        failures.append("REPLY_OPTIONS_STRATEGY_WEAK")
    if not _contains_any(normalized_options[1], RESPONSIBILITY_MARKERS):
        failures.append("REPLY_OPTIONS_STRATEGY_WEAK")
    return failures


def _addresses_incoming_message(incoming_message: str, reply: str) -> bool:
    incoming = _normalize(incoming_message)
    normalized_reply = _normalize(reply)
    if not incoming or not normalized_reply:
        return False

    if any(term in normalized_reply for term in _content_terms(incoming_message)):
        return True
    if len(_bigrams(incoming) & _bigrams(normalized_reply)) >= 2:
        return True

    return any(
        _contains_any(incoming, question_words)
        and _contains_any(normalized_reply, response_words)
        for question_words, response_words in REPLY_INTENT_CUES
    )


def _previous_reply_answers(request: GenerateRequest) -> list[str]:
    answers = [request.currentExcuse] if request.currentExcuse else []
    answers.extend(turn.message for turn in request.conversation if turn.role.value == "assistant")
    return _unique_texts(answers)


def _max_similarity(candidate: str, previous: list[str]) -> float:
    return max((_similarity(candidate, item) for item in previous), default=0.0)


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _normalize(left), _normalize(right)).ratio()


def _normalize(value: str) -> str:
    return re.sub(r"[^가-힣a-zA-Z0-9]", "", value).lower()


def _content_terms(value: str) -> set[str]:
    return {
        term.lower()
        for term in re.findall(r"[가-힣]{2,}|[a-zA-Z0-9]{2,}", value)
        if term not in STOPWORDS
    }


def _bigrams(value: str) -> set[str]:
    return {value[index : index + 2] for index in range(len(value) - 1)}


def _contains_any(value: str, markers: tuple[str, ...]) -> bool:
    return any(marker in value for marker in markers)


def _unique_texts(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(value)
    return unique


def _failure_codes(failures: list[str]) -> str:
    return ",".join(failures)


def _raise_quality_error(failures: list[str]) -> None:
    code = (
        "REPLY_QUALITY_REPEATED"
        if "REPLY_TOO_SIMILAR" in failures
        else "REPLY_QUALITY_FAILED"
    )
    raise HTTPException(
        status_code=422,
        detail={
            "code": code,
            "message": "REPLY 답변이 재생성 후에도 품질 검사를 통과하지 못했습니다.",
            "failures": [
                {"code": failure, "message": QUALITY_MESSAGES[failure]}
                for failure in failures
            ],
        },
    )


def _excerpt(value: str, limit: int = 120) -> str:
    return value.replace("\n", " ").strip()[:limit]
