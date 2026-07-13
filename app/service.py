"""AI 생성 흐름과 REPLY 품질 검증을 담당한다.

HTTP 계층은 Spring 요청을 표준 ``GenerateRequest``로 바꾸는 데만 집중하고,
이 모듈은 프롬프트 구성 → Cerebras 호출 → 결과 품질 검증을 순서대로 수행한다.
DB 저장, 사용자 인증, 대화 계보 조회는 Spring의 책임이므로 이 파일에서는 절대
직접 처리하지 않는다.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from itertools import combinations

from fastapi import HTTPException

from app.config import Settings
from app.llm import CerebrasClient
from app.models import (
    ExcuseResult,
    GenerateRequest,
    GenerationMode,
    SpringExcuseResponse,
)
from app.prompts import build_system_prompt, build_user_prompt

logger = logging.getLogger("tongchoo.service")

# 유사도는 공백·문장부호를 제거한 문장 기준으로 계산한다. 70% 이상이면 단어 몇 개만
# 바꾼 반복 답변으로 보고 한 번 더 생성한다.
REPLY_SIMILARITY_THRESHOLD = 0.70
# ``ExcuseResult`` 스키마의 최소 길이보다 한 단계 높은 품질 기준이다. 너무 짧은
# "죄송합니다"류 답장을 통과시키지 않기 위해 한글·영문·숫자만 25자 이상을 요구한다.
REPLY_MIN_EXCUSE_CHARS = 25
# replyOptions는 복사해 보낼 문장이므로 단순 감탄사나 한 단어 후보를 제외한다.
REPLY_MIN_OPTION_CHARS = 8
# REPLY UI는 각 역할이 다른 세 후보를 전제로 한다. CREATE/EVOLVE의 전역 스키마와
# 달리 이 서비스 계층에서만 정확히 세 개를 강제한다.
REPLY_OPTION_COUNT = 3

# 질문 단어가 답장에 그대로 남지 않을 수 있으므로, 자주 나오는 질문 의도와 이를
# 직접 다뤘다고 볼 수 있는 응답 단서를 함께 둔다. 이는 LLM의 의미 판단을 대체하지
# 않고 질문과 무관한 범용 사과문을 걸러내는 가벼운 안전망이다.
REPLY_INTENT_CUES = (
    (("어디", "위치"), ("위치", "여기", "장소", "도착", "집")),
    (("언제", "시간", "몇"), ("시간", "오늘", "지금", "바로", "까지", "분")),
    (("믿", "신뢰"), ("믿", "신뢰", "약속", "행동", "바꾸")),
    (("왜", "이유"), ("미리", "확인", "연락", "공유", "놓쳤", "잘못")),
)
# 조사·부사처럼 답변이 질문을 실제로 다뤘는지 판별하는 데 도움이 적은 단어는
# 키워드 겹침 검사에서 제외한다.
STOPWORDS = frozenset(
    {
        "왜",
        "어떻게",
        "무슨",
        "이번",
        "그냥",
        "정말",
        "너무",
        "이제",
        "지금",
        "하고",
        "해서",
        "안",
        "또",
    }
)
# 두 번째 후보(정중·책임 답장)에 들어가야 할 책임 인정 또는 수습 행동의 대표 어간이다.
RESPONSIBILITY_MARKERS = (
    "죄송",
    "미안",
    "잘못",
    "인정",
    "책임",
    "하겠",
    "드리겠",
    "바로",
    "확인",
    "공유",
    "보내",
    "연락",
    "정리",
)
# 오류 코드는 로그·HTTP 응답·재생성 프롬프트에서 공통으로 사용한다. 메시지를 한 곳에
# 모아 두면 검증 규칙을 추가해도 사용자 안내 문구가 빠지지 않는다.
QUALITY_MESSAGES = {
    "REPLY_TOO_SHORT": "답장 본문이 너무 짧습니다.",
    "REPLY_TOO_SIMILAR": "이전 답변과 너무 유사합니다.",
    "REPLY_INCOMING_IGNORED": "incomingMessage에 직접 반응하지 않았습니다.",
    "REPLY_OPTIONS_COUNT": "REPLY 모드의 replyOptions는 정확히 3개여야 합니다.",
    "REPLY_OPTION_TOO_SHORT": "replyOptions 중 너무 짧은 후보가 있습니다.",
    "REPLY_OPTIONS_TOO_SIMILAR": "replyOptions가 서로 너무 유사합니다.",
    "REPLY_OPTION_1_NOT_SHORTER": "1번 후보는 2번 정중한 책임 답변보다 짧아야 합니다.",
    "REPLY_OPTION_2_NO_ACTION": "2번 후보에는 책임 인정 또는 구체적인 수습 행동이 필요합니다.",
}


class ExcuseGenerationService:
    """프롬프트 구성, Cerebras 호출, REPLY 품질 재생성을 조율한다.

    ``CerebrasClient``는 제공자 통신과 형식 오류 재시도만 담당한다. 반면 이 서비스는
    이전 대화와의 중복처럼 제품 고유의 품질 규칙을 적용하므로 두 책임을 분리한다.
    """

    def __init__(self, settings: Settings):
        # 클라이언트는 서비스 수명 동안 재사용한다. 매 요청마다 설정을 다시 읽거나
        # HTTP 클라이언트 정책을 새로 만들지 않도록 FastAPI 의존성 캐시와 함께 사용한다.
        self.client = CerebrasClient(settings)

    async def generate(self, request: GenerateRequest, request_id: str) -> ExcuseResult:
        """표준 요청 하나를 제공자 결과 하나로 생성한다.

        CREATE/Evolve는 제공자가 반환한 구조화 결과를 바로 사용한다. REPLY만 이전
        답변 반복 여부를 판단해야 하므로 별도 품질 재생성 경로를 탄다.
        """
        system_prompt = build_system_prompt()
        if request.mode != GenerationMode.REPLY:
            return await self._generate_once(request, request_id, system_prompt)

        return await self._generate_reply(request, request_id, system_prompt)

    async def generate_for_spring(
        self,
        request: GenerateRequest,
        request_id: str,
    ) -> SpringExcuseResponse:
        """AI 내부 결과를 Spring이 소비하는 응답 계약으로 변환하고 성공 로그를 남긴다.

        ``ExcuseResult``에는 provider 내부 진단용 필드도 있지만, 여기서는 Spring DTO에
        정의된 필드만 반환한다. 로그도 원문 전체 대신 짧은 발췌만 남겨 개인정보 노출을
        줄인다.
        """
        result = await self.generate(request, request_id)
        logger.info(
            "ai_generation_result request_id=%s mode=%s round=%s excuse=%s",
            request_id,
            request.mode.value,
            request.roundNumber,
            _excerpt(result.excuse),
        )
        return SpringExcuseResponse.from_result(result)

    async def _generate_reply(
        self,
        request: GenerateRequest,
        request_id: str,
        system_prompt: str,
    ) -> ExcuseResult:
        """REPLY 결과를 최대 두 번 생성해 품질 기준을 만족하는 첫 결과를 반환한다.

        첫 생성이 실패하면 실패한 문장과 후보를 다음 프롬프트의 금지 문맥으로 전달한다.
        두 번째 결과도 실패하면 더 많은 호출을 반복하지 않고 422로 끝낸다. 이는 품질을
        높이면서도 사용자 요청 하나가 무한 재시도·과금 증가로 이어지는 일을 막는다.
        """
        # Spring이 전달한 현재 가지의 모든 assistant 발화를 비교 기준으로 사용한다.
        # FastAPI는 상태를 저장하지 않으므로 이 문맥이 재요청 시 중복 방지의 근거가 된다.
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
                # 재생성 결과라면 운영 로그에서 품질 회복 여부를 추적할 수 있게 남긴다.
                if attempt:
                    logger.info("reply_quality_recovered request_id=%s", request_id)
                return result

            if attempt:
                # 두 번째 실패는 같은 규칙으로 다시 요청해도 회복될 가능성이 낮다.
                # Spring이 사용자에게 재입력/재시도를 안내할 수 있도록 의미 있는 422를 반환한다.
                logger.warning(
                    "reply_quality_failed request_id=%s failures=%s",
                    request_id,
                    ",".join(failures),
                )
                _raise_quality_error(failures)

            # 첫 실패만 재생성한다. 첫 결과의 본문은 이후 유사도 비교 대상에 추가하고,
            # 본문·후보 문장은 다음 프롬프트에서 피해야 할 표현으로 전달한다.
            logger.warning(
                "reply_quality_retry request_id=%s failures=%s",
                request_id,
                ",".join(failures),
            )
            previous_answers.append(result.excuse)
            blocked_texts = _unique_texts(previous_answers + result.replyOptions)
            failure_messages = [QUALITY_MESSAGES[code] for code in failures]

        raise AssertionError("REPLY quality loop should return or raise")

    async def _generate_once(
        self,
        request: GenerateRequest,
        request_id: str,
        system_prompt: str,
        *,
        avoid_texts: list[str] | None = None,
        quality_failures: list[str] | None = None,
    ) -> ExcuseResult:
        """한 번의 Cerebras 호출에 필요한 프롬프트를 조립한다.

        재생성일 때만 ``avoid_texts``와 ``quality_failures``가 채워진다. 같은 호출 함수를
        첫 생성과 재생성에 공유해 모델·토큰·타임아웃 정책이 달라지지 않도록 한다.
        """
        user_prompt = build_user_prompt(
            request,
            max_memory_chars=self.client.settings.max_memory_chars,
            avoid_texts=avoid_texts,
            quality_failures=quality_failures,
        )
        return await self.client.generate(system_prompt, user_prompt, request_id)


def _reply_quality_failures(
    request: GenerateRequest,
    result: ExcuseResult,
    previous_answers: list[str],
) -> list[str]:
    """REPLY 본문과 후보가 재생성되어야 하는 이유를 코드 목록으로 반환한다.

    검증은 순수 함수로 유지해 실제 Cerebras 호출 없이 단위 테스트할 수 있다. 반환된
    코드는 재생성 프롬프트, 로그, 최종 HTTP 오류에 동일하게 사용된다.
    """
    failures: list[str] = []
    if len(_normalize(result.excuse)) < REPLY_MIN_EXCUSE_CHARS:
        failures.append("REPLY_TOO_SHORT")

    if _max_similarity(result.excuse, previous_answers) >= REPLY_SIMILARITY_THRESHOLD:
        failures.append("REPLY_TOO_SIMILAR")

    if not _addresses_incoming_message(request.incomingMessage or "", result.excuse):
        failures.append("REPLY_INCOMING_IGNORED")

    return failures + _reply_options_quality_failures(result.replyOptions)


def _reply_options_quality_failures(options: list[str]) -> list[str]:
    """세 replyOptions가 서로 다른 역할을 수행하는지 확인한다.

    1번은 짧은 직접 답장, 2번은 책임과 행동을 담은 정중한 답장, 3번은 관계 수습용
    대안이라는 프롬프트 규칙을 최소한의 결정적 검사로 보완한다.
    """
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
        failures.append("REPLY_OPTION_1_NOT_SHORTER")
    if not _contains_any(normalized_options[1], RESPONSIBILITY_MARKERS):
        failures.append("REPLY_OPTION_2_NO_ACTION")
    return failures


def _addresses_incoming_message(incoming_message: str, reply: str) -> bool:
    """답장이 최신 질문을 다루는지 보수적으로 판정한다.

    먼저 핵심 단어·연속 음절이 겹치는지 보고, 한국어 조사 변화 때문에 단어가 달라진
    경우에는 ``REPLY_INTENT_CUES``의 질문 의도와 응답 단서를 사용한다. 완전한 의미
    평가는 모델 프롬프트의 책임이며, 이 함수는 명백히 무관한 답변만 재생성 대상으로
    만드는 용도다.
    """
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
    """currentExcuse와 현재 대화 가지의 이전 AI 답변을 중복 비교용으로 모은다."""
    answers = [request.currentExcuse] if request.currentExcuse else []
    answers.extend(
        turn.message for turn in request.conversation if turn.role.value == "assistant"
    )
    return _unique_texts(answers)


def _max_similarity(candidate: str, previous: list[str]) -> float:
    """후보 문장과 이전 답변들 사이의 가장 높은 문자 기반 유사도를 반환한다."""
    return max((_similarity(candidate, item) for item in previous), default=0.0)


def _similarity(left: str, right: str) -> float:
    """표현 차이보다 내용 반복을 보기 위해 정규화한 문자열을 비교한다."""
    return SequenceMatcher(None, _normalize(left), _normalize(right)).ratio()


def _normalize(value: str) -> str:
    """공백·문장부호·대소문자 차이를 제거해 비교용 문자열을 만든다."""
    return re.sub(r"[^가-힣a-zA-Z0-9]", "", value).lower()


def _content_terms(value: str) -> set[str]:
    """질문 반응성 검사에 쓸 두 글자 이상 핵심 단어만 추린다."""
    return {
        term.lower()
        for term in re.findall(r"[가-힣]{2,}|[a-zA-Z0-9]{2,}", value)
        if term not in STOPWORDS
    }


def _bigrams(value: str) -> set[str]:
    """조사 변화가 있는 한국어 문장에서도 부분 겹침을 찾기 위한 두 글자 집합이다."""
    return {value[index : index + 2] for index in range(len(value) - 1)}


def _contains_any(value: str, markers: tuple[str, ...]) -> bool:
    """정규화된 문자열에 특정 질문·행동 단서 중 하나가 포함됐는지 확인한다."""
    return any(marker in value for marker in markers)


def _unique_texts(values: list[str]) -> list[str]:
    """정규화 기준으로 중복을 제거하되 첫 번째 원문 문장은 보존한다."""
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(value)
    return unique


def _raise_quality_error(failures: list[str]) -> None:
    """재생성 후에도 실패한 품질 규칙을 Spring이 처리할 수 있는 422로 변환한다."""
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
    """줄바꿈을 제거하고 로그에 안전하게 남길 최대 길이의 문장 발췌를 만든다."""
    return value.replace("\n", " ").strip()[:limit]
