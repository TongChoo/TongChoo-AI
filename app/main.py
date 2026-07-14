"""TongChoo AI 서버의 HTTP 진입점.

Spring은 인증·DB·대화 계보를 담당하고, 이 모듈은 검증된 문맥을 받아
AI 생성 결과만 반환한다. 따라서 이 파일에는 도메인 저장 로직을 두지 않는다.
"""

from __future__ import annotations

import logging
import asyncio
from functools import lru_cache
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from app.config import get_settings
from app.models import (
    ExcuseResult,
    GenerationMode,
    GenerateRequest,
    SpringCreateRequest,
    SpringExcuseResponse,
    SpringReplyRequest,
)
from app.service import ExcuseGenerationService
from app.llm import (
    api_error,
    reset_provider_attempt_budget,
    set_provider_attempt_budget,
)

logger = logging.getLogger("tongchoo.api")

# 이 서버는 Spring 전용 내부 서비스다. 사용자 인증·도메인 저장소를 직접 노출하지
# 않으므로 OpenAPI 설명도 Spring이 조립한 문맥을 받아 생성만 한다는 범위로 제한한다.
app = FastAPI(
    title="TongChoo AI Server",
    version="0.2.0",
    description="Spring이 전달한 문맥으로 Cerebras 기반 변명·답장을 생성합니다.",
)


@lru_cache
def get_service() -> ExcuseGenerationService:
    """요청마다 서비스·설정 객체를 새로 만들지 않도록 하나의 인스턴스를 재사용한다.

    서비스 안의 Cerebras 클라이언트는 타임아웃·인증 정책을 보관한다. FastAPI 의존성
    캐시를 사용하면 매 요청의 환경변수 파싱과 초기화 비용을 피할 수 있다.
    """
    return ExcuseGenerationService(get_settings())


# 엔드포인트 인자에 공통 의존성을 반복해 쓰지 않기 위한 타입 별칭이다. 테스트에서는
# ``get_service`` 의존성을 교체해 실제 Cerebras 호출 없이 API 계층을 검증할 수 있다.
GenerationService = Annotated[ExcuseGenerationService, Depends(get_service)]


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Spring이 보낸 요청 ID를 유지하고, 없으면 새 ID를 발급해 응답까지 전달한다.

    같은 ID를 FastAPI 로그, Cerebras 호출 로그, 응답 헤더에 모두 사용하면 Spring 로그와
    AI 제공자 장애 로그를 한 요청 단위로 추적할 수 있다.
    """
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


async def require_internal_token(
    authorization: str | None = Header(default=None),
) -> None:
    """Spring 외의 호출을 막는 서버 간 인증 검사.

    로컬 개발에서 토큰을 설정하지 않은 경우에는 호출을 허용한다. 배포 환경에서는
    반드시 `INTERNAL_SERVICE_TOKEN`을 설정하고 Spring도 같은 값을 Bearer 토큰으로
    전송해야 한다.
    """
    expected_token = get_settings().internal_service_token
    # 토큰이 없는 경우만 로컬 개발 모드로 간주한다. 빈 토큰과 잘못된 토큰을 동일하게
    # 허용하면 배포 설정 실수로 내부 API가 외부에 열릴 수 있으므로, 토큰이 설정된 뒤에는
    # 정확한 Bearer 값만 통과시킨다.
    if not expected_token:
        return

    if authorization != f"Bearer {expected_token}":
        raise HTTPException(
            status_code=401,
            detail={
                "code": "UNAUTHORIZED",
                "message": "유효한 내부 서비스 토큰이 필요합니다.",
            },
        )


# 모든 생성 엔드포인트에 붙는 서버 간 인증 의존성이다. health는 로드밸런서와 배포
# 점검에서 토큰 없이 호출할 수 있도록 의도적으로 제외한다.
InternalOnly = [Depends(require_internal_token)]


@app.get("/health", tags=["운영"])
async def health() -> dict[str, object]:
    """외부 AI 호출 없이 프로세스와 AI 설정 상태만 확인하는 헬스체크.

    실제 Cerebras 연결성까지 확인하면 헬스체크 자체가 외부 장애와 quota에 영향을 받으므로,
    여기서는 API 키가 주입됐는지만 노출한다. 키 값은 절대 응답에 포함하지 않는다.
    """
    settings = get_settings()
    return {
        "status": "ok",
        "ai": {
            "provider": "cerebras",
            "configured": bool(settings.cerebras_api_key),
            "model": settings.cerebras_model,
        },
    }


async def _generate_response(
    request: GenerateRequest,
    http_request: Request,
    service: ExcuseGenerationService,
) -> SpringExcuseResponse:
    """모든 생성 엔드포인트가 공유하는 로그 기록과 서비스 호출.

    요청 DTO를 여기까지 통일하면 create/reply 엔드포인트는 Spring 계약에 맞는
    입력 변환만 담당하고, 실제 생성 규칙은 서비스 계층 한 곳에서 유지할 수 있다.
    """
    # middleware가 항상 state에 넣어 준 request_id를 사용한다. 여기서 새 ID를 만들지
    # 않아야 API 진입·서비스·LLM 로그가 하나의 상관관계 ID로 이어진다.
    request_id = http_request.state.request_id
    # 원문 전체 대신 incomingMessage 앞부분만 기록한다. 대화 내용은 민감할 수 있지만,
    # 최신 질문을 확인해야 REPLY 반복 문제를 운영에서 재현·진단할 수 있다.
    logger.info(
        "ai_generation_request request_id=%s mode=%s target=%s tone=%s round=%s",
        request_id,
        request.mode.value,
        request.target.value,
        request.tone.value,
        request.roundNumber,
    )
    # REPLY 반복 문제는 AI 모델보다 Spring이 보낸 현재 변명·대화 가지가 오래된 경우에
    # 자주 발생한다. 민감한 원문 전체를 남기지 않고, 실제 수신 문맥의 발췌와 turn 수만
    # 기록하면 요청 ID 하나로 Spring 로그와 안전하게 대조할 수 있다.
    previous_assistant = next(
        (
            turn.message
            for turn in reversed(request.conversation)
            if turn.role.value == "assistant"
        ),
        "",
    )
    logger.info(
        "ai_generation_context request_id=%s mode=%s round=%s turns=%s "
        "incomingMessage=%s currentExcuse=%s previousAssistant=%s",
        request_id,
        request.mode.value,
        request.roundNumber,
        len(request.conversation),
        (request.incomingMessage or "")[:120].replace("\n", " "),
        (request.currentExcuse or "")[:120].replace("\n", " "),
        previous_assistant[:120].replace("\n", " "),
    )
    settings = get_settings()
    budget_token = set_provider_attempt_budget(settings.total_provider_attempts)
    try:
        return await asyncio.wait_for(
            service.generate_for_spring(request, request_id),
            timeout=settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise api_error(
            504,
            "AI_REQUEST_TIMEOUT",
            "AI 요청의 전체 처리 시간 상한을 초과했습니다.",
        ) from exc
    finally:
        reset_provider_attempt_budget(budget_token)


@app.post(
    "/internal/v1/excuses/create",
    response_model=SpringExcuseResponse,
    dependencies=InternalOnly,
    tags=["Spring 내부 API"],
    summary="최초 변명 생성",
)
async def create_excuse_for_spring(
    request: SpringCreateRequest,
    http_request: Request,
    service: GenerationService,
) -> SpringExcuseResponse:
    """Spring의 최초 생성 요청을 내부 표준 요청으로 바꿔 전달한다.

    CREATE mode 값은 URL이 아닌 표준 모델에 기록되므로 service.py가 모든 모드를 같은
    호출 흐름으로 다룰 수 있다.
    """
    generate_request = request.to_generate_request(GenerationMode.CREATE)
    return await _generate_response(generate_request, http_request, service)


@app.post(
    "/internal/v1/excuses/reply",
    response_model=SpringExcuseResponse,
    dependencies=InternalOnly,
    tags=["Spring 내부 API"],
    summary="상대 메시지에 답장 생성",
)
async def reply_to_excuse_for_spring(
    request: SpringReplyRequest,
    http_request: Request,
    service: GenerationService,
) -> SpringExcuseResponse:
    """상대 메시지와 현재 가지의 대화 문맥을 바탕으로 다음 답장을 만든다.

    REPLY 품질 검사는 incomingMessage와 이전 assistant 발화를 모두 사용한다. 따라서
    Spring은 다른 분기의 대화가 아닌 사용자가 선택한 현재 가지를 전달해야 한다.
    """
    return await _generate_response(
        request.to_generate_request(), http_request, service
    )


@app.post(
    "/internal/v1/excuses/generate",
    response_model=SpringExcuseResponse,
    dependencies=InternalOnly,
    tags=["호환 API"],
    summary="통합 생성 API",
)
async def generate_excuse(
    request: GenerateRequest,
    http_request: Request,
    service: GenerationService,
) -> SpringExcuseResponse:
    """기존 클라이언트를 위한 mode 기반 호환 엔드포인트.

    신규 Spring 코드는 create/reply 전용 URL을 쓰는 편이 입력 계약이 명확하다.
    이 URL은 이전 통합을 깨지 않기 위한 호환 계층으로만 유지한다.
    """
    return await _generate_response(request, http_request, service)


@app.post(
    "/internal/v1/excuses/generate/raw",
    response_model=ExcuseResult,
    dependencies=InternalOnly,
    include_in_schema=False,
)
async def generate_raw_excuse(
    request: GenerateRequest,
    http_request: Request,
    service: GenerationService,
) -> ExcuseResult:
    """운영 API에는 사용하지 않는 제공자 원본 형태의 진단용 응답.

    Spring 응답 변환 이전의 생성 결과를 확인할 때만 사용한다.
    문서에서는 숨겨 두어 일반 클라이언트가 운영 계약으로 의존하지 않게 한다.
    """
    return await service.generate(request, http_request.state.request_id)
