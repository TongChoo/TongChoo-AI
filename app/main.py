"""TongChoo AI 서버의 HTTP 진입점.

Spring은 인증·DB·대화 계보를 담당하고, 이 모듈은 검증된 문맥을 받아
AI 생성 결과만 반환한다. 따라서 이 파일에는 도메인 저장 로직을 두지 않는다.
"""

import logging
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
    SpringEvolveRequest,
    SpringExcuseResponse,
    SpringReplyRequest,
)
from app.service import ExcuseGenerationService

logger = logging.getLogger("tongchoo.api")

app = FastAPI(
    title="TongChoo AI Server",
    version="0.2.0",
    description="Spring이 전달한 문맥으로 Cerebras 기반 변명·답장을 생성합니다.",
)


@lru_cache
def get_service() -> ExcuseGenerationService:
    """설정과 HTTP 클라이언트를 요청마다 새로 만들지 않도록 서비스 인스턴스를 재사용한다."""
    return ExcuseGenerationService(get_settings())


GenerationService = Annotated[ExcuseGenerationService, Depends(get_service)]


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Spring이 보낸 요청 ID를 유지하고, 없으면 새 ID를 발급해 응답까지 전달한다."""
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
    if not expected_token:
        return

    if authorization != f"Bearer {expected_token}":
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "유효한 내부 서비스 토큰이 필요합니다."},
        )


InternalOnly = [Depends(require_internal_token)]


@app.get("/health", tags=["운영"])
async def health() -> dict[str, object]:
    """외부 AI 호출 없이 설정 상태만 확인하는 헬스체크."""
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

    요청 DTO를 여기까지 통일하면 create/evolve/reply 엔드포인트는 Spring 계약에 맞는
    입력 변환만 담당하고, 실제 생성 규칙은 서비스 계층 한 곳에서 유지할 수 있다.
    """
    request_id = http_request.state.request_id
    logger.info(
        "ai_generation_request request_id=%s mode=%s target=%s tone=%s round=%s",
        request_id,
        request.mode.value,
        request.target.value,
        request.tone.value,
        request.roundNumber,
    )
    return await service.generate_for_spring(request, request_id)


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
    """Spring의 `ExcuseCreateRequest`와 같은 필드로 최초 변명을 생성한다."""
    generate_request = request.to_generate_request(GenerationMode.CREATE)
    return await _generate_response(generate_request, http_request, service)


@app.post(
    "/internal/v1/excuses/evolve",
    response_model=SpringExcuseResponse,
    dependencies=InternalOnly,
    tags=["Spring 내부 API"],
    summary="기존 변명 수정",
)
async def evolve_excuse_for_spring(
    request: SpringEvolveRequest,
    http_request: Request,
    service: GenerationService,
) -> SpringExcuseResponse:
    """Spring이 조회한 기존 변명과 `direction`을 바탕으로 수정안을 만든다."""
    return await _generate_response(request.to_generate_request(), http_request, service)


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
    """상대 메시지와 현재 가지의 대화 문맥을 바탕으로 다음 답장을 만든다."""
    return await _generate_response(request.to_generate_request(), http_request, service)


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
    """기존 클라이언트를 위한 mode 기반 호환 엔드포인트."""
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
    """운영 API에는 사용하지 않는, 제공자 원본 형태의 진단용 응답."""
    return await service.generate(request, http_request.state.request_id)
