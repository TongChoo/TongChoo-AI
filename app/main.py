import logging
from functools import lru_cache
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from app.config import get_settings
from app.models import ExcuseResult, GenerateRequest, SpringExcuseResponse
from app.service import ExcuseGenerationService

logger = logging.getLogger("tongchoo.api")

app = FastAPI(
    title="TongChoo AI Server",
    version="0.1.0",
    description="OpenAI-backed AI service for TongChoo excuse generation.",
)


@lru_cache
def get_service() -> ExcuseGenerationService:
    return ExcuseGenerationService(get_settings())


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


async def require_internal_token(authorization: str | None = Header(default=None)) -> None:
    expected = get_settings().internal_service_token
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "유효한 내부 서비스 토큰이 필요합니다."},
        )


@app.get("/health")
async def health() -> dict[str, object]:
    settings = get_settings()
    return {
        "status": "ok",
        "ai": {
            "provider": "openai",
            "configured": bool(settings.openai_api_key),
            "model": settings.openai_model,
        },
    }


@app.post(
    "/internal/v1/excuses/generate",
    response_model=SpringExcuseResponse,
    dependencies=[Depends(require_internal_token)],
)
def generate_excuse(
    request: GenerateRequest,
    http_request: Request,
    service: ExcuseGenerationService = Depends(get_service),
) -> SpringExcuseResponse:
    request_id = http_request.state.request_id
    logger.info(
        "api_request request_id=%s mode=%s target=%s tone=%s round=%s",
        request_id,
        request.mode.value,
        request.target.value,
        request.tone.value,
        request.roundNumber,
    )
    return service.generate_for_spring(request, request_id)


@app.post(
    "/internal/v1/excuses/generate/raw",
    response_model=ExcuseResult,
    dependencies=[Depends(require_internal_token)],
    include_in_schema=False,
)
def generate_raw_excuse(
    request: GenerateRequest,
    http_request: Request,
    service: ExcuseGenerationService = Depends(get_service),
) -> ExcuseResult:
    """Provider-shaped response kept for local diagnostics and compatibility."""
    return service.generate(request, http_request.state.request_id)
