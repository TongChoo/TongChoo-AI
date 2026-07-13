"""환경변수 기반 서버 설정.

비밀 값은 `.env` 또는 배포 환경변수로만 주입한다. 이 파일과 `.env.example`에는
실제 API 키를 넣지 않는다.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Cerebras 호출과 내부 Spring 인증에 필요한 설정값."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
    )

    # Cerebras 연결 정보: FastAPI만 API 키를 보유하고 Spring에는 전달하지 않는다.
    cerebras_api_key: str = Field(default="", alias="CEREBRAS_API_KEY")
    cerebras_base_url: str = Field(
        default="https://api.cerebras.ai/v1",
        alias="CEREBRAS_BASE_URL",
    )
    cerebras_model: str = Field(default="gpt-oss-120b", alias="CEREBRAS_MODEL")
    cerebras_connect_timeout_seconds: float = Field(
        default=1.0,
        alias="CEREBRAS_CONNECT_TIMEOUT_SECONDS",
    )
    cerebras_read_timeout_seconds: float = Field(
        default=15.0,
        alias="CEREBRAS_READ_TIMEOUT_SECONDS",
    )
    # 생성 품질·지연시간 균형을 위한 재시도와 토큰 정책.
    max_attempts: int = Field(default=2, ge=1, le=3, alias="LLM_MAX_ATTEMPTS")
    max_completion_tokens: int = Field(
        default=1400,
        ge=1,
        alias="LLM_MAX_COMPLETION_TOKENS",
    )
    length_retry_completion_tokens: int = Field(
        default=1800,
        ge=1,
        alias="LLM_LENGTH_RETRY_COMPLETION_TOKENS",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, alias="LLM_TEMPERATURE")
    reasoning_effort: str = Field(default="low", alias="CEREBRAS_REASONING_EFFORT")
    # Spring이 전달한 과거 메모리는 프롬프트에 넣기 전에 이 길이로 제한한다.
    max_memory_chars: int = Field(default=12000, ge=1, alias="LLM_MAX_MEMORY_CHARS")

    # Spring → FastAPI 내부 호출에만 쓰는 별도 토큰이다. 사용자 JWT와 다르다.
    internal_service_token: str = Field(default="", alias="INTERNAL_SERVICE_TOKEN")


@lru_cache
def get_settings() -> Settings:
    """환경변수 파싱 비용을 줄이기 위해 프로세스 동안 설정 객체 하나를 재사용한다."""
    return Settings()
