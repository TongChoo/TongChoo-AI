"""환경변수 기반 서버 설정.

비밀 값은 `.env` 또는 배포 환경변수로만 주입한다. 이 파일과 `.env.example`에는
실제 API 키를 넣지 않는다.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Cerebras 호출과 내부 Spring 인증에 필요한 설정값.

    모든 값은 환경변수에서 읽고, ``Field(alias=...)``로 배포 환경의 대문자 변수명과
    Python 속성명을 분리한다. 따라서 코드에는 비밀 키나 환경별 URL을 넣지 않는다.
    """

    # 로컬 개발에서는 .env를 읽되, 배포 환경변수가 같은 이름으로 주입되면 환경변수가
    # 우선한다. extra=ignore는 다른 서비스가 공유하는 .env 값 때문에 서버가 시작하지
    # 못하는 일을 막는다.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
    )

    # Cerebras 연결 정보: FastAPI만 API 키를 보유하고 Spring에는 절대 전달하지 않는다.
    # 빈 기본값은 서버 시작 자체는 가능하게 하되, 실제 생성 요청에서는 명확한 503 설정
    # 오류를 반환하게 한다. /health로 배포 환경의 설정 상태도 확인할 수 있다.
    cerebras_api_key: str = Field(default="", alias="CEREBRAS_API_KEY")
    cerebras_base_url: str = Field(
        default="https://api.cerebras.ai/v1",
        alias="CEREBRAS_BASE_URL",
    )
    cerebras_model: str = Field(default="gpt-oss-120b", alias="CEREBRAS_MODEL")
    # connect는 네트워크 연결을 맺는 시간, read/write는 LLM 생성 응답을 기다리는 시간이다.
    # 둘을 분리해 DNS·연결 문제와 모델 생성 지연을 같은 장애로 취급하지 않는다.
    cerebras_connect_timeout_seconds: float = Field(
        default=1.0,
        alias="CEREBRAS_CONNECT_TIMEOUT_SECONDS",
    )
    cerebras_read_timeout_seconds: float = Field(
        default=15.0,
        alias="CEREBRAS_READ_TIMEOUT_SECONDS",
    )
    # 생성 품질·지연시간·비용 균형을 위한 재시도와 토큰 정책. max_attempts는 제공자
    # 네트워크/형식 오류 재시도 횟수이며, service.py의 REPLY 품질 재생성과는 별개다.
    max_attempts: int = Field(default=2, ge=1, le=3, alias="LLM_MAX_ATTEMPTS")
    reply_quality_max_attempts: int = Field(
        default=2,
        ge=1,
        le=3,
        alias="REPLY_QUALITY_MAX_ATTEMPTS",
    )
    aftermath_quality_max_attempts: int = Field(
        default=2,
        ge=1,
        le=3,
        alias="AFTERMATH_QUALITY_MAX_ATTEMPTS",
    )
    situation_quality_max_attempts: int = Field(
        default=2,
        ge=1,
        le=3,
        alias="SITUATION_QUALITY_MAX_ATTEMPTS",
    )
    reply_similarity_threshold: float = Field(
        default=0.9,
        ge=0.7,
        le=1.0,
        alias="REPLY_SIMILARITY_THRESHOLD",
    )
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
    # temperature는 같은 문맥의 표현 다양성을 조절한다. 너무 낮으면 REPLY가 반복되기
    # 쉽고, 너무 높으면 사실 일관성이 떨어질 수 있어 기본값을 중간 수준으로 둔다.
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, alias="LLM_TEMPERATURE")
    classification_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        alias="CLASSIFICATION_TEMPERATURE",
    )
    reasoning_effort: str = Field(default="low", alias="CEREBRAS_REASONING_EFFORT")
    # Spring이 전달한 과거 메모리는 프롬프트에 넣기 전에 이 길이로 제한한다. 대화
    # 문맥보다 오래된 메모리가 토큰 대부분을 차지하는 일을 막기 위한 상한이다.
    max_memory_chars: int = Field(default=12000, ge=1, alias="LLM_MAX_MEMORY_CHARS")

    # Spring → FastAPI 내부 호출에만 쓰는 별도 토큰이다. 사용자 JWT와 다르며, 빈 값은
    # 로컬 개발 편의를 위해 인증을 비활성화한다. 배포 환경에서는 반드시 설정한다.
    internal_service_token: str = Field(default="", alias="INTERNAL_SERVICE_TOKEN")


@lru_cache
def get_settings() -> Settings:
    """프로세스 동안 하나의 검증된 설정 객체를 재사용한다.

    환경변수는 런타임에 자주 바뀌는 값이 아니므로 요청마다 파싱하지 않는다. 설정 변경은
    새 프로세스를 띄워 반영하는 것을 전제로 한다.
    """
    return Settings()
