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
    demo_safe_mode: bool = Field(default=False, alias="DEMO_SAFE_MODE")
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
        le=2,
        alias="REPLY_QUALITY_MAX_ATTEMPTS",
    )
    situation_quality_max_attempts: int = Field(
        default=3,
        ge=1,
        le=3,
        alias="SITUATION_QUALITY_MAX_ATTEMPTS",
    )
    aftermath_quality_max_attempts: int = Field(
        default=2,
        ge=1,
        le=3,
        alias="AFTERMATH_QUALITY_MAX_ATTEMPTS",
    )
    reply_similarity_threshold: float = Field(
        # 같은 핵심 사실을 반복하는 답장은 2라운드에서 자연스럽게 생길 수 있다.
        # 문장 전체가 거의 같은 경우만 중복으로 보도록 보수적으로 둔다.
        default=0.92,
        ge=0.7,
        le=1.0,
        alias="REPLY_SIMILARITY_THRESHOLD",
    )
    # Judge 점수는 생성 품질을 보조하는 신호다. 짧은 메신저 답장은 길고 완결된 문장보다
    # 낮게 채점되기 쉬우므로, 사실 발명·격식 위반 같은 hardViolation과 구분해 완화된
    # 기준을 둔다.
    reply_candidate_min_score: int = Field(
        default=65,
        ge=0,
        le=100,
        alias="REPLY_CANDIDATE_MIN_SCORE",
    )
    reply_diversity_min_score: int = Field(
        default=55,
        ge=0,
        le=100,
        alias="REPLY_DIVERSITY_MIN_SCORE",
    )
    # 공개할 근거가 없어 정중히 선을 그어야 하는 질문은 세 후보가 같은 결론을 공유할
    # 수밖에 없다. 이 경우에는 표현·대화 역할의 차이만 있어도 통과할 수 있게 둔다.
    reply_privacy_diversity_min_score: int = Field(
        default=40,
        ge=0,
        le=100,
        alias="REPLY_PRIVACY_DIVERSITY_MIN_SCORE",
    )
    # Judge의 구조화 응답이 제공자 문제로 일부 누락돼도, 결정적 안전 검사를 통과한
    # 실제 답장까지 버리지 않도록 한다. Judge가 완전한 판정을 준 경우에는 계속 적용된다.
    reply_judge_fail_open: bool = Field(
        default=True,
        alias="REPLY_JUDGE_FAIL_OPEN",
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
    reply_judge_max_completion_tokens: int = Field(
        default=700,
        ge=1,
        alias="REPLY_JUDGE_MAX_COMPLETION_TOKENS",
    )
    # temperature는 같은 문맥의 표현 다양성을 조절한다. 너무 낮으면 REPLY가 반복되기
    # 쉽고, 너무 높으면 사실 일관성이 떨어질 수 있어 기본값을 중간 수준으로 둔다.
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, alias="LLM_TEMPERATURE")
    classification_temperature: float = Field(
        default=0.0,
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
