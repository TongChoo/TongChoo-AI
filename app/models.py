"""FastAPI 입출력 계약과 Cerebras Structured Outputs 스키마.

모든 필드는 camelCase를 사용한다. Spring의 Jackson 기본 설정과 FastAPI의 JSON
응답을 별도 변환 없이 연결하기 위해서다.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):
        """Python 3.9 로컬 실행을 위한 StrEnum 호환 클래스."""

        pass


class Target(StrEnum):
    """변명을 전달할 상대. Spring의 Target enum 값과 반드시 같아야 한다."""

    TEACHER = "TEACHER"
    PARENT = "PARENT"
    FRIEND = "FRIEND"
    LOVER = "LOVER"
    TEAM_LEAD = "TEAM_LEAD"
    TEAM_MEMBER = "TEAM_MEMBER"


class Tone(StrEnum):
    """문장 분위기. Spring의 Tone enum 값과 반드시 같아야 한다."""

    MILD = "MILD"
    SLICK = "SLICK"
    DESPERATE = "DESPERATE"
    BULLSHIT = "BULLSHIT"


class GenerationMode(StrEnum):
    """통합 호환 API에서 어떤 생성 작업을 할지 나타낸다."""

    CREATE = "CREATE"
    EVOLVE = "EVOLVE"
    REPLY = "REPLY"


class ConversationRole(StrEnum):
    """대화 가지에서 발화한 주체."""

    USER = "user"
    ASSISTANT = "assistant"


class SuspicionLevel(StrEnum):
    """AI가 추정한 변명의 의심도."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Aftermath(BaseModel):
    """변명을 보낸 뒤 예상되는 후속 질문 한 건.

    `dayOffset`은 Spring의 `ExcuseAftermath.dayOffset`에 그대로 저장된다.
    """

    model_config = ConfigDict(extra="forbid")

    when: Annotated[str, Field(min_length=1, max_length=100)]
    dayOffset: Annotated[int, Field(ge=0, le=365)]
    question: Annotated[str, Field(min_length=1, max_length=300)]
    collapseRate: Annotated[int, Field(ge=0, le=100)]


class ConversationTurn(BaseModel):
    """Spring이 DB 계보에서 조립한 현재 대화 가지의 한 발화."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    role: ConversationRole
    # Java record `ConversationTurn` exposes `message`. `content` remains an
    # accepted input alias for compatibility with the old README examples.
    message: Annotated[
        str,
        Field(
            min_length=1,
            max_length=2000,
            validation_alias=AliasChoices("message", "content"),
        ),
    ]

    @field_validator("message", mode="before")
    @classmethod
    def trim_message(cls, value: object) -> object:
        """공백만 있는 대화가 프롬프트에 들어가지 않도록 앞뒤 공백을 제거한다."""
        return value.strip() if isinstance(value, str) else value


class ExcuseResult(BaseModel):
    """Cerebras가 Structured Outputs로 반환해야 하는 평면 결과.

    제공자 응답은 검증하기 쉽게 평면 구조로 받고, Spring에 돌려주기 직전에
    Spring 클라이언트 응답 구조로 변환한다.
    """

    model_config = ConfigDict(extra="forbid")

    excuse: Annotated[str, Field(min_length=20, max_length=1000)]
    recommendedAction: Annotated[str, Field(min_length=1, max_length=300)]
    likelyFollowUp: Annotated[str, Field(min_length=1, max_length=300)]
    replyOptions: Annotated[list[str], Field(min_length=2, max_length=3)]
    successRate: Annotated[int, Field(ge=0, le=100)]
    realism: Annotated[int, Field(ge=1, le=5)]
    persuasion: Annotated[int, Field(ge=1, le=5)]
    suspicionLevel: SuspicionLevel
    riskFactors: Annotated[list[str], Field(min_length=1, max_length=5)]
    aftermath: Annotated[list[Aftermath], Field(min_length=1, max_length=4)]
    remember: Annotated[list[str], Field(max_length=8)]

    @field_validator("excuse", mode="before")
    @classmethod
    def trim_excuse(cls, value: object) -> object:
        """복사해 보낼 변명 문장의 불필요한 앞뒤 공백을 제거한다."""
        return value.strip() if isinstance(value, str) else value

    @field_validator("riskFactors", "remember", "replyOptions", mode="before")
    @classmethod
    def trim_items(cls, value: object) -> object:
        """배열 안 문장도 동일한 방식으로 정리한다."""
        if not isinstance(value, list):
            return value
        return [item.strip() if isinstance(item, str) else item for item in value]

    @field_validator("riskFactors", "remember", "replyOptions")
    @classmethod
    def reject_empty_items(cls, value: list[str]) -> list[str]:
        """비어 있거나 지나치게 긴 선택지를 제공자 응답으로 인정하지 않는다."""
        if any(not item for item in value):
            raise ValueError("배열 항목은 비어 있을 수 없습니다.")
        if any(len(item) > 200 for item in value):
            raise ValueError("배열 항목은 200자 이하여야 합니다.")
        return value


class GenerateRequest(BaseModel):
    """기존 mode 기반 API가 사용하는 내부 표준 요청 모델."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    mode: GenerationMode = GenerationMode.CREATE
    situation: Annotated[str, Field(min_length=1, max_length=4000)]
    target: Target
    tone: Tone
    memory: Annotated[str, Field(default="", max_length=12000)]
    rootExcuse: Annotated[str | None, Field(default=None, min_length=1, max_length=1000)]
    conversation: Annotated[list[ConversationTurn], Field(default_factory=list, max_length=10)]
    currentExcuse: Annotated[str | None, Field(default=None, min_length=1, max_length=1000)]
    incomingMessage: Annotated[str | None, Field(default=None, min_length=1, max_length=2000)]
    roundNumber: Annotated[int | None, Field(default=None, ge=1, le=10)]
    evolveDirection: Annotated[
        str | None,
        Field(
            default=None,
            min_length=1,
            max_length=300,
            validation_alias=AliasChoices("evolveDirection", "direction"),
        ),
    ]

    @model_validator(mode="after")
    def validate_mode_context(self) -> "GenerateRequest":
        """작업 모드별로 AI가 최소한 알아야 할 문맥을 강제한다."""
        if self.mode == GenerationMode.EVOLVE and not self.currentExcuse:
            raise ValueError("EVOLVE 모드에는 currentExcuse가 필요합니다.")

        if self.mode == GenerationMode.REPLY:
            if not self.incomingMessage:
                raise ValueError("REPLY 모드에는 incomingMessage가 필요합니다.")
            if not (self.currentExcuse or self.rootExcuse or self.conversation):
                raise ValueError(
                    "REPLY 모드에는 currentExcuse, rootExcuse, conversation 중 하나가 필요합니다."
                )

        return self


class SpringContextRequest(BaseModel):
    """Spring이 DB에서 조회해 내부 AI 요청에 덧붙이는 공통 문맥.

    FastAPI는 DB에 접근하지 않으므로 evolve/reply 요청의 기존 변명·대화 계보는
    반드시 Spring이 이 모델의 필드로 전달해야 한다.
    """

    model_config = ConfigDict(extra="forbid")

    situation: Annotated[str, Field(min_length=1, max_length=4000)]
    target: Target
    tone: Tone
    memory: Annotated[str, Field(default="", max_length=12000)]
    rootExcuse: Annotated[str | None, Field(default=None, min_length=1, max_length=1000)]
    conversation: Annotated[list[ConversationTurn], Field(default_factory=list, max_length=10)]
    currentExcuse: Annotated[str | None, Field(default=None, min_length=1, max_length=1000)]
    roundNumber: Annotated[int | None, Field(default=None, ge=1, le=10)]

    def to_generate_request(
        self,
        mode: GenerationMode,
        *,
        incoming_message: str | None = None,
        evolve_direction: str | None = None,
    ) -> GenerateRequest:
        """전용 Spring 엔드포인트 요청을 내부 표준 요청으로 통일한다."""
        return GenerateRequest(
            mode=mode,
            situation=self.situation,
            target=self.target,
            tone=self.tone,
            memory=self.memory,
            rootExcuse=self.rootExcuse,
            conversation=self.conversation,
            currentExcuse=self.currentExcuse,
            incomingMessage=incoming_message,
            roundNumber=self.roundNumber,
            evolveDirection=evolve_direction,
        )


class SpringCreateRequest(SpringContextRequest):
    """Spring의 `ExcuseCreateRequest`와 호환되는 최초 생성 요청.

    실제 필수 필드는 situation·target·tone뿐이고, 나머지 공통 문맥은 선택 사항이다.
    """

    pass


class SpringEvolveRequest(SpringContextRequest):
    """Spring의 기존 `direction` 필드명을 그대로 쓰는 변명 수정 요청."""

    direction: Annotated[str, Field(min_length=1, max_length=300)]

    @model_validator(mode="after")
    def validate_evolve_context(self) -> "SpringEvolveRequest":
        """원문 변명이 없으면 수정 방향만으로 일관된 결과를 만들 수 없다."""
        if not self.currentExcuse:
            raise ValueError("변명 수정에는 currentExcuse가 필요합니다.")
        return self

    def to_generate_request(self) -> GenerateRequest:
        """Spring의 direction을 내부 evolveDirection으로 옮긴다."""
        return super().to_generate_request(
            GenerationMode.EVOLVE,
            evolve_direction=self.direction,
        )


class SpringReplyRequest(SpringContextRequest):
    """Spring의 `incomingMessage`와 DB 문맥을 함께 받는 답장 생성 요청."""

    incomingMessage: Annotated[str, Field(min_length=1, max_length=2000)]

    @model_validator(mode="after")
    def validate_reply_context(self) -> "SpringReplyRequest":
        """상대 메시지만으로는 앞뒤가 맞는 답장을 만들 수 없으므로 계보를 확인한다."""
        if not (self.currentExcuse or self.rootExcuse or self.conversation):
            raise ValueError(
                "답장 생성에는 currentExcuse, rootExcuse, conversation 중 하나가 필요합니다."
            )
        return self

    def to_generate_request(self) -> GenerateRequest:
        """Spring의 incomingMessage를 내부 표준 요청에 옮긴다."""
        return super().to_generate_request(
            GenerationMode.REPLY,
            incoming_message=self.incomingMessage,
        )


class SpringItem(BaseModel):
    """Java `FastApiClient.Item` 응답과 일치하는 정렬된 문자열 항목."""

    model_config = ConfigDict(extra="forbid")

    content: Annotated[str, Field(min_length=1, max_length=200)]
    sortOrder: Annotated[int, Field(ge=0)]


class SpringAftermath(BaseModel):
    """Java `FastApiClient.Aftermath` 응답과 일치하는 후속 질문."""

    model_config = ConfigDict(extra="forbid")

    whenLabel: Annotated[str, Field(min_length=1, max_length=100)]
    dayOffset: Annotated[int, Field(ge=0, le=365)]
    question: Annotated[str, Field(min_length=1, max_length=300)]
    collapseRate: Annotated[int, Field(ge=0, le=100)]
    sortOrder: Annotated[int, Field(ge=0)]


class SpringExcuseResponse(BaseModel):
    """Java `FastApiClient.GeneratedExcuse`와 일치하는 내부 응답 계약."""

    model_config = ConfigDict(extra="forbid")

    excuse: Annotated[str, Field(min_length=20, max_length=1000)]
    successRate: Annotated[int, Field(ge=0, le=100)]
    realism: Annotated[int, Field(ge=1, le=5)]
    persuasion: Annotated[int, Field(ge=1, le=5)]
    suspicionLevel: SuspicionLevel
    riskFactors: Annotated[list[SpringItem], Field(max_length=5)]
    rememberItems: Annotated[list[SpringItem], Field(max_length=8)]
    aftermaths: Annotated[list[SpringAftermath], Field(max_length=4)]

    @classmethod
    def from_result(cls, result: ExcuseResult) -> "SpringExcuseResponse":
        """평면 제공자 결과를 Java 클라이언트의 내부 응답 구조로 변환한다."""
        return cls(
            excuse=result.excuse,
            successRate=result.successRate,
            realism=result.realism,
            persuasion=result.persuasion,
            suspicionLevel=result.suspicionLevel,
            riskFactors=[
                SpringItem(content=item, sortOrder=index)
                for index, item in enumerate(result.riskFactors)
            ],
            rememberItems=[
                SpringItem(content=item, sortOrder=index)
                for index, item in enumerate(result.remember)
            ],
            aftermaths=[
                SpringAftermath(
                    whenLabel=item.when,
                    dayOffset=item.dayOffset,
                    question=item.question,
                    collapseRate=item.collapseRate,
                    sortOrder=index,
                )
                for index, item in enumerate(result.aftermath)
            ],
        )


# 배열 개수와 문자열 길이는 제공자 응답을 받은 뒤 Pydantic이 최종 검증한다.
# Cerebras JSON Schema에는 지원 범위가 다른 제약을 과도하게 넣지 않는다.
LLM_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "excuse": {"type": "string"},
        "recommendedAction": {
            "type": "string",
            "description": "변명을 보낸 뒤 사용자가 실제로 할 복구 행동 한 가지.",
        },
        "likelyFollowUp": {
            "type": "string",
            "description": "상대가 이어서 물을 가능성이 가장 높은 짧은 질문 한 가지.",
        },
        "replyOptions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "같은 사실을 유지하면서 말투만 다른 2~3개의 짧은 선택지.",
        },
        "successRate": {
            "type": "integer",
            "description": "상대가 믿을 가능성. 0부터 100 사이의 정수.",
        },
        "realism": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5],
            "description": "현실성 점수. 1부터 5 사이의 정수.",
        },
        "persuasion": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5],
            "description": "설득력 점수. 1부터 5 사이의 정수.",
        },
        "suspicionLevel": {
            "type": "string",
            "enum": ["LOW", "MEDIUM", "HIGH"],
        },
        "riskFactors": {"type": "array", "items": {"type": "string"}},
        "aftermath": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "when": {"type": "string"},
                    "dayOffset": {
                        "type": "integer",
                        "description": "후속 질문이 발생할 예상 시점. 오늘은 0, 3일 뒤는 3, 7일 뒤는 7.",
                    },
                    "question": {"type": "string"},
                    "collapseRate": {
                        "type": "integer",
                        "description": "변명이 무너질 가능성. 0부터 100 사이의 정수.",
                    },
                },
                "required": ["when", "dayOffset", "question", "collapseRate"],
                "additionalProperties": False,
            },
        },
        "remember": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "excuse",
        "recommendedAction",
        "likelyFollowUp",
        "replyOptions",
        "successRate",
        "realism",
        "persuasion",
        "suspicionLevel",
        "riskFactors",
        "aftermath",
        "remember",
    ],
    "additionalProperties": False,
}
