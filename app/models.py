from enum import StrEnum
from typing import Annotated

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class Target(StrEnum):
    TEACHER = "TEACHER"
    PARENT = "PARENT"
    FRIEND = "FRIEND"
    LOVER = "LOVER"
    TEAM_LEAD = "TEAM_LEAD"
    TEAM_MEMBER = "TEAM_MEMBER"


class Tone(StrEnum):
    MILD = "MILD"
    SLICK = "SLICK"
    DESPERATE = "DESPERATE"
    BULLSHIT = "BULLSHIT"


class GenerationMode(StrEnum):
    CREATE = "CREATE"
    EVOLVE = "EVOLVE"
    REPLY = "REPLY"


class ConversationRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class SuspicionLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Aftermath(BaseModel):
    model_config = ConfigDict(extra="forbid")

    when: Annotated[str, Field(min_length=1, max_length=100)]
    dayOffset: Annotated[int, Field(ge=0, le=365)]
    question: Annotated[str, Field(min_length=1, max_length=300)]
    collapseRate: Annotated[int, Field(ge=0, le=100)]


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: ConversationRole
    content: Annotated[str, Field(min_length=1, max_length=2000)]

    @field_validator("content", mode="before")
    @classmethod
    def trim_content(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class ExcuseResult(BaseModel):
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
        return value.strip() if isinstance(value, str) else value

    @field_validator("riskFactors", "remember", "replyOptions", mode="before")
    @classmethod
    def trim_items(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        return [item.strip() if isinstance(item, str) else item for item in value]

    @field_validator("riskFactors", "remember", "replyOptions")
    @classmethod
    def reject_empty_items(cls, value: list[str]) -> list[str]:
        if any(not item for item in value):
            raise ValueError("array items must not be empty")
        if any(len(item) > 200 for item in value):
            raise ValueError("array items must be 200 characters or fewer")
        return value


class GenerateRequest(BaseModel):
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
        if self.mode == GenerationMode.EVOLVE and not self.currentExcuse:
            raise ValueError("EVOLVE mode requires currentExcuse")

        if self.mode == GenerationMode.REPLY:
            if not self.incomingMessage:
                raise ValueError("REPLY mode requires incomingMessage")
            if not (self.currentExcuse or self.rootExcuse or self.conversation):
                raise ValueError(
                    "REPLY mode requires currentExcuse, rootExcuse, or conversation"
                )

        return self


class SpringAnalysis(BaseModel):
    """Shape of Spring's nested ExcuseResponse.analysis object."""

    model_config = ConfigDict(extra="forbid")

    successRate: Annotated[int, Field(ge=0, le=100)]
    realism: Annotated[int, Field(ge=1, le=5)]
    persuasion: Annotated[int, Field(ge=1, le=5)]
    suspicionLevel: SuspicionLevel
    riskFactors: Annotated[list[str], Field(min_length=1, max_length=5)]


class SpringExcuseResponse(BaseModel):
    """AI-owned portion of Spring's ExcuseResponse contract.

    IDs, XP, complexity warnings, and timestamps remain Spring concerns.
    """

    model_config = ConfigDict(extra="forbid")

    incomingMessage: str | None = None
    roundNumber: Annotated[int, Field(ge=1, le=10)]
    excuse: Annotated[str, Field(min_length=20, max_length=1000)]
    target: Target
    tone: Tone
    analysis: SpringAnalysis
    aftermath: Annotated[list[Aftermath], Field(min_length=1, max_length=4)]
    remember: Annotated[list[str], Field(max_length=8)]
    recommendedAction: Annotated[str, Field(min_length=1, max_length=300)]
    likelyFollowUp: Annotated[str, Field(min_length=1, max_length=300)]
    replyOptions: Annotated[list[str], Field(min_length=2, max_length=3)]

    @classmethod
    def from_result(cls, request: GenerateRequest, result: ExcuseResult) -> "SpringExcuseResponse":
        return cls(
            incomingMessage=request.incomingMessage,
            roundNumber=request.roundNumber or 1,
            excuse=result.excuse,
            target=request.target,
            tone=request.tone,
            analysis=SpringAnalysis(
                successRate=result.successRate,
                realism=result.realism,
                persuasion=result.persuasion,
                suspicionLevel=result.suspicionLevel,
                riskFactors=result.riskFactors,
            ),
            aftermath=result.aftermath,
            remember=result.remember,
            recommendedAction=result.recommendedAction,
            likelyFollowUp=result.likelyFollowUp,
            replyOptions=result.replyOptions,
        )


# Array cardinality and string length are enforced by Pydantic after the
# provider response. minItems/maxItems are intentionally not in this schema.
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
