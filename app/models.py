"""FastAPI 입출력 계약과 Cerebras Structured Outputs 스키마.

모든 필드는 camelCase를 사용한다. Spring의 Jackson 기본 설정과 FastAPI의 JSON
응답을 별도 변환 없이 연결하기 위해서다. 이 파일은 Spring 요청, 내부 표준 요청,
제공자 결과, Spring 응답 사이의 신뢰 경계를 검증하는 유일한 계약 계층이다.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

try:
    from enum import StrEnum
except ImportError:
    # Python 3.11의 StrEnum을 쓸 수 없는 Python 3.9 로컬 환경에서도 enum 값이 JSON
    # 문자열처럼 직렬화되게 하는 최소 호환 구현이다.
    class StrEnum(str, Enum):
        """Python 3.9 로컬 실행을 위한 StrEnum 호환 클래스."""

        pass


class Target(StrEnum):
    """변명을 전달할 상대. Spring의 Target enum 값과 반드시 같아야 한다.

    값 자체가 프롬프트의 대상별 수습 전략을 선택하는 키이므로 표시용 한글 문자열이
    아닌 안정적인 대문자 계약 값을 사용한다.
    """

    TEACHER = "TEACHER"
    PARENT = "PARENT"
    FRIEND = "FRIEND"
    LOVER = "LOVER"
    TEAM_LEAD = "TEAM_LEAD"
    TEAM_MEMBER = "TEAM_MEMBER"


class Tone(StrEnum):
    """문장 분위기. Spring의 Tone enum 값과 반드시 같아야 한다.

    Tone은 표현 강도만 조절하며, 사실 발명 금지·책임 인정 같은 안전 규칙을 해제하지
    않는다. 해당 안전 규칙은 prompts.py의 공통 system prompt에 남겨 둔다.
    """

    MILD = "MILD"
    SLICK = "SLICK"
    DESPERATE = "DESPERATE"
    BULLSHIT = "BULLSHIT"


class GenerationMode(StrEnum):
    """통합 호환 API에서 어떤 생성 작업을 할지 나타낸다.

    전용 URL도 내부적으로는 이 값으로 통일한다. 따라서 service.py는 URL 종류가 아닌
    mode만 보고 REPLY 품질 검사처럼 모드별 동작을 선택할 수 있다.
    """

    CREATE = "CREATE"
    EVOLVE = "EVOLVE"
    REPLY = "REPLY"


class ConversationRole(StrEnum):
    """대화 가지에서 발화한 주체.

    소문자 값은 Chat Completions 역할 이름과 우연히 같지만, 여기서는 Spring이 저장한
    대화 계보의 발화 주체를 뜻한다. 제공자 messages로 직접 전달하지 않는다.
    """

    USER = "user"
    ASSISTANT = "assistant"


class SuspicionLevel(StrEnum):
    """AI가 추정한 변명의 의심도.

    신뢰도 점수와 달리 UI가 빠르게 색상·경고 수준을 결정할 수 있는 이산 값으로 둔다.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Aftermath(BaseModel):
    """변명을 보낸 뒤 예상되는 후속 질문 한 건.

    `dayOffset`은 Spring의 `ExcuseAftermath.dayOffset`에 그대로 저장된다.
    """

    # Structured Output에 예상하지 못한 필드가 섞이면 Spring 영속 모델과 계약이
    # 어긋날 수 있으므로 모든 내부 결과 모델은 extra 필드를 거부한다.
    model_config = ConfigDict(extra="forbid")

    # when은 UI용 라벨, dayOffset은 정렬·저장용 정수다. 라벨을 파싱해 날짜를 계산하지
    # 않도록 둘을 함께 받는다.
    when: Annotated[str, Field(min_length=1, max_length=100)]
    dayOffset: Annotated[int, Field(ge=0, le=365)]
    question: Annotated[str, Field(min_length=1, max_length=300)]
    collapseRate: Annotated[int, Field(ge=0, le=100)]


class ConversationTurn(BaseModel):
    """Spring이 DB 계보에서 조립한 현재 대화 가지의 한 발화.

    FastAPI는 대화 전체를 조회하거나 새 turn을 저장하지 않는다. Spring이 선택한 한
    가지(branch)만 전달하면 이 모델은 프롬프트와 중복 방지 검사에 사용할 최소 정보만
    검증한다.
    """

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

    # excuse는 사용자가 바로 복사해 보낼 기본안이다. 추천 행동·예상 질문·후보 답장은
    # 서비스가 품질을 설명하거나 다음 상호작용을 준비하는 데 쓰는 보조 결과다.
    excuse: Annotated[str, Field(min_length=20, max_length=1000)]
    recommendedAction: Annotated[str, Field(min_length=1, max_length=300)]
    likelyFollowUp: Annotated[str, Field(min_length=1, max_length=300)]
    replyOptions: Annotated[list[str], Field(min_length=2, max_length=3)]
    # 수치 필드는 임의의 정규화된 범위를 사용한다. Pydantic이 범위를 강제해 제공자의
    # "120점" 같은 비정상 값을 Spring에 전달하지 않는다.
    successRate: Annotated[int, Field(ge=0, le=100)]
    realism: Annotated[int, Field(ge=1, le=5)]
    persuasion: Annotated[int, Field(ge=1, le=5)]
    suspicionLevel: SuspicionLevel
    # 목록 길이를 제한하면 긴 생성 결과가 API 응답과 프롬프트 재시도 문맥을 과도하게
    # 키우는 일을 막는다.
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
        """비어 있거나 지나치게 긴 선택지를 제공자 응답으로 인정하지 않는다.

        JSON Schema가 제공자마다 배열 항목 길이 제약을 동일하게 지원하지 않을 수 있어,
        최종 방어선으로 Pydantic 검증을 한 번 더 둔다.
        """
        if any(not item for item in value):
            raise ValueError("배열 항목은 비어 있을 수 없습니다.")
        if any(len(item) > 200 for item in value):
            raise ValueError("배열 항목은 200자 이하여야 합니다.")
        return value


class GenerateRequest(BaseModel):
    """모든 생성 엔드포인트가 사용하는 내부 표준 요청 모델.

    Spring 전용 create/evolve/reply DTO는 URL마다 친숙한 입력 필드명을 유지하고, 실제
    프롬프트·LLM 계층에는 이 모델 하나만 전달한다. 생성 규칙을 모드별 URL에 중복하지
    않기 위한 어댑터 경계다.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # CREATE의 선택 입력까지 하나의 모델로 표현한다. EVOLVE/REPLY에서만 필요한 값은
    # 아래 model_validator가 조건부로 검사한다.
    mode: GenerationMode = GenerationMode.CREATE
    situation: Annotated[str, Field(min_length=1, max_length=4000)]
    target: Target
    tone: Tone
    memory: Annotated[str, Field(default="", max_length=12000)]
    rootExcuse: Annotated[
        str | None, Field(default=None, min_length=1, max_length=1000)
    ]
    conversation: Annotated[
        list[ConversationTurn], Field(default_factory=list, max_length=10)
    ]
    currentExcuse: Annotated[
        str | None, Field(default=None, min_length=1, max_length=1000)
    ]
    incomingMessage: Annotated[
        str | None, Field(default=None, min_length=1, max_length=2000)
    ]
    # 원격 협업 정책에 맞춰 답장 라운드는 최대 5회로 제한한다.
    roundNumber: Annotated[int | None, Field(default=None, ge=1, le=5)]
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
        """작업 모드별로 AI가 최소한 알아야 할 문맥을 강제한다.

        이 검증을 API 경계에서 끝내면 prompts.py가 누락값을 추측하거나 LLM이 문맥 없이
        답하도록 두지 않아도 된다.
        """
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

    # Spring 전용 DTO에도 extra=forbid를 적용해 필드명 오타가 조용히 무시되는 대신
    # 호출 시점에 422로 드러나게 한다.
    model_config = ConfigDict(extra="forbid")

    situation: Annotated[str, Field(min_length=1, max_length=4000)]
    target: Target
    tone: Tone
    memory: Annotated[str, Field(default="", max_length=12000)]
    # Spring은 DB에서 선택한 현재 가지의 문맥을 이 필드들로 전달한다. 전용
    # create/evolve/reply DTO가 모두 이 공통 기반 모델을 쓰므로, 여기서 선언해야
    # FastAPI가 `extra=forbid`로 정상적인 REPLY 문맥을 거부하지 않는다.
    rootExcuse: Annotated[
        str | None, Field(default=None, min_length=1, max_length=1000)
    ]
    conversation: Annotated[
        list[ConversationTurn], Field(default_factory=list, max_length=10)
    ]
    currentExcuse: Annotated[
        str | None, Field(default=None, min_length=1, max_length=1000)
    ]
    # Spring의 `ReplyRequest`는 다음 답장을 만들 라운드 번호를 이미 증가시켜 보낸다.
    # GenerateRequest와 같은 1~5 범위를 적용해 두 API 계약이 달라지지 않게 한다.
    roundNumber: Annotated[int | None, Field(default=None, ge=1, le=5)]

    def to_generate_request(
        self,
        mode: GenerationMode,
        *,
        incoming_message: str | None = None,
        evolve_direction: str | None = None,
    ) -> GenerateRequest:
        """전용 Spring 엔드포인트 요청을 내부 표준 요청으로 통일한다.

        변환은 순수 데이터 매핑만 수행한다. DB 조회·계보 선택은 이미 Spring이 끝낸 뒤
        이 모델을 만들었다는 전제를 유지한다.
        """
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

    # 공통 문맥만으로 충분하므로 추가 필드가 없다. pass를 남기는 이유는 FastAPI
    # OpenAPI에서 CREATE 요청의 의미 있는 이름을 유지하기 위해서다.
    pass


class SpringEvolveRequest(SpringContextRequest):
    """Spring의 기존 ``direction`` 필드명을 그대로 쓰는 변명 수정 요청."""

    direction: Annotated[str, Field(min_length=1, max_length=300)]

    @model_validator(mode="after")
    def validate_evolve_context(self) -> "SpringEvolveRequest":
        """원문 변명이 없으면 수정 방향만으로 일관된 결과를 만들 수 없다."""
        if not self.currentExcuse:
            raise ValueError("변명 수정에는 currentExcuse가 필요합니다.")
        return self

    def to_generate_request(self) -> GenerateRequest:
        """Spring의 direction을 내부 evolveDirection으로 옮긴다.

        내부 모델은 필드 의미를 명확히 하기 위해 evolveDirection을 쓰지만, 기존 Java
        클라이언트의 JSON 계약은 바꾸지 않는다.
        """
        return super().to_generate_request(
            GenerationMode.EVOLVE,
            evolve_direction=self.direction,
        )


class SpringReplyRequest(SpringContextRequest):
    """Spring의 ``incomingMessage``와 DB 문맥을 함께 받는 답장 생성 요청."""

    incomingMessage: Annotated[str, Field(min_length=1, max_length=2000)]

    @model_validator(mode="after")
    def validate_reply_context(self) -> "SpringReplyRequest":
        """상대 메시지만으로는 앞뒤가 맞는 답장을 만들 수 없으므로 계보를 확인한다.

        REPLY는 service.py에서 currentExcuse·assistant turn과 유사도를 비교한다. 따라서
        최소 한 개의 기존 문맥이 없으면 모델 호출 전에 요청을 거부한다.
        """
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
    """Java ``FastApiClient.Item`` 응답과 일치하는 정렬된 문자열 항목.

    배열 순서만 믿지 않고 sortOrder를 함께 보내면 Spring이 영속·재조회한 뒤에도 생성
    당시의 항목 순서를 안정적으로 복원할 수 있다.
    """

    model_config = ConfigDict(extra="forbid")

    content: Annotated[str, Field(min_length=1, max_length=200)]
    sortOrder: Annotated[int, Field(ge=0)]


class SpringAftermath(BaseModel):
    """Java ``FastApiClient.Aftermath`` 응답과 일치하는 후속 질문."""

    model_config = ConfigDict(extra="forbid")

    whenLabel: Annotated[str, Field(min_length=1, max_length=100)]
    dayOffset: Annotated[int, Field(ge=0, le=365)]
    question: Annotated[str, Field(min_length=1, max_length=300)]
    collapseRate: Annotated[int, Field(ge=0, le=100)]
    sortOrder: Annotated[int, Field(ge=0)]


class SpringExcuseResponse(BaseModel):
    """Java ``FastApiClient.GeneratedExcuse``와 일치하는 내부 응답 계약.

    제공자 평면 결과의 편의 필드를 그대로 노출하지 않고 Java 클라이언트가 기대하는
    정렬된 nested item 구조로만 변환한다. Spring이 DB ID·XP·부모 계보를 추가하는
    책임도 이 경계 밖에 남는다.
    """

    model_config = ConfigDict(extra="forbid")

    excuse: Annotated[str, Field(min_length=20, max_length=1000)]
    successRate: Annotated[int, Field(ge=0, le=100)]
    realism: Annotated[int, Field(ge=1, le=5)]
    persuasion: Annotated[int, Field(ge=1, le=5)]
    suspicionLevel: SuspicionLevel
    # REPLY UI는 세 후보의 순서 자체에 의미가 있다. Spring이 Item으로 재구성하면
    # 복사해 보낼 문장 API가 불필요하게 복잡해지므로, 생성 순서를 유지한 문자열 목록을
    # 그대로 전달한다.
    replyOptions: Annotated[list[str], Field(min_length=2, max_length=3)]
    riskFactors: Annotated[list[SpringItem], Field(max_length=5)]
    rememberItems: Annotated[list[SpringItem], Field(max_length=8)]
    aftermaths: Annotated[list[SpringAftermath], Field(max_length=4)]

    @classmethod
    def from_result(cls, result: ExcuseResult) -> "SpringExcuseResponse":
        """평면 제공자 결과를 Java 클라이언트의 내부 응답 구조로 변환한다.

        enumerate의 0 기반 순서는 Spring의 sortOrder 계약과 맞춘다. 목록 문자열은
        LLM이 순서를 반환하므로 여기서 임의로 정렬하지 않는다.
        """
        return cls(
            excuse=result.excuse,
            successRate=result.successRate,
            realism=result.realism,
            persuasion=result.persuasion,
            suspicionLevel=result.suspicionLevel,
            replyOptions=result.replyOptions,
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


# 이 사전은 Cerebras에 보내는 JSON Schema다. 배열 개수와 문자열 길이처럼 제공자마다
# 지원 범위가 달라질 수 있는 제약은 ``ExcuseResult`` Pydantic 모델에서 최종 검증한다.
# 즉, Schema는 모델이 알아야 할 필드 형태를 안내하고 Pydantic은 서버 신뢰 경계를 지킨다.
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
            "description": (
                "같은 사실을 유지하는 답장 선택지. REPLY 모드에서는 정확히 3개를 순서대로 "
                "작성한다: 짧은 직접 답장, 정중한 책임·수습 답장, 가벼운 관계 수습 답장. "
                "세 문장은 길이·톤·수습 전략이 서로 달라야 한다."
            ),
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
