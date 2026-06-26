import typing as tp

from pydantic import BaseModel, Field

from aiagent.constants import IntentType, MessageRole


class ConversationMessage(BaseModel):
    """대화 이력 한 건. 외부 응답 객체 대신 내부에서 사용하는 메시지 계약."""

    model_role: MessageRole = Field(description="발화 주체")
    str_content: str = Field(description="메시지 본문")


class PlannedToolCall(BaseModel):
    """LLM이 계획한 단일 도구 호출 정보."""

    str_name: str = Field(description="호출할 도구 이름")
    dict_args: tp.Dict[str, tp.Any] = Field(
        default_factory=dict,
        description="도구 인자. 도구별 스키마가 상이하여 동적 매핑이 필요하므로 Any 사용.",
    )
    str_id: tp.Optional[str] = Field(default=None, description="LLM이 부여한 호출 식별자")


class ToolExecutionResult(BaseModel):
    """단일 도구 실행 결과."""

    str_tool_name: str = Field(description="실행한 도구 이름")
    dict_tool_input: tp.Dict[str, tp.Any] = Field(
        default_factory=dict,
        description="도구에 전달한 인자. 도구별 입력 스키마가 상이하여 Any 사용.",
    )
    dict_tool_output: tp.Optional[tp.Dict[str, tp.Any]] = Field(
        default=None,
        description="도구 반환값. 구현 시점에 확정되는 외부 경계 데이터이므로 Any 사용.",
    )
    str_error: tp.Optional[str] = Field(default=None, description="실행 실패 사유")


class ToolExecutionReport(BaseModel):
    """다수 도구 실행 결과 묶음."""

    list_results: tp.List[ToolExecutionResult] = Field(
        default_factory=list, description="개별 도구 실행 결과 목록"
    )
    str_message: str = Field(description="실행 요약 메시지")


class ChatResult(BaseModel):
    """외부 시스템에 반환하는 최종 응답 계약."""

    str_answer: tp.Optional[str] = Field(default=None, description="사용자에게 전달할 자연어 답변")
    model_intent: IntentType = Field(description="분류된 사용자 의도")
    model_tool_report: tp.Optional[ToolExecutionReport] = Field(
        default=None, description="도구 실행 결과 묶음 (도구 미실행 시 None)"
    )
