import typing as tp

from pydantic import BaseModel, Field

from aiagent.constants import IntentType
from aiagent.schemas.agent_io import (
    ConversationMessage,
    PlannedToolCall,
    ToolExecutionReport,
)


class AgentState(BaseModel):
    """
    Agent의 상태를 관리하는 데이터 클래스.
    모든 상태는 직렬화 가능한 형태로 유지한다.
    """

    str_user_input: str = Field(description="사용자로부터 입력받은 원문 질의")
    model_intent: tp.Optional[IntentType] = Field(default=None, description="분류된 사용자 의도")
    dict_extracted_params: tp.Dict[str, tp.Any] = Field(
        default_factory=dict,
        description="LLM이 추출한 파라미터. 의도별 형태가 상이하여 Any 사용.",
    )
    list_messages: tp.List[ConversationMessage] = Field(
        default_factory=list, description="대화 이력"
    )
    list_planned_tool_calls: tp.List[PlannedToolCall] = Field(
        default_factory=list, description="LLM이 계획한 도구 호출 목록"
    )
    str_direct_answer: tp.Optional[str] = Field(
        default=None, description="도구 호출 없이 LLM이 직접 생성한 답변"
    )
    model_tool_report: tp.Optional[ToolExecutionReport] = Field(
        default=None, description="도구 실행 결과 묶음"
    )
    str_final_answer: tp.Optional[str] = Field(
        default=None, description="사용자에게 전달할 최종 자연어 답변"
    )
