import typing as tp

from aiagent.agents.agent_orchestrator import AgentOrchestrator
from aiagent.agents.agent_state import AgentState
from aiagent.constants import DEFAULT_INTENT
from aiagent.schemas.agent_io import ChatResult, ConversationMessage


class AgentFacade:
    """
    외부 시스템(FastAPI 등)에서 사용할 Agent 인터페이스.
    오케스트레이터는 상위 조립 계층에서 생성되어 주입.
    """

    def __init__(self, agent_orchestrator: AgentOrchestrator) -> None:
        self.agent_orchestrator: AgentOrchestrator = agent_orchestrator
        self.list_history: tp.List[ConversationMessage] = list()

    async def chat(self, str_query: str) -> ChatResult:
        """
        사용자 질문을 입력받아 분석 결과를 반환한다. (대화 이력 유지)

        Args:
            str_query: 사용자 질문

        Returns:
            분석 결과 및 답변이 담긴 ChatResult
        """
        model_state: AgentState = await self.agent_orchestrator.run(
            str_query, self.list_history
        )
        self.list_history = model_state.list_messages

        return ChatResult(
            str_answer=model_state.str_final_answer,
            model_intent=model_state.model_intent or DEFAULT_INTENT,
            model_tool_report=model_state.model_tool_report,
        )
