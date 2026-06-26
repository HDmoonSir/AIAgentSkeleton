"""
FastAPI 라우터
━━━━━━━━━━━━
엔드포인트 정의:
  - GET  /health : 헬스체크
  - POST /chat   : Agent 대화
"""

import uuid
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.agent.conversation_store import ConversationStore
from app.core.dependencies import get_agent_orchestrator, get_conversation_store
from aiagent.agents.agent_orchestrator import AgentOrchestrator
from aiagent.agents.facade import AgentFacade

router = APIRouter()


# ── Request/Response 모델 ────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., description="사용자 메시지")
    user_id: str = Field(default="dev-user", description="사용자 ID")
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    analysis_data: dict[str, Any] | None = None
    conversation_id: str | None = None


# ── 엔드포인트 ───────────────────────────────

@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "agent-ai-skeleton"}


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    agent_orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    conversation_store: ConversationStore = Depends(get_conversation_store),
) -> ChatResponse:
    """
    Agent 대화 엔드포인트.
    공유 오케스트레이터로 요청마다 facade를 만들고, conversation_id 단위 이력을 주입/저장한다.
    """
    str_conversation_id: str = request.conversation_id or uuid.uuid4().hex

    model_agent: AgentFacade = AgentFacade(agent_orchestrator)
    model_agent.list_history = conversation_store.get_history(str_conversation_id)

    model_chat_result = await model_agent.chat(request.message)

    conversation_store.save_history(str_conversation_id, model_agent.list_history)

    dict_analysis_data: dict[str, Any] | None = (
        model_chat_result.model_tool_report.model_dump()
        if model_chat_result.model_tool_report is not None
        else None
    )
    return ChatResponse(
        response=model_chat_result.str_answer or "응답을 생성할 수 없습니다.",
        analysis_data=dict_analysis_data,
        conversation_id=str_conversation_id,
    )
