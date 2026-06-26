"""
FastAPI 의존성 주입 모듈
- lifespan에서 조립된 공유 객체(app.state)를 라우트로 주입
"""

from fastapi import Request

from app.agent.conversation_store import ConversationStore
from aiagent.agents.agent_orchestrator import AgentOrchestrator


def get_agent_orchestrator(request: Request) -> AgentOrchestrator:
    """lifespan에서 조립되어 app.state에 보관된 공유 오케스트레이터를 주입."""
    return request.app.state.agent_orchestrator


def get_conversation_store(request: Request) -> ConversationStore:
    """lifespan에서 생성되어 app.state에 보관된 대화 이력 스토어를 주입."""
    return request.app.state.conversation_store
