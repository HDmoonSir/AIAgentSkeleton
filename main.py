"""
Agent AI Skeleton - FastAPI 진입점 (조립 루트)
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.adapters.http_inference_client import HttpInferenceModelClient
from app.agent.conversation_store import InMemoryConversationStore
from app.api.routes import router
from app.core.config import Settings, get_settings
from aiagent.config.loader import load_agent_config
from aiagent.config.schemas import AgentConfig
from aiagent.core.factory import assemble_orchestrator
from aiagent.ports.inference_model import InferenceModelClient

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시의 조립 루트.

    모든 의존성을 여기서 명시적으로 생성·주입한다.
    """
    app_settings: Settings = get_settings()
    agent_config: AgentConfig = load_agent_config()

    # 외부 모델 API 호출용 HTTP 클라이언트. 종료 시 명시적으로 닫는다.
    http_client: httpx.Client = httpx.Client(timeout=app_settings.model_api_timeout_seconds)
    # 모델 호출 경계(포트) 구현 주입. 사내/외부 모델 API 어댑터로 교체한다.
    inference_client: InferenceModelClient = HttpInferenceModelClient(
        app_settings.model_api_base_url, app_settings.model_api_key, http_client
    )

    app.state.agent_orchestrator = assemble_orchestrator(agent_config, inference_client)
    app.state.conversation_store = InMemoryConversationStore()

    logger.info("Agent AI Skeleton started")
    try:
        yield
    finally:
        http_client.close()
        logger.info("Agent AI Skeleton stopped")


def create_app() -> FastAPI:
    app_settings: Settings = get_settings()

    app = FastAPI(
        title="Agent AI Skeleton",
        description="AI Agent 서비스 골격 - LangGraph + FastAPI",
        version="0.1.0",
        lifespan=lifespan,
        debug=app_settings.app_debug,
    )

    app.include_router(router, prefix="/api/v1")

    # 예시 Front
    if FRONTEND_DIR.is_dir():
        app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
    )
