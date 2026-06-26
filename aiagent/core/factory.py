"""
코어 조립 (Composition) — 데이터/프레임워크 독립
- 이미 생성된 의존성(모델 호출 경계·LLM 설정)을 받아 오케스트레이터를 조립한다.
- 구체적인 모델 API 구현(HTTP 클라이언트 등)을 알지 못한다.
"""

import typing as tp

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from aiagent.agents.agent_orchestrator import AgentOrchestrator
from aiagent.config.schemas import AgentConfig
from aiagent.ports.inference_model import InferenceModelClient
from aiagent.tools.tool_registry import build_tools


def build_llm(agent_config: AgentConfig) -> ChatOpenAI:
    """명시적 설정을 바탕으로 추론 LLM 객체를 생성하여 반환."""
    return ChatOpenAI(
        openai_api_key=agent_config.str_api_key,
        base_url=agent_config.str_base_url,
        model_name=agent_config.str_model_name,
        temperature=agent_config.float_temperature,
    )


def assemble_orchestrator(
    agent_config: AgentConfig,
    inference_client: InferenceModelClient,
) -> AgentOrchestrator:
    """주입된 모델 호출 경계·LLM 설정으로 오케스트레이터를 조립한다.

    inference_client는 포트(InferenceModelClient)만 받으므로 구현(HTTP/사내 모델 API 등)을
    알 필요가 없다. 구체 의존성 생성은 상위 진입점(조립 루트)이 담당한다.
    """
    model_llm: ChatOpenAI = build_llm(agent_config)
    list_tools: tp.List[BaseTool] = build_tools(inference_client)
    return AgentOrchestrator(model_llm, list_tools)
