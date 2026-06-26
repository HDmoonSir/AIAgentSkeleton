"""
LLM 연결 설정 로더 (코어)
- .env의 LLM_* 환경변수를 읽어 AgentConfig로 변환한다.
- 코어가 필요로 하는 유일한 설정이며, 서버/외부 API 설정과는 분리한다.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from aiagent.config.schemas import AgentConfig
from aiagent.constants import DEFAULT_MODEL_NAME, DEFAULT_TEMPERATURE

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class LLMSettings(BaseSettings):
    """LLM 연결 설정(.env의 LLM_* 변수)."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        env_prefix="LLM_",
        extra="ignore",
    )

    api_base_url: str = "https://api.example.com/v1"
    api_key: str = ""
    model_name: str = DEFAULT_MODEL_NAME
    temperature: float = DEFAULT_TEMPERATURE


@lru_cache
def load_agent_config() -> AgentConfig:
    """.env에서 LLM 설정을 읽어 AgentConfig로 반환."""
    llm_settings: LLMSettings = LLMSettings()
    return AgentConfig(
        str_api_key=llm_settings.api_key,
        str_base_url=llm_settings.api_base_url,
        str_model_name=llm_settings.model_name,
        float_temperature=llm_settings.temperature,
    )
