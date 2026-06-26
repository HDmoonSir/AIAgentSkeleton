from pydantic import BaseModel, Field

from aiagent.constants import DEFAULT_MODEL_NAME, DEFAULT_TEMPERATURE


class AgentConfig(BaseModel):
    """Agent의 추론 LLM 연결 설정."""

    str_api_key: str = Field(description="LLM API 키")
    str_base_url: str = Field(description="LLM API Base URL")
    str_model_name: str = Field(default=DEFAULT_MODEL_NAME, description="사용할 모델명")
    float_temperature: float = Field(default=DEFAULT_TEMPERATURE, description="답변 생성 온도")
