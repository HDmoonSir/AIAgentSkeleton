"""
예시 앱 설정 (단독 실행용)
- 서버 구동(host/port/debug)과 외부 모델 API 접속 설정만 담는다.
- 추론 LLM 설정은 코어(aiagent/config/loader.py)로 분리되어 있다.
- 실제 모델 API(사내/외부)를 쓸 때는 MODEL_API_* 값을 교체한다.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App (예시 서버 구동)
    app_env: str = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # 외부 AI 모델 API (도구가 호출하는 대상)
    model_api_base_url: str = "https://api.example.com/v1"
    model_api_key: str = ""
    model_api_timeout_seconds: float = 30.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
