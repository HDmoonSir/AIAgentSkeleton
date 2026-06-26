"""
외부 AI 모델 API 호출 경계(포트).
- 에이전트 도구는 이 포트를 통해서만 모델 API를 호출한다.
- 구체 구현(어댑터: HTTP 클라이언트 등)은 인프라/진입점 계층에서 주입한다.
"""

import abc
import typing as tp

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """모델 API에 전달하는 추론 요청 계약."""

    str_model_name: str = Field(description="호출할 모델 식별자")
    str_input: str = Field(description="모델에 전달할 입력 텍스트")
    dict_parameters: tp.Dict[str, float] = Field(
        default_factory=dict,
        description="모델별 추론 파라미터(temperature 등). 모델마다 키가 상이하다.",
    )


class InferenceResult(BaseModel):
    """모델 API가 반환하는 추론 결과 계약."""

    str_model_name: str = Field(description="추론을 수행한 모델 식별자")
    str_output: str = Field(description="모델이 생성한 출력 텍스트")
    dict_metadata: tp.Dict[str, str] = Field(
        default_factory=dict, description="지연시간·토큰수 등 부가 메타데이터"
    )


class InferenceModelError(RuntimeError):
    """모델 API 호출 경계에서 발생한 인프라 오류."""


class InferenceModelClient(abc.ABC):
    """AI 모델 API 호출 경계(포트). 구현(어댑터)은 진입점에서 주입한다."""

    @abc.abstractmethod
    def run_inference(self, model_request: InferenceRequest) -> InferenceResult:
        """모델 API를 호출해 추론 결과를 반환한다."""
        ...
