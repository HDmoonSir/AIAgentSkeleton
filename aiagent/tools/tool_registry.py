"""
에이전트 도구 등록부.
- 외부 AI 모델 API 호출을 LangChain 도구로 노출한다.
- 모델 호출 경계(InferenceModelClient 포트)는 주입받으며, 구체 구현은 알지 못한다.
- 새 도구를 추가할 때는 이 모듈에서 포트를 주입받아 build_tools 목록에 등록한다.
"""

import typing as tp

from langchain_core.tools import BaseTool, tool

from aiagent.ports.inference_model import (
    InferenceModelClient,
    InferenceRequest,
    InferenceResult,
)

TOOL_NAME_RUN_MODEL_INFERENCE: str = "run_model_inference"


def build_tools(inference_client: InferenceModelClient) -> tp.List[BaseTool]:
    """주입된 모델 호출 경계로 에이전트 도구 목록을 구성한다."""

    @tool(TOOL_NAME_RUN_MODEL_INFERENCE)
    def run_model_inference(str_model_name: str, str_input: str) -> tp.Dict[str, tp.Any]:
        """외부 AI 모델 API를 호출해 추론 결과를 반환한다.

        Args:
            str_model_name: 실행할 모델 식별자
            str_input: 모델에 전달할 입력 텍스트
        """
        model_request: InferenceRequest = InferenceRequest(
            str_model_name=str_model_name, str_input=str_input
        )
        model_result: InferenceResult = inference_client.run_inference(model_request)
        return model_result.model_dump()

    return list([run_model_inference])
