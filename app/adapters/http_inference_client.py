"""
InferenceModelClient 포트의 HTTP 어댑터.
- 외부 AI 모델 API에 HTTP로 추론 요청을 보내는 구체 구현이다.
- httpx 클라이언트는 진입점(조립 루트)에서 생성해 주입한다.
- 실제 모델 API의 요청/응답 스키마에 맞춰 _build_payload·_parse_response를 교체한다.
"""

import typing as tp

import httpx

from aiagent.ports.inference_model import (
    InferenceModelClient,
    InferenceModelError,
    InferenceRequest,
    InferenceResult,
)

INFERENCE_ENDPOINT_PATH: str = "/inference"
RESPONSE_KEY_OUTPUT: str = "output"


class HttpInferenceModelClient(InferenceModelClient):
    """외부 AI 모델 API를 HTTP로 호출하는 어댑터."""

    def __init__(self, str_base_url: str, str_api_key: str, http_client: httpx.Client) -> None:
        self.str_base_url: str = str_base_url.rstrip("/")
        self.str_api_key: str = str_api_key
        self.http_client: httpx.Client = http_client

    def run_inference(self, model_request: InferenceRequest) -> InferenceResult:
        """모델 API에 추론 요청을 보내고 결과를 표준 계약으로 변환한다."""
        str_url: str = f"{self.str_base_url}{INFERENCE_ENDPOINT_PATH}"
        dict_headers: tp.Dict[str, str] = dict(
            {"Authorization": f"Bearer {self.str_api_key}"}
        )
        try:
            http_response: httpx.Response = self.http_client.post(
                str_url,
                json=self._build_payload(model_request),
                headers=dict_headers,
            )
            http_response.raise_for_status()
        except httpx.HTTPError as model_error:
            raise InferenceModelError(
                f"모델 API 호출 실패: {model_request.str_model_name}"
            ) from model_error

        return self._parse_response(model_request.str_model_name, http_response.json())

    @staticmethod
    def _build_payload(model_request: InferenceRequest) -> tp.Dict[str, tp.Any]:
        """포트 요청 계약을 모델 API 요청 본문으로 변환한다."""
        return dict(
            {
                "model": model_request.str_model_name,
                "input": model_request.str_input,
                "parameters": model_request.dict_parameters,
            }
        )

    @staticmethod
    def _parse_response(
        str_model_name: str, dict_body: tp.Dict[str, tp.Any]
    ) -> InferenceResult:
        """모델 API 응답 본문을 포트 결과 계약으로 변환한다."""
        str_output: tp.Optional[tp.Any] = dict_body.get(RESPONSE_KEY_OUTPUT)
        if str_output is None:
            raise InferenceModelError(
                f"모델 API 응답에 '{RESPONSE_KEY_OUTPUT}' 필드가 없습니다: {str_model_name}"
            )
        return InferenceResult(str_model_name=str_model_name, str_output=str(str_output))
