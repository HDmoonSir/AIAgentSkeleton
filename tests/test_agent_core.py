import typing as tp

from app.agent.conversation_store import InMemoryConversationStore
from aiagent.agents.prompt_builder import build_intent_classification_prompt
from aiagent.constants import IntentType, MessageRole
from aiagent.ports.inference_model import (
    InferenceModelClient,
    InferenceRequest,
    InferenceResult,
)
from aiagent.schemas.agent_io import ConversationMessage
from aiagent.tools.tool_registry import TOOL_NAME_RUN_MODEL_INFERENCE, build_tools


class FakeInferenceModelClient(InferenceModelClient):
    """테스트용 모델 호출 경계 구현. 입력을 그대로 되돌려 호출 인자를 검증한다."""

    def __init__(self) -> None:
        self.list_received_requests: tp.List[InferenceRequest] = list()

    def run_inference(self, model_request: InferenceRequest) -> InferenceResult:
        self.list_received_requests.append(model_request)
        return InferenceResult(
            str_model_name=model_request.str_model_name,
            str_output=f"echo:{model_request.str_input}",
        )


def test_intent_prompt_lists_all_intents() -> None:
    str_prompt: str = build_intent_classification_prompt("모델을 실행해줘")
    for model_intent in IntentType:
        assert model_intent.value in str_prompt


def test_conversation_store_roundtrip() -> None:
    model_store: InMemoryConversationStore = InMemoryConversationStore()
    list_messages: tp.List[ConversationMessage] = list(
        [ConversationMessage(model_role=MessageRole.USER, str_content="안녕")]
    )

    model_store.save_history("conv-1", list_messages)

    assert model_store.get_history("conv-1") == list_messages
    assert model_store.get_history("unknown") == list()


def test_build_tools_invokes_injected_client() -> None:
    model_client: FakeInferenceModelClient = FakeInferenceModelClient()
    list_tools = build_tools(model_client)

    assert len(list_tools) == 1
    model_tool = list_tools[0]
    assert model_tool.name == TOOL_NAME_RUN_MODEL_INFERENCE

    dict_output: tp.Dict[str, tp.Any] = model_tool.invoke(
        dict(str_model_name="demo-model", str_input="hello")
    )

    assert dict_output["str_output"] == "echo:hello"
    assert model_client.list_received_requests[0].str_model_name == "demo-model"
