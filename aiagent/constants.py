import enum
import typing as tp


class IntentType(str, enum.Enum):
    """사용자 질의 의도 분류값."""

    MODEL_INFERENCE = "MODEL_INFERENCE"
    GENERAL_QA = "GENERAL_QA"


class MessageRole(str, enum.Enum):
    """대화 메시지의 발화 주체."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


DEFAULT_INTENT: IntentType = IntentType.GENERAL_QA

DEFAULT_MODEL_NAME: str = "gpt-4o"
DEFAULT_TEMPERATURE: float = 0.0

DICT_INTENT_DESCRIPTION: tp.Dict[IntentType, str] = dict(
    {
        IntentType.MODEL_INFERENCE: "외부 AI 모델 API를 호출해 추론 결과를 얻어야 하는 요청",
        IntentType.GENERAL_QA: "도구 호출 없이 일반 대화/안내로 답할 수 있는 요청",
    }
)
