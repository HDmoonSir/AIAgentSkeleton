import typing as tp

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from aiagent.constants import DICT_INTENT_DESCRIPTION, IntentType, MessageRole
from aiagent.schemas.agent_io import ConversationMessage


def build_langchain_messages(
    list_messages: tp.List[ConversationMessage],
) -> tp.List[BaseMessage]:
    """내부 대화 이력 DTO를 LangChain 메시지 객체로 변환."""
    list_result: tp.List[BaseMessage] = list()
    for model_message in list_messages:
        if model_message.model_role == MessageRole.USER:
            list_result.append(HumanMessage(content=model_message.str_content))
        elif model_message.model_role == MessageRole.ASSISTANT:
            list_result.append(AIMessage(content=model_message.str_content))
        else:
            list_result.append(SystemMessage(content=model_message.str_content))
    return list_result


def build_intent_classification_prompt(str_user_input: str) -> str:
    """사용자 의도 분류용 프롬프트 생성."""
    list_intent_lines: tp.List[str] = list()
    for model_intent, str_description in DICT_INTENT_DESCRIPTION.items():
        list_intent_lines.append(f"- {model_intent.value}: {str_description}")
    str_intent_catalog: str = "\n".join(list_intent_lines)

    return (
        "당신은 사용자 요청을 분류하는 라우팅 어시스턴트입니다. "
        "사용자의 질문을 분석하여 의도를 분류하세요.\n"
        "이전 대화 맥락을 참고하여 지시어가 무엇을 지칭하는지 파악하세요.\n\n"
        f"[의도 종류]\n{str_intent_catalog}\n\n"
        "[의도 선택 가이드]\n"
        "- 외부 AI 모델을 실행해 결과를 얻어야 하는 요청 → MODEL_INFERENCE\n"
        "- 모델 실행 없이 일반 대화/안내로 답할 수 있는 요청 → GENERAL_QA\n\n"
        f"[사용자 입력]\n{str_user_input}\n\n"
        "[결과 형식]\n"
        "JSON 형식으로 'intent'와 'params'를 반환하세요. "
        "'intent' 값은 위 의도 종류 식별자 중 하나여야 합니다."
    )


def build_planning_system_message(model_intent: IntentType) -> str:
    """도구 호출 계획 수립용 시스템 메시지 생성."""
    str_base: str = (
        f"당신은 {model_intent.value} 요청을 처리하는 어시스턴트입니다. "
        "이전 대화 맥락과 제공된 도구를 사용하여 계획을 세우세요."
    )
    if model_intent == IntentType.MODEL_INFERENCE:
        str_base += (
            "\n- 외부 AI 모델을 실행해야 하면 run_model_inference 도구를 호출하세요.\n"
            "- str_model_name에는 실행할 모델 식별자를, str_input에는 모델에 전달할 입력 텍스트를 채우세요.\n"
            "- 모델 식별자가 명시되지 않았으면 대화 맥락에서 추론하거나 기본 모델을 사용하세요."
        )
    return str_base


def build_response_synthesis_prompt(model_intent: IntentType, str_evidence: str) -> str:
    """도구 실행 결과를 자연어 답변으로 변환하기 위한 프롬프트 생성."""
    return (
        "당신은 도구 실행 결과를 사용자에게 전달하는 어시스턴트입니다.\n"
        "제공된 근거(Evidence)를 바탕으로 사용자 질문에 명확하게 답변하세요.\n\n"
        "[출력 가이드라인]\n"
        "- 근거(Evidence)에 실제로 포함된 데이터만 사용하고, 없는 항목은 임의로 만들어내지 마세요.\n"
        "- 모델 추론 결과가 있으면 핵심 내용을 정리해 전달하세요.\n"
        "- 친절하고 전문적인 톤을 유지하세요.\n\n"
        f"분석 의도: {model_intent.value}\n"
        f"분석 근거(Evidence): {str_evidence}"
    )
