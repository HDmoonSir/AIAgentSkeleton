"""
대화 이력 스토어
- conversation_id 단위로 대화 이력을 보관하는 경계 인터페이스
- 로컬 대역에서는 메모리 구현을 사용하고, 운영 전환 시 DB 백엔드로 교체
"""

import abc
import typing as tp

from aiagent.schemas.agent_io import ConversationMessage


class ConversationStore(abc.ABC):
    """대화 이력 보관/조회 경계. 구현체는 메모리/DB 등으로 교체 가능."""

    @abc.abstractmethod
    def get_history(self, str_conversation_id: str) -> tp.List[ConversationMessage]:
        """해당 대화의 이력을 반환. 없으면 빈 목록."""
        ...

    @abc.abstractmethod
    def save_history(
        self, str_conversation_id: str, list_messages: tp.List[ConversationMessage]
    ) -> None:
        """해당 대화의 이력을 통째로 갱신 저장."""
        ...


class InMemoryConversationStore(ConversationStore):
    """프로세스 메모리에 대화 이력을 보관하는 구현."""

    def __init__(self) -> None:
        self.dict_history_by_conversation: tp.Dict[str, tp.List[ConversationMessage]] = dict()

    def get_history(self, str_conversation_id: str) -> tp.List[ConversationMessage]:
        return list(self.dict_history_by_conversation.get(str_conversation_id, list()))

    def save_history(
        self, str_conversation_id: str, list_messages: tp.List[ConversationMessage]
    ) -> None:
        self.dict_history_by_conversation[str_conversation_id] = list(list_messages)
