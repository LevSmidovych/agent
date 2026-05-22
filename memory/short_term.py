from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from core.llm_client import LLMClient
from memory.storage import Message, Storage

logger = logging.getLogger(__name__)


@dataclass
class BuiltContext:
    messages: list[dict]
    total_tokens: int
    truncated: int  # how many oldest messages were dropped


class ShortTermMemory:
    """Builds the chat history to send to the LLM for each request.

    Applies a message-count limit and a sliding window based on approximate
    token count when the budget exceeds half of the model's context window.
    """

    def __init__(
        self,
        storage: Storage,
        llm: LLMClient,
        message_limit: int = 20,
    ) -> None:
        self._storage = storage
        self._llm = llm
        self._message_limit = max(2, message_limit)

    @property
    def message_limit(self) -> int:
        return self._message_limit

    @message_limit.setter
    def message_limit(self, value: int) -> None:
        self._message_limit = max(2, int(value))

    def build(
        self,
        conversation_id: int,
        model: str,
        system_prompt: Optional[str] = None,
        drop_last: bool = False,
    ) -> BuiltContext:
        all_messages = self._storage.get_messages(conversation_id)
        chat_messages = [m for m in all_messages if m.role in ("user", "assistant")]
        if drop_last and chat_messages and chat_messages[-1].role == "user":
            chat_messages = chat_messages[:-1]

        truncated = max(0, len(chat_messages) - self._message_limit)
        window = chat_messages[-self._message_limit :]

        model_info = self._llm.model_info(model)
        budget = max(1024, model_info.context_length // 2)

        result_msgs: list[dict] = []
        if system_prompt:
            result_msgs.append({"role": "system", "content": system_prompt})
        for m in window:
            result_msgs.append({"role": m.role, "content": m.content})

        total = sum(self._llm.tokenize(model, m["content"]) for m in result_msgs)

        # Drop oldest user/assistant messages until we're under budget.
        while total > budget and len(result_msgs) > (1 if system_prompt else 0) + 2:
            first_idx = 1 if system_prompt else 0
            removed = result_msgs.pop(first_idx)
            total -= self._llm.tokenize(model, removed["content"])
            truncated += 1

        return BuiltContext(messages=result_msgs, total_tokens=total, truncated=truncated)
