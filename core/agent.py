from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from core.exceptions import OllamaConnectionError
from core.ollama_parsing import (
    extract_inline_tool_calls,
    get_content,
    get_message,
    get_raw_tool_calls,
    normalize_tool_calls,
    strip_inline_tool_blocks,
)
from core.tool_executor import ToolExecutor, ToolRegistry
from tools.base import ToolResult

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 8


# ---- events emitted by Agent.run ----------------------------------------


@dataclass
class TextEvent:
    text: str


@dataclass
class ToolCallEvent:
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResultEvent:
    name: str
    result: ToolResult


@dataclass
class DoneEvent:
    full_text: str


@dataclass
class ErrorEvent:
    message: str


AgentEvent = TextEvent | ToolCallEvent | ToolResultEvent | DoneEvent | ErrorEvent


@dataclass
class AgentConfig:
    model: str
    system_prompt: str = ""
    use_tools: bool = True
    allowed_tool_names: Optional[list[str]] = None
    max_iterations: int = MAX_TOOL_ITERATIONS


class Agent:
    """Minimal ReAct-style agent with native Ollama tool calling.

    Native function calling is used when the model supports it (qwen2.5, llama3.1,
    etc.). Other models receive the same request without tools and behave as
    plain chat models.
    """

    def __init__(
        self,
        llm_client,
        registry: ToolRegistry,
        executor: ToolExecutor,
        config: AgentConfig,
    ) -> None:
        self._llm = llm_client
        self._registry = registry
        self._executor = executor
        self._config = config

    def run(
        self,
        history: list[dict[str, Any]],
        user_message: str,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[AgentEvent]:
        stop_event = stop_event or threading.Event()
        messages: list[dict[str, Any]] = []
        if self._config.system_prompt:
            messages.append({"role": "system", "content": self._config.system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        if self._config.use_tools:
            all_schemas = self._registry.schemas()
            allowed = self._config.allowed_tool_names
            if allowed is None:
                tools_schema = all_schemas
            else:
                allowed_set = set(allowed)
                tools_schema = [s for s in all_schemas if s["function"]["name"] in allowed_set]
        else:
            tools_schema = []
        collected: list[str] = []

        for iteration in range(self._config.max_iterations):
            if stop_event.is_set():
                yield ErrorEvent("cancelled")
                return

            iter_text: list[str] = []
            pending_tool_calls: list[dict[str, Any]] = []

            try:
                for chunk in self._chat_stream(messages, tools_schema):
                    if stop_event.is_set():
                        yield ErrorEvent("cancelled")
                        return
                    message = get_message(chunk)
                    content = get_content(message)
                    if content:
                        iter_text.append(content)
                        yield TextEvent(text=content)
                    tool_calls = get_raw_tool_calls(message)
                    if tool_calls:
                        pending_tool_calls.extend(normalize_tool_calls(tool_calls))
            except OllamaConnectionError as exc:
                yield ErrorEvent(str(exc))
                return

            iter_text_str = "".join(iter_text)

            # Some models (notably qwen2.5 via certain Ollama builds) emit
            # tool calls as inline ``<tool_call>...</tool_call>`` text instead
            # of the native ``message.tool_calls`` field. Detect that here
            # and recover so the user does not see raw JSON in chat.
            if not pending_tool_calls:
                inline = extract_inline_tool_calls(iter_text_str)
                if inline:
                    pending_tool_calls = inline
                    iter_text_str = strip_inline_tool_blocks(iter_text_str)
                    logger.info(
                        "recovered %d inline tool call(s) from text stream", len(inline),
                    )

            collected.append(iter_text_str)

            if pending_tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": iter_text_str,
                    "tool_calls": [
                        {
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            }
                        }
                        for tc in pending_tool_calls
                    ],
                })
                for call in pending_tool_calls:
                    if stop_event.is_set():
                        yield ErrorEvent("cancelled")
                        return
                    yield ToolCallEvent(name=call["name"], arguments=call["arguments"])
                    result = self._executor.invoke(call["name"], call["arguments"])
                    yield ToolResultEvent(name=call["name"], result=result)
                    messages.append({
                        "role": "tool",
                        "name": call["name"],
                        "content": result.error if result.error else result.output,
                    })
                continue

            yield DoneEvent(full_text="".join(collected))
            return

        yield ErrorEvent(f"max iterations ({self._config.max_iterations}) exceeded")

    def _chat_stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]):
        return self._llm.chat_raw_stream(
            model=self._config.model,
            messages=messages,
            tools=tools or None,
        )


