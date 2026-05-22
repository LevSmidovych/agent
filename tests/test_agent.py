from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from core.agent import (
    Agent,
    AgentConfig,
    DoneEvent,
    ErrorEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from core.tool_executor import ToolExecutor, ToolRegistry
from tools.base import BaseTool, ToolResult


class ConstTool(BaseTool):
    name = "const"
    description = "returns its argument"
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    def run(self, stop_event, **kwargs):
        return ToolResult(output=kwargs["value"])


@pytest.fixture
def registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(ConstTool())
    return r


@pytest.fixture
def executor(registry: ToolRegistry):
    ex = ToolExecutor(registry, timeout_seconds=5)
    yield ex
    ex.close()


def _stream(chunks: list[dict]):
    return iter(chunks)


def _make_llm_stub(streams: list[list[dict]]):
    stub = MagicMock()
    stub.chat_raw_stream = MagicMock(side_effect=[_stream(s) for s in streams])
    return stub


def test_agent_streams_text_tokens_without_tool_calls(registry, executor) -> None:
    llm = _make_llm_stub([
        [
            {"message": {"content": "hello ", "tool_calls": None}},
            {"message": {"content": "there", "tool_calls": None}},
        ],
    ])
    agent = Agent(llm, registry, executor, AgentConfig(model="m"))
    events = list(agent.run([], "hi"))
    text_events = [e for e in events if isinstance(e, TextEvent)]
    assert [e.text for e in text_events] == ["hello ", "there"]
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].full_text == "hello there"


def test_agent_handles_tool_call_then_final_answer(registry, executor) -> None:
    llm = _make_llm_stub([
        [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "const", "arguments": {"value": "v1"}}}
                    ],
                }
            }
        ],
        [
            {"message": {"content": "got ", "tool_calls": None}},
            {"message": {"content": "it", "tool_calls": None}},
        ],
    ])
    agent = Agent(llm, registry, executor, AgentConfig(model="m"))
    events = list(agent.run([], "use const"))

    tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    text_events = [e for e in events if isinstance(e, TextEvent)]
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "const"
    assert tool_calls[0].arguments == {"value": "v1"}
    assert len(tool_results) == 1
    assert tool_results[0].result.output == "v1"
    assert [e.text for e in text_events] == ["got ", "it"]
    assert isinstance(events[-1], DoneEvent)


def test_agent_parses_json_string_arguments(registry, executor) -> None:
    llm = _make_llm_stub([
        [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "const", "arguments": '{"value": "json-str"}'}}
                    ],
                }
            }
        ],
        [{"message": {"content": "ok", "tool_calls": None}}],
    ])
    agent = Agent(llm, registry, executor, AgentConfig(model="m"))
    events = list(agent.run([], "go"))
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert tool_results[0].result.output == "json-str"


def test_agent_stops_on_max_iterations(registry, executor) -> None:
    def tool_stream(_i):
        return _stream([
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "const", "arguments": {"value": "x"}}}
                    ],
                }
            }
        ])

    llm = MagicMock()
    llm.chat_raw_stream = MagicMock(side_effect=lambda *a, **kw: tool_stream(0))
    agent = Agent(llm, registry, executor, AgentConfig(model="m", max_iterations=3))
    events = list(agent.run([], "go"))
    assert isinstance(events[-1], ErrorEvent)
    assert "max iterations" in events[-1].message


def test_agent_respects_stop_event(registry, executor) -> None:
    llm = _make_llm_stub([
        [{"message": {"content": "done", "tool_calls": None}}],
    ])
    stop = threading.Event()
    stop.set()
    agent = Agent(llm, registry, executor, AgentConfig(model="m"))
    events = list(agent.run([], "go", stop_event=stop))
    assert isinstance(events[-1], ErrorEvent)
    assert "cancelled" in events[-1].message


def test_agent_mixed_text_and_tool_call_in_same_turn(registry, executor) -> None:
    """Some models emit a short preamble before calling a tool. The preamble
    should still be streamed as text, then the tool should be executed."""
    llm = _make_llm_stub([
        [
            {"message": {"content": "Checking... ", "tool_calls": None}},
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "const", "arguments": {"value": "v"}}}
                    ],
                }
            },
        ],
        [{"message": {"content": "done", "tool_calls": None}}],
    ])
    agent = Agent(llm, registry, executor, AgentConfig(model="m"))
    events = list(agent.run([], "go"))

    text_events = [e for e in events if isinstance(e, TextEvent)]
    tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
    assert [e.text for e in text_events] == ["Checking... ", "done"]
    assert len(tool_calls) == 1
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].full_text == "Checking... done"


def test_agent_system_prompt_prepended(registry, executor) -> None:
    llm = _make_llm_stub([[{"message": {"content": "hi", "tool_calls": None}}]])
    agent = Agent(llm, registry, executor, AgentConfig(model="m", system_prompt="You are helpful."))
    list(agent.run([], "hello"))
    call = llm.chat_raw_stream.call_args
    messages = call.kwargs.get("messages") or call.args[1]
    assert messages[0]["role"] == "system"
    assert "helpful" in messages[0]["content"]


def test_agent_filters_tools_by_allowed_names(registry, executor) -> None:
    class Other(ConstTool):
        name = "other"

    registry.register(Other())
    llm = _make_llm_stub([[{"message": {"content": "ok", "tool_calls": None}}]])
    agent = Agent(
        llm, registry, executor,
        AgentConfig(model="m", allowed_tool_names=["const"]),
    )
    list(agent.run([], "go"))
    call = llm.chat_raw_stream.call_args
    tools = call.kwargs.get("tools")
    names = {t["function"]["name"] for t in (tools or [])}
    assert names == {"const"}


def test_agent_use_tools_false_sends_no_tools(registry, executor) -> None:
    llm = _make_llm_stub([[{"message": {"content": "ok", "tool_calls": None}}]])
    agent = Agent(
        llm, registry, executor,
        AgentConfig(model="m", use_tools=False),
    )
    list(agent.run([], "go"))
    call = llm.chat_raw_stream.call_args
    assert call.kwargs.get("tools") is None


def test_agent_recovers_inline_tool_call(registry, executor) -> None:
    """Some models emit tool calls as ``<tool_call>...</tool_call>`` text
    instead of via the native tool_calls field. The agent must detect that
    and still execute the tool."""
    inline_blob = (
        'Let me do that.\n'
        '<tool_call>\n'
        '{"name": "const", "arguments": {"value": "from-inline"}}\n'
        '</tool_call>'
    )
    llm = _make_llm_stub([
        # First iteration: model emits inline tool call as text, no native tool_calls
        [{"message": {"content": inline_blob, "tool_calls": None}}],
        # Second iteration: final answer after tool result
        [{"message": {"content": "done", "tool_calls": None}}],
    ])
    agent = Agent(llm, registry, executor, AgentConfig(model="m"))
    events = list(agent.run([], "go"))

    tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "const"
    assert tool_calls[0].arguments == {"value": "from-inline"}
    assert tool_results[0].result.output == "from-inline"
    assert isinstance(events[-1], DoneEvent)


def test_agent_recovers_bare_inline_tool_call(registry, executor) -> None:
    """Model emits only the JSON object + a stray closer (the actual bug
    seen in production with qwen2.5 via Ollama)."""
    bare = (
        'iNdEx\n'
        '{"name": "const", "arguments": {"value": "stray"}}\n'
        '</tool_call>'
    )
    llm = _make_llm_stub([
        [{"message": {"content": bare, "tool_calls": None}}],
        [{"message": {"content": "ok", "tool_calls": None}}],
    ])
    agent = Agent(llm, registry, executor, AgentConfig(model="m"))
    events = list(agent.run([], "go"))
    tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
    assert tool_calls and tool_calls[0].arguments == {"value": "stray"}
