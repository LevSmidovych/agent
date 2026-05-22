from __future__ import annotations

import threading
import time

import pytest

from core.tool_executor import ToolExecutor, ToolRegistry
from tools.base import BaseTool, ToolResult


class EchoTool(BaseTool):
    name = "echo"
    description = "echo"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, stop_event, **kwargs):
        return ToolResult(output=kwargs["text"])


class SlowTool(BaseTool):
    name = "slow"
    description = "slow"
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, delay: float = 2.0) -> None:
        self._delay = delay

    def run(self, stop_event: threading.Event, **kwargs):
        deadline = time.time() + self._delay
        while time.time() < deadline:
            if stop_event.is_set():
                return ToolResult(output="", error="cancelled")
            time.sleep(0.05)
        return ToolResult(output="done")


class CrashTool(BaseTool):
    name = "crash"
    description = "crash"
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self, stop_event, **kwargs):
        raise RuntimeError("boom")


@pytest.fixture
def registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(EchoTool())
    r.register(CrashTool())
    return r


def test_registry_schemas(registry: ToolRegistry) -> None:
    schemas = registry.schemas()
    names = {s["function"]["name"] for s in schemas}
    assert "echo" in names
    assert "crash" in names


def test_invoke_unknown_tool(registry: ToolRegistry) -> None:
    ex = ToolExecutor(registry, timeout_seconds=5)
    try:
        res = ex.invoke("doesnotexist", {})
        assert not res.ok
        assert "not found" in res.error
    finally:
        ex.close()


def test_invoke_with_dict_args(registry: ToolRegistry) -> None:
    ex = ToolExecutor(registry, timeout_seconds=5)
    try:
        res = ex.invoke("echo", {"text": "hi"})
        assert res.ok
        assert res.output == "hi"
    finally:
        ex.close()


def test_invoke_with_json_string_args(registry: ToolRegistry) -> None:
    ex = ToolExecutor(registry, timeout_seconds=5)
    try:
        res = ex.invoke("echo", '{"text": "yo"}')
        assert res.ok
        assert res.output == "yo"
    finally:
        ex.close()


def test_invoke_invalid_args(registry: ToolRegistry) -> None:
    ex = ToolExecutor(registry, timeout_seconds=5)
    try:
        res = ex.invoke("echo", "not json")
        assert not res.ok
        assert "invalid arguments" in res.error
    finally:
        ex.close()


def test_invoke_timeout_triggers_stop_event() -> None:
    registry = ToolRegistry()
    registry.register(SlowTool(delay=5.0))
    ex = ToolExecutor(registry, timeout_seconds=1)
    try:
        res = ex.invoke("slow", {})
        assert not res.ok
        assert "timed out" in res.error.lower()
    finally:
        ex.close()


def test_invoke_catches_exception(registry: ToolRegistry) -> None:
    ex = ToolExecutor(registry, timeout_seconds=5)
    try:
        res = ex.invoke("crash", {})
        assert not res.ok
        assert "boom" in res.error
    finally:
        ex.close()
