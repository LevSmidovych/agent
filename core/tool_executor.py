from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any

from tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if not tool.name:
            raise ValueError("tool has no name")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def all(self) -> list[BaseTool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def names_in_categories(self, categories: list[str]) -> list[str]:
        if not categories:
            return []
        allowed = set(categories)
        return [t.name for t in self._tools.values() if t.category in allowed]


class ToolExecutor:
    """Runs a tool in a worker thread with a timeout and cooperative cancel."""

    def __init__(self, registry: ToolRegistry, timeout_seconds: int = 30) -> None:
        self._registry = registry
        self._timeout = timeout_seconds
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")

    @property
    def timeout_seconds(self) -> int:
        return self._timeout

    @timeout_seconds.setter
    def timeout_seconds(self, value: int) -> None:
        self._timeout = max(1, int(value))

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def invoke(self, name: str, arguments: dict[str, Any] | str) -> ToolResult:
        tool = self._registry.get(name)
        if tool is None:
            return ToolResult(output="", error=f"tool '{name}' not found")

        args = _coerce_args(arguments)
        if args is None:
            return ToolResult(output="", error=f"tool '{name}' got invalid arguments: {arguments!r}")

        stop_event = threading.Event()
        future = self._pool.submit(_safe_run, tool, stop_event, args)
        try:
            return future.result(timeout=self._timeout)
        except FuturesTimeout:
            stop_event.set()
            logger.warning("tool '%s' timed out after %ss", name, self._timeout)
            return ToolResult(output="", error=f"tool '{name}' timed out after {self._timeout}s")
        except Exception as exc:
            logger.exception("tool '%s' crashed", name)
            return ToolResult(output="", error=f"tool '{name}' crashed: {exc}")


def _coerce_args(arguments: dict[str, Any] | str) -> dict[str, Any] | None:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _safe_run(tool: BaseTool, stop_event: threading.Event, args: dict[str, Any]) -> ToolResult:
    try:
        return tool.run(stop_event, **args)
    except TypeError as exc:
        return ToolResult(output="", error=f"invalid arguments: {exc}")
    except Exception as exc:
        logger.exception("tool '%s' raised", tool.name)
        return ToolResult(output="", error=str(exc))
