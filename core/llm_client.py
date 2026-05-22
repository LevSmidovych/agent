from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Iterator, Optional

import ollama

from core.exceptions import OllamaConnectionError

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    name: str
    context_length: int = 4096


class LLMClient:
    def __init__(self, host: str = "http://localhost:11434") -> None:
        self._host = host
        self._client = ollama.Client(host=host)
        self._model_info_cache: dict[str, ModelInfo] = {}

    def list_models(self) -> list[str]:
        try:
            resp = self._client.list()
        except Exception as exc:
            logger.exception("ollama list failed")
            raise OllamaConnectionError(f"cannot reach Ollama at {self._host}") from exc
        models = resp.get("models", []) if isinstance(resp, dict) else getattr(resp, "models", [])
        names: list[str] = []
        for m in models:
            if isinstance(m, dict):
                name = m.get("model") or m.get("name")
            else:
                name = getattr(m, "model", None) or getattr(m, "name", None)
            if name:
                names.append(name)
        return names

    def model_info(self, model: str) -> ModelInfo:
        cached = self._model_info_cache.get(model)
        if cached is not None:
            return cached
        try:
            resp = self._client.show(model)
        except Exception:
            logger.exception("ollama show failed for %s", model)
            info = ModelInfo(name=model)
            self._model_info_cache[model] = info
            return info
        ctx = _extract_context_length(resp)
        info = ModelInfo(name=model, context_length=ctx or 4096)
        self._model_info_cache[model] = info
        return info

    def invalidate_model_info(self, model: str | None = None) -> None:
        if model is None:
            self._model_info_cache.clear()
        else:
            self._model_info_cache.pop(model, None)

    def tokenize(self, model: str, text: str) -> int:
        """Approximate token count via a cheap chars/4 heuristic (no network call)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def chat_stream(
        self,
        model: str,
        messages: list[dict],
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        from core.ollama_parsing import get_content

        stop_event = stop_event or threading.Event()
        for chunk in self.chat_raw_stream(model, messages, stop_event=stop_event):
            content = get_content(chunk)
            if content:
                yield content

    def chat_raw_stream(
        self,
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        stop_event: Optional[threading.Event] = None,
        keep_alive: Optional[int | str] = None,
    ):
        """Streaming chat call returning raw chunks. Each chunk may carry
        ``message.content`` (text delta) and/or ``message.tool_calls``.

        ``keep_alive`` controls how long Ollama keeps the model loaded after
        this request. Pass ``0`` to force unload, ``"-1"`` to keep forever,
        ``None`` for the server default (5 minutes).
        """
        stop_event = stop_event or threading.Event()
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if tools:
            kwargs["tools"] = tools
        if keep_alive is not None:
            kwargs["keep_alive"] = keep_alive
        try:
            stream = self._client.chat(**kwargs)
            for chunk in stream:
                if stop_event.is_set():
                    break
                yield chunk
        except Exception as exc:
            logger.warning("ollama chat (stream) failed: %s", exc)
            raise OllamaConnectionError(str(exc)) from exc

    def unload_model(self, model: str) -> None:
        """Force Ollama to evict ``model`` from VRAM by issuing a zero-keep-alive
        ``generate`` call. Safe to call when the model isn't loaded.
        """
        try:
            self._client.generate(model=model, prompt="", keep_alive=0)
        except Exception as exc:
            logger.info("unload of %s failed (ignored): %s", model, exc)

    def chat_json(
        self,
        model: str,
        messages: list[dict],
        timeout_seconds: Optional[float] = None,
    ) -> str:
        """Non-streaming chat with ``format='json'``. Used by the memory
        classifier and the LLM-as-judge.

        ``timeout_seconds`` bounds the wait — if Ollama does not respond
        within that window, raise :class:`OllamaConnectionError`. The call is
        executed on a worker thread so the wait is truly preemptive, not
        cooperative; useful for batch judging where a single hung response
        would otherwise stall the whole batch.
        """
        from core.ollama_parsing import get_content

        if timeout_seconds is None:
            try:
                resp = self._client.chat(
                    model=model, messages=messages, format="json", stream=False
                )
            except Exception as exc:
                logger.exception("ollama chat (json) failed")
                raise OllamaConnectionError(str(exc)) from exc
            return get_content(resp)

        # Run in a worker thread so we can time out a slow Ollama call.
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout

        def _call():
            return self._client.chat(
                model=model, messages=messages, format="json", stream=False
            )

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call)
            try:
                resp = future.result(timeout=timeout_seconds)
            except _FutTimeout as exc:
                logger.warning("ollama chat (json) timed out after %ss", timeout_seconds)
                raise OllamaConnectionError(
                    f"judge model did not respond within {timeout_seconds}s"
                ) from exc
            except Exception as exc:
                logger.exception("ollama chat (json) failed")
                raise OllamaConnectionError(str(exc)) from exc
        return get_content(resp)


def _get(obj, key: str):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _extract_context_length(resp) -> Optional[int]:
    # Ollama show returns model_info with "<arch>.context_length" keys
    model_info = _get(resp, "model_info") or _get(resp, "modelinfo")
    if isinstance(model_info, dict):
        for key, value in model_info.items():
            if key.endswith(".context_length") and isinstance(value, int):
                return value
    params = _get(resp, "parameters")
    if isinstance(params, str):
        for line in params.splitlines():
            parts = line.strip().split()
            if len(parts) == 2 and parts[0] == "num_ctx":
                try:
                    return int(parts[1])
                except ValueError:
                    pass
    return None
