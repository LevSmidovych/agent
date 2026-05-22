from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from core.exceptions import OllamaConnectionError
from core.llm_client import LLMClient, ModelInfo


@pytest.fixture
def client() -> LLMClient:
    c = LLMClient(host="http://mock")
    c._client = MagicMock()  # type: ignore[assignment]
    return c


# ---- list_models ---------------------------------------------------------


def test_list_models_parses_dict_response(client: LLMClient) -> None:
    client._client.list.return_value = {
        "models": [{"model": "qwen2.5:14b"}, {"name": "llama3.1:8b"}]
    }
    assert client.list_models() == ["qwen2.5:14b", "llama3.1:8b"]


def test_list_models_parses_object_response(client: LLMClient) -> None:
    class M:
        model = "m1"

    class M2:
        name = "m2"

    class Resp:
        models = [M(), M2()]

    client._client.list.return_value = Resp()
    assert client.list_models() == ["m1", "m2"]


def test_list_models_raises_on_failure(client: LLMClient) -> None:
    client._client.list.side_effect = RuntimeError("network")
    with pytest.raises(OllamaConnectionError):
        client.list_models()


def test_list_models_ignores_entries_without_names(client: LLMClient) -> None:
    client._client.list.return_value = {"models": [{"foo": "bar"}, {"model": "ok"}]}
    assert client.list_models() == ["ok"]


# ---- model_info ----------------------------------------------------------


def test_model_info_caches_after_first_call(client: LLMClient) -> None:
    client._client.show.return_value = {
        "model_info": {"qwen2.arch.context_length": 32768}
    }
    info = client.model_info("qwen2.5:14b")
    info2 = client.model_info("qwen2.5:14b")
    assert info.context_length == 32768
    assert info is info2
    client._client.show.assert_called_once()


def test_model_info_falls_back_on_show_failure(client: LLMClient) -> None:
    client._client.show.side_effect = RuntimeError("network")
    info = client.model_info("m")
    assert info.context_length == 4096


def test_model_info_parses_num_ctx_from_parameters(client: LLMClient) -> None:
    client._client.show.return_value = {"parameters": "num_ctx 8192\nrepeat_penalty 1.1"}
    assert client.model_info("m").context_length == 8192


def test_invalidate_model_info(client: LLMClient) -> None:
    client._client.show.return_value = {"model_info": {"x.context_length": 1024}}
    client.model_info("m")
    client.invalidate_model_info("m")
    client._client.show.return_value = {"model_info": {"x.context_length": 2048}}
    assert client.model_info("m").context_length == 2048


def test_invalidate_all(client: LLMClient) -> None:
    client._client.show.return_value = {"model_info": {"x.context_length": 512}}
    client.model_info("a")
    client.model_info("b")
    client.invalidate_model_info()
    client._client.show.return_value = {"model_info": {"x.context_length": 4096}}
    assert client.model_info("a").context_length == 4096


# ---- tokenize ------------------------------------------------------------


def test_tokenize_heuristic() -> None:
    c = LLMClient()
    assert c.tokenize("m", "") == 0
    assert c.tokenize("m", "abcd") == 1
    assert c.tokenize("m", "a" * 400) == 100


# ---- chat_stream ---------------------------------------------------------


def test_chat_stream_yields_content(client: LLMClient) -> None:
    client._client.chat.return_value = iter([
        {"message": {"content": "hello "}},
        {"message": {"content": "world"}},
        {"message": {"content": ""}},
    ])
    tokens = list(client.chat_stream("m", [{"role": "user", "content": "hi"}]))
    assert tokens == ["hello ", "world"]


def test_chat_stream_respects_stop_event(client: LLMClient) -> None:
    stop = threading.Event()
    stop.set()
    client._client.chat.return_value = iter([{"message": {"content": "a"}}])
    tokens = list(client.chat_stream("m", [], stop_event=stop))
    assert tokens == []


def test_chat_stream_raises_on_ollama_failure(client: LLMClient) -> None:
    client._client.chat.side_effect = RuntimeError("down")
    with pytest.raises(OllamaConnectionError):
        list(client.chat_stream("m", []))


# ---- chat_raw_stream -----------------------------------------------------


def test_chat_raw_stream_passes_tools(client: LLMClient) -> None:
    client._client.chat.return_value = iter([{"message": {"content": "ok"}}])
    chunks = list(client.chat_raw_stream("m", [{"role": "user", "content": "hi"}], tools=[{"a": 1}]))
    call_kwargs = client._client.chat.call_args.kwargs
    assert call_kwargs["tools"] == [{"a": 1}]
    assert call_kwargs["stream"] is True
    assert len(chunks) == 1


def test_chat_raw_stream_without_tools_does_not_send_tools(client: LLMClient) -> None:
    client._client.chat.return_value = iter([])
    list(client.chat_raw_stream("m", []))
    call_kwargs = client._client.chat.call_args.kwargs
    assert "tools" not in call_kwargs


# ---- chat_json -----------------------------------------------------------


def test_chat_json_returns_content(client: LLMClient) -> None:
    client._client.chat.return_value = {"message": {"content": '{"save": true}'}}
    out = client.chat_json("m", [{"role": "user", "content": "hi"}])
    assert out == '{"save": true}'
    assert client._client.chat.call_args.kwargs["format"] == "json"
    assert client._client.chat.call_args.kwargs["stream"] is False


def test_chat_json_raises_on_failure(client: LLMClient) -> None:
    client._client.chat.side_effect = RuntimeError("x")
    with pytest.raises(OllamaConnectionError):
        client.chat_json("m", [])


def test_chat_json_handles_object_response(client: LLMClient) -> None:
    class Msg:
        content = '{"ok": 1}'

    class Resp:
        message = Msg()

    client._client.chat.return_value = Resp()
    assert client.chat_json("m", []) == '{"ok": 1}'


def test_chat_json_with_timeout_returns_normally(client: LLMClient) -> None:
    client._client.chat.return_value = {"message": {"content": "ok"}}
    assert client.chat_json("m", [], timeout_seconds=5.0) == "ok"


def test_chat_json_raises_on_timeout(client: LLMClient) -> None:
    import time

    def slow(*args, **kwargs):
        time.sleep(2.0)
        return {"message": {"content": "late"}}

    client._client.chat.side_effect = slow
    with pytest.raises(OllamaConnectionError, match="did not respond"):
        client.chat_json("m", [], timeout_seconds=0.1)
