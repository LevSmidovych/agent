from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.ollama_parsing import (
    coerce_args,
    get_content,
    get_eval_count,
    get_message,
    get_raw_tool_calls,
    normalize_tool_calls,
)


# ---- get_message ----------------------------------------------------------


def test_get_message_from_dict_chunk() -> None:
    chunk = {"message": {"content": "hi"}}
    assert get_message(chunk) == {"content": "hi"}


def test_get_message_from_object_chunk() -> None:
    chunk = SimpleNamespace(message={"content": "hi"})
    assert get_message(chunk) == {"content": "hi"}


def test_get_message_missing_returns_empty_dict() -> None:
    assert get_message({}) == {}
    assert get_message(SimpleNamespace()) == {}


def test_get_message_none_payload() -> None:
    # Should not crash even on weird input.
    assert get_message({"message": None}) == {}


# ---- get_content ----------------------------------------------------------


def test_get_content_from_top_level_chunk() -> None:
    assert get_content({"message": {"content": "hello"}}) == "hello"


def test_get_content_from_unwrapped_message() -> None:
    assert get_content({"content": "hello"}) == "hello"


def test_get_content_object_form() -> None:
    msg = SimpleNamespace(content="from object")
    assert get_content(SimpleNamespace(message=msg)) == "from object"


def test_get_content_empty_inputs() -> None:
    assert get_content({}) == ""
    assert get_content({"message": {}}) == ""
    assert get_content(None) == ""


# ---- get_eval_count -------------------------------------------------------


def test_eval_count_from_dict_chunk() -> None:
    assert get_eval_count({"eval_count": 42}) == 42


def test_eval_count_from_object_chunk() -> None:
    assert get_eval_count(SimpleNamespace(eval_count=99)) == 99


def test_eval_count_missing() -> None:
    assert get_eval_count({}) is None
    assert get_eval_count(SimpleNamespace()) is None


def test_eval_count_non_int_returns_none() -> None:
    assert get_eval_count({"eval_count": "ten"}) is None
    assert get_eval_count({"eval_count": 3.14}) is None


# ---- get_raw_tool_calls ---------------------------------------------------


def test_get_raw_tool_calls_from_chunk() -> None:
    chunk = {"message": {"tool_calls": [{"function": {"name": "t"}}]}}
    assert get_raw_tool_calls(chunk) == [{"function": {"name": "t"}}]


def test_get_raw_tool_calls_unwrapped_message() -> None:
    msg = {"tool_calls": [{"name": "t"}]}
    assert get_raw_tool_calls(msg) == [{"name": "t"}]


def test_get_raw_tool_calls_empty() -> None:
    assert get_raw_tool_calls({}) == []
    assert get_raw_tool_calls({"message": {}}) == []


# ---- coerce_args ----------------------------------------------------------


def test_coerce_args_passthrough_dict() -> None:
    assert coerce_args({"a": 1}) == {"a": 1}


def test_coerce_args_none_returns_empty() -> None:
    assert coerce_args(None) == {}


def test_coerce_args_parses_json_string() -> None:
    assert coerce_args('{"a": 1}') == {"a": 1}


def test_coerce_args_invalid_json_returns_empty() -> None:
    assert coerce_args("not json") == {}


def test_coerce_args_json_non_object_returns_empty() -> None:
    assert coerce_args('[1, 2]') == {}
    assert coerce_args('"string"') == {}


def test_coerce_args_unknown_type() -> None:
    assert coerce_args(42) == {}
    assert coerce_args([1, 2, 3]) == {}


# ---- normalize_tool_calls -------------------------------------------------


def test_normalize_openai_shape() -> None:
    raw = [{"function": {"name": "t1", "arguments": {"x": 1}}}]
    assert normalize_tool_calls(raw) == [{"name": "t1", "arguments": {"x": 1}}]


def test_normalize_flat_shape() -> None:
    raw = [{"name": "t2", "arguments": {"y": 2}}]
    assert normalize_tool_calls(raw) == [{"name": "t2", "arguments": {"y": 2}}]


def test_normalize_object_shape() -> None:
    fn = SimpleNamespace(name="t3", arguments={"z": 3})
    raw = [SimpleNamespace(function=fn)]
    assert normalize_tool_calls(raw) == [{"name": "t3", "arguments": {"z": 3}}]


def test_normalize_parses_json_string_arguments() -> None:
    raw = [{"function": {"name": "t", "arguments": '{"k": "v"}'}}]
    assert normalize_tool_calls(raw) == [{"name": "t", "arguments": {"k": "v"}}]


def test_normalize_drops_entries_without_name() -> None:
    raw = [{"function": {"arguments": {}}}, {"random": "x"}, None]
    assert normalize_tool_calls(raw) == []


def test_normalize_empty_input() -> None:
    assert normalize_tool_calls([]) == []
    assert normalize_tool_calls(None) == []


def test_normalize_missing_arguments_becomes_empty_dict() -> None:
    raw = [{"function": {"name": "t"}}]
    assert normalize_tool_calls(raw) == [{"name": "t", "arguments": {}}]


# ---- extract_inline_tool_calls -------------------------------------------


def test_extract_inline_wrapped_tag() -> None:
    from core.ollama_parsing import extract_inline_tool_calls

    text = (
        "Going to call:\n"
        '<tool_call>\n'
        '{"name": "notes_create", "arguments": {"title": "x"}}\n'
        "</tool_call>"
    )
    calls = extract_inline_tool_calls(text)
    assert calls == [{"name": "notes_create", "arguments": {"title": "x"}}]


def test_extract_inline_bare_json_after_stray_close_tag() -> None:
    from core.ollama_parsing import extract_inline_tool_calls

    # Reproduces the qwen2.5 quirk: missing opening tag, just JSON + closer.
    text = (
        'iNdEx\n'
        '{"name": "notes_create", "arguments": {"title": "Fix", "content": "body"}}\n'
        "</tool_call>"
    )
    calls = extract_inline_tool_calls(text)
    assert calls == [
        {"name": "notes_create", "arguments": {"title": "Fix", "content": "body"}}
    ]


def test_extract_inline_multiple_blocks() -> None:
    from core.ollama_parsing import extract_inline_tool_calls

    text = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>\n'
        '<tool_call>{"name": "b", "arguments": {"k": 1}}</tool_call>'
    )
    calls = extract_inline_tool_calls(text)
    assert [c["name"] for c in calls] == ["a", "b"]


def test_extract_inline_no_match_returns_empty() -> None:
    from core.ollama_parsing import extract_inline_tool_calls

    assert extract_inline_tool_calls("normal assistant response") == []
    assert extract_inline_tool_calls("") == []


def test_extract_inline_invalid_json_skipped() -> None:
    from core.ollama_parsing import extract_inline_tool_calls

    text = '<tool_call>{"name": "broken", "arguments": {</tool_call>'
    assert extract_inline_tool_calls(text) == []


def test_strip_inline_tool_blocks_removes_wrapped() -> None:
    from core.ollama_parsing import strip_inline_tool_blocks

    text = (
        "Let me check:\n"
        '<tool_call>{"name": "x"}</tool_call>\n'
        "Done."
    )
    cleaned = strip_inline_tool_blocks(text)
    assert "<tool_call>" not in cleaned
    assert "Let me check" in cleaned
    assert "Done." in cleaned


def test_strip_inline_tool_blocks_removes_stray_closer() -> None:
    from core.ollama_parsing import strip_inline_tool_blocks

    text = "intro\n</tool_call>"
    cleaned = strip_inline_tool_blocks(text)
    assert "</tool_call>" not in cleaned
    assert "intro" in cleaned


def test_strip_inline_tool_blocks_empty_input() -> None:
    from core.ollama_parsing import strip_inline_tool_blocks

    assert strip_inline_tool_blocks("") == ""
    assert strip_inline_tool_blocks(None) is None


def test_strip_inline_tool_blocks_removes_bare_json() -> None:
    """When the model emits only the JSON + stray closer with NO opening
    tag (the actual qwen2.5 quirk), the JSON must still be stripped."""
    from core.ollama_parsing import strip_inline_tool_blocks

    text = (
        'HeaderCode:\n'
        '{"name": "notes_create", "arguments": {"title": "X", "content": "Body"}}\n'
        "</tool_call>"
    )
    cleaned = strip_inline_tool_blocks(text)
    assert '"name"' not in cleaned
    assert "</tool_call>" not in cleaned
    assert "HeaderCode" not in cleaned


def test_strip_inline_tool_blocks_keeps_real_prose() -> None:
    """Don't accidentally strip JSON-looking content that isn't a tool call."""
    from core.ollama_parsing import strip_inline_tool_blocks

    text = (
        "Here is a config example:\n"
        '{"key": "value"}\n'
        "End of example."
    )
    cleaned = strip_inline_tool_blocks(text)
    # No "name"/"arguments" → not matched, kept
    assert '"key": "value"' in cleaned
    assert "Here is a config" in cleaned
