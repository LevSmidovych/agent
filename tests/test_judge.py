from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from benchmarks.judge import (
    DEFAULT_JUDGE_PREFERENCES,
    JudgeItem,
    JudgeVerdict,
    LLMJudge,
    parse_verdict,
)
from benchmarks.prompts import BenchmarkPrompt
from core.exceptions import OllamaConnectionError


def _stub_llm(content):
    llm = MagicMock()
    if isinstance(content, list):
        llm.chat_json = MagicMock(side_effect=content)
    elif isinstance(content, Exception):
        llm.chat_json = MagicMock(side_effect=content)
    else:
        llm.chat_json = MagicMock(return_value=content)
    return llm


def _qa_prompt() -> BenchmarkPrompt:
    return BenchmarkPrompt(
        id="p", prompt="Explain X", category="quality_ua",
        expected_behavior="thorough, in Ukrainian",
    )


# ---- model selection ------------------------------------------------------


def test_select_model_prefers_qwen_32b_over_14b() -> None:
    chosen = LLMJudge.select_model([
        "llama3.1:8b", "qwen2.5:14b-instruct-q8_0", "qwen2.5:32b-instruct-q4_K_M",
    ])
    assert chosen.startswith("qwen2.5:32b")


def test_select_model_falls_back_to_largest_pref_present() -> None:
    chosen = LLMJudge.select_model([
        "llama3.1:8b", "qwen2.5:14b-instruct-q8_0",
    ])
    assert chosen.startswith("qwen2.5:14b")


def test_select_model_falls_back_to_qwen_7b() -> None:
    chosen = LLMJudge.select_model([
        "mistral-nemo:latest", "qwen2.5:7b-instruct-q4_K_M",
    ])
    assert chosen.startswith("qwen2.5:7b")


def test_select_model_returns_first_when_no_match() -> None:
    chosen = LLMJudge.select_model(["custom-model:latest", "weird:v1"])
    assert chosen == "custom-model:latest"


def test_select_model_empty_list_returns_none() -> None:
    assert LLMJudge.select_model([]) is None


def test_select_model_accepts_exact_match() -> None:
    chosen = LLMJudge.select_model(["qwen2.5:32b"])
    assert chosen == "qwen2.5:32b"


def test_select_model_picks_mistral_when_only_option() -> None:
    chosen = LLMJudge.select_model(["mistral:7b-instruct", "llama3.1:8b"])
    # mistral:7b comes before llama3.1:8b in DEFAULT_JUDGE_PREFERENCES
    assert chosen.startswith("mistral:7b")


def test_select_model_prefers_qwen_over_mistral() -> None:
    chosen = LLMJudge.select_model([
        "mistral:7b-instruct", "qwen2.5:7b-instruct-q4_K_M",
    ])
    assert chosen.startswith("qwen2.5:7b")


def test_select_model_custom_preferences() -> None:
    chosen = LLMJudge.select_model(
        ["custom-a", "custom-b"],
        preferences=["custom-b"],
    )
    assert chosen == "custom-b"


# ---- parse_verdict --------------------------------------------------------


def test_parse_verdict_strict_json() -> None:
    v = parse_verdict('{"score": 4, "rationale": "good"}')
    assert v.score == 4.0
    assert v.rationale == "good"
    assert v.error is None
    assert v.ok


def test_parse_verdict_float_score() -> None:
    v = parse_verdict('{"score": 4.5, "rationale": "ok"}')
    assert v.score == 4.5


def test_parse_verdict_score_as_string() -> None:
    v = parse_verdict('{"score": "3", "rationale": "x"}')
    assert v.score == 3.0


def test_parse_verdict_clamps_above_5() -> None:
    v = parse_verdict('{"score": 9, "rationale": "x"}')
    assert v.score == 5.0


def test_parse_verdict_clamps_below_1() -> None:
    v = parse_verdict('{"score": 0, "rationale": "x"}')
    assert v.score == 1.0


def test_parse_verdict_extracts_from_markdown_fence() -> None:
    raw = "Here is the verdict:\n```json\n{\"score\": 5, \"rationale\": \"perfect\"}\n```"
    v = parse_verdict(raw)
    assert v.score == 5.0
    assert v.rationale == "perfect"


def test_parse_verdict_extracts_from_prose() -> None:
    raw = "I think: {\"score\": 2, \"rationale\": \"meh\"}. Hope that helps."
    v = parse_verdict(raw)
    assert v.score == 2.0


def test_parse_verdict_empty_response() -> None:
    v = parse_verdict("")
    assert v.score is None
    assert "empty" in v.error


def test_parse_verdict_invalid_json() -> None:
    v = parse_verdict("complete garbage")
    assert v.score is None
    assert "parse" in v.error.lower()


def test_parse_verdict_missing_score_field() -> None:
    v = parse_verdict('{"rationale": "ok"}')
    assert v.score is None
    assert v.error is not None


def test_parse_verdict_score_not_numeric() -> None:
    v = parse_verdict('{"score": "abc", "rationale": "x"}')
    assert v.score is None


def test_parse_verdict_score_boolean_rejected() -> None:
    v = parse_verdict('{"score": true, "rationale": "x"}')
    assert v.score is None


def test_parse_verdict_accepts_reason_alias() -> None:
    v = parse_verdict('{"score": 4, "reason": "alt key"}')
    assert v.score == 4.0
    assert v.rationale == "alt key"


# ---- judge() --------------------------------------------------------------


def test_judge_returns_verdict_on_success() -> None:
    llm = _stub_llm('{"score": 5, "rationale": "great"}')
    j = LLMJudge(llm, "qwen2.5:14b")
    v = j.judge(_qa_prompt(), "відповідь")
    assert v.score == 5.0
    assert v.rationale == "great"


def test_judge_passes_rubric_in_user_message() -> None:
    llm = _stub_llm('{"score": 3, "rationale": "x"}')
    j = LLMJudge(llm, "qwen2.5:14b")
    j.judge(_qa_prompt(), "test response")
    call = llm.chat_json.call_args
    messages = call.kwargs.get("messages") or call.args[1]
    user = messages[1]["content"]
    assert "thorough, in Ukrainian" in user
    assert "test response" in user


def test_judge_handles_empty_response() -> None:
    llm = _stub_llm('{"score": 1, "rationale": "empty"}')
    j = LLMJudge(llm, "m")
    v = j.judge(_qa_prompt(), "")
    assert v.score == 1.0
    # User message should mark the empty response explicitly
    call = llm.chat_json.call_args
    messages = call.kwargs.get("messages") or call.args[1]
    assert "(empty)" in messages[1]["content"]


def test_judge_returns_error_on_connection_failure() -> None:
    llm = _stub_llm(OllamaConnectionError("network down"))
    j = LLMJudge(llm, "m")
    v = j.judge(_qa_prompt(), "x")
    assert v.score is None
    assert "network" in v.error


# ---- judge_batch ----------------------------------------------------------


def _make_items(n):
    return [
        JudgeItem(result_id=i, prompt=_qa_prompt(), response=f"resp {i}")
        for i in range(n)
    ]


def test_judge_batch_yields_verdicts_in_order() -> None:
    llm = _stub_llm([
        '{"score": 5, "rationale": "a"}',
        '{"score": 3, "rationale": "b"}',
        '{"score": 1, "rationale": "c"}',
    ])
    j = LLMJudge(llm, "m")
    results = list(j.judge_batch(_make_items(3)))
    assert [v.score for _, v in results] == [5.0, 3.0, 1.0]
    assert [item.result_id for item, _ in results] == [0, 1, 2]


def test_judge_batch_invokes_progress_callback() -> None:
    llm = _stub_llm('{"score": 4, "rationale": "ok"}')
    j = LLMJudge(llm, "m")
    items = _make_items(2)
    reports: list[tuple[int, int]] = []
    list(j.judge_batch(items, progress=lambda i, t, _: reports.append((i, t))))
    # progress before each + one final
    assert reports[0] == (0, 2)
    assert reports[-1] == (2, 2)


def test_judge_batch_stops_on_event() -> None:
    llm = _stub_llm('{"score": 4, "rationale": "ok"}')
    j = LLMJudge(llm, "m")
    stop = threading.Event()
    stop.set()
    results = list(j.judge_batch(_make_items(3), stop_event=stop))
    assert results == []


def test_judge_batch_empty_input() -> None:
    j = LLMJudge(_stub_llm('{"score": 5}'), "m")
    assert list(j.judge_batch([])) == []
