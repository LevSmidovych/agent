from __future__ import annotations

from benchmarks.prompts import BenchmarkPrompt
from benchmarks.scoring import (
    ScoreResult,
    _extract_choice,
    _extract_first_number,
    _normalize,
    score_prompt,
    score_reasoning,
    score_tool_use,
)


# ---- helpers --------------------------------------------------------------


def test_normalize_lowercase_and_punct() -> None:
    assert _normalize("  Hello, World!  ") == "hello world"
    assert _normalize("12:00.") == "12:00"
    assert _normalize("Тарас Шевченко") == "тарас шевченко"


def test_extract_first_number() -> None:
    assert _extract_first_number("the answer is 42 actually") == 42.0
    assert _extract_first_number("price: 3.14 EUR") == 3.14
    assert _extract_first_number("temperature -5 degrees") == -5.0
    assert _extract_first_number("no digits here") is None


def test_extract_choice_finds_standalone_letter() -> None:
    assert _extract_choice("The answer is C.") == "C"
    assert _extract_choice("c) some flowers") == "C"
    assert _extract_choice("first A then B") == "A"
    assert _extract_choice("nothing relevant") is None
    # Standalone letter only — "Apple" should not match "A"
    assert _extract_choice("Apple is a fruit") is None


# ---- reasoning grader -----------------------------------------------------


def _make(match_type, expected):
    return BenchmarkPrompt(
        id="t", prompt="p", category="reasoning",
        expected=expected, match_type=match_type,
    )


def test_reasoning_returns_none_without_ground_truth() -> None:
    p = BenchmarkPrompt(id="t", prompt="p", category="reasoning")
    assert score_reasoning(p, "anything") is None


def test_reasoning_exact_positive() -> None:
    res = score_reasoning(_make("exact", "carrot"), "Carrot.")
    assert res.passed
    assert "exact" in res.rationale


def test_reasoning_exact_negative() -> None:
    res = score_reasoning(_make("exact", "carrot"), "potato")
    assert not res.passed


def test_reasoning_contains_positive() -> None:
    res = score_reasoning(_make("contains", "juice"), "Anna drinks juice.")
    assert res.passed


def test_reasoning_contains_case_insensitive() -> None:
    res = score_reasoning(_make("contains", "JUICE"), "anna drinks juice")
    assert res.passed


def test_reasoning_contains_negative() -> None:
    res = score_reasoning(_make("contains", "tea"), "Anna drinks juice")
    assert not res.passed


def test_reasoning_number_positive() -> None:
    res = score_reasoning(_make("number", "5"), "Bob is 5 years old.")
    assert res.passed


def test_reasoning_number_with_decimal() -> None:
    res = score_reasoning(_make("number", "3.14"), "pi is approximately 3.14159")
    assert res.passed


def test_reasoning_number_negative_value() -> None:
    res = score_reasoning(_make("number", "-5"), "temperature is -5 degrees")
    assert res.passed


def test_reasoning_number_mismatch() -> None:
    res = score_reasoning(_make("number", "42"), "I think it is 7")
    assert not res.passed


def test_reasoning_number_no_digit_in_response() -> None:
    res = score_reasoning(_make("number", "42"), "I don't know")
    assert not res.passed


def test_reasoning_choice_positive() -> None:
    res = score_reasoning(_make("choice", "C"), "The answer is C.")
    assert res.passed


def test_reasoning_choice_lowercase_accepted() -> None:
    res = score_reasoning(_make("choice", "B"), "b)")
    assert res.passed


def test_reasoning_choice_wrong_letter() -> None:
    res = score_reasoning(_make("choice", "C"), "A")
    assert not res.passed


def test_reasoning_choice_no_letter() -> None:
    res = score_reasoning(_make("choice", "C"), "I'm not sure honestly")
    assert not res.passed


def test_reasoning_unknown_match_type() -> None:
    p = BenchmarkPrompt(id="t", prompt="p", category="reasoning",
                        expected="x", match_type="weird")
    res = score_reasoning(p, "x")
    assert not res.passed


# ---- tool use grader ------------------------------------------------------


def _tool_prompt(tool: str, args=None):
    return BenchmarkPrompt(
        id="t", prompt="p", category="tool_use",
        expected_tool=tool, expected_args_contains=args,
    )


def test_tool_use_returns_none_without_expected_tool() -> None:
    p = BenchmarkPrompt(id="t", prompt="p", category="tool_use")
    assert score_tool_use(p, []) is None


def test_tool_use_no_calls_made() -> None:
    p = _tool_prompt("notes_search")
    res = score_tool_use(p, [])
    assert not res.passed
    assert "no tool calls" in res.rationale


def test_tool_use_correct_tool_no_args() -> None:
    p = _tool_prompt("list_directory")
    calls = [{"name": "list_directory", "arguments": {"path": ""}}]
    res = score_tool_use(p, calls)
    assert res.passed


def test_tool_use_wrong_tool() -> None:
    p = _tool_prompt("notes_search")
    calls = [{"name": "notes_create", "arguments": {"title": "x"}}]
    res = score_tool_use(p, calls)
    assert not res.passed
    assert "notes_create" in res.rationale


def test_tool_use_args_match_one_of_accepted() -> None:
    p = _tool_prompt("notes_search", {"query": ["python", "Python"]})
    calls = [{"name": "notes_search", "arguments": {"query": "my Python notes"}}]
    res = score_tool_use(p, calls)
    assert res.passed


def test_tool_use_args_missing_substring() -> None:
    p = _tool_prompt("notes_search", {"query": ["python"]})
    calls = [{"name": "notes_search", "arguments": {"query": "javascript"}}]
    res = score_tool_use(p, calls)
    assert not res.passed
    assert "query" in res.rationale


def test_tool_use_partial_credit_for_some_args() -> None:
    p = _tool_prompt(
        "notes_create",
        {"title": ["shopping"], "content": ["milk"]},
    )
    # title matches, content does not
    calls = [{"name": "notes_create", "arguments": {"title": "shopping list", "content": "eggs"}}]
    res = score_tool_use(p, calls)
    assert 0.0 < res.pass_rate < 1.0
    assert "content" in res.rationale


def test_tool_use_multiple_calls_picks_best() -> None:
    p = _tool_prompt("notes_search", {"query": ["python"]})
    calls = [
        {"name": "notes_search", "arguments": {"query": "java"}},
        {"name": "notes_search", "arguments": {"query": "python tips"}},
    ]
    res = score_tool_use(p, calls)
    assert res.passed


def test_tool_use_openai_shaped_call() -> None:
    p = _tool_prompt("notes_search", {"query": ["python"]})
    calls = [{"function": {"name": "notes_search", "arguments": {"query": "python"}}}]
    res = score_tool_use(p, calls)
    assert res.passed


def test_tool_use_json_string_arguments() -> None:
    p = _tool_prompt("notes_search", {"query": ["python"]})
    calls = [{"function": {"name": "notes_search", "arguments": '{"query": "python"}'}}]
    res = score_tool_use(p, calls)
    assert res.passed


def test_tool_use_args_non_string_value_coerced() -> None:
    p = _tool_prompt("notes_create", {"title": ["42"]})
    calls = [{"name": "notes_create", "arguments": {"title": 42}}]
    res = score_tool_use(p, calls)
    assert res.passed


def test_normalize_tool_calls_drops_invalid_entries() -> None:
    from core.ollama_parsing import normalize_tool_calls

    assert normalize_tool_calls([{"random": "x"}, None, {"name": "ok"}]) == [
        {"name": "ok", "arguments": {}},
    ]


# ---- dispatch -------------------------------------------------------------


def test_score_prompt_routes_reasoning() -> None:
    p = BenchmarkPrompt(
        id="t", prompt="p", category="reasoning",
        expected="5", match_type="number",
    )
    res = score_prompt(p, response="5")
    assert res.passed


def test_score_prompt_routes_tool_use() -> None:
    p = BenchmarkPrompt(
        id="t", prompt="p", category="tool_use",
        expected_tool="notes_search",
    )
    res = score_prompt(p, tool_calls=[{"name": "notes_search", "arguments": {}}])
    assert res.passed


def test_score_prompt_returns_none_for_freeform() -> None:
    p = BenchmarkPrompt(id="t", prompt="p", category="quality_ua")
    assert score_prompt(p, response="anything") is None


def test_score_prompt_returns_none_for_speed() -> None:
    p = BenchmarkPrompt(id="t", prompt="p", category="speed")
    assert score_prompt(p, response="42") is None
