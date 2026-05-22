from __future__ import annotations

import pytest

from benchmarks.prompts import (
    CODE,
    PROMPT_SETS,
    QUALITY_UA,
    REASONING,
    SPEED,
    TOOL_USE,
    BenchmarkPrompt,
    all_prompts,
    get_set,
    make_tool_schemas_for_prompts,
    prompts_by_category,
    set_keys,
)


# ---- registry shape ------------------------------------------------------


def test_five_sets_present() -> None:
    assert set(set_keys()) == {"speed", "quality_ua", "code", "reasoning", "tool_use"}


def test_required_set_sizes() -> None:
    assert len(SPEED) >= 5
    assert len(QUALITY_UA) >= 8
    assert len(CODE) >= 5
    assert len(REASONING) >= 6
    assert len(TOOL_USE) >= 5


def test_get_set_returns_copy() -> None:
    a = get_set("speed")
    b = get_set("speed")
    a.clear()
    assert b, "get_set must return an independent list"


def test_get_set_unknown_returns_empty() -> None:
    assert get_set("nonexistent") == []


def test_all_prompts_aggregates() -> None:
    total = sum(len(v) for v in PROMPT_SETS.values())
    assert len(all_prompts()) == total


def test_prompts_by_category_filters() -> None:
    reasoning = prompts_by_category("reasoning")
    assert all(p.category == "reasoning" for p in reasoning)
    assert len(reasoning) == len(REASONING)


# ---- prompt invariants ---------------------------------------------------


def test_all_prompt_ids_unique() -> None:
    ids = [p.id for p in all_prompts()]
    assert len(ids) == len(set(ids)), "prompt IDs must be unique across all sets"


def test_all_prompts_have_nonempty_text() -> None:
    for p in all_prompts():
        assert p.prompt.strip(), f"prompt {p.id} is empty"
        assert p.id, "prompt id must be non-empty"
        assert p.category, f"prompt {p.id} has no category"


def test_prompt_category_matches_set() -> None:
    for key, prompts in PROMPT_SETS.items():
        for p in prompts:
            assert p.category == key, f"{p.id}: category {p.category!r} != set {key!r}"


# ---- reasoning prompts have rule-based scoring ---------------------------


def test_reasoning_prompts_have_match_type() -> None:
    valid = {"exact", "contains", "number", "choice"}
    for p in REASONING:
        assert p.match_type in valid, f"{p.id}: unknown match_type {p.match_type!r}"
        assert p.expected is not None, f"{p.id}: must have expected value"


def test_reasoning_choice_has_single_letter_expected() -> None:
    choice = [p for p in REASONING if p.match_type == "choice"]
    assert choice, "expected at least one multi-choice reasoning prompt"
    for p in choice:
        assert p.expected in {"A", "B", "C", "D"}


# ---- tool_use prompts have tool expectations -----------------------------


def test_tool_use_prompts_have_expected_tool() -> None:
    for p in TOOL_USE:
        assert p.expected_tool, f"{p.id}: must declare expected_tool"


def test_tool_use_args_are_lists_of_strings() -> None:
    for p in TOOL_USE:
        if p.expected_args_contains is None:
            continue
        assert isinstance(p.expected_args_contains, dict)
        for key, val in p.expected_args_contains.items():
            assert isinstance(val, list) and all(isinstance(v, str) for v in val), (
                f"{p.id}: expected_args_contains[{key!r}] must be list of strings"
            )


# ---- quality_ua + code prompts use judge ---------------------------------


def test_quality_ua_prompts_have_expected_behavior() -> None:
    for p in QUALITY_UA:
        assert p.expected_behavior, f"{p.id}: needs expected_behavior for judge"


def test_code_prompts_have_expected_behavior() -> None:
    for p in CODE:
        assert p.expected_behavior, f"{p.id}: needs expected_behavior for judge"


# ---- speed prompts are short and don't need ground truth -----------------


def test_speed_prompts_have_no_expected() -> None:
    for p in SPEED:
        assert p.expected is None
        assert p.expected_tool is None


# ---- make_tool_schemas_for_prompts ---------------------------------------


def _tool_prompt(pid: str, tool: str) -> BenchmarkPrompt:
    return BenchmarkPrompt(
        id=pid, prompt="x", category="tool_use", expected_tool=tool,
    )


def test_tool_schemas_returns_none_for_no_tool_use_prompts() -> None:
    assert make_tool_schemas_for_prompts(list(SPEED)) is None


def test_tool_schemas_collects_unique_tools() -> None:
    prompts = [
        _tool_prompt("t1", "notes_search"),
        _tool_prompt("t2", "notes_create"),
        _tool_prompt("t3", "notes_search"),  # dup
    ]
    schemas = make_tool_schemas_for_prompts(prompts)
    assert schemas is not None
    names = {s["function"]["name"] for s in schemas}
    assert names == {"notes_search", "notes_create"}


def test_tool_schemas_use_openai_shape() -> None:
    prompts = [_tool_prompt("t1", "read_file")]
    schemas = make_tool_schemas_for_prompts(prompts)
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "read_file"
    assert "parameters" in schemas[0]["function"]


def test_tool_schemas_empty_input() -> None:
    assert make_tool_schemas_for_prompts([]) is None
