from __future__ import annotations

import pytest

from benchmarks.quantization import (
    base_name,
    find_quantizations,
    group_by_base,
    parse_quantization,
    quantization_bits,
)


# ---- parse_quantization ---------------------------------------------------


@pytest.mark.parametrize("model,expected", [
    ("qwen2.5:14b-instruct-q4_K_M", "q4_k_m"),
    ("qwen2.5:14b-instruct-q8_0", "q8_0"),
    ("qwen2.5:14b-instruct-fp16", "fp16"),
    ("qwen2.5:14b-instruct-bf16", "bf16"),
    ("llama3.1:8b-instruct-q4_0", "q4_0"),
    ("qwen2.5:14b-instruct-q5_K_M", "q5_k_m"),
    ("model-q2_K", "q2_k"),
    ("model-Q6_K", "q6_k"),
    ("model-FP16", "fp16"),
])
def test_parse_quantization_known_tokens(model, expected) -> None:
    assert parse_quantization(model) == expected


@pytest.mark.parametrize("model", [
    "qwen2.5:14b",
    "qwen2.5:14b-instruct",
    "llama3.1:latest",
    "mistral:7b",
    "",
])
def test_parse_quantization_returns_none_without_suffix(model) -> None:
    assert parse_quantization(model) is None


def test_parse_quantization_unknown_q_token_kept() -> None:
    # Forward-compat: q7_k may appear in future Ollama releases.
    assert parse_quantization("model-q7_k") == "q7_k"


def test_parse_quantization_ignores_non_quant_dash_segment() -> None:
    # "-instruct" is not a quant tag.
    assert parse_quantization("qwen2.5:14b-instruct") is None


# ---- quantization_bits ----------------------------------------------------


def test_bits_ordering_matches_expectations() -> None:
    # Strict monotone ordering: more bits → higher value.
    order = ["q2_k", "q3_k_m", "q4_k_m", "q5_k_m", "q6_k", "q8_0", "fp16", "fp32"]
    bits = [quantization_bits(q) for q in order]
    assert all(a < b for a, b in zip(bits, bits[1:])), bits


def test_bits_for_fp_aliases() -> None:
    assert quantization_bits("fp16") == quantization_bits("f16") == quantization_bits("bf16")
    assert quantization_bits("fp32") == quantization_bits("f32")


def test_bits_unknown_returns_none() -> None:
    assert quantization_bits(None) is None
    assert quantization_bits("") is None
    assert quantization_bits("nonsense") is None


def test_bits_case_insensitive() -> None:
    assert quantization_bits("Q4_K_M") == quantization_bits("q4_k_m")


# ---- base_name -----------------------------------------------------------


@pytest.mark.parametrize("model,expected", [
    ("qwen2.5:14b-instruct-q4_K_M", "qwen2.5:14b-instruct"),
    ("qwen2.5:14b-instruct-Q8_0", "qwen2.5:14b-instruct"),
    ("qwen2.5:14b-instruct-fp16", "qwen2.5:14b-instruct"),
    ("qwen2.5:14b", "qwen2.5:14b"),
    ("llama3.1:latest", "llama3.1:latest"),
    ("model-instruct", "model-instruct"),  # no quant suffix
    ("", ""),
])
def test_base_name(model, expected) -> None:
    assert base_name(model) == expected


# ---- find_quantizations --------------------------------------------------


def test_find_quantizations_returns_matching_sorted_by_bits() -> None:
    available = [
        "qwen2.5:14b-instruct-q4_K_M",
        "qwen2.5:14b-instruct-q8_0",
        "qwen2.5:14b-instruct-fp16",
        "qwen2.5:7b-instruct-q4_K_M",       # different base
        "llama3.1:8b-instruct-q4_K_M",       # different model
    ]
    result = find_quantizations(available, "qwen2.5:14b-instruct-q4_K_M")
    quants = [q for _, q in result]
    assert quants == ["q4_k_m", "q8_0", "fp16"]


def test_find_quantizations_base_without_suffix_also_works() -> None:
    available = [
        "qwen2.5:14b-instruct-q4_K_M",
        "qwen2.5:14b-instruct-q8_0",
    ]
    result = find_quantizations(available, "qwen2.5:14b-instruct")
    assert len(result) == 2


def test_find_quantizations_includes_default_no_suffix() -> None:
    available = [
        "qwen2.5:14b",                       # default
        "qwen2.5:14b-instruct-q8_0",         # different base (has -instruct)
    ]
    result = find_quantizations(available, "qwen2.5:14b")
    assert result == [("qwen2.5:14b", "default")]


def test_find_quantizations_empty_for_unknown_base() -> None:
    assert find_quantizations(["qwen2.5:14b"], "nonexistent") == []


def test_find_quantizations_empty_input() -> None:
    assert find_quantizations([], "anything") == []


# ---- group_by_base -------------------------------------------------------


def test_group_by_base_groups_quantizations_together() -> None:
    available = [
        "qwen2.5:14b-instruct-q4_K_M",
        "qwen2.5:14b-instruct-q8_0",
        "llama3.1:8b-instruct-q4_0",
        "qwen2.5:7b-instruct-q4_K_M",
    ]
    groups = group_by_base(available)
    assert set(groups.keys()) == {
        "qwen2.5:14b-instruct",
        "llama3.1:8b-instruct",
        "qwen2.5:7b-instruct",
    }
    qwen_14b = [q for _, q in groups["qwen2.5:14b-instruct"]]
    assert qwen_14b == ["q4_k_m", "q8_0"]


def test_group_by_base_each_group_sorted_by_bits() -> None:
    available = [
        "model-fp16",
        "model-q4_K_M",
        "model-q8_0",
        "model-q2_K",
    ]
    groups = group_by_base(available)
    quants = [q for _, q in groups["model"]]
    assert quants == ["q2_k", "q4_k_m", "q8_0", "fp16"]


def test_group_by_base_handles_default_models() -> None:
    available = ["qwen2.5:14b", "qwen2.5:14b-instruct-q8_0"]
    groups = group_by_base(available)
    # Default-quant model has different "base" because no -instruct suffix
    assert "qwen2.5:14b" in groups
    assert "qwen2.5:14b-instruct" in groups
