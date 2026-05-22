"""Quantization detection for Ollama model names.

Ollama model names follow the pattern ``<base>:<tag>`` where ``<tag>`` often
encodes the quantization, e.g.::

    qwen2.5:14b-instruct-q8_0
    qwen2.5:14b-instruct-q4_K_M
    qwen2.5:14b-instruct-fp16
    llama3.1:8b-instruct-q4_0
    qwen2.5:14b                       # default (usually q4_K_M)

This module knows how to:
  * extract the quantization suffix from a model name;
  * normalize a model name to its quantization-free "base";
  * map quantization tokens to approximate bits-per-weight (useful for
    plotting trade-off curves);
  * find all installed quantizations of a given base in a list of available
    models.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

# Tokens we recognise as quantization markers. Order does not matter — we
# match the trailing dash-separated segment against this set.
_KNOWN_QUANT_TOKENS: frozenset[str] = frozenset({
    "q2_k",
    "q3_k_s", "q3_k_m", "q3_k_l",
    "q4_0", "q4_1", "q4_k_s", "q4_k_m", "q4_k_l",
    "q5_0", "q5_1", "q5_k_s", "q5_k_m",
    "q6_k",
    "q8_0",
    "fp16", "f16", "bf16",
    "fp32", "f32",
})

# Suffix at the end of a model name, separated by a dash. Captures both
# "-q4_K_M" and "-FP16" (case-insensitive).
_TRAILING_QUANT_RE = re.compile(
    r"-(q\d+(?:_[a-z0-9]+)*|fp16|f16|bf16|fp32|f32)$",
    re.IGNORECASE,
)

# Approximate bits-per-weight per quantization. K-quants store some weights
# at higher precision than the nominal bit count, so the effective rate is
# slightly above the integer — these values are good enough for plotting
# size/quality trade-off curves on a linear axis.
_BITS_PER_WEIGHT: dict[str, float] = {
    "q2_k":    2.6,
    "q3_k_s":  3.4,
    "q3_k_m":  3.7,
    "q3_k_l":  4.1,
    "q4_0":    4.5,
    "q4_1":    5.0,
    "q4_k_s":  4.4,
    "q4_k_m":  4.8,
    "q4_k_l":  5.1,
    "q5_0":    5.5,
    "q5_1":    6.0,
    "q5_k_s":  5.5,
    "q5_k_m":  5.7,
    "q6_k":    6.6,
    "q8_0":    8.5,
    "fp16":    16.0,
    "f16":     16.0,
    "bf16":    16.0,
    "fp32":    32.0,
    "f32":     32.0,
}


def parse_quantization(model_name: str) -> Optional[str]:
    """Return the lower-cased quantization tag (e.g. ``"q4_k_m"``) or ``None``
    if the model name has no recognisable quantization suffix.
    """
    if not model_name:
        return None
    match = _TRAILING_QUANT_RE.search(model_name)
    if not match:
        return None
    token = match.group(1).lower()
    if token in _KNOWN_QUANT_TOKENS:
        return token
    # Generic q-token we don't have in the lookup but is still a quantization
    # (e.g. an experimental q7_k).
    if token.startswith("q") and any(ch.isdigit() for ch in token):
        return token
    return None


def base_name(model_name: str) -> str:
    """Strip the quantization suffix from a model name.

    Examples:
        >>> base_name("qwen2.5:14b-instruct-q4_K_M")
        'qwen2.5:14b-instruct'
        >>> base_name("qwen2.5:14b")
        'qwen2.5:14b'
    """
    if not model_name:
        return model_name
    quant = parse_quantization(model_name)
    if quant is None:
        return model_name
    suffix_re = re.compile(r"-" + re.escape(quant) + r"$", re.IGNORECASE)
    return suffix_re.sub("", model_name)


def quantization_bits(quant: Optional[str]) -> Optional[float]:
    """Approximate bits-per-weight for plotting. Returns ``None`` for unknown
    or ``None`` quantization (caller may treat as the model's default).
    """
    if not quant:
        return None
    return _BITS_PER_WEIGHT.get(quant.lower())


def find_quantizations(
    available_models: Iterable[str],
    base: str,
) -> list[tuple[str, str]]:
    """Find all installed models matching the given base name.

    Returns ``[(model_name, quantization), ...]`` sorted by bits-per-weight
    ascending. ``quantization`` is ``"default"`` for entries where the tag
    has no recognisable quant suffix (those are typically the ``q4_K_M``
    default that Ollama publishes).
    """
    base_norm = base_name(base).lower()
    out: list[tuple[str, str]] = []
    for m in available_models:
        m_base = base_name(m).lower()
        if m_base == base_norm:
            quant = parse_quantization(m) or "default"
            out.append((m, quant))
    out.sort(key=_quant_sort_key)
    return out


def group_by_base(models: Iterable[str]) -> dict[str, list[tuple[str, str]]]:
    """Group installed models by their quantization-free base name.

    Returns ``{base: [(model_name, quant), ...]}`` with each list sorted by
    bits-per-weight ascending.
    """
    groups: dict[str, list[tuple[str, str]]] = {}
    for m in models:
        base = base_name(m)
        quant = parse_quantization(m) or "default"
        groups.setdefault(base, []).append((m, quant))
    for base in groups:
        groups[base].sort(key=_quant_sort_key)
    return groups


def _quant_sort_key(entry: tuple[str, str]) -> float:
    _, quant = entry
    bits = quantization_bits(quant)
    if bits is not None:
        return bits
    # Place unknown / "default" between Q4 and Q5 — Ollama's default tag is
    # usually q4_K_M, so this lines up with the observed bit width.
    return 4.8 if quant == "default" else 99.0
