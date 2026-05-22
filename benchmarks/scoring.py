"""Rule-based scoring for benchmark prompts.

The benchmark runner produces a free-form text response from each model. For
``reasoning`` and ``tool_use`` prompts we can grade automatically without
involving another LLM. This module contains the deterministic graders.

Free-form prompts (``speed``, ``quality_ua``, ``code``) are graded elsewhere
by the LLM-as-judge module (see ``benchmarks/judge.py``) or manually.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from benchmarks.prompts import BenchmarkPrompt
from core.ollama_parsing import normalize_tool_calls


@dataclass
class ScoreResult:
    """Outcome of a rule-based check.

    ``pass_rate`` is in [0.0, 1.0]. For binary checks it is exactly 0 or 1.
    """

    pass_rate: float
    rationale: str

    @property
    def passed(self) -> bool:
        return self.pass_rate >= 0.999


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
# Uppercase letter standing alone (not part of a word like "Apple").
_UPPER_CHOICE_RE = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")
# Lowercase a/b/c/d only when followed by a typical answer marker (`)`, `.`,
# `:`). Avoids matching the English article "a" in free prose.
_LOWER_CHOICE_RE = re.compile(r"(?<![A-Za-z])([a-d])(?=[).:])")


# ---- helpers -------------------------------------------------------------


_PUNCT_KEEP = {":", "-"}


def _normalize(text: str) -> str:
    """Lower-case, replace punctuation with space, collapse whitespace.

    Keeps all unicode letters and digits — including Cyrillic — so that we
    can match Ukrainian responses too. ``:`` and ``-`` are kept so times and
    negative numbers survive.
    """
    if not text:
        return ""
    text = text.lower()
    out: list[str] = []
    for ch in text:
        if ch in string.punctuation and ch not in _PUNCT_KEEP:
            out.append(" ")
        else:
            out.append(ch)
    return " ".join("".join(out).split())


def _extract_first_number(text: str) -> Optional[float]:
    if not text:
        return None
    m = _NUMBER_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _extract_choice(text: str) -> Optional[str]:
    """Find the first answer letter A-D in the response.

    Prefers a standalone uppercase letter (``"C"``, ``"The answer is C."``).
    Falls back to lowercase only when followed by ``)``, ``.``, or ``:`` —
    the standard multi-choice answer markers. This avoids matching the
    English article "a" in regular prose.
    """
    if not text:
        return None
    m = _UPPER_CHOICE_RE.search(text)
    if m:
        return m.group(1)
    m = _LOWER_CHOICE_RE.search(text)
    return m.group(1).upper() if m else None


# ---- reasoning -----------------------------------------------------------


def score_reasoning(prompt: BenchmarkPrompt, response: str) -> Optional[ScoreResult]:
    """Grade a reasoning prompt.

    Returns ``None`` if the prompt has no rule-based ground truth — callers
    should fall back to LLM-as-judge or manual scoring.
    """
    if not prompt.expected or not prompt.match_type:
        return None
    match = prompt.match_type
    expected = prompt.expected
    response = response or ""

    if match == "exact":
        if _normalize(response) == _normalize(expected):
            return ScoreResult(1.0, "exact normalized match")
        return ScoreResult(0.0, f"expected exact {expected!r}, got {response[:60]!r}")

    if match == "contains":
        if _normalize(expected) in _normalize(response):
            return ScoreResult(1.0, f"response contains {expected!r}")
        return ScoreResult(0.0, f"expected substring {expected!r} not found")

    if match == "number":
        expected_num = _extract_first_number(expected)
        got_num = _extract_first_number(response)
        if expected_num is None:
            return ScoreResult(0.0, f"expected value {expected!r} is not numeric")
        if got_num is None:
            return ScoreResult(0.0, "no number found in response")
        if abs(expected_num - got_num) < 0.01:
            return ScoreResult(1.0, f"number match: {got_num}")
        return ScoreResult(0.0, f"expected {expected_num}, got {got_num}")

    if match == "choice":
        expected_letter = (expected or "").strip().upper()
        got_letter = _extract_choice(response)
        if got_letter is None:
            return ScoreResult(0.0, "no choice letter (A/B/C/D) in response")
        if got_letter == expected_letter:
            return ScoreResult(1.0, f"choice match: {got_letter}")
        return ScoreResult(0.0, f"expected {expected_letter}, got {got_letter}")

    return ScoreResult(0.0, f"unknown match_type {match!r}")


# ---- tool use ------------------------------------------------------------


def score_tool_use(
    prompt: BenchmarkPrompt,
    tool_calls: Iterable[Any] | None,
) -> Optional[ScoreResult]:
    """Grade a tool-use prompt.

    Pass criteria:
      1. At least one tool call has ``name == expected_tool``.
      2. For each ``(arg_name, accepted_values)`` in ``expected_args_contains``
         that matching call's arg (case-insensitively stringified) contains
         at least one of the accepted substrings.

    Returns ``None`` if the prompt has no tool expectation.
    """
    if not prompt.expected_tool:
        return None

    calls = normalize_tool_calls(tool_calls)

    if not calls:
        return ScoreResult(0.0, f"no tool calls; expected {prompt.expected_tool!r}")

    matching = [c for c in calls if c["name"] == prompt.expected_tool]
    if not matching:
        names = ", ".join(c["name"] for c in calls)
        return ScoreResult(
            0.0,
            f"expected tool {prompt.expected_tool!r}, got [{names}]",
        )

    requirements = prompt.expected_args_contains or {}
    if not requirements:
        return ScoreResult(1.0, f"called {prompt.expected_tool}")

    # Find the best-matching call (one that satisfies most requirements).
    best_satisfied = -1
    best_rationale = ""
    for call in matching:
        args = call.get("arguments") or {}
        ok = 0
        failed_keys: list[str] = []
        for key, accepted in requirements.items():
            value = args.get(key)
            value_str = str(value).lower() if value is not None else ""
            if any(needle.lower() in value_str for needle in accepted):
                ok += 1
            else:
                failed_keys.append(key)
        if ok > best_satisfied:
            best_satisfied = ok
            if ok == len(requirements):
                best_rationale = f"all {ok} arg checks passed"
            else:
                missing = ", ".join(failed_keys)
                best_rationale = (
                    f"satisfied {ok}/{len(requirements)} arg checks; missing: {missing}"
                )

    if best_satisfied == len(requirements):
        return ScoreResult(1.0, best_rationale)
    fraction = best_satisfied / max(len(requirements), 1)
    return ScoreResult(fraction, best_rationale)


# ---- dispatch ------------------------------------------------------------


def score_prompt(
    prompt: BenchmarkPrompt,
    response: str = "",
    tool_calls: Iterable[Any] | None = None,
) -> Optional[ScoreResult]:
    """Pick the appropriate rule-based grader for the prompt.

    Returns ``None`` when no rule-based scoring applies — the caller should
    use LLM-as-judge or manual scoring.
    """
    if prompt.category == "tool_use" or prompt.expected_tool:
        return score_tool_use(prompt, tool_calls)
    if prompt.category == "reasoning" or (prompt.expected and prompt.match_type):
        return score_reasoning(prompt, response)
    return None
