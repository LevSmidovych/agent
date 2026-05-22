"""LLM-as-judge for free-form benchmark prompts (speed / quality_ua / code).

We deliberately use a *local* Ollama model — typically the largest available
on the machine — to avoid sending data to external APIs. The judge model is
loaded into VRAM only once for the entire batch of evaluations (separate
phase after all generation completes), which avoids thrashing.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional

from core.llm_client import LLMClient
from core.exceptions import OllamaConnectionError
from benchmarks.prompts import BenchmarkPrompt

logger = logging.getLogger(__name__)


# Preference order — first match in available_models wins. The list reflects
# expected quality of judge output: bigger / more recent → better grading.
DEFAULT_JUDGE_PREFERENCES: tuple[str, ...] = (
    "qwen2.5:32b",
    "qwen2.5:14b",
    "mixtral:8x7b",
    "qwen2.5:7b",
    "llama3.1:70b",
    "mistral-nemo",
    "mistral:7b",
    "llama3.1:8b",
)


JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator of AI assistant responses.\n"
    "Given a user question, the rubric (expected behavior), and the model's "
    "response, score the response from 1 to 5:\n"
    " - 5: Excellent — fully correct, well-written, addresses everything\n"
    " - 4: Good — minor flaws but mostly correct and useful\n"
    " - 3: Acceptable — partially correct or missing some aspects\n"
    " - 2: Poor — major issues but some value\n"
    " - 1: Bad — incorrect or unhelpful\n\n"
    "Return STRICT JSON of the form:\n"
    '  {"score": <integer 1-5>, "rationale": "<one short sentence>"}\n'
    "Return nothing else, no preamble, no markdown."
)


@dataclass
class JudgeVerdict:
    score: Optional[float]
    rationale: str
    raw_response: str = ""
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.score is not None and self.error is None


@dataclass
class JudgeItem:
    """Bundle of what the batch judge needs: the prompt and the response to grade.

    ``result_id`` is propagated through so the caller can map verdicts back to
    the corresponding ``benchmark_results`` row.
    """

    result_id: int
    prompt: BenchmarkPrompt
    response: str


DEFAULT_JUDGE_TIMEOUT_SECONDS: float = 120.0


class LLMJudge:
    def __init__(
        self,
        llm: LLMClient,
        model: str,
        timeout_seconds: float = DEFAULT_JUDGE_TIMEOUT_SECONDS,
    ) -> None:
        self._llm = llm
        self._model = model
        self._timeout = timeout_seconds

    @property
    def model(self) -> str:
        return self._model

    @property
    def timeout_seconds(self) -> float:
        return self._timeout

    # ---- selection ---------------------------------------------------------

    @classmethod
    def select_model(
        cls,
        available_models: Iterable[str],
        preferences: Iterable[str] | None = None,
    ) -> Optional[str]:
        """Pick the best judge model from ``available_models`` by preference order.

        Falls back to the first available model if no preferred match is
        found. Returns ``None`` when the list is empty.
        """
        models = list(available_models)
        if not models:
            return None
        prefs = list(preferences or DEFAULT_JUDGE_PREFERENCES)
        for pref in prefs:
            for m in models:
                if m == pref or m.startswith(pref + ":") or m.startswith(pref + "-"):
                    return m
        return models[0]

    # ---- single grade ------------------------------------------------------

    def judge(self, prompt: BenchmarkPrompt, response: str) -> JudgeVerdict:
        messages = self._build_messages(prompt, response)
        try:
            content = self._llm.chat_json(
                model=self._model,
                messages=messages,
                timeout_seconds=self._timeout,
            )
        except OllamaConnectionError as exc:
            return JudgeVerdict(score=None, rationale="", error=str(exc))
        return parse_verdict(content)

    def _build_messages(self, prompt: BenchmarkPrompt, response: str) -> list[dict]:
        rubric = prompt.expected_behavior.strip() or "(no specific rubric provided)"
        user = (
            f"Question:\n{prompt.prompt}\n\n"
            f"Rubric / expected behavior:\n{rubric}\n\n"
            f"Model response:\n{response or '(empty)'}\n\n"
            "Score the model response according to the rubric."
        )
        return [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    # ---- batch -------------------------------------------------------------

    def judge_batch(
        self,
        items: Iterable[JudgeItem],
        stop_event: Optional[threading.Event] = None,
        progress: Optional[Callable[[int, int, JudgeItem], None]] = None,
    ) -> Iterator[tuple[JudgeItem, JudgeVerdict]]:
        items = list(items)
        total = len(items)
        for idx, item in enumerate(items):
            if stop_event is not None and stop_event.is_set():
                return
            if progress is not None:
                progress(idx, total, item)
            verdict = self.judge(item.prompt, item.response)
            yield item, verdict
        if progress is not None and total > 0:
            progress(total, total, items[-1])


# ---- JSON parsing --------------------------------------------------------


_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*?\}", re.MULTILINE)


def parse_verdict(content: str) -> JudgeVerdict:
    """Extract a verdict from the judge model's reply.

    Many local models emit a small preamble or wrap the JSON in markdown
    fences even when told not to. We try a strict parse first, then fall
    back to extracting the first ``{...}`` block.
    """
    raw = (content or "").strip()
    if not raw:
        return JudgeVerdict(score=None, rationale="", raw_response=raw, error="empty response")

    payload = _try_load(raw)
    if payload is None:
        match = _JSON_OBJECT_RE.search(raw)
        if match is not None:
            payload = _try_load(match.group(0))
    if payload is None:
        return JudgeVerdict(score=None, rationale="", raw_response=raw, error="could not parse JSON")

    score = _coerce_score(payload.get("score"))
    rationale_value = payload.get("rationale") or payload.get("reason") or ""
    rationale = str(rationale_value).strip()
    if score is None:
        return JudgeVerdict(score=None, rationale=rationale, raw_response=raw,
                            error="score missing or non-numeric")
    return JudgeVerdict(score=score, rationale=rationale, raw_response=raw)


def _try_load(text: str) -> Optional[dict]:
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _coerce_score(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _clamp(float(value))
    if isinstance(value, str):
        try:
            return _clamp(float(value.strip()))
        except ValueError:
            return None
    return None


def _clamp(value: float, lo: float = 1.0, hi: float = 5.0) -> float:
    return max(lo, min(hi, value))
