"""Multi-model benchmark runner.

Drives generation across a list of models × prompts sequentially. Captures
TTFT, throughput, total time, output size, VRAM peak, and (for tool-use
prompts) the tool calls the model emitted *without executing them* — we are
grading whether the model can pick the right tool, not its side effects.

Quality grading happens in two passes:

* Rule-based scoring (``benchmarks.scoring``) is applied inline for
  reasoning and tool_use prompts — the result lands in ``pass_rate``.
* LLM-as-judge (``benchmarks.judge``) runs as a separate batch phase after
  generation completes, so the judge model is loaded into VRAM only once.

Resilience:

* Each model gets a *warmup* prompt whose result is discarded — avoids the
  "first call loads model into VRAM" outlier dominating TTFT/tokens_per_sec.
* If three consecutive prompts on the same model produce an error, the
  remaining prompts on that model are skipped and the runner moves on.
* Between models we ask Ollama to unload the previous one (``keep_alive=0``)
  so VRAM measurements for the next model start from a clean baseline.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from benchmarks.constants import CATEGORY_TOOL_USE, RUN_TYPE_STANDARD
from benchmarks.prompts import BenchmarkPrompt
from benchmarks.resources import VRAMMonitor
from benchmarks.scoring import score_prompt
from benchmarks.storage import BenchmarkStorage
from core.exceptions import OllamaConnectionError
from core.llm_client import LLMClient
from core.ollama_parsing import (
    get_content as _ollama_content,
    get_eval_count as _ollama_eval_count,
    get_raw_tool_calls,
    normalize_tool_calls,
)

logger = logging.getLogger(__name__)


# ---- public dataclasses ---------------------------------------------------


@dataclass
class BenchmarkConfig:
    models: list[str]
    prompts: list[BenchmarkPrompt]
    notes: Optional[str] = None
    warmup: bool = True
    max_consecutive_errors: int = 3
    tool_schemas: Optional[list[dict]] = None  # for tool_use prompts
    run_type: str = RUN_TYPE_STANDARD
    prompt_set_label: Optional[str] = None  # override the auto-derived label


@dataclass
class GenerationResult:
    model_name: str
    prompt: BenchmarkPrompt
    ttft_ms: Optional[float]
    tokens_per_sec: Optional[float]
    total_time_ms: Optional[float]
    output_tokens: Optional[int]
    output_text: str
    tool_calls: list[dict]
    vram_peak_mb: Optional[int]
    error: Optional[str]
    pass_rate: Optional[float] = None
    rule_rationale: Optional[str] = None


@dataclass
class GenerationProgress:
    model_index: int
    model_count: int
    model_name: str
    prompt_index: int
    prompt_count: int
    prompt_id: str
    overall_index: int   # 0-based across the whole run
    overall_total: int
    is_warmup: bool = False


# ---- runner --------------------------------------------------------------


class MultiModelRunner:
    def __init__(
        self,
        llm: LLMClient,
        storage: BenchmarkStorage,
        vram_monitor: Optional[VRAMMonitor] = None,
    ) -> None:
        self._llm = llm
        self._storage = storage
        self._vram = vram_monitor

    def run(
        self,
        config: BenchmarkConfig,
        progress: Optional[Callable[[GenerationProgress], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> int:
        if not config.models:
            raise ValueError("at least one model is required")
        if not config.prompts:
            raise ValueError("at least one prompt is required")

        run_id = self._storage.create_run(
            prompt_set=self._derive_prompt_set_label(config),
            notes=config.notes,
            run_type=config.run_type,
        )
        try:
            self._run_models(run_id, config, progress, stop_event)
        finally:
            self._storage.finish_run(run_id)
        return run_id

    def _derive_prompt_set_label(self, config: BenchmarkConfig) -> str:
        if config.prompt_set_label:
            return config.prompt_set_label
        cats = sorted({p.category for p in config.prompts})
        return ",".join(cats)

    def _run_models(
        self,
        run_id: int,
        config: BenchmarkConfig,
        progress: Optional[Callable[[GenerationProgress], None]],
        stop_event: Optional[threading.Event],
    ) -> None:
        total_pairs = len(config.models) * len(config.prompts)
        overall_idx = 0
        last_model: Optional[str] = None
        last_prompt: Optional[BenchmarkPrompt] = None
        for model_idx, model in enumerate(config.models):
            if stop_event is not None and stop_event.is_set():
                return

            # Optional warmup — pick the first prompt, discard result.
            if config.warmup:
                warmup_prompt = config.prompts[0]
                if progress:
                    progress(GenerationProgress(
                        model_index=model_idx, model_count=len(config.models),
                        model_name=model,
                        prompt_index=-1, prompt_count=len(config.prompts),
                        prompt_id=warmup_prompt.id,
                        overall_index=overall_idx, overall_total=total_pairs,
                        is_warmup=True,
                    ))
                self._run_one(model, warmup_prompt, config, stop_event)

            consecutive_errors = 0
            for prompt_idx, prompt in enumerate(config.prompts):
                if stop_event is not None and stop_event.is_set():
                    return
                if consecutive_errors >= config.max_consecutive_errors:
                    logger.warning(
                        "skipping rest of %s after %d consecutive errors",
                        model, consecutive_errors,
                    )
                    overall_idx += (len(config.prompts) - prompt_idx)
                    break

                if progress:
                    progress(GenerationProgress(
                        model_index=model_idx, model_count=len(config.models),
                        model_name=model,
                        prompt_index=prompt_idx, prompt_count=len(config.prompts),
                        prompt_id=prompt.id,
                        overall_index=overall_idx, overall_total=total_pairs,
                    ))

                # On the last prompt of this model, ask Ollama to unload right
                # after the call. This is a cleaner unload than a separate
                # generate(keep_alive=0) for streaming-only setups.
                last_for_model = prompt_idx == len(config.prompts) - 1
                keep_alive = 0 if last_for_model else None

                result = self._run_one(
                    model, prompt, config, stop_event, keep_alive=keep_alive,
                )

                if result.error:
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0

                self._persist(run_id, result)
                overall_idx += 1
                last_model, last_prompt = model, prompt

            # Belt-and-suspenders: if the model errored out and we never sent
            # a keep_alive=0 chat (broke out before the last prompt), unload
            # explicitly so the next model starts with clean VRAM.
            if consecutive_errors >= config.max_consecutive_errors:
                self._llm.unload_model(model)

        # Emit a final 100% progress update so the UI doesn't sit at
        # (total-1)/total. Use the last completed (model, prompt) for context.
        if progress is not None and last_prompt is not None:
            progress(GenerationProgress(
                model_index=len(config.models) - 1,
                model_count=len(config.models),
                model_name=last_model or "",
                prompt_index=len(config.prompts) - 1,
                prompt_count=len(config.prompts),
                prompt_id=last_prompt.id,
                overall_index=total_pairs,
                overall_total=total_pairs,
            ))

    # ---- single call ------------------------------------------------------

    def _run_one(
        self,
        model: str,
        prompt: BenchmarkPrompt,
        config: BenchmarkConfig,
        stop_event: Optional[threading.Event],
        keep_alive: Optional[int] = None,
    ) -> GenerationResult:
        messages = [{"role": "user", "content": prompt.prompt}]
        tools = config.tool_schemas if prompt.category == CATEGORY_TOOL_USE else None

        if self._vram is not None:
            self._vram.reset_peak()

        start = time.perf_counter()
        ttft: Optional[float] = None
        text_chunks: list[str] = []
        chunk_count = 0
        eval_count: Optional[int] = None
        tool_calls: list[dict] = []

        try:
            for chunk in self._llm.chat_raw_stream(
                model=model,
                messages=messages,
                tools=tools,
                stop_event=stop_event,
                keep_alive=keep_alive,
            ):
                if stop_event is not None and stop_event.is_set():
                    break
                if ttft is None:
                    ttft = time.perf_counter() - start

                content = _ollama_content(chunk)
                if content:
                    text_chunks.append(content)
                    chunk_count += 1

                raw_calls = get_raw_tool_calls(chunk)
                if raw_calls:
                    tool_calls.extend(normalize_tool_calls(raw_calls))

                ec = _ollama_eval_count(chunk)
                if ec is not None:
                    eval_count = ec
        except OllamaConnectionError as exc:
            total = (time.perf_counter() - start) * 1000
            vram = self._vram.peak_mb if self._vram is not None else None
            return GenerationResult(
                model_name=model, prompt=prompt,
                ttft_ms=None, tokens_per_sec=None, total_time_ms=total,
                output_tokens=None, output_text="", tool_calls=[],
                vram_peak_mb=vram, error=str(exc),
            )

        total_sec = time.perf_counter() - start
        output_text = "".join(text_chunks)
        output_tokens = eval_count if eval_count is not None else chunk_count
        ttft_sec = ttft if ttft is not None else 0.0
        gen_sec = max(total_sec - ttft_sec, 1e-6)
        tps = output_tokens / gen_sec if output_tokens else 0.0
        vram = self._vram.peak_mb if self._vram is not None else None

        result = GenerationResult(
            model_name=model, prompt=prompt,
            ttft_ms=ttft_sec * 1000,
            tokens_per_sec=tps,
            total_time_ms=total_sec * 1000,
            output_tokens=output_tokens,
            output_text=output_text,
            tool_calls=tool_calls,
            vram_peak_mb=vram,
            error=None,
        )

        # Inline rule-based grading.
        grade = score_prompt(prompt, response=output_text, tool_calls=tool_calls)
        if grade is not None:
            result.pass_rate = grade.pass_rate
            result.rule_rationale = grade.rationale
        return result

    # ---- persistence ------------------------------------------------------

    def _persist(self, run_id: int, result: GenerationResult) -> None:
        prompt = result.prompt
        self._storage.add_result(
            run_id=run_id,
            model_name=result.model_name,
            prompt_id=prompt.id,
            category=prompt.category,
            ttft_ms=result.ttft_ms,
            tokens_per_sec=result.tokens_per_sec,
            total_time_ms=result.total_time_ms,
            output_tokens=result.output_tokens,
            output_text=result.output_text,
            error=result.error,
            pass_rate=result.pass_rate,
            vram_peak_mb=result.vram_peak_mb,
            expected=prompt.expected,
            match_type=prompt.match_type,
            expected_tool=prompt.expected_tool,
            expected_args=prompt.expected_args_contains,
            tool_calls=result.tool_calls or None,
        )


