from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchmarks.prompts import BenchmarkPrompt
from benchmarks.runner import (
    BenchmarkConfig,
    GenerationProgress,
    MultiModelRunner,
)
from benchmarks.storage import BenchmarkStorage
from core.exceptions import OllamaConnectionError
from memory.storage import Storage


# ---- fixtures -------------------------------------------------------------


@pytest.fixture
def storage(tmp_db_path: Path):
    s = Storage(tmp_db_path)
    yield BenchmarkStorage(s.connection)
    s.close()


def _stream(chunks):
    return iter(chunks)


def _make_llm(stream_responses):
    """``stream_responses`` is a list of lists. Each top-level entry will be
    returned for one ``chat_raw_stream`` call, in order.
    """
    llm = MagicMock()
    streams = list(stream_responses)

    def _next_stream(*args, **kwargs):
        if not streams:
            raise AssertionError("no more stub streams")
        return _stream(streams.pop(0))

    llm.chat_raw_stream = MagicMock(side_effect=_next_stream)
    llm.unload_model = MagicMock()
    return llm


def _speed_prompt(pid="p1", text="hi"):
    return BenchmarkPrompt(id=pid, prompt=text, category="speed")


def _reasoning_prompt(pid="r1", expected="42", match_type="number"):
    return BenchmarkPrompt(
        id=pid, prompt="What is the answer?", category="reasoning",
        expected=expected, match_type=match_type,
    )


def _tool_prompt(pid="t1", tool="notes_search", args=None):
    return BenchmarkPrompt(
        id=pid, prompt="Find Python notes", category="tool_use",
        expected_tool=tool, expected_args_contains=args or {"query": ["python"]},
    )


def _text(*texts):
    """Build a streaming response producing the given text chunks."""
    return [{"message": {"content": t}} for t in texts] + [{"eval_count": sum(len(t) for t in texts)}]


def _tool_call(name, args):
    return {"message": {"content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}}


# ---- basic runs -----------------------------------------------------------


def test_run_requires_models(storage) -> None:
    runner = MultiModelRunner(_make_llm([]), storage)
    with pytest.raises(ValueError):
        runner.run(BenchmarkConfig(models=[], prompts=[_speed_prompt()]))


def test_run_requires_prompts(storage) -> None:
    runner = MultiModelRunner(_make_llm([]), storage)
    with pytest.raises(ValueError):
        runner.run(BenchmarkConfig(models=["m"], prompts=[]))


def test_single_model_single_prompt(storage) -> None:
    # warmup + actual
    llm = _make_llm([_text("warm"), _text("hello")])
    runner = MultiModelRunner(llm, storage)
    run_id = runner.run(BenchmarkConfig(models=["m"], prompts=[_speed_prompt()]))
    results = storage.results_for_run(run_id)
    assert len(results) == 1, "warmup must not be recorded"
    assert results[0].output_text == "hello"
    assert results[0].model_name == "m"


def test_warmup_can_be_disabled(storage) -> None:
    llm = _make_llm([_text("hello")])
    runner = MultiModelRunner(llm, storage)
    run_id = runner.run(BenchmarkConfig(
        models=["m"], prompts=[_speed_prompt()], warmup=False,
    ))
    results = storage.results_for_run(run_id)
    assert len(results) == 1
    # only one chat_raw_stream call → no warmup
    assert llm.chat_raw_stream.call_count == 1


def test_two_models_two_prompts(storage) -> None:
    # 2 models × (1 warmup + 2 real) = 6 streams
    llm = _make_llm([
        _text("warm-a"), _text("a-p1"), _text("a-p2"),
        _text("warm-b"), _text("b-p1"), _text("b-p2"),
    ])
    runner = MultiModelRunner(llm, storage)
    run_id = runner.run(BenchmarkConfig(
        models=["mA", "mB"],
        prompts=[_speed_prompt("p1"), _speed_prompt("p2")],
    ))
    results = storage.results_for_run(run_id)
    assert len(results) == 4
    by_model = {(r.model_name, r.prompt_id): r.output_text for r in results}
    assert by_model[("mA", "p1")] == "a-p1"
    assert by_model[("mA", "p2")] == "a-p2"
    assert by_model[("mB", "p1")] == "b-p1"
    assert by_model[("mB", "p2")] == "b-p2"


# ---- keep_alive between models -------------------------------------------


def test_last_prompt_per_model_sets_keep_alive_zero(storage) -> None:
    llm = _make_llm([
        _text("warm"), _text("a1"), _text("a2"),  # model A
        _text("warm"), _text("b1"), _text("b2"),  # model B
    ])
    runner = MultiModelRunner(llm, storage)
    runner.run(BenchmarkConfig(
        models=["A", "B"],
        prompts=[_speed_prompt("p1"), _speed_prompt("p2")],
    ))
    calls = llm.chat_raw_stream.call_args_list
    keep_alives = [c.kwargs.get("keep_alive") for c in calls]
    # warmup (None), p1 (None), p2 (0) → for each model
    assert keep_alives == [None, None, 0, None, None, 0]


# ---- error handling -------------------------------------------------------


def test_three_consecutive_errors_skips_remaining_prompts(storage) -> None:
    err = OllamaConnectionError("boom")

    def stream_factory(*args, **kwargs):
        raise err

    llm = MagicMock()
    llm.chat_raw_stream = MagicMock(side_effect=stream_factory)
    llm.unload_model = MagicMock()

    runner = MultiModelRunner(llm, storage)
    prompts = [_speed_prompt(f"p{i}") for i in range(5)]
    run_id = runner.run(BenchmarkConfig(
        models=["m"], prompts=prompts, warmup=False,
    ))
    results = storage.results_for_run(run_id)
    # After 3 consecutive errors, remaining 2 prompts should be skipped.
    assert len(results) == 3
    assert all(r.error for r in results)
    # Model should be force-unloaded after errors.
    llm.unload_model.assert_called_once_with("m")


def test_error_counter_resets_on_success(storage) -> None:
    # 1 success, 2 errors, 1 success, 1 success → no skip, all recorded
    calls = []

    def factory(*args, **kwargs):
        idx = len(calls)
        calls.append(idx)
        if idx in (1, 2):  # second and third call error
            raise OllamaConnectionError("transient")
        return _stream(_text(f"ok-{idx}"))

    llm = MagicMock()
    llm.chat_raw_stream = MagicMock(side_effect=factory)
    llm.unload_model = MagicMock()

    runner = MultiModelRunner(llm, storage)
    prompts = [_speed_prompt(f"p{i}") for i in range(4)]
    run_id = runner.run(BenchmarkConfig(
        models=["m"], prompts=prompts, warmup=False,
    ))
    results = storage.results_for_run(run_id)
    assert len(results) == 4
    errors = [r for r in results if r.error]
    assert len(errors) == 2
    llm.unload_model.assert_not_called()


# ---- inline scoring -------------------------------------------------------


def test_reasoning_prompt_gets_pass_rate(storage) -> None:
    llm = _make_llm([_text("warm"), _text("The answer is 42")])
    runner = MultiModelRunner(llm, storage)
    run_id = runner.run(BenchmarkConfig(
        models=["m"], prompts=[_reasoning_prompt()],
    ))
    results = storage.results_for_run(run_id)
    assert results[0].pass_rate == 1.0
    assert results[0].category == "reasoning"
    assert results[0].expected == "42"
    assert results[0].match_type == "number"


def test_reasoning_wrong_answer_pass_rate_zero(storage) -> None:
    llm = _make_llm([_text("warm"), _text("I think 99")])
    runner = MultiModelRunner(llm, storage)
    run_id = runner.run(BenchmarkConfig(
        models=["m"], prompts=[_reasoning_prompt(expected="42", match_type="number")],
    ))
    results = storage.results_for_run(run_id)
    assert results[0].pass_rate == 0.0


def test_speed_prompt_has_no_pass_rate(storage) -> None:
    llm = _make_llm([_text("warm"), _text("hello")])
    runner = MultiModelRunner(llm, storage)
    run_id = runner.run(BenchmarkConfig(
        models=["m"], prompts=[_speed_prompt()],
    ))
    results = storage.results_for_run(run_id)
    assert results[0].pass_rate is None


# ---- tool_use capture (no execution) -------------------------------------


def test_tool_use_captures_calls_without_execution(storage) -> None:
    llm = _make_llm([
        _text("warm"),
        [_tool_call("notes_search", {"query": "python tips"})],
    ])
    runner = MultiModelRunner(llm, storage)
    run_id = runner.run(BenchmarkConfig(
        models=["m"], prompts=[_tool_prompt()],
        tool_schemas=[{"type": "function", "function": {"name": "notes_search"}}],
    ))
    results = storage.results_for_run(run_id)
    assert results[0].tool_calls == [{"name": "notes_search", "arguments": {"query": "python tips"}}]
    assert results[0].pass_rate == 1.0
    assert results[0].expected_tool == "notes_search"


def test_tool_use_wrong_tool_pass_rate_zero(storage) -> None:
    llm = _make_llm([
        _text("warm"),
        [_tool_call("notes_create", {"title": "x"})],
    ])
    runner = MultiModelRunner(llm, storage)
    run_id = runner.run(BenchmarkConfig(
        models=["m"], prompts=[_tool_prompt()],
        tool_schemas=[{"type": "function", "function": {"name": "notes_search"}}],
    ))
    results = storage.results_for_run(run_id)
    assert results[0].pass_rate == 0.0


# ---- VRAM monitor integration --------------------------------------------


def test_vram_monitor_records_peak_per_prompt(storage) -> None:
    llm = _make_llm([_text("warm"), _text("hi"), _text("yo")])
    monitor = MagicMock()
    # peak_mb is a property — make it return increasing values
    peaks = iter([0, 5120, 6000, 0, 7200])
    type(monitor).peak_mb = property(lambda self: next(peaks))
    runner = MultiModelRunner(llm, storage, vram_monitor=monitor)
    run_id = runner.run(BenchmarkConfig(
        models=["m"],
        prompts=[_speed_prompt("p1"), _speed_prompt("p2")],
    ))
    # reset_peak called: before warmup, before p1, before p2 = 3 times
    assert monitor.reset_peak.call_count >= 3
    results = storage.results_for_run(run_id)
    vrams = [r.vram_peak_mb for r in results]
    assert all(v is not None for v in vrams)


# ---- stop_event ----------------------------------------------------------


def test_stop_event_interrupts_run(storage) -> None:
    llm = _make_llm([
        _text("warm"), _text("ok-1"),
        _text("ok-2"),  # extra in case needed
    ])
    runner = MultiModelRunner(llm, storage)
    stop = threading.Event()

    progress_calls = []

    def progress(p):
        progress_calls.append(p)
        if p.prompt_index >= 0 and not p.is_warmup:
            stop.set()  # stop right after first real prompt is reported

    run_id = runner.run(
        BenchmarkConfig(models=["m"], prompts=[_speed_prompt("p1"), _speed_prompt("p2"), _speed_prompt("p3")]),
        progress=progress,
        stop_event=stop,
    )
    results = storage.results_for_run(run_id)
    assert len(results) <= 2  # interrupted before all prompts


# ---- progress callback ---------------------------------------------------


def test_progress_callback_emits_correct_indices(storage) -> None:
    llm = _make_llm([_text("warm"), _text("a"), _text("b")])
    runner = MultiModelRunner(llm, storage)
    events: list[GenerationProgress] = []
    runner.run(
        BenchmarkConfig(models=["m"], prompts=[_speed_prompt("p1"), _speed_prompt("p2")]),
        progress=events.append,
    )
    # warmup + p1 + p2 + final-100% = 4 events
    assert events[0].is_warmup is True
    assert events[1].is_warmup is False and events[1].prompt_index == 0
    assert events[2].prompt_index == 1
    # The last event must report overall_index == overall_total so the UI
    # progress bar reaches 100% (no off-by-one at the end of a run).
    assert events[-1].overall_index == events[-1].overall_total == 2


# ---- run record ----------------------------------------------------------


def test_run_is_finished(storage) -> None:
    llm = _make_llm([_text("warm"), _text("ok")])
    runner = MultiModelRunner(llm, storage)
    run_id = runner.run(BenchmarkConfig(models=["m"], prompts=[_speed_prompt()]))
    run = storage.get_run(run_id)
    assert run.finished_at is not None


def test_run_type_propagated(storage) -> None:
    llm = _make_llm([_text("warm"), _text("ok")])
    runner = MultiModelRunner(llm, storage)
    run_id = runner.run(BenchmarkConfig(
        models=["m"], prompts=[_speed_prompt()],
        run_type="quantization",
    ))
    assert storage.get_run(run_id).run_type == "quantization"
