from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.storage import BenchmarkStorage
from memory.storage import Storage


@pytest.fixture
def storage(tmp_db_path: Path):
    s = Storage(tmp_db_path)
    yield BenchmarkStorage(s.connection)
    s.close()


def test_create_and_list_run(storage: BenchmarkStorage) -> None:
    run_id = storage.create_run(prompt_set="speed", notes="first")
    assert run_id > 0
    runs = storage.list_runs()
    assert len(runs) == 1
    assert runs[0].prompt_set == "speed"
    assert runs[0].notes == "first"
    assert runs[0].finished_at is None


def test_finish_run_sets_timestamp(storage: BenchmarkStorage) -> None:
    run_id = storage.create_run(prompt_set="code")
    storage.finish_run(run_id)
    assert storage.get_run(run_id).finished_at is not None


def test_add_result_and_fetch(storage: BenchmarkStorage) -> None:
    run_id = storage.create_run(prompt_set="speed")
    storage.add_result(
        run_id=run_id, model_name="m1", prompt_id="p1",
        ttft_ms=120.5, tokens_per_sec=25.0, total_time_ms=2000.0,
        output_tokens=50, output_text="hello", error=None,
    )
    storage.add_result(
        run_id=run_id, model_name="m2", prompt_id="p1",
        ttft_ms=80.0, tokens_per_sec=40.0, total_time_ms=1200.0,
        output_tokens=40, output_text="hi", error=None,
    )
    results = storage.results_for_run(run_id)
    assert len(results) == 2
    assert {r.model_name for r in results} == {"m1", "m2"}


def test_delete_run_cascades(storage: BenchmarkStorage) -> None:
    run_id = storage.create_run(prompt_set="code")
    storage.add_result(
        run_id=run_id, model_name="m", prompt_id="p",
        ttft_ms=1.0, tokens_per_sec=1.0, total_time_ms=1.0,
        output_tokens=1, output_text="x", error=None,
    )
    storage.delete_run(run_id)
    assert storage.get_run(run_id) is None
    assert storage.results_for_run(run_id) == []


def test_list_runs_ordered_desc(storage: BenchmarkStorage) -> None:
    a = storage.create_run(prompt_set="speed")
    b = storage.create_run(prompt_set="code")
    runs = storage.list_runs()
    assert [r.id for r in runs] == [b, a]


def test_store_error_result(storage: BenchmarkStorage) -> None:
    run_id = storage.create_run(prompt_set="speed")
    storage.add_result(
        run_id=run_id, model_name="m", prompt_id="p",
        ttft_ms=None, tokens_per_sec=None, total_time_ms=500.0,
        output_tokens=None, output_text="", error="timeout",
    )
    results = storage.results_for_run(run_id)
    assert results[0].error == "timeout"
    assert results[0].ttft_ms is None


# ---- v3: scoring + resources ---------------------------------------------


def test_run_type_default_standard(storage: BenchmarkStorage) -> None:
    run_id = storage.create_run(prompt_set="speed")
    assert storage.get_run(run_id).run_type == "standard"


def test_run_type_quantization(storage: BenchmarkStorage) -> None:
    run_id = storage.create_run(prompt_set="speed", run_type="quantization")
    assert storage.get_run(run_id).run_type == "quantization"


def test_add_result_with_scoring_fields(storage: BenchmarkStorage) -> None:
    run_id = storage.create_run(prompt_set="reasoning")
    storage.add_result(
        run_id=run_id, model_name="m", prompt_id="r1",
        category="reasoning",
        ttft_ms=100.0, tokens_per_sec=20.0, total_time_ms=1500.0,
        output_tokens=30, output_text="42",
        pass_rate=1.0, vram_peak_mb=8200,
        expected="42", match_type="exact",
    )
    r = storage.results_for_run(run_id)[0]
    assert r.category == "reasoning"
    assert r.pass_rate == 1.0
    assert r.vram_peak_mb == 8200
    assert r.expected == "42"
    assert r.match_type == "exact"


def test_add_result_with_tool_use_fields(storage: BenchmarkStorage) -> None:
    run_id = storage.create_run(prompt_set="tool_use")
    storage.add_result(
        run_id=run_id, model_name="m", prompt_id="t1",
        category="tool_use",
        ttft_ms=100.0, tokens_per_sec=20.0, total_time_ms=500.0,
        output_tokens=5, output_text="",
        expected_tool="notes_search",
        expected_args={"query": ["python", "Python"]},
        tool_calls=[{"name": "notes_search", "arguments": {"query": "python"}}],
        pass_rate=1.0,
    )
    r = storage.results_for_run(run_id)[0]
    assert r.expected_tool == "notes_search"
    assert r.expected_args == {"query": ["python", "Python"]}
    assert r.tool_calls == [{"name": "notes_search", "arguments": {"query": "python"}}]


def test_update_judge_score(storage: BenchmarkStorage) -> None:
    run_id = storage.create_run(prompt_set="quality_ua")
    rid = storage.add_result(
        run_id=run_id, model_name="m", prompt_id="ua1",
        output_text="відповідь",
    )
    storage.update_judge_score(rid, score=4.5, rationale="природна мова, грамотна")
    r = storage.results_for_run(run_id)[0]
    assert r.llm_judge_score == 4.5
    assert "грамотна" in r.llm_judge_rationale


def test_update_manual_score(storage: BenchmarkStorage) -> None:
    run_id = storage.create_run(prompt_set="quality_ua")
    rid = storage.add_result(
        run_id=run_id, model_name="m", prompt_id="ua1",
        output_text="x",
    )
    storage.update_manual_score(rid, score=4)
    r = storage.results_for_run(run_id)[0]
    assert r.manual_score == 4


def test_update_manual_score_to_none_clears(storage: BenchmarkStorage) -> None:
    run_id = storage.create_run(prompt_set="quality_ua")
    rid = storage.add_result(run_id=run_id, model_name="m", prompt_id="p")
    storage.update_manual_score(rid, score=5)
    storage.update_manual_score(rid, score=None)
    assert storage.results_for_run(run_id)[0].manual_score is None
