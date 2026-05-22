from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmarks.exporter import (
    ExportBundle,
    aggregated_rows,
    export_aggregated_csv,
    export_csv,
    export_json,
    export_run_bundle,
    export_summary_markdown,
)
from benchmarks.storage import BenchmarkResult, BenchmarkRun


def _sample_run(run_type="standard") -> BenchmarkRun:
    return BenchmarkRun(
        id=1, started_at="2026-04-22T10:00:00+00:00",
        finished_at="2026-04-22T10:05:00+00:00",
        prompt_set="speed", notes="test", run_type=run_type,
    )


def _result(
    *,
    rid=1, model="m1", prompt_id="p1", category="speed",
    ttft=100.0, tps=30.0, total=1500.0, tokens=45,
    output="hello world", error=None,
    pass_rate=None, judge=None, judge_rationale=None, manual=None,
    vram=None, expected=None, match_type=None, expected_tool=None,
    tool_calls=None,
) -> BenchmarkResult:
    return BenchmarkResult(
        id=rid, run_id=1, model_name=model, prompt_id=prompt_id,
        ttft_ms=ttft, tokens_per_sec=tps, total_time_ms=total,
        output_tokens=tokens, output_text=output, error=error,
        category=category, pass_rate=pass_rate,
        llm_judge_score=judge, llm_judge_rationale=judge_rationale,
        manual_score=manual, vram_peak_mb=vram,
        expected=expected, match_type=match_type,
        expected_tool=expected_tool, tool_calls=tool_calls,
    )


# ---- raw CSV / JSON -------------------------------------------------------


def test_export_csv_writes_rows(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    export_csv(path, _sample_run(), [
        _result(model="m1", output="hello\nworld"),
        _result(rid=2, model="m2", ttft=None, tps=None, total=200.0,
                tokens=None, output="", error="timeout"),
    ])
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["model_name"] == "m1"
    assert rows[0]["output_text"] == "hello\\nworld"
    assert rows[1]["error"] == "timeout"


def test_export_csv_includes_v3_fields(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    export_csv(path, _sample_run(), [
        _result(category="reasoning", pass_rate=1.0, judge=4.5, manual=5,
                vram=8200, expected="42", match_type="number"),
    ])
    row = next(csv.DictReader(path.open(encoding="utf-8")))
    assert row["category"] == "reasoning"
    assert row["pass_rate"] == "1.0"
    assert row["llm_judge_score"] == "4.5"
    assert row["manual_score"] == "5"
    assert row["vram_peak_mb"] == "8200"
    assert row["expected"] == "42"
    assert row["match_type"] == "number"


def test_export_json_payload(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    export_json(path, _sample_run(), [
        _result(),
        _result(rid=2, model="m2", error="boom"),
    ])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run"]["id"] == 1
    assert data["run"]["run_type"] == "standard"
    assert len(data["results"]) == 2
    assert data["results"][1]["error"] == "boom"


def test_export_json_includes_tool_calls(tmp_path: Path) -> None:
    calls = [{"name": "notes_search", "arguments": {"query": "py"}}]
    path = tmp_path / "out.json"
    export_json(path, _sample_run(), [
        _result(category="tool_use", expected_tool="notes_search", tool_calls=calls),
    ])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["results"][0]["tool_calls"] == calls


def test_export_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "out.csv"
    export_csv(path, _sample_run(), [_result()])
    assert path.exists()


# ---- aggregated CSV ------------------------------------------------------


def test_aggregated_rows_per_model() -> None:
    results = [
        _result(model="A", category="speed", tps=40.0, ttft=80.0, vram=5000),
        _result(model="A", rid=2, category="reasoning",
                tps=None, ttft=None, pass_rate=1.0),
        _result(model="B", rid=3, category="speed", tps=20.0, ttft=150.0, vram=3000),
        _result(model="B", rid=4, category="reasoning",
                tps=None, ttft=None, pass_rate=0.5),
    ]
    rows = aggregated_rows(results)
    by_model = {r["model"]: r for r in rows}
    assert by_model["A"]["mean_tokens_per_sec"] == 40.0
    assert by_model["A"]["peak_vram_mb"] == 5000
    assert by_model["A"]["score_reasoning"] == 100.0
    assert by_model["B"]["score_reasoning"] == 50.0


def test_aggregated_rows_count_errors() -> None:
    results = [
        _result(model="X"),
        _result(model="X", rid=2, error="boom"),
        _result(model="X", rid=3, error="boom"),
    ]
    rows = aggregated_rows(results)
    assert rows[0]["n_results"] == 3
    assert rows[0]["n_errors"] == 2


def test_export_aggregated_csv(tmp_path: Path) -> None:
    path = tmp_path / "agg.csv"
    export_aggregated_csv(path, [
        _result(model="A", category="speed", tps=40.0, ttft=80.0, vram=5000),
        _result(model="B", category="speed", tps=20.0, ttft=150.0, vram=3000),
    ])
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["model"] == "A"
    assert rows[0]["score_speed"] == "100.0"  # normalised
    assert rows[1]["score_speed"] == "50.0"


def test_export_aggregated_csv_empty(tmp_path: Path) -> None:
    path = tmp_path / "agg.csv"
    export_aggregated_csv(path, [])
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows == []


# ---- summary markdown ---------------------------------------------------


def test_summary_markdown_includes_metadata(tmp_path: Path) -> None:
    path = tmp_path / "summary.md"
    export_summary_markdown(path, _sample_run(), [_result()])
    content = path.read_text(encoding="utf-8")
    assert "Benchmark run #1" in content
    assert "speed" in content
    assert "## Per-model summary" in content


def test_summary_markdown_lists_models_in_table(tmp_path: Path) -> None:
    path = tmp_path / "summary.md"
    export_summary_markdown(path, _sample_run(), [
        _result(model="qwen-A"),
        _result(rid=2, model="llama-B"),
    ])
    content = path.read_text(encoding="utf-8")
    assert "qwen-A" in content
    assert "llama-B" in content


def test_summary_markdown_with_charts(tmp_path: Path) -> None:
    path = tmp_path / "summary.md"
    export_summary_markdown(
        path, _sample_run(), [_result()],
        chart_files={"tps": "tps.png", "radar": "radar.png"},
    )
    content = path.read_text(encoding="utf-8")
    assert "![Tokens/sec](tps.png)" in content
    assert "![Quality by category](radar.png)" in content
    # Charts that weren't provided are not referenced
    assert "ttft.png" not in content


def test_summary_markdown_picks_notable_rationales(tmp_path: Path) -> None:
    path = tmp_path / "summary.md"
    results = [
        _result(rid=1, category="quality_ua", judge=5.0, judge_rationale="excellent"),
        _result(rid=2, category="quality_ua", judge=1.0, judge_rationale="bad"),
        _result(rid=3, category="quality_ua", judge=3.0, judge_rationale="meh"),
    ]
    export_summary_markdown(path, _sample_run(), results)
    content = path.read_text(encoding="utf-8")
    assert "Notable judge rationales" in content
    assert "excellent" in content
    assert "bad" in content


def test_summary_markdown_no_errors_section_when_clean(tmp_path: Path) -> None:
    path = tmp_path / "summary.md"
    export_summary_markdown(path, _sample_run(), [_result()])
    content = path.read_text(encoding="utf-8")
    assert "**Errors:**" not in content


def test_summary_markdown_shows_errors_count(tmp_path: Path) -> None:
    path = tmp_path / "summary.md"
    export_summary_markdown(path, _sample_run(), [
        _result(),
        _result(rid=2, error="boom"),
    ])
    content = path.read_text(encoding="utf-8")
    assert "**Errors:** 1" in content


# ---- bundle ---------------------------------------------------------------


def test_export_run_bundle_writes_all_files(tmp_path: Path) -> None:
    from benchmarks.charts import make_bar_figure

    target = tmp_path / "run_1"
    figs = {
        "tps": make_bar_figure({"A": 30}, title="t", ylabel="y"),
        "radar": make_bar_figure({"A": 80}, title="r", ylabel="y"),
    }
    bundle = export_run_bundle(target, _sample_run(), [_result()], figures=figs)

    assert isinstance(bundle, ExportBundle)
    assert bundle.target_dir == target
    assert bundle.results_csv.exists()
    assert bundle.results_json.exists()
    assert bundle.aggregated_csv.exists()
    assert bundle.summary_md.exists()
    assert set(bundle.charts.keys()) == {"tps", "radar"}
    for path in bundle.charts.values():
        assert path.exists()


def test_export_run_bundle_summary_links_charts(tmp_path: Path) -> None:
    from benchmarks.charts import make_bar_figure

    target = tmp_path / "run_x"
    bundle = export_run_bundle(target, _sample_run(), [_result()], figures={
        "tps": make_bar_figure({"A": 30}, title="t", ylabel="y"),
    })
    content = bundle.summary_md.read_text(encoding="utf-8")
    assert "tps.png" in content


def test_export_run_bundle_no_figures(tmp_path: Path) -> None:
    bundle = export_run_bundle(tmp_path / "run_y", _sample_run(), [_result()])
    assert bundle.charts == {}
    assert "## Charts" not in bundle.summary_md.read_text(encoding="utf-8")


def test_export_run_bundle_creates_target_dir(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "run"
    bundle = export_run_bundle(target, _sample_run(), [_result()])
    assert bundle.target_dir.exists()
