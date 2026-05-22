from __future__ import annotations

import io

import pytest
from matplotlib.figure import Figure

from benchmarks.charts import (
    QuantizationPoint,
    aggregate_category_scores,
    aggregate_tokens_per_sec,
    aggregate_ttft,
    aggregate_vram_peak,
    make_bar_figure,
    make_quality_vs_speed_scatter,
    make_quantization_figure,
    make_radar_figure,
    quantization_curve,
    save_figure,
    _quality_score,
)
from benchmarks.storage import BenchmarkResult


# ---- result factory ------------------------------------------------------


def _result(
    model: str,
    prompt_id: str,
    *,
    ttft: float | None = 100.0,
    tps: float | None = 30.0,
    total: float | None = 1500.0,
    output_tokens: int | None = 45,
    error: str | None = None,
    pass_rate: float | None = None,
    llm_judge_score: float | None = None,
    manual_score: int | None = None,
    vram_peak_mb: int | None = None,
    category: str | None = "speed",
) -> BenchmarkResult:
    return BenchmarkResult(
        id=0, run_id=1, model_name=model, prompt_id=prompt_id,
        ttft_ms=ttft, tokens_per_sec=tps, total_time_ms=total,
        output_tokens=output_tokens, output_text="", error=error,
        category=category, pass_rate=pass_rate,
        llm_judge_score=llm_judge_score, manual_score=manual_score,
        vram_peak_mb=vram_peak_mb,
    )


# ---- aggregations --------------------------------------------------------


def test_aggregate_tokens_per_sec_mean_per_model() -> None:
    rs = [
        _result("A", "p1", tps=20.0),
        _result("A", "p2", tps=40.0),
        _result("B", "p1", tps=10.0),
    ]
    assert aggregate_tokens_per_sec(rs) == {"A": 30.0, "B": 10.0}


def test_aggregate_tokens_per_sec_ignores_errored_rows() -> None:
    rs = [
        _result("A", "p1", tps=20.0),
        _result("A", "p2", tps=None, error="boom"),
    ]
    assert aggregate_tokens_per_sec(rs) == {"A": 20.0}


def test_aggregate_tokens_per_sec_skips_models_with_no_data() -> None:
    rs = [_result("A", "p1", tps=None)]
    assert aggregate_tokens_per_sec(rs) == {}


def test_aggregate_ttft_mean_per_model() -> None:
    rs = [
        _result("A", "p1", ttft=200.0),
        _result("A", "p2", ttft=400.0),
        _result("B", "p1", ttft=None),
    ]
    assert aggregate_ttft(rs) == {"A": 300.0}


def test_aggregate_vram_peak_takes_max() -> None:
    rs = [
        _result("A", "p1", vram_peak_mb=5000),
        _result("A", "p2", vram_peak_mb=8200),
        _result("B", "p1", vram_peak_mb=3000),
    ]
    assert aggregate_vram_peak(rs) == {"A": 8200, "B": 3000}


def test_aggregate_vram_peak_skips_none() -> None:
    rs = [_result("A", "p1", vram_peak_mb=None)]
    assert aggregate_vram_peak(rs) == {}


# ---- quality score mapping -----------------------------------------------


def test_quality_score_manual_takes_priority() -> None:
    r = _result("A", "p", llm_judge_score=4.0, manual_score=2, pass_rate=1.0)
    assert _quality_score(r) == 25.0  # (2-1)*25


def test_quality_score_judge_when_no_manual() -> None:
    r = _result("A", "p", llm_judge_score=5.0, pass_rate=0.5)
    assert _quality_score(r) == 100.0  # (5-1)*25


def test_quality_score_pass_rate_fallback() -> None:
    r = _result("A", "p", pass_rate=0.8)
    assert _quality_score(r) == 80.0


def test_quality_score_none_when_no_signals() -> None:
    r = _result("A", "p")
    assert _quality_score(r) is None


# ---- category aggregation ------------------------------------------------


def test_aggregate_category_scores_quality_categories() -> None:
    rs = [
        _result("A", "ua1", category="quality_ua", llm_judge_score=5.0),
        _result("A", "ua2", category="quality_ua", llm_judge_score=3.0),
        _result("A", "r1", category="reasoning", pass_rate=1.0),
        _result("A", "tool1", category="tool_use", pass_rate=0.5),
    ]
    scores = aggregate_category_scores(rs)
    # judge=5 → 100, judge=3 → 50; mean = 75
    assert scores["A"]["quality_ua"] == 75.0
    assert scores["A"]["reasoning"] == 100.0
    assert scores["A"]["tool_use"] == 50.0


def test_aggregate_category_scores_speed_normalized_to_fastest() -> None:
    rs = [
        _result("A", "s1", category="speed", tps=40.0),
        _result("B", "s1", category="speed", tps=20.0),
    ]
    scores = aggregate_category_scores(rs)
    assert scores["A"]["speed"] == 100.0
    assert scores["B"]["speed"] == 50.0


def test_aggregate_category_scores_handles_empty() -> None:
    assert aggregate_category_scores([]) == {}


# ---- quantization curve --------------------------------------------------


def test_quantization_curve_builds_sorted_points() -> None:
    rs = [
        _result("qwen2.5:14b-instruct-q8_0", "p1", tps=20.0, pass_rate=0.8, category="reasoning"),
        _result("qwen2.5:14b-instruct-q4_K_M", "p1", tps=40.0, pass_rate=0.6, category="reasoning"),
        _result("qwen2.5:14b-instruct-fp16", "p1", tps=10.0, pass_rate=0.9, category="reasoning"),
    ]
    points = quantization_curve(rs)
    bits = [p.bits for p in points]
    assert bits == sorted(bits)  # ordered by bits ascending
    assert {p.quant for p in points} == {"q4_k_m", "q8_0", "fp16"}


def test_quantization_curve_omits_models_without_quality() -> None:
    rs = [_result("qwen2.5:14b-instruct-q8_0", "p1", tps=20.0, category="speed")]
    # speed-only result has no quality score → no curve point
    assert quantization_curve(rs) == []


def test_quantization_curve_default_quant_keeps_point() -> None:
    rs = [_result("qwen2.5:14b", "p1", tps=30.0, pass_rate=0.5, category="reasoning")]
    points = quantization_curve(rs)
    assert len(points) == 1
    assert points[0].quant == "default"


# ---- plot factories — smoke tests ---------------------------------------


def test_make_bar_figure_returns_figure() -> None:
    fig = make_bar_figure({"A": 10, "B": 20}, title="TPS", ylabel="tok/s")
    assert isinstance(fig, Figure)


def test_make_bar_figure_empty_data_no_crash() -> None:
    fig = make_bar_figure({}, title="empty", ylabel="x")
    assert isinstance(fig, Figure)


def test_make_radar_figure_returns_figure() -> None:
    fig = make_radar_figure({
        "A": {"speed": 80, "quality_ua": 60, "code": 70, "reasoning": 90, "tool_use": 50},
        "B": {"speed": 60, "quality_ua": 80},
    })
    assert isinstance(fig, Figure)


def test_make_radar_figure_empty_data_no_crash() -> None:
    fig = make_radar_figure({})
    assert isinstance(fig, Figure)


def test_make_quantization_figure_returns_figure() -> None:
    points = [
        QuantizationPoint("m-q4_K_M", "q4_k_m", 4.8, 70.0, 50.0),
        QuantizationPoint("m-q8_0", "q8_0", 8.5, 85.0, 30.0),
        QuantizationPoint("m-fp16", "fp16", 16.0, 90.0, 18.0),
    ]
    fig = make_quantization_figure(points)
    assert isinstance(fig, Figure)


def test_make_quantization_figure_empty_no_crash() -> None:
    fig = make_quantization_figure([])
    assert isinstance(fig, Figure)


def test_make_scatter_returns_figure() -> None:
    fig = make_quality_vs_speed_scatter(
        model_scores={"A": {"quality_ua": 80, "code": 60}, "B": {"quality_ua": 60}},
        tps_by_model={"A": 30.0, "B": 50.0},
    )
    assert isinstance(fig, Figure)


def test_make_scatter_empty_no_crash() -> None:
    fig = make_quality_vs_speed_scatter({}, {})
    assert isinstance(fig, Figure)


def test_save_figure_writes_png_bytes(tmp_path) -> None:
    fig = make_bar_figure({"A": 10}, title="t", ylabel="y")
    path = tmp_path / "out.png"
    save_figure(fig, path)
    assert path.exists()
    assert path.stat().st_size > 100  # something non-trivial got written


def test_save_figure_to_bytesio() -> None:
    fig = make_bar_figure({"A": 10}, title="t", ylabel="y")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    assert buf.getbuffer().nbytes > 100
