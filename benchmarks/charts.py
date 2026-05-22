"""Data aggregation + matplotlib plot factories for benchmark results.

The module is intentionally Qt-free — figures are produced as ``matplotlib.figure.Figure``
objects which the UI layer wraps in ``FigureCanvasQTAgg``.  This separation
keeps the heavy plotting logic unit-testable on a non-display ``Agg`` backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

# Use the Agg backend by default; the UI module switches to QtAgg explicitly.
import matplotlib

if matplotlib.get_backend().lower() not in {"qtagg", "agg"}:
    matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from benchmarks.constants import (
    CATEGORY_SPEED,
    RADAR_CATEGORIES,
)
from benchmarks.quantization import quantization_bits
from benchmarks.storage import BenchmarkResult


# ---------------------------------------------------------------------------
# Aggregation — pure functions; trivially testable
# ---------------------------------------------------------------------------


@dataclass
class QuantizationPoint:
    model: str
    quant: str
    bits: float
    quality_pct: float       # 0–100, averaged judge/manual/pass_rate
    tokens_per_sec: float


def _valid(results: Iterable[BenchmarkResult]) -> list[BenchmarkResult]:
    """Filter out errored / placeholder rows so means/maxes are not skewed."""
    return [r for r in results if not r.error]


def aggregate_tokens_per_sec(results: Iterable[BenchmarkResult]) -> dict[str, float]:
    """Mean tokens/sec per model. Models with no successful runs are omitted."""
    buckets: dict[str, list[float]] = {}
    for r in _valid(results):
        if r.tokens_per_sec is None:
            continue
        buckets.setdefault(r.model_name, []).append(r.tokens_per_sec)
    return {m: sum(v) / len(v) for m, v in buckets.items() if v}


def aggregate_ttft(results: Iterable[BenchmarkResult]) -> dict[str, float]:
    """Mean TTFT (ms) per model."""
    buckets: dict[str, list[float]] = {}
    for r in _valid(results):
        if r.ttft_ms is None:
            continue
        buckets.setdefault(r.model_name, []).append(r.ttft_ms)
    return {m: sum(v) / len(v) for m, v in buckets.items() if v}


def aggregate_vram_peak(results: Iterable[BenchmarkResult]) -> dict[str, int]:
    """Highest VRAM peak (MB) observed per model across all its prompts."""
    out: dict[str, int] = {}
    for r in _valid(results):
        if r.vram_peak_mb is None:
            continue
        out[r.model_name] = max(out.get(r.model_name, 0), int(r.vram_peak_mb))
    return out


def _quality_score(r: BenchmarkResult) -> Optional[float]:
    """Map a single result to a 0–100 quality score.

    Priority:
      1. ``manual_score`` (1–5 → 0–100)  — human-reviewed wins
      2. ``llm_judge_score`` (1–5 → 0–100)
      3. ``pass_rate`` (0–1 → 0–100) — rule-based for reasoning / tool_use
    """
    if r.manual_score is not None:
        return (float(r.manual_score) - 1) * 25.0
    if r.llm_judge_score is not None:
        return (float(r.llm_judge_score) - 1) * 25.0
    if r.pass_rate is not None:
        return float(r.pass_rate) * 100.0
    return None


def aggregate_category_scores(
    results: Iterable[BenchmarkResult],
) -> dict[str, dict[str, float]]:
    """Per-model average quality (0–100) for each category.

    Speed is treated separately: instead of a quality score we normalise each
    model's mean tokens/sec against the fastest model in the set.
    """
    results = list(_valid(results))

    # Quality categories (manual / judge / pass_rate).
    per_model_cat: dict[str, dict[str, list[float]]] = {}
    for r in results:
        if not r.category:
            continue
        q = _quality_score(r)
        if q is None:
            continue
        per_model_cat.setdefault(r.model_name, {}).setdefault(r.category, []).append(q)

    scores: dict[str, dict[str, float]] = {}
    for model, cats in per_model_cat.items():
        scores[model] = {cat: sum(v) / len(v) for cat, v in cats.items() if v}

    # Speed: normalise tokens/sec across all models to 0–100.
    tps = aggregate_tokens_per_sec(results)
    if tps:
        max_tps = max(tps.values()) or 1.0
        for model, v in tps.items():
            scores.setdefault(model, {})[CATEGORY_SPEED] = (v / max_tps) * 100.0

    return scores


def quantization_curve(results: Iterable[BenchmarkResult]) -> list[QuantizationPoint]:
    """Build the data series for the quantization trade-off line chart.

    One point per model. ``quality_pct`` is the model's mean quality across
    *all* prompts that produced a score (judge/manual/pass_rate).
    """
    results = list(_valid(results))
    tps = aggregate_tokens_per_sec(results)

    model_quality: dict[str, list[float]] = {}
    for r in results:
        q = _quality_score(r)
        if q is not None:
            model_quality.setdefault(r.model_name, []).append(q)

    points: list[QuantizationPoint] = []
    from benchmarks.quantization import parse_quantization
    for model, qs in model_quality.items():
        quant = parse_quantization(model) or "default"
        bits = quantization_bits(quant)
        if bits is None and quant == "default":
            bits = 4.8  # Ollama's typical default — see quantization._quant_sort_key
        if bits is None:
            continue
        if model not in tps:
            continue
        points.append(QuantizationPoint(
            model=model, quant=quant, bits=bits,
            quality_pct=sum(qs) / len(qs),
            tokens_per_sec=tps[model],
        ))
    points.sort(key=lambda p: p.bits)
    return points


# ---------------------------------------------------------------------------
# Plot factories — take aggregated data, return a Figure
# ---------------------------------------------------------------------------


def make_bar_figure(
    data: dict[str, float],
    title: str,
    ylabel: str,
    color: str = "#2563eb",
    decimals: int = 1,
) -> Figure:
    fig = Figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    if not data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes,
                fontsize=14, color="#888")
        ax.axis("off")
        return fig

    items = sorted(data.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    values = [float(v) for _, v in items]
    bars = ax.bar(names, values, color=color, edgecolor="#1e40af")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.{decimals}f}",
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    return fig


def make_radar_figure(
    model_scores: dict[str, dict[str, float]],
    categories: tuple[str, ...] = RADAR_CATEGORIES,
    category_labels: Optional[dict[str, str]] = None,
) -> Figure:
    fig = Figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="polar")
    if not model_scores:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="#888")
        ax.axis("off")
        return fig

    angles = np.linspace(0, 2 * math.pi, len(categories), endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    palette = ["#2563eb", "#16a34a", "#dc2626", "#a855f7", "#f59e0b", "#0891b2"]
    for i, (model, scores) in enumerate(sorted(model_scores.items())):
        values = [float(scores.get(cat, 0.0)) for cat in categories]
        values_closed = values + values[:1]
        color = palette[i % len(palette)]
        ax.plot(angles_closed, values_closed, label=model, color=color, linewidth=2)
        ax.fill(angles_closed, values_closed, alpha=0.12, color=color)

    ax.set_xticks(angles)
    labels = [(category_labels or {}).get(c, c) for c in categories]
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20", "40", "60", "80"], fontsize=8)
    ax.set_rlabel_position(180 / len(categories))
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), fontsize=9)
    fig.tight_layout()
    return fig


def make_quantization_figure(points: list[QuantizationPoint]) -> Figure:
    fig = Figure(figsize=(8, 5))
    ax_q = fig.add_subplot(111)
    if not points:
        ax_q.text(0.5, 0.5, "No data", ha="center", va="center",
                  transform=ax_q.transAxes, fontsize=14, color="#888")
        ax_q.axis("off")
        return fig

    bits = [p.bits for p in points]
    quality = [p.quality_pct for p in points]
    tps = [p.tokens_per_sec for p in points]

    ax_q.plot(bits, quality, "o-", color="#2563eb", linewidth=2, label="Quality (%)")
    ax_q.set_xlabel("Bits per weight")
    ax_q.set_ylabel("Quality (0–100)", color="#2563eb")
    ax_q.set_ylim(0, 100)
    ax_q.tick_params(axis="y", labelcolor="#2563eb")

    ax_tps = ax_q.twinx()
    ax_tps.plot(bits, tps, "s--", color="#f59e0b", linewidth=2, label="Tokens/sec")
    ax_tps.set_ylabel("Tokens/sec", color="#f59e0b")
    ax_tps.tick_params(axis="y", labelcolor="#f59e0b")

    # Annotate each point with its quant tag.
    for p in points:
        ax_q.annotate(
            p.quant, xy=(p.bits, p.quality_pct),
            xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=8, color="#1f2937",
        )

    # Compose a combined legend.
    lines_q, labels_q = ax_q.get_legend_handles_labels()
    lines_t, labels_t = ax_tps.get_legend_handles_labels()
    ax_q.legend(lines_q + lines_t, labels_q + labels_t, loc="lower right")
    ax_q.set_title("Quality vs Speed across quantizations")
    fig.tight_layout()
    return fig


def make_quality_vs_speed_scatter(
    model_scores: dict[str, dict[str, float]],
    tps_by_model: dict[str, float],
    overall_quality_by_model: Optional[dict[str, float]] = None,
) -> Figure:
    """Scatter plot: each model = one point at (tokens/sec, mean quality)."""
    fig = Figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    quality_by_model: dict[str, float] = {}
    if overall_quality_by_model is not None:
        quality_by_model = dict(overall_quality_by_model)
    else:
        for model, cats in model_scores.items():
            quality_categories = [v for k, v in cats.items() if k != "speed"]
            if quality_categories:
                quality_by_model[model] = sum(quality_categories) / len(quality_categories)

    if not quality_by_model or not tps_by_model:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="#888")
        ax.axis("off")
        return fig

    palette = ["#2563eb", "#16a34a", "#dc2626", "#a855f7", "#f59e0b", "#0891b2"]
    for i, model in enumerate(sorted(quality_by_model.keys())):
        if model not in tps_by_model:
            continue
        x = tps_by_model[model]
        y = quality_by_model[model]
        color = palette[i % len(palette)]
        ax.scatter(x, y, s=110, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(
            model, xy=(x, y), xytext=(6, 6),
            textcoords="offset points", fontsize=9,
        )

    ax.set_xlabel("Tokens/sec (mean)")
    ax.set_ylabel("Quality (0–100, mean of non-speed categories)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.set_title("Quality vs Speed")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience helper for "save figure as PNG"
# ---------------------------------------------------------------------------


def save_figure(figure: Figure, path, dpi: int = 150) -> None:
    figure.savefig(str(path), dpi=dpi, bbox_inches="tight")
