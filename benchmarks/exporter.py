"""Benchmark export helpers — CSV, JSON, aggregated summary, and the
all-in-one bundle that the UI's "Export All" button produces.

Designed for inclusion in diploma reports:
  * ``results.csv``     — raw row-per-(model, prompt) data
  * ``results.json``    — same as JSON for round-tripping in code
  * ``aggregated.csv``  — per-model summary stats (mean TPS, mean TTFT, peak
                           VRAM, per-category mean quality). Easy to paste
                           into LaTeX via ``pandas.read_csv(...).to_latex()``
                           or to import into Excel/Word.
  * ``summary.md``      — markdown report with the metadata, the aggregated
                           table, links to the saved charts, and notable
                           judge rationales.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from benchmarks.charts import (
    aggregate_category_scores,
    aggregate_tokens_per_sec,
    aggregate_ttft,
    aggregate_vram_peak,
    save_figure,
)
from benchmarks.storage import BenchmarkResult, BenchmarkRun


# ---------------------------------------------------------------------------
# Raw row-level exports
# ---------------------------------------------------------------------------


_CSV_FIELDS = [
    "run_id",
    "started_at",
    "prompt_set",
    "model_name",
    "prompt_id",
    "category",
    "ttft_ms",
    "tokens_per_sec",
    "total_time_ms",
    "output_tokens",
    "vram_peak_mb",
    "pass_rate",
    "llm_judge_score",
    "manual_score",
    "expected",
    "match_type",
    "expected_tool",
    "error",
    "output_text",
]


def export_csv(
    path: Path,
    run: BenchmarkRun,
    results: Iterable[BenchmarkResult],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "run_id": run.id,
                "started_at": run.started_at,
                "prompt_set": run.prompt_set,
                "model_name": r.model_name,
                "prompt_id": r.prompt_id,
                "category": r.category or "",
                "ttft_ms": r.ttft_ms,
                "tokens_per_sec": r.tokens_per_sec,
                "total_time_ms": r.total_time_ms,
                "output_tokens": r.output_tokens,
                "vram_peak_mb": r.vram_peak_mb,
                "pass_rate": r.pass_rate,
                "llm_judge_score": r.llm_judge_score,
                "manual_score": r.manual_score,
                "expected": r.expected or "",
                "match_type": r.match_type or "",
                "expected_tool": r.expected_tool or "",
                "error": r.error or "",
                "output_text": (r.output_text or "").replace("\n", "\\n"),
            })


def export_json(
    path: Path,
    run: BenchmarkRun,
    results: Iterable[BenchmarkResult],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run": {
            "id": run.id,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "prompt_set": run.prompt_set,
            "run_type": run.run_type,
            "notes": run.notes,
        },
        "results": [_result_dict(r) for r in results],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _result_dict(r: BenchmarkResult) -> dict[str, Any]:
    return {
        "model_name": r.model_name,
        "prompt_id": r.prompt_id,
        "category": r.category,
        "ttft_ms": r.ttft_ms,
        "tokens_per_sec": r.tokens_per_sec,
        "total_time_ms": r.total_time_ms,
        "output_tokens": r.output_tokens,
        "vram_peak_mb": r.vram_peak_mb,
        "pass_rate": r.pass_rate,
        "llm_judge_score": r.llm_judge_score,
        "llm_judge_rationale": r.llm_judge_rationale,
        "manual_score": r.manual_score,
        "expected": r.expected,
        "match_type": r.match_type,
        "expected_tool": r.expected_tool,
        "tool_calls": r.tool_calls,
        "output_text": r.output_text,
        "error": r.error,
    }


# ---------------------------------------------------------------------------
# Aggregated per-model CSV
# ---------------------------------------------------------------------------


_CATEGORIES = ("speed", "quality_ua", "code", "reasoning", "tool_use")
_AGG_FIELDS = [
    "model",
    "mean_ttft_ms",
    "mean_tokens_per_sec",
    "peak_vram_mb",
    "score_speed",
    "score_quality_ua",
    "score_code",
    "score_reasoning",
    "score_tool_use",
    "n_results",
    "n_errors",
]


def aggregated_rows(results: list[BenchmarkResult]) -> list[dict[str, Any]]:
    """Compute per-model summary rows. Output is ready for CSV / LaTeX."""
    by_model: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        by_model.setdefault(r.model_name, []).append(r)

    tps = aggregate_tokens_per_sec(results)
    ttft = aggregate_ttft(results)
    vram = aggregate_vram_peak(results)
    cat_scores = aggregate_category_scores(results)

    rows: list[dict[str, Any]] = []
    for model in sorted(by_model.keys()):
        bucket = by_model[model]
        row: dict[str, Any] = {
            "model": model,
            "mean_ttft_ms": _round(ttft.get(model), 1),
            "mean_tokens_per_sec": _round(tps.get(model), 2),
            "peak_vram_mb": vram.get(model),
            "n_results": len(bucket),
            "n_errors": sum(1 for r in bucket if r.error),
        }
        scores = cat_scores.get(model, {})
        for cat in _CATEGORIES:
            row[f"score_{cat}"] = _round(scores.get(cat), 1)
        rows.append(row)
    return rows


def export_aggregated_csv(path: Path, results: list[BenchmarkResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = aggregated_rows(results)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_AGG_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in _AGG_FIELDS})


def _round(value: Optional[float], decimals: int) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), decimals)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def export_summary_markdown(
    path: Path,
    run: BenchmarkRun,
    results: list[BenchmarkResult],
    chart_files: Optional[dict[str, str]] = None,
) -> None:
    """Write a human-readable summary suitable for diploma inclusion.

    ``chart_files`` maps a chart key to a filename **relative to** the summary
    file's directory. Recognised keys: ``tps``, ``ttft``, ``vram``, ``radar``,
    ``scatter``, ``quantization``. Missing keys are simply omitted from the
    report.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chart_files = chart_files or {}

    lines: list[str] = []
    lines.append(f"# Benchmark run #{run.id}")
    lines.append("")
    lines.append(f"- **Started:** {run.started_at}")
    lines.append(f"- **Finished:** {run.finished_at or '—'}")
    lines.append(f"- **Prompt set:** `{run.prompt_set}`")
    lines.append(f"- **Run type:** `{run.run_type}`")
    if run.notes:
        lines.append(f"- **Notes:** {run.notes}")
    lines.append(f"- **Results:** {len(results)}")
    n_errors = sum(1 for r in results if r.error)
    if n_errors:
        lines.append(f"- **Errors:** {n_errors}")
    lines.append("")

    lines.append("## Per-model summary")
    lines.append("")
    lines.append("| Model | TTFT (ms) | tok/s | VRAM (MB) | Speed | UA | Code | Reasoning | Tool Use | Errors |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in aggregated_rows(results):
        lines.append(
            "| {model} | {ttft} | {tps} | {vram} | {speed} | {ua} | {code} | {reason} | {tool} | {err} |".format(
                model=row["model"],
                ttft=_fmt(row.get("mean_ttft_ms")),
                tps=_fmt(row.get("mean_tokens_per_sec")),
                vram=_fmt(row.get("peak_vram_mb")),
                speed=_fmt(row.get("score_speed")),
                ua=_fmt(row.get("score_quality_ua")),
                code=_fmt(row.get("score_code")),
                reason=_fmt(row.get("score_reasoning")),
                tool=_fmt(row.get("score_tool_use")),
                err=row.get("n_errors", 0),
            )
        )
    lines.append("")

    if chart_files:
        lines.append("## Charts")
        lines.append("")
        for key, title in (
            ("tps", "Tokens/sec"),
            ("ttft", "TTFT (ms)"),
            ("vram", "VRAM peak (MB)"),
            ("radar", "Quality by category"),
            ("scatter", "Quality vs Speed"),
            ("quantization", "Quantization: quality vs speed"),
        ):
            filename = chart_files.get(key)
            if not filename:
                continue
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"![{title}]({filename})")
            lines.append("")

    rationales = _notable_rationales(results)
    if rationales:
        lines.append("## Notable judge rationales")
        lines.append("")
        for category, items in rationales.items():
            lines.append(f"### {category}")
            lines.append("")
            for r in items:
                score = r.llm_judge_score if r.llm_judge_score is not None else r.manual_score
                lines.append(
                    f"- **{r.model_name}** on `{r.prompt_id}` "
                    f"(score {score}): {r.llm_judge_rationale or '—'}"
                )
            lines.append("")

    lines.append("## Data")
    lines.append("")
    lines.append("- [results.csv](results.csv) — raw row-per-prompt data")
    lines.append("- [results.json](results.json) — same data in JSON")
    lines.append("- [aggregated.csv](aggregated.csv) — per-model summary stats")

    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(v: Any) -> str:
    if v is None or v == "":
        return "—"
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def _notable_rationales(
    results: list[BenchmarkResult],
    per_category: int = 3,
) -> dict[str, list[BenchmarkResult]]:
    """Pick best/worst-scored results per category for the report."""
    by_cat: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        if not r.category or not r.llm_judge_rationale:
            continue
        by_cat.setdefault(r.category, []).append(r)

    notable: dict[str, list[BenchmarkResult]] = {}
    for cat, items in by_cat.items():
        scored = [r for r in items if r.llm_judge_score is not None]
        if not scored:
            continue
        scored.sort(key=lambda r: r.llm_judge_score, reverse=True)
        # Top & bottom — up to ``per_category`` each, de-duplicated.
        picks: list[BenchmarkResult] = []
        seen: set[int] = set()
        for r in scored[:per_category] + scored[-per_category:]:
            if r.id in seen:
                continue
            picks.append(r)
            seen.add(r.id)
        if picks:
            notable[cat] = picks
    return notable


# ---------------------------------------------------------------------------
# Bundle: full "Export All" pipeline
# ---------------------------------------------------------------------------


@dataclass
class ExportBundle:
    """Files produced by ``export_run_bundle``. All paths are absolute."""

    target_dir: Path
    results_csv: Path
    results_json: Path
    aggregated_csv: Path
    summary_md: Path
    charts: dict[str, Path]


def export_run_bundle(
    target_dir: Path,
    run: BenchmarkRun,
    results: list[BenchmarkResult],
    figures: Optional[dict[str, Any]] = None,
) -> ExportBundle:
    """Render a complete export bundle into ``target_dir``.

    ``figures`` maps chart keys to ``matplotlib.figure.Figure`` instances.
    Each figure is saved as ``<key>.png``. Recognised keys:
    ``tps``, ``ttft``, ``vram``, ``radar``, ``scatter``, ``quantization``.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    results_csv = target / "results.csv"
    results_json = target / "results.json"
    aggregated_csv = target / "aggregated.csv"
    summary_md = target / "summary.md"

    export_csv(results_csv, run, results)
    export_json(results_json, run, results)
    export_aggregated_csv(aggregated_csv, results)

    chart_paths: dict[str, Path] = {}
    chart_files: dict[str, str] = {}
    for key, figure in (figures or {}).items():
        if figure is None:
            continue
        out = target / f"{key}.png"
        save_figure(figure, out)
        chart_paths[key] = out
        chart_files[key] = out.name

    export_summary_markdown(summary_md, run, results, chart_files=chart_files)

    return ExportBundle(
        target_dir=target,
        results_csv=results_csv,
        results_json=results_json,
        aggregated_csv=aggregated_csv,
        summary_md=summary_md,
        charts=chart_paths,
    )
