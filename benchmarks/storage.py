from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BenchmarkRun:
    id: int
    started_at: str
    finished_at: Optional[str]
    prompt_set: str
    notes: Optional[str]
    run_type: str = "standard"  # "standard" | "quantization"


@dataclass
class BenchmarkResult:
    id: int
    run_id: int
    model_name: str
    prompt_id: str
    ttft_ms: Optional[float]
    tokens_per_sec: Optional[float]
    total_time_ms: Optional[float]
    output_tokens: Optional[int]
    output_text: Optional[str]
    error: Optional[str]
    category: Optional[str] = None
    pass_rate: Optional[float] = None
    llm_judge_score: Optional[float] = None
    llm_judge_rationale: Optional[str] = None
    manual_score: Optional[int] = None
    vram_peak_mb: Optional[int] = None
    expected: Optional[str] = None
    match_type: Optional[str] = None
    expected_tool: Optional[str] = None
    expected_args: Optional[dict] = None
    tool_calls: Optional[list] = None


class BenchmarkStorage:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection

    # --- runs --------------------------------------------------------------

    def create_run(
        self,
        prompt_set: str,
        notes: Optional[str] = None,
        run_type: str = "standard",
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO benchmark_runs(started_at, prompt_set, notes, run_type) VALUES (?, ?, ?, ?)",
            (_now(), prompt_set, notes, run_type),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int) -> None:
        self._conn.execute(
            "UPDATE benchmark_runs SET finished_at = ? WHERE id = ?",
            (_now(), run_id),
        )
        self._conn.commit()

    def list_runs(self, limit: int = 100) -> list[BenchmarkRun]:
        rows = self._conn.execute(
            "SELECT * FROM benchmark_runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_run(r) for r in rows]

    def get_run(self, run_id: int) -> Optional[BenchmarkRun]:
        row = self._conn.execute(
            "SELECT * FROM benchmark_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return _row_to_run(row) if row else None

    def delete_run(self, run_id: int) -> None:
        self._conn.execute("DELETE FROM benchmark_results WHERE run_id = ?", (run_id,))
        self._conn.execute("DELETE FROM benchmark_runs WHERE id = ?", (run_id,))
        self._conn.commit()

    # --- results -----------------------------------------------------------

    def add_result(
        self,
        run_id: int,
        model_name: str,
        prompt_id: str,
        *,
        category: Optional[str] = None,
        ttft_ms: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
        total_time_ms: Optional[float] = None,
        output_tokens: Optional[int] = None,
        output_text: Optional[str] = None,
        error: Optional[str] = None,
        pass_rate: Optional[float] = None,
        vram_peak_mb: Optional[int] = None,
        expected: Optional[str] = None,
        match_type: Optional[str] = None,
        expected_tool: Optional[str] = None,
        expected_args: Optional[dict] = None,
        tool_calls: Optional[list] = None,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO benchmark_results(
                run_id, model_name, prompt_id,
                ttft_ms, tokens_per_sec, total_time_ms, output_tokens,
                output_text, error,
                category, pass_rate, vram_peak_mb,
                expected, match_type,
                expected_tool, expected_args_json, tool_calls_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, model_name, prompt_id,
                ttft_ms, tokens_per_sec, total_time_ms, output_tokens,
                output_text, error,
                category, pass_rate, vram_peak_mb,
                expected, match_type,
                expected_tool,
                json.dumps(expected_args) if expected_args is not None else None,
                json.dumps(tool_calls) if tool_calls is not None else None,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def results_for_run(self, run_id: int) -> list[BenchmarkResult]:
        rows = self._conn.execute(
            "SELECT * FROM benchmark_results WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()
        return [_row_to_result(r) for r in rows]

    def update_judge_score(
        self,
        result_id: int,
        score: Optional[float],
        rationale: Optional[str] = None,
    ) -> None:
        self._conn.execute(
            "UPDATE benchmark_results SET llm_judge_score = ?, llm_judge_rationale = ? WHERE id = ?",
            (score, rationale, result_id),
        )
        self._conn.commit()

    def update_manual_score(self, result_id: int, score: Optional[int]) -> None:
        self._conn.execute(
            "UPDATE benchmark_results SET manual_score = ? WHERE id = ?",
            (score, result_id),
        )
        self._conn.commit()


def _row_to_run(row: sqlite3.Row) -> BenchmarkRun:
    keys = row.keys()
    return BenchmarkRun(
        id=row["id"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        prompt_set=row["prompt_set"],
        notes=row["notes"],
        run_type=row["run_type"] if "run_type" in keys else "standard",
    )


def _row_to_result(row: sqlite3.Row) -> BenchmarkResult:
    keys = row.keys()

    def opt(name: str) -> Any:
        return row[name] if name in keys else None

    expected_args_json = opt("expected_args_json")
    tool_calls_json = opt("tool_calls_json")
    return BenchmarkResult(
        id=row["id"],
        run_id=row["run_id"],
        model_name=row["model_name"],
        prompt_id=row["prompt_id"],
        ttft_ms=row["ttft_ms"],
        tokens_per_sec=row["tokens_per_sec"],
        total_time_ms=row["total_time_ms"],
        output_tokens=row["output_tokens"],
        output_text=row["output_text"],
        error=row["error"],
        category=opt("category"),
        pass_rate=opt("pass_rate"),
        llm_judge_score=opt("llm_judge_score"),
        llm_judge_rationale=opt("llm_judge_rationale"),
        manual_score=opt("manual_score"),
        vram_peak_mb=opt("vram_peak_mb"),
        expected=opt("expected"),
        match_type=opt("match_type"),
        expected_tool=opt("expected_tool"),
        expected_args=json.loads(expected_args_json) if expected_args_json else None,
        tool_calls=json.loads(tool_calls_json) if tool_calls_json else None,
    )
