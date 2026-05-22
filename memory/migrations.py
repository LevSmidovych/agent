from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Callable

from core.exceptions import MigrationError

logger = logging.getLogger(__name__)


def _migrate_v0_to_v1(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            profile_name TEXT NOT NULL,
            model_name TEXT NOT NULL
        );

        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tool_name TEXT,
            tool_args TEXT,
            tool_result TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        );

        CREATE INDEX idx_messages_conv ON messages(conversation_id);
        """
    )


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            prompt_set TEXT NOT NULL,
            notes TEXT
        );

        CREATE TABLE benchmark_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            prompt_id TEXT NOT NULL,
            ttft_ms REAL,
            tokens_per_sec REAL,
            total_time_ms REAL,
            output_tokens INTEGER,
            output_text TEXT,
            error TEXT,
            FOREIGN KEY (run_id) REFERENCES benchmark_runs(id)
        );

        CREATE INDEX idx_bench_results_run ON benchmark_results(run_id);
        """
    )


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    """Extend benchmark_results with auto/manual scoring and resource metrics."""
    conn.executescript(
        """
        ALTER TABLE benchmark_results ADD COLUMN category TEXT;
        ALTER TABLE benchmark_results ADD COLUMN pass_rate REAL;
        ALTER TABLE benchmark_results ADD COLUMN llm_judge_score REAL;
        ALTER TABLE benchmark_results ADD COLUMN llm_judge_rationale TEXT;
        ALTER TABLE benchmark_results ADD COLUMN manual_score INTEGER;
        ALTER TABLE benchmark_results ADD COLUMN vram_peak_mb INTEGER;
        ALTER TABLE benchmark_results ADD COLUMN expected TEXT;
        ALTER TABLE benchmark_results ADD COLUMN match_type TEXT;
        ALTER TABLE benchmark_results ADD COLUMN expected_tool TEXT;
        ALTER TABLE benchmark_results ADD COLUMN expected_args_json TEXT;
        ALTER TABLE benchmark_results ADD COLUMN tool_calls_json TEXT;
        ALTER TABLE benchmark_runs ADD COLUMN run_type TEXT DEFAULT 'standard';
        """
    )


MIGRATIONS: list[tuple[int, Callable[[sqlite3.Connection], None]]] = [
    (1, _migrate_v0_to_v1),
    (2, _migrate_v1_to_v2),
    (3, _migrate_v2_to_v3),
]


def _ensure_version_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )


def _current_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return row[0] or 0


def apply_migrations(conn: sqlite3.Connection) -> int:
    _ensure_version_table(conn)
    current = _current_version(conn)
    applied = 0
    for target_version, migration in MIGRATIONS:
        if target_version <= current:
            continue
        logger.info("applying migration v%d", target_version)
        try:
            with conn:
                migration(conn)
                conn.execute(
                    "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
                    (target_version, datetime.now(timezone.utc).isoformat()),
                )
        except sqlite3.Error as exc:
            raise MigrationError(f"migration v{target_version} failed: {exc}") from exc
        applied += 1
    return applied
