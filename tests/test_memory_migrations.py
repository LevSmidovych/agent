from __future__ import annotations

import sqlite3

import pytest

from core.exceptions import MigrationError
from memory import migrations


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys = ON")
    yield c
    c.close()


def test_apply_migrations_on_fresh_db(conn: sqlite3.Connection) -> None:
    applied = migrations.apply_migrations(conn)
    assert applied == len(migrations.MIGRATIONS)

    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {
        "conversations", "messages", "schema_version",
        "benchmark_runs", "benchmark_results",
    }.issubset(tables)
    # v3 columns must exist on benchmark_results
    bench_cols = {r[1] for r in conn.execute("PRAGMA table_info(benchmark_results)")}
    assert {
        "category", "pass_rate", "llm_judge_score", "llm_judge_rationale",
        "manual_score", "vram_peak_mb", "expected", "match_type",
        "expected_tool", "expected_args_json", "tool_calls_json",
    }.issubset(bench_cols)
    runs_cols = {r[1] for r in conn.execute("PRAGMA table_info(benchmark_runs)")}
    assert "run_type" in runs_cols


def test_apply_migrations_is_idempotent(conn: sqlite3.Connection) -> None:
    migrations.apply_migrations(conn)
    applied_again = migrations.apply_migrations(conn)
    assert applied_again == 0


def test_schema_version_recorded(conn: sqlite3.Connection) -> None:
    migrations.apply_migrations(conn)
    versions = [r[0] for r in conn.execute("SELECT version FROM schema_version ORDER BY version")]
    assert versions == [v for v, _ in migrations.MIGRATIONS]


def test_migration_failure_wraps_in_migration_error(monkeypatch, conn: sqlite3.Connection) -> None:
    def boom(c: sqlite3.Connection) -> None:
        c.execute("CREATE TABLE x (id INTEGER)")
        c.execute("CREATE TABLE x (id INTEGER)")  # duplicate -> error

    monkeypatch.setattr(migrations, "MIGRATIONS", [(99, boom)])
    with pytest.raises(MigrationError):
        migrations.apply_migrations(conn)
