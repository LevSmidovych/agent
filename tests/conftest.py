from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Silence ChromaDB's anonymized telemetry — background threads can race with
# rapid fixture setup/teardown and cause flaky test failures.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")

# Force matplotlib to use the non-interactive Agg backend before any test
# imports a Figure. Without this, importing matplotlib.pyplot anywhere in
# the test tree spins up a Qt event loop and blocks headless CI runs.
import matplotlib  # noqa: E402
matplotlib.use("Agg", force=True)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests that require a live Ollama instance",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip_marker = pytest.mark.skip(reason="needs --run-integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"
