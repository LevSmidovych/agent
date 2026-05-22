from __future__ import annotations

import sys
import threading
import time
import types
from unittest.mock import MagicMock

import pytest

from benchmarks import resources


# ---- helpers --------------------------------------------------------------


def _install_fake_pynvml(monkeypatch, used_mb_values):
    """Install a fake pynvml module that returns predetermined VRAM usage.

    ``used_mb_values`` is either an int (constant) or a callable returning the
    next value to report.
    """
    fake = types.SimpleNamespace()
    fake.nvmlInit = MagicMock()
    fake.nvmlShutdown = MagicMock()
    fake.nvmlDeviceGetHandleByIndex = MagicMock(return_value="handle")

    if callable(used_mb_values):
        next_val = used_mb_values
    else:
        next_val = lambda: used_mb_values  # noqa: E731

    def mem_info(_handle):
        mb = next_val()
        return types.SimpleNamespace(used=mb * 1024 * 1024, total=24 * 1024 * 1024 * 1024)

    fake.nvmlDeviceGetMemoryInfo = MagicMock(side_effect=mem_info)
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    return fake


# ---- fallback behavior ----------------------------------------------------


def test_monitor_unavailable_when_pynvml_missing(monkeypatch) -> None:
    """If pynvml cannot be imported, the monitor stays in no-op mode."""

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    m = resources.VRAMMonitor()
    assert m.available is False
    assert m.sample() == 0
    assert m.peak_mb == 0
    m.start()  # no-op
    assert m.is_running is False
    m.stop()  # safe
    m.shutdown()  # safe


def test_monitor_unavailable_when_nvml_init_raises(monkeypatch) -> None:
    fake = _install_fake_pynvml(monkeypatch, 1000)
    fake.nvmlInit.side_effect = RuntimeError("no GPU")

    m = resources.VRAMMonitor()
    assert m.available is False
    assert m.sample() == 0


# ---- sampling -------------------------------------------------------------


def test_sample_returns_used_mb(monkeypatch) -> None:
    _install_fake_pynvml(monkeypatch, 5120)
    m = resources.VRAMMonitor()
    assert m.available is True
    assert m.sample() == 5120


def test_reset_peak_to_current_sample(monkeypatch) -> None:
    _install_fake_pynvml(monkeypatch, 4096)
    m = resources.VRAMMonitor()
    m.reset_peak()
    assert m.peak_mb == 4096


def test_sample_failure_returns_zero(monkeypatch) -> None:
    fake = _install_fake_pynvml(monkeypatch, 1000)
    m = resources.VRAMMonitor()
    fake.nvmlDeviceGetMemoryInfo.side_effect = RuntimeError("transient")
    assert m.sample() == 0


# ---- polling loop ---------------------------------------------------------


def test_polling_captures_peak(monkeypatch) -> None:
    samples = iter([1000, 5000, 3000, 8000, 4000] + [4000] * 100)
    _install_fake_pynvml(monkeypatch, lambda: next(samples))

    m = resources.VRAMMonitor(poll_hz=100.0)  # 10 ms
    m.start()
    try:
        # give the thread time to pull several samples
        time.sleep(0.2)
    finally:
        m.stop()

    assert m.peak_mb >= 5000  # at minimum captured one of the spikes


def test_session_context_manager_records_peak(monkeypatch) -> None:
    samples = iter([1000, 7000, 7000, 4000] + [4000] * 100)
    _install_fake_pynvml(monkeypatch, lambda: next(samples))

    m = resources.VRAMMonitor(poll_hz=100.0)
    m.start()
    try:
        with m.session() as s:
            time.sleep(0.1)
        assert s.peak_mb >= 1000
    finally:
        m.stop()


def test_reset_peak_clears_old_max(monkeypatch) -> None:
    # First sample is 9000 (peak), then 2000 (low) after reset
    state = {"value": 9000}

    def reader():
        return state["value"]

    _install_fake_pynvml(monkeypatch, reader)
    m = resources.VRAMMonitor()
    m.reset_peak()
    assert m.peak_mb == 9000
    state["value"] = 2000
    m.reset_peak()
    assert m.peak_mb == 2000


# ---- lifecycle ------------------------------------------------------------


def test_start_is_idempotent(monkeypatch) -> None:
    _install_fake_pynvml(monkeypatch, 1000)
    m = resources.VRAMMonitor(poll_hz=20.0)
    m.start()
    first = m._thread
    m.start()  # second call should not spawn a new thread
    assert m._thread is first
    m.stop()


def test_stop_without_start_is_safe(monkeypatch) -> None:
    _install_fake_pynvml(monkeypatch, 1000)
    m = resources.VRAMMonitor()
    m.stop()  # must not raise


def test_shutdown_disables_monitor(monkeypatch) -> None:
    fake = _install_fake_pynvml(monkeypatch, 1000)
    m = resources.VRAMMonitor()
    m.shutdown()
    assert m.available is False
    fake.nvmlShutdown.assert_called_once()


def test_context_manager_starts_and_shuts_down(monkeypatch) -> None:
    fake = _install_fake_pynvml(monkeypatch, 2048)
    with resources.VRAMMonitor(poll_hz=50.0) as m:
        assert m.is_running is True
        assert m.sample() == 2048
    fake.nvmlShutdown.assert_called_once()
