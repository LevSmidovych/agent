from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUSnapshot:
    """Single point-in-time GPU sample."""

    vram_used_mb: int
    vram_total_mb: int


class VRAMMonitor:
    """Polls the GPU at a configurable rate (default 10 Hz) and tracks peak
    VRAM usage between ``reset_peak()`` calls.

    Designed to be created once per benchmark session and reused across many
    prompts: before each prompt call ``reset_peak()``, read ``peak_mb`` after
    the prompt finishes.

    Falls back to a no-op when ``nvidia-ml-py`` is not installed or no NVIDIA
    GPU is present — the monitor stays usable, ``available`` returns False
    and all samples return 0.
    """

    def __init__(self, poll_hz: float = 10.0, device_index: int = 0) -> None:
        self._interval = 1.0 / max(poll_hz, 0.1)
        self._device_index = device_index
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._peak_mb = 0
        self._pynvml = None
        self._handle = None
        self._available = self._init_nvml()

    # ---- init -------------------------------------------------------------

    def _init_nvml(self) -> bool:
        try:
            import pynvml  # type: ignore
        except ImportError:
            logger.info("pynvml not installed; VRAM monitoring disabled")
            return False
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            self._pynvml = pynvml
            return True
        except Exception as exc:
            logger.info("NVML init failed (%s); VRAM monitoring disabled", exc)
            return False

    # ---- public API -------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._available

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def peak_mb(self) -> int:
        with self._lock:
            return self._peak_mb

    def sample(self) -> int:
        """Return current VRAM used in MB (0 when unavailable)."""
        if not self._available:
            return 0
        try:
            info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return int(info.used // (1024 * 1024))
        except Exception:
            logger.exception("nvml sample failed")
            return 0

    def reset_peak(self) -> None:
        """Reset peak tracker to the current sample. Call before each prompt."""
        current = self.sample()
        with self._lock:
            self._peak_mb = current

    def start(self) -> None:
        if not self._available or self.is_running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="vram-monitor"
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def shutdown(self) -> None:
        """Stop polling and tear down NVML. Safe to call multiple times."""
        self.stop()
        if self._available and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
            self._available = False
            self._pynvml = None
            self._handle = None

    # ---- internals --------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            mb = self.sample()
            with self._lock:
                if mb > self._peak_mb:
                    self._peak_mb = mb
            self._stop.wait(self._interval)

    # context-manager helper for a single tracked window
    def session(self) -> "VRAMSession":
        return VRAMSession(self)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()


class VRAMSession:
    """Context manager that resets the peak on enter and captures it on exit.

    Usage:
        with monitor.session() as s:
            # ... run prompt ...
        peak = s.peak_mb
    """

    def __init__(self, monitor: VRAMMonitor) -> None:
        self._monitor = monitor
        self.peak_mb = 0

    def __enter__(self) -> "VRAMSession":
        self._monitor.reset_peak()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.peak_mb = self._monitor.peak_mb
