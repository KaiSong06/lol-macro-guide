"""dxcam-backed screen capture with minimap ROI cropping.

The Capture class owns a single dxcam camera and produces cropped numpy
arrays of the League minimap on demand. dxcam is Windows-only, so the
module imports it lazily and the Capture class takes the camera as a
constructor argument — tests inject a fake camera and the real runtime
uses :meth:`Capture.create` which builds a dxcam instance.

Transient capture failures raise :class:`CaptureError`; after
:data:`MAX_CONSECUTIVE_FAILURES` in a row the next failure escalates to
:class:`CaptureExhausted`, which the orchestration layer translates into a
one-shot TTS announcement ("Screen capture failed. Coach paused.").
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# dxcam is Windows-only. Import lazily so the module is usable on macOS/Linux
# for unit tests that inject a fake camera.
if sys.platform == "win32":  # pragma: no cover - Windows-only import path
    try:
        import dxcam  # type: ignore[import-not-found]

        _DXCAM_IMPORT_ERROR: Exception | None = None
    except ImportError as exc:  # pragma: no cover
        dxcam = None  # type: ignore[assignment]
        _DXCAM_IMPORT_ERROR = exc
else:
    dxcam = None  # type: ignore[assignment]
    _DXCAM_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

#: Number of consecutive ``grab()`` returning None before escalating.
MAX_CONSECUTIVE_FAILURES = 10


class CaptureError(Exception):
    """Raised on a transient single-frame capture failure."""


class CaptureExhausted(Exception):
    """Raised after too many consecutive capture failures."""


@dataclass(frozen=True)
class ROI:
    """A minimap region-of-interest in screen coordinates."""

    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


# ---------------------------------------------------------------------------
# Calibration persistence
# ---------------------------------------------------------------------------
def load_calibration(path: Path) -> ROI | None:
    """Load a calibrated minimap ROI from ``calibration.json``.

    Returns ``None`` if the file is missing, corrupt, or has an invalid
    schema — callers treat any of those as "needs calibration" and prompt
    the user to run ``python -m lolcoach.calibrate``.
    """
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("calibration file %s is corrupt: %s", path, exc)
        return None
    try:
        return ROI(
            x=int(data["x"]),
            y=int(data["y"]),
            w=int(data["w"]),
            h=int(data["h"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("calibration file %s has invalid schema: %s", path, exc)
        return None


def validate_roi_against_screen(roi: ROI, screen_width: int, screen_height: int) -> None:
    """Raise :class:`ValueError` if the ROI extends outside the screen.

    Used at coach startup to fail fast if the persisted calibration predates
    a monitor / resolution change.
    """
    if roi.x < 0 or roi.y < 0:
        raise ValueError(f"ROI {roi} has negative offset")
    if roi.x + roi.w > screen_width:
        raise ValueError(
            f"ROI {roi} extends past screen width {screen_width}"
        )
    if roi.y + roi.h > screen_height:
        raise ValueError(
            f"ROI {roi} extends past screen height {screen_height}"
        )


# ---------------------------------------------------------------------------
# Latest-frame buffer (1-slot queue, drain-then-put)
# ---------------------------------------------------------------------------
class LatestFrameBuffer:
    """A single-slot queue where the producer always replaces any pending frame.

    Used to pass frames from the Capture thread to the Inference worker.
    The inference worker only cares about the newest frame, so stale frames
    are dropped on the producer side without blocking.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)

    def put_latest(self, frame: np.ndarray) -> None:
        """Drain any pending frame, then put *frame*. Never blocks."""
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(frame)
        except queue.Full:  # pragma: no cover - defensive; drain+put is atomic enough
            pass

    def get(self, timeout: float | None = None) -> np.ndarray | None:
        """Block for *timeout* seconds waiting for a frame. Return None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ---------------------------------------------------------------------------
# Capture class
# ---------------------------------------------------------------------------
class Capture:
    """Owns a dxcam-like camera and produces minimap crops on ``grab_minimap()``.

    The camera is injected to keep the class testable on non-Windows hosts.
    Real runtime uses :meth:`create` which instantiates a real ``dxcam.DXCamera``.
    """

    def __init__(self, roi: ROI, camera: Any | None = None) -> None:
        self._roi = roi
        self._camera = camera
        self._consecutive_failures = 0
        self._lock = threading.Lock()

    @classmethod
    def create(cls, roi: ROI) -> Capture:  # pragma: no cover - Windows-only hardware path
        """Factory that instantiates a real dxcam camera. Windows-only."""
        if sys.platform != "win32":
            raise RuntimeError(
                f"Capture.create() is Windows-only; got sys.platform={sys.platform!r}"
            )
        if dxcam is None:
            raise RuntimeError(
                "dxcam failed to import on this Windows build"
            ) from _DXCAM_IMPORT_ERROR
        camera = dxcam.create()
        return cls(roi=roi, camera=camera)

    @property
    def roi(self) -> ROI:
        return self._roi

    def grab_minimap(self) -> np.ndarray:
        """Grab a single frame and return the minimap crop.

        Raises :class:`CaptureError` on a single transient failure, escalating
        to :class:`CaptureExhausted` after :data:`MAX_CONSECUTIVE_FAILURES`
        consecutive failures.
        """
        if self._camera is None:
            raise RuntimeError(
                "Capture has no camera; use Capture.create() on Windows or inject one"
            )

        frame = self._camera.grab()
        if frame is None:
            with self._lock:
                self._consecutive_failures += 1
                count = self._consecutive_failures
            if count >= MAX_CONSECUTIVE_FAILURES:
                raise CaptureExhausted(
                    f"dxcam returned None {MAX_CONSECUTIVE_FAILURES} times in a row"
                )
            raise CaptureError("dxcam grab returned None")

        with self._lock:
            self._consecutive_failures = 0
        return self._crop(frame)

    def _crop(self, frame: np.ndarray) -> np.ndarray:
        r = self._roi
        return frame[r.y : r.y + r.h, r.x : r.x + r.w]
