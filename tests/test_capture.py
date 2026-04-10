"""Tests for ``lolcoach.capture`` — dxcam-backed capture and calibration.

Scenarios mapped from the implementation plan's Unit 2 test list:

1. Happy: ``grab_minimap()`` with a fake camera + valid ROI returns the
   expected crop shape.
2. Happy: ``LatestFrameBuffer.put_latest`` drain-then-put keeps the newest
   frame only.
3. Edge: ``load_calibration`` with a missing file returns None.
4. Edge: ``load_calibration`` with a valid file returns an ROI.
5. Edge: ``load_calibration`` with corrupt/invalid JSON returns None + warns.
6. Edge: ``validate_roi_against_screen`` rejects out-of-bounds ROIs.
7. Error: first None frame raises :class:`CaptureError`.
8. Error: 10 consecutive None frames raise :class:`CaptureExhausted`.
9. Integration: the calibrate subcommand ``--corners`` flow writes a
   valid calibration.json (and the ``--test-clicks`` alias still works).

dxcam is Windows-only. These tests mock the camera object so they run on
any platform.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np
import pytest

from lolcoach.capture import (
    MAX_CONSECUTIVE_FAILURES,
    ROI,
    Capture,
    CaptureError,
    CaptureExhausted,
    LatestFrameBuffer,
    load_calibration,
    validate_roi_against_screen,
)


# ---------------------------------------------------------------------------
# Helpers: fake dxcam-like camera
# ---------------------------------------------------------------------------
class _FakeCamera:
    """Minimal dxcam stand-in. Yields frames from a queue on ``grab()``."""

    def __init__(self, frames: list[np.ndarray | None]) -> None:
        self._frames = list(frames)

    def grab(self) -> np.ndarray | None:
        if not self._frames:
            return None
        return self._frames.pop(0)


# ---------------------------------------------------------------------------
# Happy path: grab_minimap returns expected crop
# ---------------------------------------------------------------------------
def test_grab_minimap_returns_crop_with_expected_shape() -> None:
    full_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Paint the minimap region so we can assert the crop is non-trivial.
    full_frame[830:1030, 50:250, :] = 123
    camera = _FakeCamera([full_frame])
    capture = Capture(roi=ROI(x=50, y=830, w=200, h=200), camera=camera)

    crop = capture.grab_minimap()

    assert crop.shape == (200, 200, 3)
    assert (crop == 123).all()


# ---------------------------------------------------------------------------
# LatestFrameBuffer drain-then-put
# ---------------------------------------------------------------------------
def test_latest_frame_buffer_keeps_only_newest_frame() -> None:
    buf = LatestFrameBuffer()
    frame_a = np.full((4, 4, 3), 10, dtype=np.uint8)
    frame_b = np.full((4, 4, 3), 20, dtype=np.uint8)

    buf.put_latest(frame_a)
    buf.put_latest(frame_b)
    out = buf.get(timeout=0.1)

    assert out is not None
    assert (out == 20).all()


def test_latest_frame_buffer_get_times_out_when_empty() -> None:
    buf = LatestFrameBuffer()
    assert buf.get(timeout=0.01) is None


def test_latest_frame_buffer_concurrent_put_and_get() -> None:
    buf = LatestFrameBuffer()
    barrier = threading.Barrier(2)
    received: list[np.ndarray | None] = []

    def producer() -> None:
        barrier.wait()
        for i in range(100):
            buf.put_latest(np.full((2, 2, 3), i, dtype=np.uint8))

    def consumer() -> None:
        barrier.wait()
        for _ in range(10):
            frame = buf.get(timeout=0.1)
            if frame is not None:
                received.append(frame)

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    # The consumer may race ahead of the producer; it's fine. What matters is
    # that no exceptions were raised and received frames are valid arrays.
    for frame in received:
        assert frame.shape == (2, 2, 3)


# ---------------------------------------------------------------------------
# load_calibration: missing / valid / corrupt / bad schema
# ---------------------------------------------------------------------------
def test_load_calibration_missing_file_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    assert load_calibration(path) is None


def test_load_calibration_valid_file_returns_roi(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"x": 100, "y": 200, "w": 300, "h": 400}))

    roi = load_calibration(path)

    assert roi == ROI(x=100, y=200, w=300, h=400)


def test_load_calibration_corrupt_json_returns_none_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "calibration.json"
    path.write_text("{not valid json")

    with caplog.at_level(logging.WARNING, logger="lolcoach.capture"):
        result = load_calibration(path)

    assert result is None
    assert any("corrupt" in rec.message.lower() for rec in caplog.records)


def test_load_calibration_missing_schema_fields_returns_none_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"x": 100, "y": 200}))  # missing w, h

    with caplog.at_level(logging.WARNING, logger="lolcoach.capture"):
        result = load_calibration(path)

    assert result is None
    assert any("schema" in rec.message.lower() for rec in caplog.records)


# ---------------------------------------------------------------------------
# validate_roi_against_screen
# ---------------------------------------------------------------------------
def test_validate_roi_within_bounds_is_ok() -> None:
    validate_roi_against_screen(ROI(x=0, y=0, w=200, h=200), 1920, 1080)
    validate_roi_against_screen(ROI(x=1720, y=880, w=200, h=200), 1920, 1080)


def test_validate_roi_extending_past_width_raises() -> None:
    with pytest.raises(ValueError, match="width"):
        validate_roi_against_screen(ROI(x=1800, y=800, w=200, h=200), 1920, 1080)


def test_validate_roi_extending_past_height_raises() -> None:
    with pytest.raises(ValueError, match="height"):
        validate_roi_against_screen(ROI(x=100, y=900, w=200, h=200), 1920, 1080)


def test_validate_roi_with_negative_offset_raises() -> None:
    with pytest.raises(ValueError, match="negative"):
        validate_roi_against_screen(ROI(x=-10, y=0, w=100, h=100), 1920, 1080)


# ---------------------------------------------------------------------------
# Failure counting: transient error vs exhaustion
# ---------------------------------------------------------------------------
def test_grab_minimap_none_frame_raises_capture_error() -> None:
    camera = _FakeCamera([None])
    capture = Capture(roi=ROI(x=0, y=0, w=10, h=10), camera=camera)

    with pytest.raises(CaptureError):
        capture.grab_minimap()


def test_grab_minimap_resets_failure_counter_on_successful_frame() -> None:
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    # 3 failures, then a success, then 3 more failures — should NOT escalate.
    camera = _FakeCamera([None, None, None, frame, None, None, None])
    capture = Capture(roi=ROI(x=0, y=0, w=10, h=10), camera=camera)

    for _ in range(3):
        with pytest.raises(CaptureError):
            capture.grab_minimap()
    # Successful frame resets the counter.
    assert capture.grab_minimap().shape == (10, 10, 3)
    for _ in range(3):
        with pytest.raises(CaptureError):
            capture.grab_minimap()
    # Still alive — we never hit MAX_CONSECUTIVE_FAILURES.


def test_grab_minimap_exhausts_after_max_consecutive_failures() -> None:
    camera = _FakeCamera([None] * MAX_CONSECUTIVE_FAILURES)
    capture = Capture(roi=ROI(x=0, y=0, w=10, h=10), camera=camera)

    # First MAX-1 failures are transient CaptureError.
    for _ in range(MAX_CONSECUTIVE_FAILURES - 1):
        with pytest.raises(CaptureError):
            capture.grab_minimap()
    # The MAX-th failure escalates to CaptureExhausted.
    with pytest.raises(CaptureExhausted):
        capture.grab_minimap()


def test_grab_minimap_without_camera_raises_runtime_error() -> None:
    capture = Capture(roi=ROI(x=0, y=0, w=10, h=10), camera=None)
    with pytest.raises(RuntimeError, match="camera"):
        capture.grab_minimap()


# ---------------------------------------------------------------------------
# Calibration subcommand: non-interactive --corners flow
# ---------------------------------------------------------------------------
def test_calibrate_compute_roi_handles_ordered_corners() -> None:
    from lolcoach.calibrate import compute_roi

    # Top-left then bottom-right (the expected order)
    roi = compute_roi((100, 200), (300, 500))
    assert roi == ROI(x=100, y=200, w=200, h=300)


def test_calibrate_compute_roi_handles_reversed_corners() -> None:
    from lolcoach.calibrate import compute_roi

    # User clicked bottom-right first by accident
    roi = compute_roi((300, 500), (100, 200))
    assert roi == ROI(x=100, y=200, w=200, h=300)


def test_calibrate_save_calibration_roundtrip(tmp_path: Path) -> None:
    from lolcoach.calibrate import save_calibration

    path = tmp_path / "nested" / "calibration.json"
    roi = ROI(x=10, y=20, w=30, h=40)
    save_calibration(path, roi)

    assert path.exists()
    assert load_calibration(path) == roi


def test_calibrate_main_with_corners_writes_expected_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from lolcoach.calibrate import main

    output = tmp_path / "calibration.json"
    exit_code = main(["--output", str(output), "--corners", "100,200,300,400"])

    assert exit_code == 0
    assert output.exists()
    loaded = json.loads(output.read_text())
    assert loaded == {"x": 100, "y": 200, "w": 200, "h": 200}


def test_calibrate_main_test_clicks_alias_still_works(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Backward-compat: --test-clicks is kept as an alias for --corners."""
    from lolcoach.calibrate import main

    output = tmp_path / "calibration.json"
    exit_code = main(
        ["--output", str(output), "--test-clicks", "100,200,300,400"]
    )

    assert exit_code == 0
    assert output.exists()
    loaded = json.loads(output.read_text())
    assert loaded == {"x": 100, "y": 200, "w": 200, "h": 200}


def test_calibrate_main_with_bad_corners_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from lolcoach.calibrate import main

    output = tmp_path / "calibration.json"
    exit_code = main(["--output", str(output), "--corners", "100,200"])  # only 2 coords

    assert exit_code == 2
    assert not output.exists()
    captured = capsys.readouterr()
    assert "--corners" in captured.err


def test_calibrate_main_with_non_integer_corners_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """F13 bonus: exercise the _parse_corners ValueError branch."""
    from lolcoach.calibrate import main

    output = tmp_path / "calibration.json"
    exit_code = main(["--output", str(output), "--corners", "a,b,c,d"])

    assert exit_code == 2
    assert not output.exists()


def test_calibrate_main_interactive_without_display_returns_error(
    tmp_path: Path,
) -> None:
    from lolcoach.calibrate import main

    output = tmp_path / "calibration.json"
    # Without --corners we fall into the interactive path which is
    # Windows-only in practice; on the test runner it should refuse cleanly.
    exit_code = main(["--output", str(output)])
    assert exit_code != 0
    assert not output.exists()
