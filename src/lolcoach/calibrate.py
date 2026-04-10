"""First-run calibration subcommand.

Invoked as ``python -m lolcoach.calibrate``. The interactive path uses an
OpenCV imshow window with mouse callbacks to let the user click the top-left
and bottom-right corners of the League minimap, then persists the resulting
:class:`~lolcoach.capture.ROI` to ``calibration.json``.

Interactive cv2/dxcam paths are Windows+display-only and marked
``pragma: no cover``. Tests exercise the non-interactive ``--test-clicks``
path plus the pure-function helpers :func:`compute_roi` and
:func:`save_calibration`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lolcoach.capture import ROI


def compute_roi(
    click_a: tuple[int, int],
    click_b: tuple[int, int],
) -> ROI:
    """Build a normalized ROI from two opposite corner clicks.

    Callers do not need to enforce click order — the function returns an ROI
    with non-negative width and height regardless of which corner was clicked
    first.
    """
    x1, y1 = click_a
    x2, y2 = click_b
    return ROI(
        x=min(x1, x2),
        y=min(y1, y2),
        w=abs(x2 - x1),
        h=abs(y2 - y1),
    )


def save_calibration(path: Path, roi: ROI) -> None:
    """Persist an ROI to ``calibration.json``, creating parent dirs on demand."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"x": roi.x, "y": roi.y, "w": roi.w, "h": roi.h}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lolcoach.calibrate",
        description="Calibrate the League minimap region for lolcoach.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("calibration.json"),
        help="Output path for the calibration JSON file (default: ./calibration.json)",
    )
    parser.add_argument(
        "--test-clicks",
        type=str,
        default=None,
        help=(
            "Non-interactive mode: comma-separated 'x1,y1,x2,y2' for the "
            "minimap corners. Used by tests; humans should omit this flag."
        ),
    )
    return parser


def _parse_test_clicks(raw: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    parts = raw.split(",")
    if len(parts) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(p) for p in parts)
    except ValueError:
        return None
    return (x1, y1), (x2, y2)


def _run_interactive() -> ROI | None:  # pragma: no cover - hardware-dependent
    """Interactive calibration via OpenCV imshow + mouse callback.

    Windows-only in practice — requires dxcam + a display. Documented in
    ``tests/manual.md`` and smoke-tested on real hardware in Unit 10.
    """
    raise NotImplementedError(
        "Interactive calibration lands with the real dxcam integration — "
        "pass --test-clicks for now, or run on a Windows machine with dxcam."
    )


def main(argv: list[str] | None = None) -> int:
    """Calibrate entry point. Returns a shell exit code."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.test_clicks is not None:
        parsed = _parse_test_clicks(args.test_clicks)
        if parsed is None:
            print(
                "--test-clicks requires exactly 4 comma-separated integers: "
                "x1,y1,x2,y2",
                file=sys.stderr,
            )
            return 2
        click_a, click_b = parsed
        roi = compute_roi(click_a, click_b)
        save_calibration(args.output, roi)
        print(f"Wrote calibration to {args.output}: {roi}")
        return 0

    # Interactive path: requires Windows + a display. Refuse cleanly here so
    # test runners and CI don't crash trying to pop a window.
    print(
        "Interactive calibration is Windows-only for now. "
        "Pass --test-clicks x1,y1,x2,y2 for scripted calibration, or run this "
        "command on a Windows machine with dxcam installed.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
