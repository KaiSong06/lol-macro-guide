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
        description=(
            "Calibrate the League minimap region for lolcoach. Writes a "
            "calibration.json file that capture.py loads to know which "
            "region of the screen to crop."
        ),
        epilog=(
            "Scripted mode (no display required):\n"
            "  python -m lolcoach.calibrate --corners 1720,880,1920,1080\n\n"
            "Interactive mode (Windows with a display, Phase 2+):\n"
            "  python -m lolcoach.calibrate\n"
            "  (click the minimap's top-left, then bottom-right corner)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("calibration.json"),
        help=(
            "Output path for the calibration JSON file "
            "(default: ./calibration.json). Parent directories are "
            "created if needed."
        ),
    )
    parser.add_argument(
        "--corners",
        "--test-clicks",  # deprecated alias, still accepted
        dest="corners",
        type=str,
        default=None,
        metavar="X1,Y1,X2,Y2",
        help=(
            "Scripted/headless calibration mode: pass the minimap's "
            "top-left and bottom-right corners as comma-separated "
            "integers. This is the only supported mode outside Windows "
            "with a display and is the recommended path for agents and "
            "CI. --test-clicks is kept as an alias for backward "
            "compatibility. Example: --corners 1720,880,1920,1080"
        ),
    )
    return parser


def _parse_corners(raw: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
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
    if sys.platform != "win32":
        return None

    import cv2
    import dxcam

    camera = dxcam.create()
    frame = camera.grab()
    if frame is None:
        return None

    clicks: list[tuple[int, int]] = []
    window_name = "lolcoach minimap calibration"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        clicks.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), thickness=-1)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    while len(clicks) < 2:
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(50) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyWindow(window_name)
            return None
    cv2.destroyWindow(window_name)
    return compute_roi(clicks[0], clicks[1])


def main(argv: list[str] | None = None) -> int:
    """Calibrate entry point. Returns a shell exit code."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.corners is not None:
        parsed = _parse_corners(args.corners)
        if parsed is None:
            print(
                "--corners requires exactly 4 comma-separated integers: "
                "x1,y1,x2,y2",
                file=sys.stderr,
            )
            return 2
        click_a, click_b = parsed
        roi = compute_roi(click_a, click_b)
        save_calibration(args.output, roi)
        print(f"Wrote calibration to {args.output}: {roi}")
        return 0

    roi = _run_interactive()
    if roi is None:
        print(
            "Interactive calibration requires Windows with dxcam and a display. "
            "Pass --corners x1,y1,x2,y2 for scripted calibration.",
            file=sys.stderr,
        )
        return 1
    save_calibration(args.output, roi)
    print(f"Wrote calibration to {args.output}: {roi}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
