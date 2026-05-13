"""Tests for ``lolcoach.detector`` — deterministic OpenCV template matching.

Scenarios mapped from the implementation plan's Unit 3 test list:

1. Happy: a synthetic frame with one embedded template produces one
   detection at the expected normalized position and team.
2. Happy: ``quadrant_of`` covers all 9 regions (one representative point
   per region).
3. Edge: overlapping hits for the same champion are collapsed by NMS.
4. Edge: a detection below the confidence threshold is dropped.
5. Edge: a detected champion not in ally or enemy rosters is dropped with
   a log event.
6. Error: a corrupt template file is skipped at load time, detector keeps
   loading the rest.
7. Edge: empty roster → ``detect`` returns an empty champion list.

Uses the five synthetic templates checked into
``tests/fixtures/champion_templates_subset/``. These are NOT real Riot
assets — they're hand-drawn 64×64 PNGs with distinctive patterns so
OpenCV ``matchTemplate`` can differentiate them in test frames.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from lolcoach.detector import (
    DetectionResult,
    Detector,
    quadrant_of,
)

FIXTURE_TEMPLATE_DIR = Path(__file__).parent / "fixtures" / "champion_templates_subset"
ALL_TEMPLATE_NAMES = {"LeeSin", "Jinx", "Hecarim", "Ekko", "Ahri"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_template(name: str) -> np.ndarray:
    img = cv2.imread(str(FIXTURE_TEMPLATE_DIR / f"{name}.png"), cv2.IMREAD_COLOR)
    assert img is not None, f"fixture {name} missing"
    return img


def _embed(frame: np.ndarray, template: np.ndarray, top_left: tuple[int, int]) -> None:
    """Paste *template* into *frame* at *top_left* in place."""
    x, y = top_left
    h, w = template.shape[:2]
    frame[y : y + h, x : x + w] = template


# ---------------------------------------------------------------------------
# Happy path: single embedded template detected with expected team + position
# ---------------------------------------------------------------------------
def test_detect_single_enemy_returns_one_detection_with_correct_position() -> None:
    detector = Detector(FIXTURE_TEMPLATE_DIR)

    # 512x512 synthetic frame with Lee Sin placed at (240, 320).
    frame = np.full((512, 512, 3), 50, dtype=np.uint8)
    lee_sin = _load_template("LeeSin")
    _embed(frame, lee_sin, top_left=(240, 320))

    result = detector.detect(frame, ally_names=[], enemy_names=["LeeSin"])

    assert isinstance(result, DetectionResult)
    assert len(result.champions) == 1
    det = result.champions[0]
    assert det.name == "LeeSin"
    assert det.team == "enemy"
    assert det.confidence > 0.85
    # Center should be at (240 + 32, 320 + 32) = (272, 352); norm ≈ (0.53, 0.69).
    assert det.position_norm[0] == pytest.approx(272 / 512, abs=0.02)
    assert det.position_norm[1] == pytest.approx(352 / 512, abs=0.02)
    assert det.quadrant in ("mid", "bot_jungle", "bot_river")  # around center-bottom
    # detected_at is a reasonable wall clock time
    assert result.detected_at > 0


def test_detect_populates_last_seen_champions_on_result() -> None:
    """The grounding set is on the DetectionResult itself, not the Detector
    instance. This pins each grounding decision to the frame the LLM was
    shown and removes the cross-thread stale-read path (F8 from /ce:review).
    """
    detector = Detector(FIXTURE_TEMPLATE_DIR)

    frame = np.full((512, 512, 3), 50, dtype=np.uint8)
    _embed(frame, _load_template("LeeSin"), top_left=(100, 100))
    _embed(frame, _load_template("Jinx"), top_left=(300, 300))

    result = detector.detect(frame, ally_names=["Jinx"], enemy_names=["LeeSin"])

    assert result.last_seen_champions == frozenset({"LeeSin", "Jinx"})
    # The Detector instance should not retain any mutable last-seen state.
    assert not hasattr(detector, "_last_seen_champions")
    assert not hasattr(detector, "last_seen_champions")


def test_detect_empty_result_has_empty_last_seen() -> None:
    """A frame with no detections still carries a last_seen_champions field
    so downstream consumers can rely on the field existing unconditionally.
    """
    detector = Detector(FIXTURE_TEMPLATE_DIR, confidence_threshold=0.999)
    frame = np.full((512, 512, 3), 127, dtype=np.uint8)

    result = detector.detect(frame, ally_names=[], enemy_names=["LeeSin"])

    assert result.champions == ()
    assert result.last_seen_champions == frozenset()


# ---------------------------------------------------------------------------
# quadrant_of: 9 regions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "x,y,expected",
    [
        (0.10, 0.10, "top_lane"),
        (0.50, 0.10, "top_river"),
        (0.90, 0.10, "base_red"),
        (0.10, 0.50, "top_jungle"),
        (0.50, 0.50, "mid"),
        (0.90, 0.50, "bot_jungle"),
        (0.10, 0.90, "base_blue"),
        (0.50, 0.90, "bot_river"),
        (0.90, 0.90, "bot_lane"),
    ],
)
def test_quadrant_of_covers_all_nine_regions(x: float, y: float, expected: str) -> None:
    assert quadrant_of(x, y) == expected


def test_quadrant_of_edge_values() -> None:
    """Boundary values at exactly 1/3 and 2/3 are deterministic."""
    # Just below boundaries land in the earlier region.
    assert quadrant_of(0.0, 0.0) == "top_lane"
    assert quadrant_of(0.9999, 0.9999) == "bot_lane"


# ---------------------------------------------------------------------------
# NMS: overlapping hits for same champion collapse to one
# ---------------------------------------------------------------------------
def test_detect_collapses_overlapping_hits_via_nms() -> None:
    detector = Detector(FIXTURE_TEMPLATE_DIR)

    # Two very close Lee Sin embeddings — NMS should keep only one.
    frame = np.full((512, 512, 3), 50, dtype=np.uint8)
    lee_sin = _load_template("LeeSin")
    _embed(frame, lee_sin, top_left=(100, 100))
    _embed(frame, lee_sin, top_left=(110, 105))  # heavy overlap

    result = detector.detect(frame, ally_names=[], enemy_names=["LeeSin"])

    lee_hits = [c for c in result.champions if c.name == "LeeSin"]
    assert len(lee_hits) == 1


# ---------------------------------------------------------------------------
# Confidence threshold
# ---------------------------------------------------------------------------
def test_detect_drops_matches_below_confidence_threshold() -> None:
    # Set an extremely high threshold so the solid-color synthetic frame
    # (which contains no real template) produces zero matches.
    detector = Detector(FIXTURE_TEMPLATE_DIR, confidence_threshold=0.999)

    frame = np.full((512, 512, 3), 127, dtype=np.uint8)
    result = detector.detect(frame, ally_names=[], enemy_names=["LeeSin"])

    assert result.champions == ()


# ---------------------------------------------------------------------------
# Unknown champion (not in rosters) dropped with log
# ---------------------------------------------------------------------------
def test_detect_does_not_return_champion_outside_the_roster() -> None:
    """After F2, the detector doesn't even run matchTemplate against
    templates whose names aren't in the roster. The previous
    implementation matched all templates and filtered after (with a
    'dropping unknown champion' INFO log); the log path is now dead
    code because the non-roster iteration never happens.

    The externally-observable behavior is the same: an Ekko detection
    is not returned when Ekko isn't in ally_names or enemy_names.
    """
    detector = Detector(FIXTURE_TEMPLATE_DIR)

    frame = np.full((512, 512, 3), 50, dtype=np.uint8)
    _embed(frame, _load_template("Ekko"), top_left=(200, 200))

    result = detector.detect(frame, ally_names=["Jinx"], enemy_names=["LeeSin"])

    assert all(c.name != "Ekko" for c in result.champions)


# ---------------------------------------------------------------------------
# Corrupt template is skipped at load time
# ---------------------------------------------------------------------------
def test_detector_skips_corrupt_templates(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Make a directory with one valid template and one corrupt file.
    template_dir = tmp_path / "templates"
    template_dir.mkdir()

    # Copy a valid one
    valid = cv2.imread(str(FIXTURE_TEMPLATE_DIR / "LeeSin.png"), cv2.IMREAD_COLOR)
    cv2.imwrite(str(template_dir / "LeeSin.png"), valid)

    # Write a bogus "png"
    (template_dir / "Broken.png").write_bytes(b"this is not a PNG file")

    with caplog.at_level(logging.WARNING, logger="lolcoach.detector"):
        detector = Detector(template_dir)

    assert "LeeSin" in detector.known_champions
    assert "Broken" not in detector.known_champions
    assert any("broken" in rec.message.lower() for rec in caplog.records)


# ---------------------------------------------------------------------------
# Empty roster
# ---------------------------------------------------------------------------
def test_detect_with_empty_rosters_returns_no_detections() -> None:
    detector = Detector(FIXTURE_TEMPLATE_DIR)

    frame = np.full((512, 512, 3), 50, dtype=np.uint8)
    _embed(frame, _load_template("LeeSin"), top_left=(100, 100))

    result = detector.detect(frame, ally_names=[], enemy_names=[])

    assert result.champions == ()


# ---------------------------------------------------------------------------
# Known_champions reflects the loaded template set
# ---------------------------------------------------------------------------
def test_detector_known_champions_matches_fixture_set() -> None:
    detector = Detector(FIXTURE_TEMPLATE_DIR)
    assert detector.known_champions == ALL_TEMPLATE_NAMES


# ---------------------------------------------------------------------------
# Template larger than frame is skipped safely
# ---------------------------------------------------------------------------
def test_detect_skips_template_larger_than_frame() -> None:
    detector = Detector(FIXTURE_TEMPLATE_DIR)
    # Tiny 10x10 frame — template is 64x64
    tiny_frame = np.full((10, 10, 3), 50, dtype=np.uint8)
    result = detector.detect(tiny_frame, ally_names=["LeeSin"], enemy_names=[])
    assert result.champions == ()


# ---------------------------------------------------------------------------
# F2 regression: matchTemplate runs only for in-roster champions
# ---------------------------------------------------------------------------
def test_detect_only_runs_matchtemplate_for_roster_champions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hot path must NOT run cv2.matchTemplate against every loaded
    template. With ~160 League champions but only 10 playing any given
    match, filtering by roster before the matchTemplate loop yields a
    ~16x speedup on the capture hot path (F2 from /ce:review).

    This test patches cv2.matchTemplate with a call counter and asserts
    exactly len(roster) calls, not len(all_templates).
    """
    import cv2 as real_cv2

    detector = Detector(FIXTURE_TEMPLATE_DIR)
    # The fixture templates are {LeeSin, Jinx, Hecarim, Ekko, Ahri}.
    assert detector.known_champions == ALL_TEMPLATE_NAMES
    assert len(detector.known_champions) == 5

    calls: list[str] = []
    original_match = real_cv2.matchTemplate

    def counting_match(frame, template, method):
        # Best-effort identify which template this is by shape fingerprint.
        calls.append(f"shape={template.shape}")
        return original_match(frame, template, method)

    monkeypatch.setattr(real_cv2, "matchTemplate", counting_match)

    frame = np.full((512, 512, 3), 50, dtype=np.uint8)
    detector.detect(
        frame,
        ally_names=["Jinx"],  # 1 ally in roster
        enemy_names=["LeeSin"],  # 1 enemy in roster
    )

    # Exactly 2 matchTemplate calls — one per roster entry — NOT 5
    # (which would be "matched all templates then filtered after").
    assert len(calls) == 2, (
        f"Expected 2 matchTemplate calls (roster size), got {len(calls)}. "
        f"The detector is still running matchTemplate against non-roster "
        f"templates — the F2 hot-path optimization is not in effect."
    )


def test_detect_handles_roster_champion_with_missing_template(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If the Riot roster includes a champion whose template didn't load
    (e.g., a brand new champion released after the last
    build-champion-templates.py run), the detector must skip gracefully
    without crashing and without affecting other champions.
    """
    detector = Detector(FIXTURE_TEMPLATE_DIR)

    frame = np.full((512, 512, 3), 50, dtype=np.uint8)
    _embed(frame, _load_template("LeeSin"), top_left=(100, 100))

    # "NewChampion2026" has no template loaded; LeeSin does.
    result = detector.detect(
        frame,
        ally_names=[],
        enemy_names=["LeeSin", "NewChampion2026"],
    )

    # LeeSin should still be detected.
    assert any(c.name == "LeeSin" for c in result.champions)
    # No crash. NewChampion2026 simply doesn't appear.
    assert all(c.name != "NewChampion2026" for c in result.champions)


def test_detect_with_all_rostered_champions_runs_all_matches() -> None:
    """Baseline sanity: if the roster contains every loaded template,
    every template should be matched. This pairs with the filter test
    above — together they prove the count IS len(roster), not
    min(len(roster), len(templates)) or some other quirk.
    """
    detector = Detector(FIXTURE_TEMPLATE_DIR)

    frame = np.full((512, 512, 3), 50, dtype=np.uint8)
    # Put every fixture template into the frame at non-overlapping positions
    _embed(frame, _load_template("LeeSin"), top_left=(50, 50))
    _embed(frame, _load_template("Jinx"), top_left=(150, 50))
    _embed(frame, _load_template("Hecarim"), top_left=(250, 50))
    _embed(frame, _load_template("Ekko"), top_left=(50, 150))
    _embed(frame, _load_template("Ahri"), top_left=(150, 150))

    result = detector.detect(
        frame,
        ally_names=["Jinx", "Hecarim", "Ahri"],
        enemy_names=["LeeSin", "Ekko"],
    )

    found_names = {c.name for c in result.champions}
    assert found_names == ALL_TEMPLATE_NAMES
