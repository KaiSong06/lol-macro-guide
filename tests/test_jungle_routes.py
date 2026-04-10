"""Tests for ``lolcoach.jungle_routes``.

Scenarios from the implementation plan's Unit 5 test list (8 total):

1-6: one test per standard jungle path — verify that walking along the path
     for a known elapsed time lands on the expected quadrant.
7:   cold start — ``predict_quadrant(None, ...)`` returns ``None``.
8:   tail overshoot — very large elapsed values return the last camp's
     quadrant (best-effort tail).
"""

from __future__ import annotations

from lolcoach.jungle_routes import (
    STANDARD_PATHS,
    predict_along_path,
    predict_quadrant,
)


# ---------------------------------------------------------------------------
# Per-path walks
# ---------------------------------------------------------------------------
def test_blue_top_path_walks_forward_correctly() -> None:
    # Path starts in top_jungle. After a short time, still there.
    assert predict_along_path("blue_top", elapsed_seconds=10) == "top_jungle"
    # After ~75 seconds (blue + gromp + wolves ≈ 70) we're still in top_jungle.
    assert predict_along_path("blue_top", elapsed_seconds=70) == "top_jungle"
    # After crossing to raptors (~95s in) we're in mid.
    assert predict_along_path("blue_top", elapsed_seconds=100) == "mid"


def test_blue_bot_path_walks_forward_correctly() -> None:
    assert predict_along_path("blue_bot", elapsed_seconds=0) == "top_jungle"
    # After raptors, end up in mid at cross-over
    assert predict_along_path("blue_bot", elapsed_seconds=90) == "mid"
    # After red + krugs, end up in bot_jungle
    assert predict_along_path("blue_bot", elapsed_seconds=180) == "bot_jungle"


def test_red_bot_path_walks_forward_correctly() -> None:
    assert predict_along_path("red_bot", elapsed_seconds=10) == "bot_jungle"
    # RedBuff(0) + Krugs(25) + Raptors(30) = 55s; crossing at idx 2 (mid)
    assert predict_along_path("red_bot", elapsed_seconds=80) == "mid"
    # All the way to BlueBuff at top_jungle
    assert predict_along_path("red_bot", elapsed_seconds=180) == "top_jungle"


def test_red_top_path_walks_forward_correctly() -> None:
    assert predict_along_path("red_top", elapsed_seconds=10) == "bot_jungle"
    assert predict_along_path("red_top", elapsed_seconds=40) == "mid"
    assert predict_along_path("red_top", elapsed_seconds=150) == "top_jungle"


def test_invade_blue_path_walks_forward_correctly() -> None:
    assert predict_along_path("invade_blue", elapsed_seconds=10) == "bot_jungle"
    # After a while, invader has crossed into the enemy top jungle
    assert predict_along_path("invade_blue", elapsed_seconds=80) == "top_jungle"


def test_invade_red_path_walks_forward_correctly() -> None:
    assert predict_along_path("invade_red", elapsed_seconds=10) == "top_jungle"
    assert predict_along_path("invade_red", elapsed_seconds=80) == "bot_jungle"


# ---------------------------------------------------------------------------
# Cold start + tail overshoot + unknown path
# ---------------------------------------------------------------------------
def test_predict_quadrant_cold_start_returns_none() -> None:
    assert predict_quadrant(last_seen=None, elapsed_seconds=60.0) is None


def test_predict_quadrant_short_elapsed_returns_last_seen() -> None:
    # Under 30 seconds, assume they're still in the last-seen quadrant.
    assert predict_quadrant(last_seen="top_jungle", elapsed_seconds=10.0) == "top_jungle"


def test_predict_quadrant_long_elapsed_walks_a_path() -> None:
    # For long elapsed values, expect a plausible quadrant that isn't None.
    result = predict_quadrant(last_seen="top_jungle", elapsed_seconds=120.0)
    assert result in {"top_jungle", "mid", "bot_jungle", "top_river", "bot_river"}


def test_predict_along_path_tail_overshoot_returns_last_camp() -> None:
    last_camp_quadrant = STANDARD_PATHS["blue_top"][-1].quadrant
    assert predict_along_path("blue_top", elapsed_seconds=99999.0) == last_camp_quadrant


def test_predict_along_path_rejects_unknown_path() -> None:
    import pytest

    with pytest.raises(KeyError):
        predict_along_path("not_a_real_path", elapsed_seconds=10.0)


def test_standard_paths_has_six_named_paths() -> None:
    assert set(STANDARD_PATHS.keys()) == {
        "blue_top",
        "blue_bot",
        "red_bot",
        "red_top",
        "invade_blue",
        "invade_red",
    }
