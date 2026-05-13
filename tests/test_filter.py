"""Tests for the DecisionFilter policy gate."""

from __future__ import annotations

import pytest

from lolcoach.config import DecisionsConfig
from lolcoach.detector import ChampionDetection, DetectionResult
from lolcoach.filter import DecisionFilter, find_champion_mentions
from lolcoach.inference import Callout
from lolcoach.state import GameState


class FakeClock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _callout(
    *,
    decision: str = "Path to bot river, Lee Sin is missing",
    category: str = "pathing",
    target_lane: str = "bot",
    confidence: int = 8,
    reason: str = "Lee Sin was last seen bot river.",
    generated_at: float = 1_000.0,
) -> Callout:
    return Callout(
        decision=decision,
        category=category,
        target_lane=target_lane,
        confidence=confidence,
        reason=reason,
        generated_at=generated_at,
    )


def _state(champions: frozenset[str] | None = None) -> GameState:
    champions = champions or frozenset({"Hecarim", "Jinx", "Ahri", "LeeSin"})
    return GameState(
        game_time_seconds=600.0,
        active_player_champion="Hecarim",
        ally_champions=frozenset({"Hecarim", "Jinx", "Ahri"}),
        enemy_champions=frozenset(champions - {"Hecarim", "Jinx", "Ahri"}),
        all_champions=champions,
    )


def _detections(names: tuple[str, ...] = ("LeeSin",)) -> DetectionResult:
    return DetectionResult(
        champions=tuple(
            ChampionDetection(
                name=name,
                team="enemy",
                position_norm=(0.5, 0.5),
                quadrant="mid",
                confidence=0.9,
            )
            for name in names
        ),
        detected_at=1_000.0,
    )


def _filter(clock: FakeClock) -> DecisionFilter:
    return DecisionFilter(
        DecisionsConfig(
            cooldown_seconds=5,
            confidence_threshold=6,
            dedup_window_seconds=30,
            staleness_threshold_seconds=10,
        ),
        known_champions={
            "Ahri",
            "Ekko",
            "Hecarim",
            "Jinx",
            "Kaisa",
            "LeeSin",
            "MonkeyKing",
            "Velkoz",
            "Vi",
        },
        clock=clock,
    )


def test_accepts_grounded_callout() -> None:
    clock = FakeClock()
    decision = _filter(clock).should_speak(_callout(), _state(), _detections())

    assert decision.accepted is True
    assert decision.reason == "accepted"


@pytest.mark.parametrize(
    ("callout", "reason"),
    [
        (_callout(confidence=5), "low_confidence"),
        (_callout(generated_at=980.0), "stale"),
    ],
)
def test_rejects_policy_gates(callout: Callout, reason: str) -> None:
    clock = FakeClock()
    decision = _filter(clock).should_speak(callout, _state(), _detections())

    assert decision.accepted is False
    assert decision.reason == reason


def test_cooldown_applies_after_record_spoken() -> None:
    clock = FakeClock()
    filt = _filter(clock)
    filt.record_spoken(_callout())

    decision = filt.should_speak(
        _callout(category="vision", target_lane="mid"),
        _state(),
        _detections(),
    )

    assert decision.accepted is False
    assert decision.reason == "cooldown"


def test_dedup_window_applies_after_cooldown_expires() -> None:
    clock = FakeClock()
    filt = _filter(clock)
    filt.record_spoken(_callout())
    clock.advance(6)

    decision = filt.should_speak(_callout(generated_at=clock.now), _state(), _detections())

    assert decision.accepted is False
    assert decision.reason == "dedup"


def test_dedup_entry_expires() -> None:
    clock = FakeClock()
    filt = _filter(clock)
    filt.record_spoken(_callout())
    clock.advance(31)

    decision = filt.should_speak(_callout(generated_at=clock.now), _state(), _detections())

    assert decision.accepted is True


def test_rejects_hallucinated_champion_not_in_game_or_detector() -> None:
    clock = FakeClock()
    callout = _callout(
        decision="Ekko is invading your blue",
        reason="Ekko appears to be moving into your jungle.",
    )

    decision = _filter(clock).should_speak(
        callout,
        _state(frozenset({"Hecarim", "Jinx", "Ahri", "LeeSin"})),
        _detections(("LeeSin",)),
    )

    assert decision.accepted is False
    assert decision.reason == "ungrounded_champion:ekko"


def test_detector_seen_champion_is_grounded_even_if_not_in_roster() -> None:
    clock = FakeClock()
    callout = _callout(
        decision="Ekko is showing mid",
        reason="Ekko is visible on the minimap.",
    )

    decision = _filter(clock).should_speak(
        callout,
        _state(frozenset({"Hecarim", "Jinx", "Ahri", "LeeSin"})),
        _detections(("Ekko",)),
    )

    assert decision.accepted is True


def test_champion_aliases_are_detected() -> None:
    text = "Lee Sin paths toward Kai'Sa while Vel Koz and Wukong reset."

    mentions = find_champion_mentions(
        text,
        {"LeeSin", "Kaisa", "Velkoz", "MonkeyKing"},
    )

    assert mentions == frozenset({"leesin", "kaisa", "velkoz", "monkeyking"})


def test_short_champion_names_do_not_match_inside_words() -> None:
    mentions = find_champion_mentions(
        "Place vision before the river fight.",
        {"Vi"},
    )

    assert mentions == frozenset()
