"""Tests for ``lolcoach.state`` — thread-safe game state store.

Scenarios from the implementation plan's Unit 5 test list (15+ total):

1.  ``update_from_riot`` parses an allgamedata fixture into populated fields.
2.  ``update_from_detector`` with an enemy jungler detection sets
    ``enemy_jungler_last_seen``.
3.  ``snapshot()`` returns a deep copy — callers cannot mutate stored state.
4.  Kill feed is bounded to ``KILL_FEED_MAX`` (50) via deque(maxlen=).
5.  An enemy-jungler kill event clears ``enemy_jungler_last_seen``.
6.  Cold start: fresh StateManager has all derived fields set to ``None``.
7.  ``reset()`` clears everything — no carryover between games.
8.  ``update_from_riot`` preserves detector-sourced fields that Riot doesn't
    provide.
9.  Mirror match: enemy Hecarim detection does not overwrite ally Hecarim.
10. Paused game (game_time frozen across two updates) accepts updates.
11. Unknown summoner in kill event still appends to the feed with a log.
12. ``all_champions`` is populated on the first Riot API update.
13. Thread safety: concurrent writers + reader → consistent snapshots.
14. Thread safety: concurrent reset + writer → valid final state.
15. ``update_from_detector`` with no enemies does not mutate last_seen.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from lolcoach.detector import ChampionDetection, DetectionResult
from lolcoach.state import KILL_FEED_MAX, GameState, StateManager

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())


# ---------------------------------------------------------------------------
# Basic update + snapshot
# ---------------------------------------------------------------------------
def test_fresh_state_manager_has_cold_start_defaults() -> None:
    sm = StateManager()
    snap = sm.snapshot()

    assert isinstance(snap, GameState)
    assert snap.game_time_seconds == 0.0
    assert snap.enemy_jungler_last_seen is None
    assert snap.enemy_jungler_predicted_quadrant is None
    assert snap.all_champions == frozenset()
    assert snap.kill_feed == ()


def test_update_from_riot_populates_fields() -> None:
    sm = StateManager()
    data = _load_fixture("allgamedata_ingame.json")
    sm.update_from_riot(data)

    snap = sm.snapshot()
    assert snap.game_time_seconds == 742.5
    assert snap.active_summoner_name == "ActivePlayer#NA1"
    assert "Hecarim" in snap.ally_champions
    assert "Jinx" in snap.ally_champions
    assert "LeeSin" in snap.enemy_champions
    assert "Ekko" in snap.enemy_champions
    assert snap.all_champions == frozenset(
        {"Hecarim", "Jinx", "Ahri", "LeeSin", "Ekko"}
    )
    assert snap.enemy_jungler_champion_name == "LeeSin"


def test_update_from_detector_sets_enemy_jungler_last_seen() -> None:
    sm = StateManager()
    data = _load_fixture("allgamedata_ingame.json")
    sm.update_from_riot(data)

    detection = DetectionResult(
        champions=(
            ChampionDetection(
                name="LeeSin",
                team="enemy",
                position_norm=(0.47, 0.62),
                quadrant="bot_jungle",
                confidence=0.91,
            ),
        ),
        detected_at=1712750000.0,
    )
    sm.update_from_detector(detection)

    snap = sm.snapshot()
    assert snap.enemy_jungler_last_seen is not None
    quad, ts = snap.enemy_jungler_last_seen
    assert quad == "bot_jungle"
    assert ts == 1712750000.0


def test_snapshot_is_independent_of_stored_state() -> None:
    sm = StateManager()
    data = _load_fixture("allgamedata_ingame.json")
    sm.update_from_riot(data)

    snap1 = sm.snapshot()
    # Mutate the returned snapshot's kill_feed tuple... we can't, it's a tuple.
    # Mutate the data we passed in and verify stored state is unaffected.
    data["gameData"]["gameTime"] = 99999.0
    snap2 = sm.snapshot()
    assert snap1.game_time_seconds == snap2.game_time_seconds == 742.5


# ---------------------------------------------------------------------------
# Kill feed + bounded deque
# ---------------------------------------------------------------------------
def test_kill_feed_is_bounded_to_max() -> None:
    sm = StateManager()
    for i in range(KILL_FEED_MAX + 20):
        sm.record_kill({"KillerName": "A", "VictimName": f"V{i}", "EventTime": float(i)})

    snap = sm.snapshot()
    assert len(snap.kill_feed) == KILL_FEED_MAX
    # The oldest 20 should have been dropped; the newest is V<max+19>.
    newest = snap.kill_feed[-1]
    assert newest["VictimName"] == f"V{KILL_FEED_MAX + 19}"


def test_enemy_jungler_kill_clears_last_seen() -> None:
    sm = StateManager()
    data = _load_fixture("allgamedata_ingame.json")
    sm.update_from_riot(data)
    sm.update_from_detector(
        DetectionResult(
            champions=(
                ChampionDetection(
                    name="LeeSin",
                    team="enemy",
                    position_norm=(0.5, 0.6),
                    quadrant="bot_jungle",
                    confidence=0.9,
                ),
            ),
            detected_at=100.0,
        )
    )
    assert sm.snapshot().enemy_jungler_last_seen is not None

    # The Riot kill feed includes the enemy jungler's summoner name as VictimName.
    sm.record_kill({"KillerName": "Ally2#NA1", "VictimName": "Enemy1#NA1", "EventTime": 120.0})

    snap = sm.snapshot()
    assert snap.enemy_jungler_last_seen is None


def test_unknown_summoner_kill_is_still_appended() -> None:
    sm = StateManager()  # no Riot update, so no pre-existing kill events
    sm.record_kill(
        {"KillerName": "Unknown#XX1", "VictimName": "AlsoUnknown#XX2", "EventTime": 5.0}
    )
    snap = sm.snapshot()
    assert len(snap.kill_feed) == 1
    assert snap.kill_feed[0]["KillerName"] == "Unknown#XX1"


# ---------------------------------------------------------------------------
# Reset + mirror match + preservation
# ---------------------------------------------------------------------------
def test_reset_clears_everything() -> None:
    sm = StateManager()
    sm.update_from_riot(_load_fixture("allgamedata_ingame.json"))
    sm.update_from_detector(
        DetectionResult(
            champions=(
                ChampionDetection(
                    name="LeeSin",
                    team="enemy",
                    position_norm=(0.5, 0.5),
                    quadrant="mid",
                    confidence=0.9,
                ),
            ),
            detected_at=10.0,
        )
    )
    sm.record_kill({"KillerName": "A", "VictimName": "B", "EventTime": 1.0})
    assert sm.snapshot().all_champions != frozenset()

    sm.reset()

    snap = sm.snapshot()
    assert snap.game_time_seconds == 0.0
    assert snap.all_champions == frozenset()
    assert snap.ally_champions == frozenset()
    assert snap.enemy_champions == frozenset()
    assert snap.enemy_jungler_last_seen is None
    assert snap.kill_feed == ()


def test_update_from_riot_dedupes_events_by_id() -> None:
    """Riot's Live Client Data API returns the cumulative event history on
    every poll. The StateManager must not re-process the same ChampionKill
    events repeatedly, or the kill_feed fills with duplicates and (worse)
    enemy_jungler_last_seen gets permanently wiped after the first enemy
    jungler death.
    """
    sm = StateManager()
    data = _load_fixture("allgamedata_ingame.json")
    # The fixture has 2 ChampionKill events (EventID 2 and 3).
    expected_kills = [
        e for e in data["events"]["Events"] if e["EventName"] == "ChampionKill"
    ]
    assert len(expected_kills) == 2, "fixture should have 2 ChampionKill events"

    sm.update_from_riot(data)
    assert len(sm.snapshot().kill_feed) == 2

    # Poll the same fixture twice more. No new events — kill_feed must not grow.
    sm.update_from_riot(data)
    assert len(sm.snapshot().kill_feed) == 2

    sm.update_from_riot(data)
    assert len(sm.snapshot().kill_feed) == 2


def test_enemy_jungler_kill_only_clears_last_seen_once() -> None:
    """The regression test for F1. The fixture's EventID=3 event is
    'Ally2#NA1 kills Enemy1#NA1' — Enemy1#NA1 is the enemy jungler.
    Before the EventID dedup fix, every repeat poll re-matched this event
    against self._enemy_jungler_summoner_name and re-cleared last_seen.
    """
    import time as _time

    sm = StateManager()
    data = _load_fixture("allgamedata_ingame.json")
    sm.update_from_riot(data)  # processes both kills; last_seen was None

    # Detector sees the jungler (as if they respawned and walked back to jungle)
    sm.update_from_detector(
        DetectionResult(
            champions=(
                ChampionDetection(
                    name="LeeSin",
                    team="enemy",
                    position_norm=(0.5, 0.5),
                    quadrant="bot_jungle",
                    confidence=0.9,
                ),
            ),
            detected_at=_time.time(),
        )
    )
    assert sm.snapshot().enemy_jungler_last_seen is not None

    # Every subsequent Riot poll carries the same historical kill event.
    # Without EventID dedup, _append_kill_locked re-matches VictimName and
    # re-clears last_seen on every poll. With dedup, last_seen survives.
    sm.update_from_riot(data)
    sm.update_from_riot(data)
    sm.update_from_riot(data)

    assert sm.snapshot().enemy_jungler_last_seen is not None


def test_reset_clears_processed_event_ids() -> None:
    """reset() must also clear the event-id high-water mark so a new game's
    events are processed correctly (Riot resets EventIDs per game).
    """
    sm = StateManager()
    data = _load_fixture("allgamedata_ingame.json")
    sm.update_from_riot(data)
    assert len(sm.snapshot().kill_feed) == 2

    sm.reset()

    # After reset, polling the same events again should re-process them
    # (they look like fresh events from a new game).
    sm.update_from_riot(data)
    assert len(sm.snapshot().kill_feed) == 2


def test_update_from_riot_preserves_detector_last_seen() -> None:
    sm = StateManager()
    sm.update_from_riot(_load_fixture("allgamedata_ingame.json"))
    sm.update_from_detector(
        DetectionResult(
            champions=(
                ChampionDetection(
                    name="LeeSin",
                    team="enemy",
                    position_norm=(0.5, 0.5),
                    quadrant="bot_jungle",
                    confidence=0.9,
                ),
            ),
            detected_at=50.0,
        )
    )

    # A second Riot update arrives; last_seen must not be clobbered.
    sm.update_from_riot(_load_fixture("allgamedata_ingame.json"))
    snap = sm.snapshot()
    assert snap.enemy_jungler_last_seen is not None
    assert snap.enemy_jungler_last_seen[0] == "bot_jungle"


def test_mirror_match_ally_hecarim_not_overwritten_by_enemy_hecarim() -> None:
    sm = StateManager()
    data = _load_fixture("allgamedata_ingame.json")
    # Add an enemy Hecarim (same champion as the ally active player).
    data["allPlayers"].append(
        {
            "summonerName": "EnemyHecarim#NA1",
            "championName": "Hecarim",
            "position": "TOP",
            "team": "CHAOS",
            "summonerSpells": {
                "summonerSpellOne": {"displayName": "Teleport"},
                "summonerSpellTwo": {"displayName": "Flash"},
            },
            "level": 8,
            "isBot": False,
            "isDead": False,
            "respawnTimer": 0.0,
            "items": [],
            "scores": {"kills": 0, "deaths": 0, "assists": 0, "creepScore": 0, "wardScore": 0.0},
        }
    )
    sm.update_from_riot(data)

    snap = sm.snapshot()
    # Both teams have Hecarim; the set contains it once but both rosters
    # should correctly categorize.
    assert "Hecarim" in snap.ally_champions
    assert "Hecarim" in snap.enemy_champions


def test_paused_game_update_with_same_gametime_is_allowed() -> None:
    sm = StateManager()
    data = _load_fixture("allgamedata_ingame.json")
    sm.update_from_riot(data)
    # Second update with the same gameTime — no error, snapshot still valid.
    sm.update_from_riot(data)
    snap = sm.snapshot()
    assert snap.game_time_seconds == 742.5


# ---------------------------------------------------------------------------
# Predicted quadrant integration with jungle_routes
# ---------------------------------------------------------------------------
def test_snapshot_predicted_quadrant_is_none_at_cold_start() -> None:
    sm = StateManager()
    snap = sm.snapshot()
    assert snap.enemy_jungler_predicted_quadrant is None


def test_snapshot_predicted_quadrant_after_detection() -> None:
    import time as _time

    sm = StateManager()
    sm.update_from_riot(_load_fixture("allgamedata_ingame.json"))
    # detected_at must be close to the current wall clock so that snapshot's
    # ``time.time() - ts`` is small (under 30s) and predict_quadrant returns
    # the last-seen quadrant unchanged.
    sm.update_from_detector(
        DetectionResult(
            champions=(
                ChampionDetection(
                    name="LeeSin",
                    team="enemy",
                    position_norm=(0.5, 0.5),
                    quadrant="top_jungle",
                    confidence=0.9,
                ),
            ),
            detected_at=_time.time() - 5.0,  # 5 seconds ago
        )
    )
    snap = sm.snapshot()
    # Short elapsed (we just saw them), predicted should be the last-seen.
    assert snap.enemy_jungler_predicted_quadrant == "top_jungle"


# ---------------------------------------------------------------------------
# Detector update with no enemies does not mutate last_seen
# ---------------------------------------------------------------------------
def test_update_from_detector_with_no_enemies_does_not_clobber() -> None:
    sm = StateManager()
    sm.update_from_riot(_load_fixture("allgamedata_ingame.json"))
    sm.update_from_detector(
        DetectionResult(
            champions=(
                ChampionDetection(
                    name="LeeSin",
                    team="enemy",
                    position_norm=(0.5, 0.5),
                    quadrant="bot_jungle",
                    confidence=0.9,
                ),
            ),
            detected_at=50.0,
        )
    )
    # Next detection has zero enemies — they went into fog.
    sm.update_from_detector(
        DetectionResult(champions=(), detected_at=60.0)
    )

    snap = sm.snapshot()
    # last_seen should remain at bot_jungle — the detector's silence is not
    # a signal to forget.
    assert snap.enemy_jungler_last_seen is not None
    assert snap.enemy_jungler_last_seen[0] == "bot_jungle"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------
def test_concurrent_writers_and_readers_produce_consistent_snapshots() -> None:
    sm = StateManager()
    data = _load_fixture("allgamedata_ingame.json")
    sm.update_from_riot(data)

    num_iterations = 200
    errors: list[str] = []

    def writer_riot() -> None:
        for _ in range(num_iterations):
            sm.update_from_riot(data)

    def writer_detector() -> None:
        for i in range(num_iterations):
            sm.update_from_detector(
                DetectionResult(
                    champions=(
                        ChampionDetection(
                            name="LeeSin",
                            team="enemy",
                            position_norm=(0.5, 0.5),
                            quadrant="bot_jungle",
                            confidence=0.9,
                        ),
                    ),
                    detected_at=float(i),
                )
            )

    def reader() -> None:
        for _ in range(num_iterations):
            snap = sm.snapshot()
            # Invariant: if all_champions is populated, ally+enemy unions to all.
            if snap.all_champions:
                if snap.ally_champions | snap.enemy_champions != snap.all_champions:
                    errors.append("union inconsistency")
                    return

    threads = [
        threading.Thread(target=writer_riot),
        threading.Thread(target=writer_detector),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # Final state still valid
    snap = sm.snapshot()
    assert snap.all_champions == frozenset(
        {"Hecarim", "Jinx", "Ahri", "LeeSin", "Ekko"}
    )


def test_concurrent_reset_and_writer_leave_valid_state() -> None:
    sm = StateManager()
    data = _load_fixture("allgamedata_ingame.json")

    def writer() -> None:
        for _ in range(50):
            sm.update_from_riot(data)

    def resetter() -> None:
        for _ in range(50):
            sm.reset()

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=resetter)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Final state must be internally consistent regardless of interleaving.
    snap = sm.snapshot()
    # It's either the full updated state or the fresh reset state, never a mix.
    if snap.all_champions:
        assert snap.ally_champions | snap.enemy_champions == snap.all_champions
    else:
        assert snap.game_time_seconds == 0.0
        assert snap.enemy_jungler_last_seen is None
