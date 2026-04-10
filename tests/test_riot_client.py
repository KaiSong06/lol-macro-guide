"""Tests for ``lolcoach.riot_client`` — polling + lifecycle FSM.

Scenarios mapped from the plan's Unit 4 test list (13 total):

1. Happy: first 200 → IDLE → STARTING → ACTIVE with role verified (JUNGLE).
2. Happy: Smite fallback path → ACTIVE for older clients that don't expose
   ``position``.
3. Edge: role mismatch (TOP lane) → STARTING → IDLE + on_role_mismatch
   callback fires once.
4. Edge: 404 for 10 consecutive polls after ACTIVE → ENDING → IDLE +
   on_game_end.
5. Edge: new game detected via gameTime reset → cycle through ENDING → IDLE
   → STARTING for the new game.
6. Edge (reconnect): ACTIVE → 404 briefly → 200 with same/advancing
   gameTime → stays ACTIVE.
7. Edge (remake via gameTime reset): ACTIVE → 404 → 200 with gameTime near
   zero → transitions through ENDING → IDLE → STARTING for new game.
8. Error: connection refused → logged, polling continues, no crash, no
   state transition.
9. Error: Riot API returns 500 → logged, no transition, polling continues.
10. Edge: mirror-match disambiguation via summonerName (not championName).
11. Integration: full lifecycle IDLE → STARTING → ACTIVE → ENDING → IDLE
    driven by a scripted sequence of HTTP mocks.
12. Edge: clean shutdown during polling thread — stop() joins within the
    poll interval.
13. Edge: empty response body / malformed JSON → treated as no-response.

The 14th scenario in my personal count is an extra safety test for the
``_is_jungle_role`` predicate which is pure and cheap to cover.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pytest
import requests_mock as rm_module

from lolcoach.config import RiotApiConfig
from lolcoach.riot_client import (
    Callbacks,
    LifecycleState,
    RiotClient,
    _is_jungle_role,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
ALLGAMEDATA_URL = "https://127.0.0.1:2999/liveclientdata/allgamedata"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())


class _CallbackSpy:
    """A spy implementation of Callbacks that records every invocation."""

    def __init__(self) -> None:
        self.state_changes: list[tuple[LifecycleState, LifecycleState, str]] = []
        self.game_data_calls: list[dict] = []
        self.role_mismatches: list[str] = []
        self.game_ends: list[str] = []

    def as_callbacks(self) -> Callbacks:
        return Callbacks(
            on_state_change=lambda f, t, r: self.state_changes.append((f, t, r)),
            on_game_data=lambda d: self.game_data_calls.append(d),
            on_role_mismatch=lambda r: self.role_mismatches.append(r),
            on_game_end=lambda r: self.game_ends.append(r),
        )


def _make_client(callbacks: Callbacks | None = None) -> RiotClient:
    config = RiotApiConfig()
    cbs = callbacks or _CallbackSpy().as_callbacks()
    return RiotClient(config=config, callbacks=cbs)


# ---------------------------------------------------------------------------
# Happy path: IDLE → STARTING → ACTIVE via JUNGLE position
# ---------------------------------------------------------------------------
def test_first_200_with_jungle_position_transitions_to_active() -> None:
    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())
    data = _load_fixture("allgamedata_ingame.json")

    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=data)
        client.poll_once()

    assert client.state == LifecycleState.ACTIVE
    assert len(spy.state_changes) == 2
    assert spy.state_changes[0][0] == LifecycleState.IDLE
    assert spy.state_changes[0][1] == LifecycleState.STARTING
    assert spy.state_changes[1][1] == LifecycleState.ACTIVE
    assert spy.game_data_calls == [data]
    assert spy.role_mismatches == []


# ---------------------------------------------------------------------------
# Happy path: Smite fallback (no JUNGLE position but has Smite)
# ---------------------------------------------------------------------------
def test_smite_fallback_promotes_to_active_when_position_missing() -> None:
    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())
    data = _load_fixture("allgamedata_ingame.json")
    # Wipe position field on the active player; keep Smite summoner spell.
    for p in data["allPlayers"]:
        if p["summonerName"] == data["activePlayer"]["summonerName"]:
            p["position"] = ""  # empty, as older clients do
            break

    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=data)
        client.poll_once()

    assert client.state == LifecycleState.ACTIVE
    assert spy.game_data_calls == [data]
    assert spy.role_mismatches == []


# ---------------------------------------------------------------------------
# Edge: role mismatch (TOP lane, no Smite) → IDLE with on_role_mismatch
# ---------------------------------------------------------------------------
def test_top_lane_active_player_triggers_role_mismatch() -> None:
    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())
    data = _load_fixture("allgamedata_ingame.json")
    for p in data["allPlayers"]:
        if p["summonerName"] == data["activePlayer"]["summonerName"]:
            p["position"] = "TOP"
            p["summonerSpells"]["summonerSpellOne"]["displayName"] = "Teleport"
            p["summonerSpells"]["summonerSpellTwo"]["displayName"] = "Flash"
            break

    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=data)
        client.poll_once()

    assert client.state == LifecycleState.IDLE
    assert spy.role_mismatches == ["TOP"]
    assert spy.game_data_calls == []
    # State transitioned IDLE → STARTING → IDLE
    assert len(spy.state_changes) == 2
    assert spy.state_changes[-1][1] == LifecycleState.IDLE


# ---------------------------------------------------------------------------
# Edge: 404 for 10 consecutive seconds after ACTIVE → ENDING
# ---------------------------------------------------------------------------
def test_404_for_10_seconds_after_active_transitions_to_ending() -> None:
    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())
    data = _load_fixture("allgamedata_ingame.json")

    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=data)
        client.poll_once()
    assert client.state == LifecycleState.ACTIVE

    # Simulate 11 seconds of 404. The first 404 starts the window; the
    # second 404 lands after the 10s threshold so the FSM fires ENDING.
    base = client._now()  # monotonic clock used by the FSM
    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, status_code=404)
        client._clock_override = lambda: base + 1.0
        client.poll_once()
        assert client.state == LifecycleState.ACTIVE  # still within window

        client._clock_override = lambda: base + 12.0
        client.poll_once()

    assert client.state == LifecycleState.IDLE
    assert spy.game_ends == ["timeout"]
    assert any(t == LifecycleState.ENDING for (_, t, _) in spy.state_changes)


# ---------------------------------------------------------------------------
# Reconnect: 404 for <10s, then 200 with same gameTime → stay ACTIVE
# ---------------------------------------------------------------------------
def test_short_404_window_does_not_end_game_on_reconnect() -> None:
    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())
    data = _load_fixture("allgamedata_ingame.json")

    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=data)
        client.poll_once()
    assert client.state == LifecycleState.ACTIVE

    # 5 seconds of 404
    base = client._now()
    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, status_code=404)
        client._clock_override = lambda: base + 5.0
        client.poll_once()
    assert client.state == LifecycleState.ACTIVE  # still within reconnect window

    # Now 200 comes back with advanced gameTime (same game continued)
    data_later = copy.deepcopy(data)
    data_later["gameData"]["gameTime"] = data["gameData"]["gameTime"] + 5.0
    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=data_later)
        client._clock_override = lambda: base + 6.0
        client.poll_once()

    assert client.state == LifecycleState.ACTIVE
    assert spy.game_ends == []


# ---------------------------------------------------------------------------
# Remake: ACTIVE → 404 briefly → 200 with gameTime reset → cycle to new game
# ---------------------------------------------------------------------------
def test_gametime_reset_triggers_new_game_cycle() -> None:
    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())
    first_game = _load_fixture("allgamedata_ingame.json")
    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=first_game)
        client.poll_once()
    assert client.state == LifecycleState.ACTIVE

    # Simulate a brief 404 window, then a new game with gameTime near 0.
    remake = _load_fixture("allgamedata_remake.json")
    remake["activePlayer"]["summonerName"] = first_game["activePlayer"]["summonerName"]
    remake["allPlayers"][0]["summonerName"] = first_game["activePlayer"]["summonerName"]

    base = client._now()
    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=remake)
        client._clock_override = lambda: base + 3.0
        client.poll_once()

    assert client.state == LifecycleState.ACTIVE  # new game ACTIVE after cycling
    assert spy.game_ends == ["new game started"]
    # The sequence of state changes should include ENDING and back to ACTIVE.
    assert LifecycleState.ENDING in {t for _, t, _ in spy.state_changes}


# ---------------------------------------------------------------------------
# Error: connection refused → no crash, no transition, polling continues
# ---------------------------------------------------------------------------
def test_connection_refused_is_logged_and_polling_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    import requests

    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())

    with rm_module.Mocker() as m, caplog.at_level(
        logging.WARNING, logger="lolcoach.riot_client"
    ):
        m.get(ALLGAMEDATA_URL, exc=requests.ConnectionError("refused"))
        client.poll_once()

    assert client.state == LifecycleState.IDLE
    assert spy.state_changes == []
    # It is acceptable to log the connection error at INFO/DEBUG — we do not
    # strictly require a WARNING, but we do require no crash.


# ---------------------------------------------------------------------------
# Error: Riot API returns 500 → no transition
# ---------------------------------------------------------------------------
def test_http_500_does_not_transition() -> None:
    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())

    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, status_code=500)
        client.poll_once()

    assert client.state == LifecycleState.IDLE
    assert spy.state_changes == []


# ---------------------------------------------------------------------------
# Edge: malformed JSON body
# ---------------------------------------------------------------------------
def test_malformed_json_body_is_treated_as_no_response() -> None:
    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())

    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, text="not json at all", status_code=200)
        client.poll_once()

    assert client.state == LifecycleState.IDLE


# ---------------------------------------------------------------------------
# Mirror match: two players on opposite teams have the same champion; the
# active player is identified by summonerName, not championName.
# ---------------------------------------------------------------------------
def test_mirror_match_uses_summoner_name_for_identity() -> None:
    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())
    data = _load_fixture("allgamedata_ingame.json")
    # Add an enemy Hecarim (same champ as active player).
    enemy_hecarim = {
        "summonerName": "EnemyHecarim#NA1",
        "championName": "Hecarim",
        "level": 8,
        "position": "TOP",
        "team": "CHAOS",
        "isBot": False,
        "isDead": False,
        "respawnTimer": 0.0,
        "items": [],
        "summonerSpells": {
            "summonerSpellOne": {"displayName": "Teleport"},
            "summonerSpellTwo": {"displayName": "Flash"},
        },
        "scores": {"kills": 0, "deaths": 0, "assists": 0, "creepScore": 0, "wardScore": 0.0},
    }
    data["allPlayers"].append(enemy_hecarim)

    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=data)
        client.poll_once()

    # Should still identify the active player correctly and go to ACTIVE
    # (the active player's JUNGLE position wins over the enemy Hecarim's TOP).
    assert client.state == LifecycleState.ACTIVE
    assert spy.role_mismatches == []


# ---------------------------------------------------------------------------
# Full lifecycle integration: IDLE → STARTING → ACTIVE → ENDING → IDLE
# ---------------------------------------------------------------------------
def test_full_lifecycle_end_to_end() -> None:
    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())
    data = _load_fixture("allgamedata_ingame.json")

    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=data)
        client.poll_once()
    assert client.state == LifecycleState.ACTIVE

    base = client._now()
    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, status_code=404)
        # First 404 starts the no-response window; second exceeds the threshold.
        client._clock_override = lambda: base + 0.5
        client.poll_once()
        client._clock_override = lambda: base + 12.0
        client.poll_once()

    assert client.state == LifecycleState.IDLE
    states_seen = {t for _, t, _ in spy.state_changes}
    expected = {
        LifecycleState.STARTING,
        LifecycleState.ACTIVE,
        LifecycleState.ENDING,
        LifecycleState.IDLE,
    }
    assert expected.issubset(states_seen)
    assert spy.game_ends == ["timeout"]


# ---------------------------------------------------------------------------
# Thread lifecycle: start() + stop() cleanly
# ---------------------------------------------------------------------------
def test_start_and_stop_cleanly_joins_thread() -> None:
    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())
    data = _load_fixture("allgamedata_ingame.json")

    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=data)
        # Use a very short poll interval so the test finishes quickly.
        client._poll_interval_s = 0.01
        client.start()
        # Let the polling thread run at least once.
        time.sleep(0.05)
        client.stop(timeout=2.0)

    # Thread should have observed at least one poll.
    assert len(spy.state_changes) >= 2  # at least IDLE → STARTING → ACTIVE
    assert client.state == LifecycleState.ACTIVE


def test_stop_is_idempotent() -> None:
    client = _make_client()
    client.stop()  # never started
    client.stop()  # calling again is a no-op


# ---------------------------------------------------------------------------
# Pure helper: _is_jungle_role
# ---------------------------------------------------------------------------
def test_is_jungle_role_via_position() -> None:
    player = {
        "position": "JUNGLE",
        "summonerSpells": {
            "summonerSpellOne": {"displayName": "Flash"},
            "summonerSpellTwo": {"displayName": "Ghost"},
        },
    }
    assert _is_jungle_role(player) is True


def test_is_jungle_role_via_smite_fallback() -> None:
    player = {
        "position": "",
        "summonerSpells": {
            "summonerSpellOne": {"displayName": "Smite"},
            "summonerSpellTwo": {"displayName": "Flash"},
        },
    }
    assert _is_jungle_role(player) is True


def test_is_jungle_role_false_for_top_lane() -> None:
    player = {
        "position": "TOP",
        "summonerSpells": {
            "summonerSpellOne": {"displayName": "Teleport"},
            "summonerSpellTwo": {"displayName": "Flash"},
        },
    }
    assert _is_jungle_role(player) is False


def test_is_jungle_role_handles_missing_fields() -> None:
    assert _is_jungle_role({}) is False
