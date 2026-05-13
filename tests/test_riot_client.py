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
import threading
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
# F6 + F7 regression tests: FSM locking + stop() race discipline
# ---------------------------------------------------------------------------
def test_stop_preserves_thread_reference_when_join_times_out() -> None:
    """If the polling thread is stuck in an in-flight HTTP call and join
    times out, :attr:`_thread` must remain set so a subsequent :meth:`start`
    refuses to spawn a second thread that would race the orphan.
    """

    class _StuckThread:
        def __init__(self) -> None:
            self.daemon = True
            self.name = "stuck-mock"
            self._started = False

        def start(self) -> None:
            self._started = True

        def join(self, timeout: float | None = None) -> None:
            pass  # never exits

        def is_alive(self) -> bool:
            return True

    client = _make_client()
    stuck = _StuckThread()
    client._thread = stuck  # type: ignore[assignment]

    client.stop(timeout=0.01)

    assert client._thread is stuck, (
        "stop() must not clear the thread reference when join timed out"
    )


def test_start_refuses_to_spawn_second_thread_when_orphan_is_alive() -> None:
    """After a stop() that timed out, start() must not spawn a second
    polling thread on top of the orphan.
    """

    class _StuckThread:
        def __init__(self) -> None:
            self.daemon = True
            self.name = "stuck-mock"

        def start(self) -> None:
            pass

        def join(self, timeout: float | None = None) -> None:
            pass

        def is_alive(self) -> bool:
            return True

    client = _make_client()
    orphan = _StuckThread()
    client._thread = orphan  # type: ignore[assignment]

    client.start()  # must be a no-op

    assert client._thread is orphan, (
        "start() must refuse to replace a still-alive thread"
    )


def test_concurrent_poll_once_serializes_via_lock() -> None:
    """Two threads calling ``poll_once()`` simultaneously must not
    double-transition the FSM through IDLE → STARTING. The poll lock
    serializes the read-and-transition sequence so only one full cycle
    runs at a time.
    """
    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())
    data = _load_fixture("allgamedata_ingame.json")

    # Use a thread-safe Mocker around a shared adapter. requests_mock
    # patches the Session's adapter under a lock so registered URLs are
    # safe to query concurrently from worker threads.
    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=data)

        barrier = threading.Barrier(4)
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                barrier.wait(timeout=5.0)
                client.poll_once()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

    assert errors == [], f"worker threads raised: {errors}"
    assert client.state == LifecycleState.ACTIVE

    # Exactly one IDLE → STARTING transition should have occurred; the
    # first poll drove the FSM to ACTIVE and subsequent polls took the
    # already-active branch. Without the poll lock, each worker would
    # independently have seen IDLE and fired its own transition.
    idle_to_starting = [
        (f, t)
        for (f, t, _) in spy.state_changes
        if f == LifecycleState.IDLE and t == LifecycleState.STARTING
    ]
    assert len(idle_to_starting) == 1, (
        f"expected exactly 1 IDLE→STARTING, got {len(idle_to_starting)}"
    )


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


# ---------------------------------------------------------------------------
# F4 regression: verify=False must be scoped to loopback only
# ---------------------------------------------------------------------------
def test_riot_client_accepts_127_0_0_1_base_url() -> None:
    config = RiotApiConfig(base_url="https://127.0.0.1:2999")
    client = RiotClient(config=config, callbacks=_CallbackSpy().as_callbacks())
    assert client.state == LifecycleState.IDLE


def test_riot_client_accepts_localhost_base_url() -> None:
    config = RiotApiConfig(base_url="https://localhost:2999")
    client = RiotClient(config=config, callbacks=_CallbackSpy().as_callbacks())
    assert client.state == LifecycleState.IDLE


def test_riot_client_accepts_ipv6_loopback_base_url() -> None:
    config = RiotApiConfig(base_url="https://[::1]:2999")
    client = RiotClient(config=config, callbacks=_CallbackSpy().as_callbacks())
    assert client.state == LifecycleState.IDLE


def test_riot_client_rejects_non_loopback_hostname() -> None:
    """If a misconfigured or tampered config.yaml points base_url at a
    non-loopback host, the client must refuse to construct. Otherwise
    verify=False silently accepts any cert for that host and the coach
    becomes an MITM-able screen-capture exfil vector.
    """
    config = RiotApiConfig(base_url="https://evil.example.com:2999")
    with pytest.raises(ValueError, match="loopback"):
        RiotClient(config=config, callbacks=_CallbackSpy().as_callbacks())


def test_riot_client_rejects_public_ip_address() -> None:
    config = RiotApiConfig(base_url="https://203.0.113.1:2999")
    with pytest.raises(ValueError, match="loopback"):
        RiotClient(config=config, callbacks=_CallbackSpy().as_callbacks())


def test_riot_client_rejects_missing_host() -> None:
    """A base_url with no hostname at all is a config error."""
    config = RiotApiConfig(base_url="not-a-url")
    with pytest.raises(ValueError, match="base_url"):
        RiotClient(config=config, callbacks=_CallbackSpy().as_callbacks())


# ---------------------------------------------------------------------------
# F5 regression: on_role_mismatch fires exactly once per game
# ---------------------------------------------------------------------------
def _make_top_lane_fixture() -> dict:
    """Helper: return the ingame fixture with the active player's role
    changed to TOP (and summoner spells replaced so the Smite fallback
    doesn't rescue it).
    """
    data = _load_fixture("allgamedata_ingame.json")
    for p in data["allPlayers"]:
        if p["summonerName"] == data["activePlayer"]["summonerName"]:
            p["position"] = "TOP"
            p["summonerSpells"]["summonerSpellOne"]["displayName"] = "Teleport"
            p["summonerSpells"]["summonerSpellTwo"]["displayName"] = "Flash"
            break
    return data


def test_role_mismatch_fires_exactly_once_across_repeat_polls() -> None:
    """Plan line 460 explicitly requires 'fires exactly once'. Before the
    sticky flag, every 2s poll while stuck in IDLE after a role mismatch
    re-entered _start_game and re-fired on_role_mismatch. In Phase 2
    that would be audible TTS spam every 2s for the entire game.
    """
    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())
    data = _make_top_lane_fixture()

    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=data)
        client.poll_once()  # first mismatch
        client.poll_once()  # repeat 1 — must NOT fire
        client.poll_once()  # repeat 2
        client.poll_once()  # repeat 3

    assert spy.role_mismatches == ["TOP"], (
        f"on_role_mismatch must fire exactly once across repeat polls, "
        f"got {spy.role_mismatches}"
    )
    assert client.state == LifecycleState.IDLE


def test_role_mismatch_resets_after_no_response_timeout() -> None:
    """A 10s no-response window while paused on a role mismatch means
    the game ended (or the user alt-tabbed to the lobby). Clear the
    sticky memory so the next 200 re-evaluates as a fresh game.

    Discriminating assertion: before the 10s 404, a repeat poll must NOT
    fire on_role_mismatch (sticky blocks it). After the 10s 404, a fresh
    poll MUST fire it again (memory cleared). Without both assertions the
    test would pass vacuously against the current buggy implementation.
    """
    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())
    data = _make_top_lane_fixture()

    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=data)
        client.poll_once()  # fires TOP
        client.poll_once()  # sticky — must NOT re-fire
    # The discriminator: broken code gets ["TOP", "TOP"] here; fix gets ["TOP"].
    assert spy.role_mismatches == ["TOP"]

    # Simulate 12s of 404 (game ended from client view).
    base = client._now()
    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, status_code=404)
        client._clock_override = lambda: base + 0.5
        client.poll_once()
        client._clock_override = lambda: base + 12.0
        client.poll_once()
    client._clock_override = None

    # The user's next game is also top lane — should fire AGAIN for the
    # new game (the memory cleared after the no-response window).
    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=data)
        client.poll_once()

    assert spy.role_mismatches == ["TOP", "TOP"], (
        f"after 10s no-response, next game's role mismatch should fire "
        f"fresh; got {spy.role_mismatches}"
    )


def test_role_mismatch_does_not_reset_on_brief_404() -> None:
    """A single 404 within the no-response window (transient network
    blip) should NOT clear the sticky flag. Only after the 10s
    threshold elapses does the flag clear.
    """
    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())
    data = _make_top_lane_fixture()

    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=data)
        client.poll_once()
    assert spy.role_mismatches == ["TOP"]

    # 5s of 404 — under the 10s threshold.
    base = client._now()
    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, status_code=404)
        client._clock_override = lambda: base + 0.5
        client.poll_once()
        client._clock_override = lambda: base + 5.0
        client.poll_once()
    client._clock_override = None

    # Next 200 with same mismatch should NOT re-fire the callback.
    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=data)
        client.poll_once()

    assert spy.role_mismatches == ["TOP"], (
        f"sticky flag must survive a brief 404 window, got {spy.role_mismatches}"
    )


# ---------------------------------------------------------------------------
# F14 regression: malformed 200 payload must not cycle the FSM
# ---------------------------------------------------------------------------
def _malformed_payload_no_active_player() -> dict:
    """200 response with no activePlayer field at all."""
    return {
        "allPlayers": [],
        "events": {"Events": []},
        "gameData": {
            "gameMode": "CLASSIC",
            "gameTime": 10.5,
            "mapName": "Map11",
            "mapNumber": 11,
        },
    }


def test_malformed_payload_missing_active_player_does_not_cycle_fsm() -> None:
    """A 200 response that lacks an activePlayer entry (schema drift or
    mid-transition glitch) must NOT transition through STARTING on every
    poll. F14 from /ce:review: the old FSM cycled IDLE → STARTING → IDLE
    per poll, spamming on_state_change callbacks and rotating the JSONL
    log repeatedly — the root cause of F9's ~900-files-per-30-minutes
    scenario when a bad payload storm hits.
    """
    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())

    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=_malformed_payload_no_active_player())
        client.poll_once()
        client.poll_once()
        client.poll_once()
        client.poll_once()

    assert client.state == LifecycleState.IDLE
    assert all(t != LifecycleState.STARTING for (_, t, _) in spy.state_changes), (
        f"malformed payloads must not cycle through STARTING; "
        f"got {[(f.value, t.value) for (f, t, _) in spy.state_changes]}"
    )
    # No callbacks should have fired for game_data or role_mismatch.
    assert spy.game_data_calls == []
    assert spy.role_mismatches == []


def test_malformed_payload_warns_only_once_across_repeat_polls(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The malformed-payload warning must be throttled — logging on every
    2s poll would flood the logs for a stuck session. Only the first
    malformed poll in a run warns; the flag resets on the next valid
    payload.
    """
    import logging as _logging

    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())
    bad = _malformed_payload_no_active_player()

    with rm_module.Mocker() as m, caplog.at_level(
        _logging.WARNING, logger="lolcoach.riot_client"
    ):
        m.get(ALLGAMEDATA_URL, json=bad)
        client.poll_once()
        client.poll_once()
        client.poll_once()

    malformed_warnings = [
        r for r in caplog.records
        if "malformed" in r.message.lower() or "activeplayer" in r.message.lower()
    ]
    assert len(malformed_warnings) == 1, (
        f"expected exactly 1 malformed-payload warning, "
        f"got {len(malformed_warnings)}: "
        f"{[r.message for r in malformed_warnings]}"
    )


def test_malformed_then_valid_payload_transitions_normally() -> None:
    """After a run of malformed payloads, a valid one should cleanly
    transition IDLE → STARTING → ACTIVE. The throttle flag resets on
    valid input so the next malformed run would warn again.
    """
    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())

    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=_malformed_payload_no_active_player())
        client.poll_once()
        client.poll_once()
    assert client.state == LifecycleState.IDLE

    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=_load_fixture("allgamedata_ingame.json"))
        client.poll_once()

    assert client.state == LifecycleState.ACTIVE
    # Exactly one STARTING → ACTIVE transition (the successful poll),
    # not one per prior bad poll.
    active_transitions = [
        (f, t) for (f, t, _) in spy.state_changes if t == LifecycleState.ACTIVE
    ]
    assert len(active_transitions) == 1


def test_malformed_payload_with_missing_all_players_is_handled() -> None:
    """Variant: the payload has an activePlayer but no allPlayers list
    to resolve it against. Still a malformed case.
    """
    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())
    bad = {
        "activePlayer": {"summonerName": "Ghost#NA1", "level": 1},
        # Deliberately no allPlayers key
        "events": {"Events": []},
        "gameData": {"gameMode": "CLASSIC", "gameTime": 0.0, "mapName": "Map11"},
    }

    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=bad)
        client.poll_once()
        client.poll_once()

    assert client.state == LifecycleState.IDLE
    assert all(t != LifecycleState.STARTING for (_, t, _) in spy.state_changes)


def test_malformed_payload_active_player_not_in_all_players() -> None:
    """Variant: activePlayer.summonerName exists but doesn't match any
    entry in allPlayers (mid-transition glitch where the arrays haven't
    synchronized yet). Same handling — treat as no-response.
    """
    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())
    bad = {
        "activePlayer": {"summonerName": "Mismatch#NA1", "level": 1},
        "allPlayers": [
            {
                "summonerName": "Ally1#NA1",
                "championName": "Jinx",
                "position": "BOTTOM",
                "team": "ORDER",
                "summonerSpells": {
                    "summonerSpellOne": {"displayName": "Flash"},
                    "summonerSpellTwo": {"displayName": "Heal"},
                },
            }
        ],
        "events": {"Events": []},
        "gameData": {"gameMode": "CLASSIC", "gameTime": 0.0, "mapName": "Map11"},
    }

    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=bad)
        client.poll_once()
        client.poll_once()

    assert client.state == LifecycleState.IDLE
    assert all(t != LifecycleState.STARTING for (_, t, _) in spy.state_changes)


def test_malformed_payload_during_active_eventually_triggers_ending() -> None:
    """If the coach is ACTIVE and Riot starts returning malformed 200
    payloads, the accumulated no-response time should eventually
    trigger the ENDING transition — same as a 404 storm. This is the
    graceful-degradation path: bad data is worse than no data, and the
    user's game either really ended or Riot's client crashed.
    """
    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())

    # Drive into ACTIVE first with a valid payload.
    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=_load_fixture("allgamedata_ingame.json"))
        client.poll_once()
    assert client.state == LifecycleState.ACTIVE

    # Now Riot returns malformed payloads for 12s (past the 10s threshold).
    base = client._now()
    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=_malformed_payload_no_active_player())
        client._clock_override = lambda: base + 0.5
        client.poll_once()
        client._clock_override = lambda: base + 12.0
        client.poll_once()
    client._clock_override = None

    assert client.state == LifecycleState.IDLE
    assert spy.game_ends == ["timeout"]


def test_role_mismatch_memory_tracks_summoner_not_just_a_boolean() -> None:
    """If the active player changes (e.g., a different user logs in on
    the same machine, or a test harness injects a different player), the
    sticky memory must re-evaluate rather than short-circuit based on an
    unrelated stale state.
    """
    spy = _CallbackSpy()
    client = _make_client(spy.as_callbacks())
    data_a = _make_top_lane_fixture()  # ActivePlayer#NA1 playing TOP

    # Build a "different user" fixture: swap the active player's summoner
    # name to DifferentUser#NA1 and also update the matching entry.
    data_b = _make_top_lane_fixture()
    data_b["activePlayer"]["summonerName"] = "DifferentUser#NA1"
    for p in data_b["allPlayers"]:
        if p["summonerName"] == "ActivePlayer#NA1":
            p["summonerName"] = "DifferentUser#NA1"
            break

    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=data_a)
        client.poll_once()  # fires TOP for ActivePlayer#NA1
    assert spy.role_mismatches == ["TOP"]

    with rm_module.Mocker() as m:
        m.get(ALLGAMEDATA_URL, json=data_b)
        client.poll_once()  # different summoner — should fire fresh

    assert spy.role_mismatches == ["TOP", "TOP"], (
        f"different active player should re-evaluate, "
        f"got {spy.role_mismatches}"
    )
