"""Polls Riot's Live Client Data API and drives the game lifecycle FSM.

Owns four responsibilities per the design doc:

1. Poll ``https://127.0.0.1:2999/liveclientdata/allgamedata`` on a timer.
2. Identify the active player and verify they are the jungler.
3. Drive the ``IDLE → STARTING → ACTIVE → ENDING`` state machine, including
   reconnect windows and remake cycles.
4. Fire typed callbacks on state transitions, fresh game data, role
   mismatches, and game ends so ``main.py`` can react without polling.

**Deferred-to-implementation discovery:** the design doc assumed Riot's
endpoint exposes a ``gameData.gameId`` field for remake detection. In
practice the Live Client Data API does not include ``gameId``. This module
uses ``gameData.gameTime`` as a game-identity proxy instead: if we see a
200 response whose ``gameTime`` has jumped backward by more than a small
delta, that's a new game and we cycle through ``ENDING → IDLE → STARTING``.
This is equivalent to the original intent since ``gameTime`` resets to near
zero on every new game and only increases monotonically within a single
game.

Self-signed cert handling: the endpoint uses a self-signed localhost
certificate, so every ``requests`` call sets ``verify=False``. The module
also suppresses urllib3's ``InsecureRequestWarning`` once at import time —
otherwise every poll would log a noisy warning.
"""

from __future__ import annotations

import ipaddress
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import urlparse

import requests
import urllib3

from lolcoach.config import RiotApiConfig

logger = logging.getLogger(__name__)

# Silence the one-time warning about our self-signed localhost cert.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

#: Seconds of consecutive no-response before we declare the game ended.
NO_RESPONSE_ENDING_THRESHOLD_S = 10.0

#: If a 200 response shows a ``gameTime`` that's more than this far behind
#: the last one we saw, treat it as a new game.
GAMETIME_RESET_DELTA_S = 30.0

#: Request timeout for every poll — keep tight so the poll thread doesn't
#: block the orchestration layer on a stuck socket.
REQUEST_TIMEOUT_S = 5.0

ALLGAMEDATA_PATH = "/liveclientdata/allgamedata"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
class LifecycleState(Enum):
    IDLE = "idle"
    STARTING = "starting"
    ACTIVE = "active"
    ENDING = "ending"


@dataclass(frozen=True)
class Callbacks:
    """Typed callback sink passed in by ``main.py``.

    Every field is required so the orchestration layer can't forget to wire
    one. Tests supply spies that record every invocation.
    """

    on_state_change: Callable[[LifecycleState, LifecycleState, str], None]
    on_game_data: Callable[[dict], None]
    on_role_mismatch: Callable[[str], None]
    on_game_end: Callable[[str], None]


# ---------------------------------------------------------------------------
# Pure helpers (module-level for easy testing)
# ---------------------------------------------------------------------------
def _is_jungle_role(player: dict[str, Any]) -> bool:
    """Return True if the player is playing jungle.

    Primary signal: ``position == "JUNGLE"`` (modern client). Fallback
    signal: ``SummonerSmite`` on either summoner spell slot (older clients
    or game modes that don't populate ``position``).
    """
    position = str(player.get("position") or "").upper()
    if position == "JUNGLE":
        return True
    spells = player.get("summonerSpells") or {}
    for slot in ("summonerSpellOne", "summonerSpellTwo"):
        spell = spells.get(slot) or {}
        display = str(spell.get("displayName") or "").strip()
        if display.lower() == "smite":
            return True
    return False


def _find_active_player_entry(data: dict[str, Any]) -> dict[str, Any] | None:
    """Return the ``allPlayers`` entry matching the active player.

    Uses ``summonerName`` for identity — this is important for mirror
    matches where two players on opposite teams have the same champion.
    """
    active = data.get("activePlayer") or {}
    name = active.get("summonerName")
    if not name:
        return None
    for entry in data.get("allPlayers", []):
        if entry.get("summonerName") == name:
            return entry
    return None


def _extract_role(player: dict[str, Any]) -> str:
    """Return a human-readable role label from an ``allPlayers`` entry."""
    pos = str(player.get("position") or "").upper()
    return pos or "UNKNOWN"


def _assert_loopback_base_url(base_url: str) -> None:
    """Reject any ``base_url`` whose host is not a loopback address.

    The client ships with ``verify=False`` because the Live Client Data API
    runs on a self-signed localhost certificate. That bypass is only safe
    when the target is the loopback interface — an attacker who can tamper
    with ``config.yaml`` could otherwise point ``base_url`` at a remote
    host and silently exfiltrate summoner names over an unverified TLS
    connection. This check enforces the loopback invariant at construction
    time so the failure mode is impossible, not merely discouraged.
    """
    parsed = urlparse(base_url)
    host = parsed.hostname
    if not host:
        raise ValueError(
            f"RiotClient base_url must have a hostname; got {base_url!r}"
        )
    # Accept the literal ``localhost`` alias even though it resolves at
    # runtime — DNS resolution against /etc/hosts's localhost entry is
    # effectively loopback for any sane system.
    if host.lower() == "localhost":
        return
    try:
        ip = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError(
            f"RiotClient base_url host must be an IP address or 'localhost' "
            f"(got {host!r}); verify=False TLS bypass is scoped to loopback only"
        ) from exc
    if not ip.is_loopback:
        raise ValueError(
            f"RiotClient base_url must be a loopback address "
            f"(got {host!r}); verify=False TLS bypass is scoped to loopback only"
        )


# ---------------------------------------------------------------------------
# RiotClient
# ---------------------------------------------------------------------------
class RiotClient:
    """Polls the Live Client Data API and drives the lifecycle FSM.

    The client exposes :meth:`poll_once` for synchronous test control and
    :meth:`start` / :meth:`stop` for the real runtime polling thread.
    """

    def __init__(
        self,
        config: RiotApiConfig,
        callbacks: Callbacks,
        session: requests.Session | None = None,
    ) -> None:
        _assert_loopback_base_url(config.base_url)
        self._config = config
        self._callbacks = callbacks
        self._session = session or requests.Session()
        self._url = f"{config.base_url}{ALLGAMEDATA_PATH}"

        self._state_lock = threading.Lock()
        self._state = LifecycleState.IDLE
        self._last_game_time: float | None = None
        self._no_response_since: float | None = None

        # Serializes entire poll cycles so concurrent poll_once() calls
        # (tests or a rogue thread) cannot double-transition the FSM. The
        # _state_lock stays a short lock for external reads via the
        # ``state`` property; this one covers the full read-and-decide
        # block inside a single poll.
        self._poll_lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._poll_interval_s = config.poll_interval_ms / 1000.0

        # Test hook: an override for the monotonic clock. Real code uses
        # :meth:`_now` which returns ``time.monotonic()`` unless overridden.
        self._clock_override: Callable[[], float] | None = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    @property
    def state(self) -> LifecycleState:
        with self._state_lock:
            return self._state

    def start(self) -> None:
        """Spawn the polling thread. Idempotent.

        If an earlier thread is still alive (e.g., a previous :meth:`stop`
        timed out waiting for an in-flight HTTP call to finish), this is
        a no-op — spawning a second thread would race the orphan for the
        state lock and double-transition the FSM.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_poll_loop, name="riot-poll", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = REQUEST_TIMEOUT_S + 2.0) -> None:
        """Signal the polling thread to stop and join it.

        The default timeout is ``REQUEST_TIMEOUT_S + 2.0`` so a poll that's
        mid-HTTP-call has enough slack to finish and the subsequent
        ``stop_event.wait()`` can exit the loop cleanly. A bare
        ``timeout=REQUEST_TIMEOUT_S`` would race the HTTP call and leave
        an orphan thread alive after this returns.

        If the thread is still alive after the join, :attr:`_thread` is
        **not** cleared — a subsequent :meth:`start` will refuse to spawn
        a second thread rather than race the orphan.
        """
        self._stop_event.set()
        thread = self._thread
        if thread is None:
            return
        if thread.is_alive():
            thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning(
                "riot poll thread did not exit within %.1fs; leaving reference "
                "in place (start() will refuse until it exits)",
                timeout,
            )
            return
        self._thread = None

    def poll_once(self) -> None:
        """Execute one poll cycle. Exposed for test control.

        Serialized by ``_poll_lock`` so concurrent calls (tests + real
        thread, or two tests in parallel) cannot race the FSM through a
        double transition.
        """
        with self._poll_lock:
            self._poll_once_locked()

    def _poll_once_locked(self) -> None:
        """Body of :meth:`poll_once`, assumed to run under ``_poll_lock``."""
        try:
            resp = self._session.get(
                self._url, timeout=REQUEST_TIMEOUT_S, verify=False
            )
        except requests.RequestException as exc:
            logger.debug("riot api request failed: %s", exc)
            self._handle_no_response()
            return

        if resp.status_code == 404:
            self._handle_no_response()
            return

        if resp.status_code != 200:
            logger.warning("riot api returned HTTP %d", resp.status_code)
            self._handle_no_response()
            return

        try:
            data = resp.json()
        except ValueError:
            logger.warning("riot api returned non-JSON body")
            self._handle_no_response()
            return
        if not isinstance(data, dict):
            logger.warning("riot api returned unexpected payload type")
            self._handle_no_response()
            return

        self._handle_game_data(data)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _now(self) -> float:
        if self._clock_override is not None:
            return self._clock_override()
        return time.monotonic()

    def _run_poll_loop(self) -> None:  # pragma: no cover - thread loop
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception:
                logger.exception("unexpected error in riot poll loop")
            self._stop_event.wait(self._poll_interval_s)

    def _transition(self, new_state: LifecycleState, reason: str) -> None:
        with self._state_lock:
            from_state = self._state
            self._state = new_state
        logger.info(
            "lifecycle %s -> %s (%s)", from_state.value, new_state.value, reason
        )
        self._callbacks.on_state_change(from_state, new_state, reason)

    def _handle_no_response(self) -> None:
        now = self._now()
        if self._no_response_since is None:
            self._no_response_since = now
        elapsed = now - self._no_response_since
        if (
            self._state == LifecycleState.ACTIVE
            and elapsed >= NO_RESPONSE_ENDING_THRESHOLD_S
        ):
            self._end_current_game("timeout")

    def _handle_game_data(self, data: dict[str, Any]) -> None:
        # Successful 200 clears the no-response window.
        self._no_response_since = None

        game_time = self._extract_game_time(data)

        if self._state == LifecycleState.IDLE:
            self._start_game(data, game_time)
            return

        if self._state == LifecycleState.ACTIVE:
            # Detect a new game: gameTime jumped backward more than the
            # reset delta. This is our proxy for the missing gameId field.
            if (
                self._last_game_time is not None
                and game_time < self._last_game_time - GAMETIME_RESET_DELTA_S
            ):
                self._end_current_game("new game started")
                self._start_game(data, game_time)
                return
            self._last_game_time = game_time
            self._callbacks.on_game_data(data)
            return

        # STARTING or ENDING should not normally receive more game data
        # here — treat defensively.
        if self._state == LifecycleState.STARTING:  # pragma: no cover
            self._start_game(data, game_time)

    def _start_game(self, data: dict[str, Any], game_time: float) -> None:
        """Run the STARTING → ACTIVE (or → IDLE on mismatch) transition."""
        self._transition(LifecycleState.STARTING, "first game data")
        player = _find_active_player_entry(data)
        if player is None:
            logger.warning("could not locate active player in allgamedata")
            self._transition(LifecycleState.IDLE, "active player not found")
            return
        if not _is_jungle_role(player):
            role = _extract_role(player)
            self._callbacks.on_role_mismatch(role)
            self._transition(LifecycleState.IDLE, f"role mismatch: {role}")
            return
        self._last_game_time = game_time
        self._transition(LifecycleState.ACTIVE, "role verified as jungle")
        self._callbacks.on_game_data(data)

    def _end_current_game(self, reason: str) -> None:
        self._transition(LifecycleState.ENDING, reason)
        self._callbacks.on_game_end(reason)
        self._last_game_time = None
        self._no_response_since = None
        self._transition(LifecycleState.IDLE, "cleanup complete")

    @staticmethod
    def _extract_game_time(data: dict[str, Any]) -> float:
        gd = data.get("gameData") or {}
        try:
            return float(gd.get("gameTime", 0.0))
        except (TypeError, ValueError):
            return 0.0
