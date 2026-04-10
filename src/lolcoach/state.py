"""Thread-safe game state store.

The :class:`StateManager` aggregates the two sources of truth:

1. Riot's Live Client Data API — which every 2 seconds gives us game time,
   active player stats, team rosters, summoner spells, and the event feed.
2. The OpenCV template-match detector — which every 500 ms tells us where
   each known champion is on the minimap (and therefore updates
   ``enemy_jungler_last_seen`` whenever the enemy jungler appears).

All mutations take a single :class:`threading.RLock`. Reads go through
:meth:`StateManager.snapshot` which deep-copies the state under the lock
and releases before returning — the inference worker can run a 5-7 second
Ollama call on its snapshot without blocking the capture or Riot threads.

Every derived field is either computed on snapshot (e.g.,
``enemy_jungler_predicted_quadrant``) or stored as an immutable value
type. :class:`GameState` itself is frozen.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from lolcoach.detector import DetectionResult
from lolcoach.jungle_routes import predict_quadrant

logger = logging.getLogger(__name__)

#: Maximum number of kill-feed events retained in the bounded deque.
KILL_FEED_MAX = 50


# ---------------------------------------------------------------------------
# Immutable snapshot value type
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GameState:
    """Immutable snapshot of the aggregated game state.

    Returned by :meth:`StateManager.snapshot`. Callers may stash this and
    pass it across threads — it is a pure value type and never shares
    mutable state with the manager.
    """

    game_time_seconds: float = 0.0
    active_summoner_name: str | None = None
    ally_champions: frozenset[str] = frozenset()
    enemy_champions: frozenset[str] = frozenset()
    all_champions: frozenset[str] = frozenset()
    enemy_jungler_champion_name: str | None = None
    enemy_jungler_summoner_name: str | None = None
    enemy_jungler_last_seen: tuple[str, float] | None = None
    enemy_jungler_predicted_quadrant: str | None = None
    kill_feed: tuple[dict[str, Any], ...] = ()
    riot_events: tuple[dict[str, Any], ...] = ()
    raw_riot_data: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------
class StateManager:
    """Mutable, thread-safe game state. Reads go through :meth:`snapshot`."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._game_time_seconds: float = 0.0
        self._active_summoner_name: str | None = None
        self._ally_champions: set[str] = set()
        self._enemy_champions: set[str] = set()
        self._enemy_jungler_champion_name: str | None = None
        self._enemy_jungler_summoner_name: str | None = None
        self._enemy_jungler_last_seen: tuple[str, float] | None = None
        self._kill_feed: deque[dict[str, Any]] = deque(maxlen=KILL_FEED_MAX)
        self._riot_events: list[dict[str, Any]] = []
        self._raw_riot_data: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Public updates
    # ------------------------------------------------------------------
    def update_from_riot(self, data: dict[str, Any]) -> None:
        """Merge a fresh Live Client Data API response into state.

        Does NOT clobber detector-sourced fields (``enemy_jungler_last_seen``).
        """
        with self._lock:
            self._raw_riot_data = data
            game_data = data.get("gameData") or {}
            try:
                self._game_time_seconds = float(game_data.get("gameTime", 0.0))
            except (TypeError, ValueError):
                self._game_time_seconds = 0.0

            active = data.get("activePlayer") or {}
            self._active_summoner_name = active.get("summonerName")

            all_players = data.get("allPlayers") or []
            active_player_entry = None
            for p in all_players:
                if p.get("summonerName") == self._active_summoner_name:
                    active_player_entry = p
                    break

            active_team = (active_player_entry or {}).get("team")
            ally: set[str] = set()
            enemy: set[str] = set()
            for p in all_players:
                champ = p.get("championName")
                if not champ:
                    continue
                if active_team is not None and p.get("team") == active_team:
                    ally.add(champ)
                elif active_team is not None:
                    enemy.add(champ)
                else:
                    # Couldn't determine teams — treat everyone as ally (defensive)
                    ally.add(champ)
            self._ally_champions = ally
            self._enemy_champions = enemy

            # Identify the enemy jungler.
            enemy_jungler_champ = None
            enemy_jungler_summoner = None
            for p in all_players:
                if active_team is None or p.get("team") == active_team:
                    continue
                if _is_jungle_entry(p):
                    enemy_jungler_champ = p.get("championName")
                    enemy_jungler_summoner = p.get("summonerName")
                    break
            self._enemy_jungler_champion_name = enemy_jungler_champ
            self._enemy_jungler_summoner_name = enemy_jungler_summoner

            events = (data.get("events") or {}).get("Events") or []
            self._riot_events = list(events)
            # Append new kill events to the bounded kill feed.
            for ev in events:
                if ev.get("EventName") == "ChampionKill":
                    self._append_kill_locked(ev)

    def update_from_detector(self, result: DetectionResult) -> None:
        """Update enemy jungler last-seen from the detector output.

        Only fires when the detector saw a champion whose name matches the
        identified enemy jungler. Empty-detection frames do not clobber
        the last-seen value.
        """
        with self._lock:
            enemy_jungler_champ = self._enemy_jungler_champion_name
            if not enemy_jungler_champ:
                return
            for champ in result.champions:
                if champ.team == "enemy" and champ.name == enemy_jungler_champ:
                    self._enemy_jungler_last_seen = (champ.quadrant, result.detected_at)
                    return

    def record_kill(self, event: dict[str, Any]) -> None:
        """Append a kill event manually (also called from update_from_riot)."""
        with self._lock:
            self._append_kill_locked(event)

    def reset(self) -> None:
        """Clear all state back to fresh defaults."""
        with self._lock:
            self._game_time_seconds = 0.0
            self._active_summoner_name = None
            self._ally_champions = set()
            self._enemy_champions = set()
            self._enemy_jungler_champion_name = None
            self._enemy_jungler_summoner_name = None
            self._enemy_jungler_last_seen = None
            self._kill_feed.clear()
            self._riot_events = []
            self._raw_riot_data = None

    def snapshot(self) -> GameState:
        """Return an immutable deep-copy snapshot of the current state.

        Safe to call from any thread. The returned value does not share any
        mutable structure with the manager.
        """
        with self._lock:
            # Compute the predicted quadrant at snapshot time so it reflects
            # the latest game time.
            predicted = None
            if self._enemy_jungler_last_seen is not None:
                quad, ts = self._enemy_jungler_last_seen
                elapsed = max(0.0, time.time() - ts)
                predicted = predict_quadrant(quad, elapsed)

            return GameState(
                game_time_seconds=self._game_time_seconds,
                active_summoner_name=self._active_summoner_name,
                ally_champions=frozenset(self._ally_champions),
                enemy_champions=frozenset(self._enemy_champions),
                all_champions=frozenset(
                    self._ally_champions | self._enemy_champions
                ),
                enemy_jungler_champion_name=self._enemy_jungler_champion_name,
                enemy_jungler_summoner_name=self._enemy_jungler_summoner_name,
                enemy_jungler_last_seen=self._enemy_jungler_last_seen,
                enemy_jungler_predicted_quadrant=predicted,
                kill_feed=tuple(dict(ev) for ev in self._kill_feed),
                riot_events=tuple(dict(ev) for ev in self._riot_events),
                raw_riot_data=_deepish_copy(self._raw_riot_data),
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _append_kill_locked(self, event: dict[str, Any]) -> None:
        """Append a kill event under the lock."""
        self._kill_feed.append(dict(event))
        # If the enemy jungler was killed, clear their last-seen.
        victim = event.get("VictimName")
        if (
            victim
            and self._enemy_jungler_summoner_name
            and victim == self._enemy_jungler_summoner_name
        ):
            self._enemy_jungler_last_seen = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_jungle_entry(player: dict[str, Any]) -> bool:
    """Local copy of the jungle-role predicate to avoid importing riot_client.

    Duplication is cheap and breaks the import cycle (riot_client already
    depends on logging_utils and config; state is downstream).
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


def _deepish_copy(data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a defensive copy of a Riot API dict.

    Not a true ``copy.deepcopy`` because the Riot payload is plain JSON with
    no nested mutable objects beyond dicts and lists — a recursive dict/list
    clone is sufficient and much cheaper than ``copy.deepcopy``.
    """
    if data is None:
        return None
    return _clone(data)  # type: ignore[no-any-return]


def _clone(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone(v) for v in value]
    return value
