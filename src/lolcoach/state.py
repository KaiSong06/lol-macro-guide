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
    #: The active player's champion name (e.g. "Hecarim"). Sourced from
    #: ``allPlayers[i].championName`` for the entry whose summonerName
    #: matches the active player. ``None`` until the first Riot poll
    #: identifies the active player; the prompt builder uses this to
    #: tell the LLM "you are coaching a Hecarim jungler".
    active_player_champion: str | None = None
    #: The active player's current gold (integer-truncated). Sourced from
    #: ``activePlayer.currentGold``. ``None`` if the field is missing or
    #: non-numeric — the prompt builder treats ``None`` as "unknown" and
    #: omits the gold line rather than telling the LLM the player is broke.
    active_player_gold: int | None = None
    ally_champions: frozenset[str] = frozenset()
    enemy_champions: frozenset[str] = frozenset()
    all_champions: frozenset[str] = frozenset()
    enemy_jungler_champion_name: str | None = None
    enemy_jungler_summoner_name: str | None = None
    #: Last confirmed sighting of the enemy jungler as
    #: ``(quadrant_name, game_time_seconds_at_detection)``. ``None`` until
    #: the first detector hit — the cold-start rule (R6 precondition)
    #: says the prompt builder must OMIT the "Enemy jungler" line entirely
    #: in that case, never substitute a guess. The timestamp half is in
    #: **game-clock** seconds (not wall-clock), so ``state.game_time_seconds
    #: - ts`` gives a correct "~Ns ago" delta for the user prompt.
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
        self._active_player_champion: str | None = None
        self._active_player_gold: int | None = None
        self._ally_champions: set[str] = set()
        self._enemy_champions: set[str] = set()
        self._enemy_jungler_champion_name: str | None = None
        self._enemy_jungler_summoner_name: str | None = None
        self._enemy_jungler_last_seen: tuple[str, float] | None = None
        self._kill_feed: deque[dict[str, Any]] = deque(maxlen=KILL_FEED_MAX)
        self._riot_events: list[dict[str, Any]] = []
        self._raw_riot_data: dict[str, Any] | None = None
        # EventID high-water mark: Riot's Live Client Data API returns the
        # cumulative event history on every poll, so we must track which
        # events we've already processed or every repeat poll will re-fire
        # the same ChampionKill and corrupt state (F1 regression).
        self._processed_event_ids: set[int] = set()

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

            # Coerce currentGold to int defensively. Real Riot payloads return
            # a float (e.g. 3200.456), but older clients sometimes omit the
            # field entirely and external data is external data, so a missing
            # or unparseable value collapses to None rather than 0 — the
            # prompt builder treats None as "unknown" and omits the line.
            gold_raw = active.get("currentGold")
            if gold_raw is None:
                self._active_player_gold = None
            else:
                try:
                    self._active_player_gold = int(gold_raw)
                except (TypeError, ValueError):
                    self._active_player_gold = None

            all_players = data.get("allPlayers") or []
            active_player_entry = None
            for p in all_players:
                if p.get("summonerName") == self._active_summoner_name:
                    active_player_entry = p
                    break

            self._active_player_champion = (
                (active_player_entry or {}).get("championName")
            )

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
            # Riot returns the full cumulative event history on every poll.
            # Skip any event whose EventID we've already seen — otherwise
            # every ChampionKill re-fires on every poll, corrupting
            # kill_feed with duplicates and permanently wiping
            # enemy_jungler_last_seen after the first enemy jungler death.
            for ev in events:
                event_id = ev.get("EventID")
                if not isinstance(event_id, int):
                    # Missing/non-int EventID shouldn't happen, but fall
                    # back to a per-call replay rather than crashing —
                    # better to over-log than to silently drop.
                    if ev.get("EventName") == "ChampionKill":
                        self._append_kill_locked(ev)
                    continue
                if event_id in self._processed_event_ids:
                    continue
                self._processed_event_ids.add(event_id)
                if ev.get("EventName") == "ChampionKill":
                    self._append_kill_locked(ev)

    def update_from_detector(self, result: DetectionResult) -> None:
        """Update enemy jungler last-seen from the detector output.

        Only fires when the detector saw a champion whose name matches the
        identified enemy jungler. Empty-detection frames do not clobber
        the last-seen value.

        The stored timestamp uses **game-clock** (``self._game_time_seconds``)
        rather than wall-clock (``result.detected_at``). Every downstream
        consumer — snapshot's ``predict_quadrant`` call AND
        :func:`lolcoach.prompts.build_user_prompt`'s age computation — needs
        to compute "how much game-time has elapsed since last-seen", and
        mixing wall-clock and game-clock produced a silent ~0s ago clamp
        in production that defeated the whole purpose of telling the LLM
        how stale the sighting is. Using game-clock everywhere makes the
        "age" field mean what it says.

        The trade-off: ``self._game_time_seconds`` is refreshed by Riot
        polling every ~2s, so the stored game-clock can lag the true
        detection moment by up to the Riot poll interval. That's inside
        jungle coaching granularity ("~30s ago" vs "~32s ago" is the same
        coaching decision) and the wall-clock alternative silently clamped
        everything to 0s, which was much worse.
        """
        with self._lock:
            enemy_jungler_champ = self._enemy_jungler_champion_name
            if not enemy_jungler_champ:
                return
            for champ in result.champions:
                if champ.team == "enemy" and champ.name == enemy_jungler_champ:
                    self._enemy_jungler_last_seen = (
                        champ.quadrant,
                        self._game_time_seconds,
                    )
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
            self._active_player_champion = None
            self._active_player_gold = None
            self._ally_champions = set()
            self._enemy_champions = set()
            self._enemy_jungler_champion_name = None
            self._enemy_jungler_summoner_name = None
            self._enemy_jungler_last_seen = None
            self._kill_feed.clear()
            self._riot_events = []
            self._raw_riot_data = None
            self._processed_event_ids.clear()

    def snapshot(self) -> GameState:
        """Return an immutable deep-copy snapshot of the current state.

        Safe to call from any thread. The returned value does not share any
        mutable structure with the manager.
        """
        with self._lock:
            # Compute the predicted quadrant at snapshot time so it reflects
            # the latest game time. ``last_seen`` is stored in game-clock
            # by ``update_from_detector``, and ``_game_time_seconds`` is
            # game-clock, so the subtraction is in consistent units. During
            # active play the game clock advances at real-time rate, so
            # ``predict_quadrant`` still receives real-world elapsed
            # seconds (which is what the jungle-path timings assume).
            # During a pause, game_time freezes and "elapsed" correctly
            # stops advancing — the enemy jungler isn't actually moving.
            predicted = None
            if self._enemy_jungler_last_seen is not None:
                quad, ts = self._enemy_jungler_last_seen
                elapsed = max(0.0, self._game_time_seconds - ts)
                predicted = predict_quadrant(quad, elapsed)

            return GameState(
                game_time_seconds=self._game_time_seconds,
                active_summoner_name=self._active_summoner_name,
                active_player_champion=self._active_player_champion,
                active_player_gold=self._active_player_gold,
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
