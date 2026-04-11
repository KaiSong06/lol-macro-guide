"""System and user prompt builders for the local vision LLM (Ollama).

The inference module sends two prompts to ``/api/chat``:

* ``build_system_prompt()`` — a static instruction block that defines the
  response schema, the allowed CATEGORY enum, the allowed TARGET_LANE enum,
  the confidence scale, and the "you are a jungle coach" framing. Constant
  output, never depends on game state.

* ``build_user_prompt(state, detections)`` — a structured context block built
  from the latest ``GameState`` snapshot and the latest ``DetectionResult``.
  The prompt is composed of one line per known context field, in fixed order.

The single load-bearing rule for the user prompt:
**missing data → omit the line entirely.** When the enemy jungler has never
been seen, the "Enemy jungler" line is dropped — it is never replaced with
"unknown" or a guessed default. The same rule applies to ``active_player_gold``,
``active_player_champion``, and the detector visibility line. The exception
is event-stream summaries (kill feed, objectives) where an empty collection
is itself information ("nothing has happened yet"), so we show an explicit
``none`` marker rather than dropping the line.

Why omit instead of "unknown": the LLM tends to reason against any value it
is told. Telling it "Enemy jungler: unknown" produces a callout like "Watch
for the unknown jungler in your jungle". Showing no line at all produces a
clean "no information" signal that the model gracefully ignores.
"""

from __future__ import annotations

from typing import Any

from lolcoach.detector import DetectionResult
from lolcoach.state import GameState

# ---------------------------------------------------------------------------
# Schema enums — exposed so tests and downstream consumers (DecisionFilter
# in Unit 8, the inference parser in Unit 7) share a single source of truth.
# ---------------------------------------------------------------------------
#: Allowed values for the CATEGORY response field. The first two are
#: priority-interrupt-eligible by default in ``TtsConfig.interrupt_categories``.
ALLOWED_CATEGORIES: frozenset[str] = frozenset(
    {
        "gank_warning",
        "counter_gank",
        "pathing",
        "objective_call",
        "recall",
        "vision",
    }
)

#: Allowed values for the TARGET_LANE response field. ``global`` is for
#: callouts that don't target a specific lane (e.g., "Recall now").
ALLOWED_LANES: frozenset[str] = frozenset(
    {"top", "mid", "bot", "jungle", "global"}
)

#: Riot ``EventName`` values that count as objectives in the user prompt.
#: Filtered out of the full ``riot_events`` stream so the LLM only sees
#: meaningful map events, not the full GameStart/MinionsSpawning noise.
_OBJECTIVE_EVENT_NAMES: frozenset[str] = frozenset(
    {
        "DragonKill",
        "BaronKill",
        "HeraldKill",
        "AtakhanKill",
        "TurretKilled",
        "InhibKilled",
    }
)


# ---------------------------------------------------------------------------
# System prompt — static template
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_TEMPLATE = """\
You are a real-time League of Legends jungle coach. You see one minimap \
screenshot per call plus a structured game-state context block. Your job is \
to generate exactly one short, actionable callout for the jungler — \
information they could act on in the next 30 seconds — and nothing else.

Reason about spatial intent (where champions are, where they are likely to \
go, what objective is contested) — do not identify champions from the image. \
The names of champions in this game are listed in the context block. Never \
mention a champion that is not in the context block.

Respond with exactly five lines, in this order, no preamble and no markdown:

DECISION: <one short sentence the coach will speak out loud>
CATEGORY: <one of: {categories}>
TARGET_LANE: <one of: {lanes}>
CONFIDENCE: <integer from 1 to 10, where 10 is highest confidence>
REASON: <one sentence explaining the spatial reasoning behind the decision>

Rules:
- DECISION must be a single sentence under 15 words.
- CATEGORY must be exactly one of the listed values.
- TARGET_LANE must be exactly one of the listed values.
- CONFIDENCE must be an integer in [1, 10].
- REASON must be a single sentence and must only mention champions from \
the context block.
- If you have no useful callout for this frame, return CONFIDENCE: 1 and a \
short generic DECISION — never refuse to answer.
"""

# Pre-compute the formatted prompt once at import time. The enums are
# frozensets, so we sort them to keep the rendered system prompt
# deterministic across Python invocations.
_SYSTEM_PROMPT: str = _SYSTEM_PROMPT_TEMPLATE.format(
    categories=", ".join(sorted(ALLOWED_CATEGORIES)),
    lanes=", ".join(sorted(ALLOWED_LANES)),
)


def build_system_prompt() -> str:
    """Return the static system prompt that defines the response schema.

    The result is constant — every call returns byte-identical output —
    so the LLM sees the same schema definition on every request.
    """
    return _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# User prompt — composed from GameState + DetectionResult
# ---------------------------------------------------------------------------
def build_user_prompt(state: GameState, detections: DetectionResult) -> str:
    """Build the per-frame structured context block for the LLM.

    The output is a fixed-shape text block with one line per known field.
    Lines are appended in fixed order so the LLM sees a stable layout, but
    a line is **omitted entirely** when its underlying data is unknown
    (``None``). Event-stream summaries (kill feed, objectives) always show
    a line, with an explicit ``none`` value if the collection is empty.

    The function is pure: same inputs always produce the same string, no
    side effects, no mutation of inputs.
    """
    lines: list[str] = ["=== GAME STATE ==="]

    # Game time — always present (defaults to 0 in cold start, which still
    # tells the LLM "we are very early in the game").
    lines.append(f"Game time: {_format_game_time(state.game_time_seconds)}")

    # Active player champion — omit if unknown.
    if state.active_player_champion is not None:
        lines.append(f"Your champion: {state.active_player_champion} (jungler)")

    # Active player gold — omit if unknown.
    if state.active_player_gold is not None:
        lines.append(f"Your gold: {state.active_player_gold}g")

    # Recent objectives — always present, even if empty (event-stream summary).
    objectives_text = _format_objectives(state.riot_events)
    lines.append(f"Recent objectives: {objectives_text}")

    # Enemy jungler — COLD-START RULE: omit entirely if last_seen is None.
    if state.enemy_jungler_last_seen is not None:
        lines.append(
            _format_enemy_jungler_line(
                champion_name=state.enemy_jungler_champion_name,
                last_seen=state.enemy_jungler_last_seen,
                predicted_quadrant=state.enemy_jungler_predicted_quadrant,
                game_time_seconds=state.game_time_seconds,
            )
        )

    # Recent kills — always present (event-stream summary).
    lines.append(f"Recent kills: {_format_kill_feed(state.kill_feed)}")

    # Currently visible from the latest detector frame — omit if no hits.
    if detections.champions:
        lines.append(
            "Currently visible on minimap: "
            + _format_visible_champions(detections)
        )

    lines.append("=== END GAME STATE ===")
    lines.append("")
    lines.append(
        "What is the most useful 1-sentence callout for the jungler RIGHT NOW? "
        "Respond using the exact format from the system prompt."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Formatting helpers — pure functions, easy to test in isolation if needed.
# ---------------------------------------------------------------------------
def _format_game_time(seconds: float) -> str:
    """Format game time as ``MM:SS`` from a raw seconds float.

    Negative values (which shouldn't happen but might if the FSM glitches)
    clamp to 0:00 rather than producing a negative minute.
    """
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


def _format_objectives(riot_events: tuple[dict[str, Any], ...]) -> str:
    """Filter ``riot_events`` for objective-relevant entries and join them.

    Only events whose ``EventName`` is in ``_OBJECTIVE_EVENT_NAMES`` are
    included. Each entry is rendered as ``<EventName> at <MM:SS>``. If
    nothing matches, returns the literal string ``none``.
    """
    items: list[str] = []
    for event in riot_events:
        name = event.get("EventName")
        if name not in _OBJECTIVE_EVENT_NAMES:
            continue
        when = _format_game_time(float(event.get("EventTime", 0.0)))
        items.append(f"{name} at {when}")
    if not items:
        return "none"
    return "; ".join(items)


def _format_enemy_jungler_line(
    champion_name: str | None,
    last_seen: tuple[str, float],
    predicted_quadrant: str | None,
    game_time_seconds: float,
) -> str:
    """Render the enemy jungler line. Caller verifies last_seen is not None."""
    quadrant, ts = last_seen
    age = max(0.0, game_time_seconds - ts)
    age_text = f"~{int(age)}s ago"
    name = champion_name or "enemy jungler"
    line = f"Enemy jungler: {name}, last seen in {quadrant} {age_text}"
    if predicted_quadrant:
        line += f", predicted in {predicted_quadrant}"
    return line


def _format_kill_feed(kill_feed: tuple[dict[str, Any], ...]) -> str:
    """Render the kill feed as a ``;``-separated list of ``A killed B at MM:SS``.

    Empty kill feed returns the literal string ``none``.
    """
    if not kill_feed:
        return "none"
    items: list[str] = []
    for kill in kill_feed:
        killer = kill.get("KillerName") or "unknown"
        victim = kill.get("VictimName") or "unknown"
        when = _format_game_time(float(kill.get("EventTime", 0.0)))
        items.append(f"{killer} killed {victim} at {when}")
    return "; ".join(items)


def _format_visible_champions(detections: DetectionResult) -> str:
    """Render detector hits as ``Name (quadrant)`` pairs separated by ``,``."""
    return ", ".join(
        f"{c.name} ({c.quadrant})" for c in detections.champions
    )
