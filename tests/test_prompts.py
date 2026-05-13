"""Tests for ``lolcoach.prompts`` — system + user prompt builders.

The system prompt is a static instruction block. The user prompt is a
structured context block built from a ``GameState`` snapshot plus the
latest ``DetectionResult``. Both are consumed by ``inference.run`` and sent
to the local Ollama vision model via ``/api/chat``.

Critical invariant tested here: **the cold-start rule.** When
``state.enemy_jungler_last_seen is None``, the prompt builder must omit
the "Enemy jungler" line entirely — never substitute a default position,
never write "Enemy jungler: unknown". Same rule applies to every other
optional field: missing data → line omitted; empty collection → line
shown with an explicit ``none`` marker.

Tests assert on **substring presence and structural shape**, not exact
byte equality. Locking in the entire prompt as a golden snapshot would
make every word-level wording tweak a multi-test refactor; that's the
wrong tradeoff for a prompt that will be tuned against benchmark data.
"""

from __future__ import annotations

from typing import Any

from lolcoach.detector import ChampionDetection, DetectionResult
from lolcoach.prompts import (
    ALLOWED_CATEGORIES,
    ALLOWED_LANES,
    build_system_prompt,
    build_user_prompt,
)
from lolcoach.state import GameState


# ---------------------------------------------------------------------------
# Helpers — synthetic GameState + DetectionResult builders
# ---------------------------------------------------------------------------
def _make_full_state() -> GameState:
    """Build a fully-populated GameState that exercises every prompt line."""
    return GameState(
        game_time_seconds=742.5,  # 12:22
        active_summoner_name="ActivePlayer#NA1",
        active_player_champion="Hecarim",
        active_player_gold=3200,
        ally_champions=frozenset({"Hecarim", "Jinx", "Ahri"}),
        enemy_champions=frozenset({"LeeSin", "Ekko"}),
        all_champions=frozenset({"Hecarim", "Jinx", "Ahri", "LeeSin", "Ekko"}),
        enemy_jungler_champion_name="LeeSin",
        enemy_jungler_summoner_name="Enemy1#NA1",
        enemy_jungler_last_seen=("bot_jungle", 700.0),
        enemy_jungler_predicted_quadrant="mid",
        kill_feed=(
            {
                "EventName": "ChampionKill",
                "EventTime": 540.0,
                "KillerName": "Enemy1#NA1",
                "VictimName": "Ally1#NA1",
            },
            {
                "EventName": "ChampionKill",
                "EventTime": 620.0,
                "KillerName": "Ally2#NA1",
                "VictimName": "Enemy1#NA1",
            },
        ),
        riot_events=(
            {"EventID": 0, "EventName": "GameStart", "EventTime": 0.028},
            {"EventID": 1, "EventName": "MinionsSpawning", "EventTime": 65.0},
            {
                "EventID": 4,
                "EventName": "DragonKill",
                "EventTime": 360.0,
                "KillerName": "Enemy1#NA1",
                "DragonType": "Infernal",
            },
            {
                "EventID": 5,
                "EventName": "HeraldKill",
                "EventTime": 510.0,
                "KillerName": "Enemy1#NA1",
            },
            {
                "EventID": 6,
                "EventName": "TurretKilled",
                "EventTime": 700.0,
                "TurretKilled": "Turret_T1_R_03_A",
            },
        ),
    )


def _make_cold_start_state() -> GameState:
    """Build a fresh-game GameState — first 30 seconds, no detector hits yet."""
    return GameState(
        game_time_seconds=30.0,
        active_summoner_name="ActivePlayer#NA1",
        active_player_champion="Hecarim",
        active_player_gold=500,
        ally_champions=frozenset({"Hecarim", "Jinx", "Ahri"}),
        enemy_champions=frozenset({"LeeSin", "Ekko"}),
        all_champions=frozenset({"Hecarim", "Jinx", "Ahri", "LeeSin", "Ekko"}),
        enemy_jungler_champion_name="LeeSin",
        enemy_jungler_summoner_name="Enemy1#NA1",
        enemy_jungler_last_seen=None,  # COLD START
        enemy_jungler_predicted_quadrant=None,
        kill_feed=(),
        riot_events=(),
    )


def _empty_detections() -> DetectionResult:
    return DetectionResult(champions=(), detected_at=1000.0)


def _populated_detections() -> DetectionResult:
    return DetectionResult(
        champions=(
            ChampionDetection(
                name="LeeSin",
                team="enemy",
                position_norm=(0.6, 0.7),
                quadrant="bot_jungle",
                confidence=0.92,
            ),
            ChampionDetection(
                name="Ekko",
                team="enemy",
                position_norm=(0.5, 0.5),
                quadrant="mid",
                confidence=0.88,
            ),
        ),
        detected_at=1000.0,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
def test_build_system_prompt_lists_all_required_response_fields() -> None:
    prompt = build_system_prompt()

    for field in ("DECISION", "CATEGORY", "TARGET_LANE", "CONFIDENCE", "REASON"):
        assert field in prompt, f"system prompt must mention required field {field}"


def test_build_system_prompt_lists_allowed_categories() -> None:
    prompt = build_system_prompt()

    for category in ALLOWED_CATEGORIES:
        assert category in prompt, (
            f"system prompt must list allowed category {category}"
        )


def test_build_system_prompt_lists_allowed_lanes() -> None:
    prompt = build_system_prompt()

    for lane in ALLOWED_LANES:
        assert lane in prompt, f"system prompt must list allowed lane {lane}"


def test_build_system_prompt_specifies_confidence_range() -> None:
    prompt = build_system_prompt()

    # Range may be expressed as "1-10", "1 to 10", or "1..10" — accept any.
    assert "1" in prompt and "10" in prompt
    assert "confidence" in prompt.lower()


def test_build_system_prompt_frames_llm_as_jungle_coach() -> None:
    prompt = build_system_prompt().lower()
    assert "jungle" in prompt
    assert "coach" in prompt or "coaching" in prompt


def test_build_system_prompt_is_deterministic() -> None:
    """The system prompt is a constant template, not a function of state.
    Two consecutive builds must return identical strings — otherwise the
    LLM gets a different schema definition on every call.
    """
    assert build_system_prompt() == build_system_prompt()


# ---------------------------------------------------------------------------
# User prompt — happy path (fully populated)
# ---------------------------------------------------------------------------
def test_build_user_prompt_fully_populated_includes_all_context_lines() -> None:
    state = _make_full_state()
    detections = _populated_detections()

    prompt = build_user_prompt(state, detections)

    # 6 specced context lines + the currently-visible-detections line.
    assert "Game time:" in prompt
    assert "Your champion:" in prompt
    assert "Hecarim" in prompt
    assert "Your gold:" in prompt
    assert "3200" in prompt
    assert "Recent objectives:" in prompt
    assert "Enemy jungler:" in prompt
    assert "LeeSin" in prompt or "Lee Sin" in prompt
    assert "Recent kills:" in prompt
    assert "Currently visible on minimap:" in prompt


def test_build_user_prompt_formats_game_time_as_minutes_and_seconds() -> None:
    state = _make_full_state()  # 742.5 seconds == 12:22
    prompt = build_user_prompt(state, _empty_detections())

    assert "12:22" in prompt, (
        "game time must be formatted as MM:SS, not raw float seconds"
    )


def test_build_user_prompt_lists_objectives_filtered_from_riot_events() -> None:
    """Only DragonKill / BaronKill / HeraldKill / TurretKilled events should
    appear in the objectives line. GameStart and MinionsSpawning should not.
    """
    state = _make_full_state()
    prompt = build_user_prompt(state, _empty_detections())

    assert "Dragon" in prompt
    assert "Herald" in prompt
    assert "Turret" in prompt or "turret" in prompt
    # Non-objective events from the same riot_events list must NOT appear.
    assert "GameStart" not in prompt
    assert "MinionsSpawning" not in prompt


def test_build_user_prompt_lists_detections_with_quadrants() -> None:
    state = _make_full_state()
    detections = _populated_detections()

    prompt = build_user_prompt(state, detections)

    # Each detected champion should appear with its quadrant.
    assert "LeeSin" in prompt or "Lee Sin" in prompt
    assert "bot_jungle" in prompt
    assert "Ekko" in prompt
    assert "mid" in prompt


# ---------------------------------------------------------------------------
# User prompt — cold start rule (CRITICAL invariant)
# ---------------------------------------------------------------------------
def test_build_user_prompt_cold_start_omits_enemy_jungler_line_entirely() -> None:
    """The plan's load-bearing cold-start rule: when last_seen is None,
    the 'Enemy jungler' line is omitted entirely. Never replaced with a
    placeholder, never substituted with a guess. The LLM must see a clean
    'no information' signal so it doesn't reason against a hallucination.
    """
    state = _make_cold_start_state()  # enemy_jungler_last_seen is None

    prompt = build_user_prompt(state, _empty_detections())

    assert "Enemy jungler:" not in prompt, (
        "cold-start rule violated: 'Enemy jungler:' line must be omitted "
        "entirely when last_seen is None"
    )
    # Other lines that DO have data must still be present.
    assert "Game time:" in prompt
    assert "Your champion:" in prompt
    assert "Recent kills:" in prompt  # empty kill feed shows "none"


def test_build_user_prompt_cold_start_kill_feed_shows_none_explicitly() -> None:
    """Empty kill feed is honest information ('no kills yet'), not missing
    data — show it with an explicit 'none' so the LLM knows nothing has
    happened rather than wondering whether the field was omitted.
    """
    state = _make_cold_start_state()
    prompt = build_user_prompt(state, _empty_detections())

    # The line is present and the value is 'none' (case-insensitive).
    assert "Recent kills:" in prompt
    kills_line = next(
        line for line in prompt.splitlines() if line.startswith("Recent kills:")
    )
    assert "none" in kills_line.lower()


def test_build_user_prompt_cold_start_omits_currently_visible_when_no_detections() -> None:
    """No detector hits → omit the line entirely. Showing 'visible: none'
    would mislead the LLM into thinking the minimap was scanned and empty;
    in reality, the detector might have lost the frame or the capture
    queue is stale.
    """
    state = _make_cold_start_state()
    prompt = build_user_prompt(state, _empty_detections())

    assert "Currently visible on minimap:" not in prompt


def test_build_user_prompt_cold_start_objectives_line_shows_none_explicitly() -> None:
    """Empty riot_events is information ('no objectives taken yet'), not
    missing data — production unconditionally appends 'Recent objectives: none'
    so the LLM has a positive signal rather than wondering whether the
    field was omitted. Pin the contract: the line MUST appear and the
    value MUST be 'none' (case-insensitive).
    """
    state = _make_cold_start_state()  # riot_events is empty
    prompt = build_user_prompt(state, _empty_detections())

    assert "Recent objectives:" in prompt, (
        "production unconditionally emits the line; tightening the test "
        "from a vacuous conditional to a hard assertion (review finding 8)"
    )
    objectives_line = next(
        line for line in prompt.splitlines()
        if line.startswith("Recent objectives:")
    )
    assert "none" in objectives_line.lower()


def test_build_user_prompt_with_last_seen_but_no_predicted_quadrant() -> None:
    """The ``if predicted_quadrant:`` false branch in build_user_prompt's
    inlined enemy-jungler section is hit only when last_seen is set but
    enemy_jungler_predicted_quadrant is None. The full-state and cold-start
    fixtures don't cover this combination — tests/test_prompts.py partial
    branch coverage was the missing case (review finding 14).
    """
    state = GameState(
        game_time_seconds=600.0,
        active_summoner_name="ActivePlayer#NA1",
        active_player_champion="Hecarim",
        enemy_jungler_champion_name="LeeSin",
        enemy_jungler_last_seen=("bot_jungle", 540.0),
        enemy_jungler_predicted_quadrant=None,  # the missing case
    )

    prompt = build_user_prompt(state, _empty_detections())

    # The line is present (last_seen is not None).
    enemy_line = next(
        line for line in prompt.splitlines()
        if line.startswith("Enemy jungler:")
    )
    # The "predicted in" suffix is omitted.
    assert "predicted in" not in enemy_line
    assert "last seen in bot_jungle" in enemy_line


def test_build_user_prompt_objectives_line_capped_at_recency_limit() -> None:
    """state.riot_events is unbounded (Riot returns the full cumulative
    event history every poll), so the rendered objectives line could grow
    linearly with game length and inflate prompt token count on every
    500ms call. Verify the trim caps the rendered count regardless of
    how many objective events accumulate (review finding 12).
    """
    # Construct a synthetic event stream with 30 dragon kills — well
    # above the cap. Mixed with non-objective events that should be
    # filtered out before the cap is applied.
    riot_events: list[dict[str, Any]] = []
    for i in range(30):
        riot_events.append(
            {
                "EventID": i * 2,
                "EventName": "DragonKill",
                "EventTime": float(60 + i * 30),
                "DragonType": "Infernal",
            }
        )
        riot_events.append(
            {
                "EventID": i * 2 + 1,
                "EventName": "MinionsSpawning",  # filtered out
                "EventTime": float(60 + i * 30 + 5),
            }
        )

    state = GameState(
        game_time_seconds=2000.0,
        active_summoner_name="P",
        active_player_champion="Hecarim",
        riot_events=tuple(riot_events),
    )

    prompt = build_user_prompt(state, _empty_detections())
    objectives_line = next(
        line for line in prompt.splitlines()
        if line.startswith("Recent objectives:")
    )
    rendered_count = objectives_line.count("DragonKill")
    # The cap should keep rendered count bounded — definitely fewer than
    # the 30 events we pushed in. We don't pin the exact number to keep
    # the test resilient to a future cap tweak.
    assert rendered_count > 0, "objectives line should still render some events"
    assert rendered_count <= 12, (
        f"objectives line is supposed to be capped to a recency window, "
        f"but rendered {rendered_count} of 30 events — the trim is missing "
        f"or the limit is too generous"
    )

    # Additionally verify recency: the rendered line should contain the
    # LATEST event timestamp, not the earliest one.
    latest_event_time = max(
        float(ev["EventTime"]) for ev in riot_events
        if ev["EventName"] == "DragonKill"
    )
    minutes, secs = divmod(int(latest_event_time), 60)
    latest_timestamp = f"{minutes}:{secs:02d}"
    assert latest_timestamp in objectives_line, (
        f"objectives line must include the most-recent event "
        f"({latest_timestamp}); got: {objectives_line}"
    )


# ---------------------------------------------------------------------------
# User prompt — defensive omission of unknown soft fields
# ---------------------------------------------------------------------------
def test_build_user_prompt_omits_gold_line_when_unknown() -> None:
    """active_player_gold is None (older Riot client, missing currentGold) →
    omit the gold line entirely. Don't write 'Your gold: 0' — that's a lie.
    """
    state = GameState(
        game_time_seconds=120.0,
        active_summoner_name="Player",
        active_player_champion="Hecarim",
        active_player_gold=None,  # unknown
        ally_champions=frozenset({"Hecarim"}),
        enemy_champions=frozenset({"LeeSin"}),
        all_champions=frozenset({"Hecarim", "LeeSin"}),
    )

    prompt = build_user_prompt(state, _empty_detections())

    assert "Your gold:" not in prompt
    # Champion line should still be present.
    assert "Your champion:" in prompt


def test_build_user_prompt_omits_champion_line_when_unknown() -> None:
    state = GameState(
        game_time_seconds=120.0,
        active_summoner_name="Player",
        active_player_champion=None,  # unknown
        active_player_gold=500,
    )

    prompt = build_user_prompt(state, _empty_detections())

    assert "Your champion:" not in prompt


# ---------------------------------------------------------------------------
# User prompt — output is well-formed
# ---------------------------------------------------------------------------
def test_build_user_prompt_returns_a_str_with_no_placeholder_fragments() -> None:
    """Defensive: catch f-string drift where a missing field substitutes
    'None' or '{var}' verbatim into the prompt body."""
    state = _make_cold_start_state()
    prompt = build_user_prompt(state, _empty_detections())

    assert isinstance(prompt, str)
    assert "None" not in prompt, (
        "prompt must not contain literal 'None' — every optional field must "
        "be omitted via conditional logic, not f-string interpolation"
    )
    assert "{" not in prompt and "}" not in prompt, (
        "prompt must not contain unsubstituted f-string placeholders"
    )


def test_build_user_prompt_does_not_mutate_inputs() -> None:
    """The prompt builder is a pure function — building the prompt twice
    against the same state must produce the same string and must not
    mutate the state or detections."""
    state = _make_full_state()
    detections = _populated_detections()

    first = build_user_prompt(state, detections)
    second = build_user_prompt(state, detections)

    assert first == second
