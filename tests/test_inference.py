"""Tests for ``lolcoach.inference`` — Ollama vision client + response parser.

Scenarios mapped from the plan's Unit 7 test list:

Parser scenarios:
1.  Happy: VALID_RESPONSE → Callout with all 5 fields parsed.
2.  Happy: TRAILING_WHITESPACE → parses cleanly despite trailing CRLF/spaces.
3.  Happy: VALID_OUT_OF_ORDER → parses cleanly despite reordered fields.
4.  Edge: MALFORMED_NO_CATEGORY → returns None.
5.  Edge: UNKNOWN_CATEGORY → returns None.
6.  Edge: UNKNOWN_LANE → returns None.
7.  Edge: CONFIDENCE_OUT_OF_RANGE → returns None.
8.  Edge: CONFIDENCE_NOT_INTEGER → returns None.
9.  Edge: EMPTY_DECISION → returns None.
10. Edge: GARBAGE_RESPONSE → returns None.

HTTP scenarios:
11. Happy: run() with mocked Ollama 200 → returns Callout, request body
    contains keep_alive: -1 and base64-encoded image.
12. Edge: run() with mocked HTTP 500 → returns None, logs.
13. Edge: run() with mocked requests.Timeout → returns None, logs.
14. Edge: run() with mocked requests.ConnectionError → returns None, logs.
15. Edge: run() with mocked Ollama returning malformed content → returns None.
16. Happy: warmup() with mocked 200 → returns True, request body has
    keep_alive: -1 and a tiny synthetic image.
17. Edge: warmup() with mocked 500 → returns False.
18. Edge: warmup() with mocked timeout → returns False.

Concurrency:
19. Single-worker lock: two concurrent run() calls → exactly one HTTP
    request fires, the second returns None immediately without contacting
    the network (stale-frame drop policy).
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import requests
import requests_mock as rm_module

from lolcoach.config import InferenceConfig
from lolcoach.detector import ChampionDetection, DetectionResult
from lolcoach.inference import Callout, InferenceEngine, _parse_response
from lolcoach.state import GameState
from tests.fixtures.ollama_responses import (
    CONFIDENCE_NOT_INTEGER,
    CONFIDENCE_OUT_OF_RANGE,
    EMPTY_DECISION,
    GARBAGE_RESPONSE,
    MALFORMED_NO_CATEGORY,
    TRAILING_WHITESPACE,
    UNKNOWN_CATEGORY,
    UNKNOWN_LANE,
    VALID_OUT_OF_ORDER,
    VALID_RESPONSE,
)

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wrap(content: str) -> dict:
    """Wrap raw model content text in Ollama's /api/chat JSON envelope."""
    return {
        "model": "gemma3:4b",
        "created_at": "2026-04-10T18:30:00Z",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }


def _make_engine(
    timeout_ms: int = 10_000,
    model: str = "gemma3:4b",
) -> InferenceEngine:
    config = InferenceConfig(
        model=model,
        timeout_ms=timeout_ms,
        ollama_host="http://localhost:11434",
        keep_alive=-1,
    )
    return InferenceEngine(config=config)


def _make_state() -> GameState:
    return GameState(
        game_time_seconds=600.0,
        active_summoner_name="ActivePlayer#NA1",
        active_player_champion="Hecarim",
        active_player_gold=2000,
        ally_champions=frozenset({"Hecarim", "Jinx", "Ahri"}),
        enemy_champions=frozenset({"LeeSin", "Ekko"}),
        all_champions=frozenset({"Hecarim", "Jinx", "Ahri", "LeeSin", "Ekko"}),
        enemy_jungler_champion_name="LeeSin",
        enemy_jungler_summoner_name="Enemy1#NA1",
        enemy_jungler_last_seen=("bot_jungle", 580.0),
        enemy_jungler_predicted_quadrant="mid",
    )


def _make_detections() -> DetectionResult:
    return DetectionResult(
        champions=(
            ChampionDetection(
                name="LeeSin",
                team="enemy",
                position_norm=(0.6, 0.7),
                quadrant="bot_jungle",
                confidence=0.91,
            ),
        ),
        detected_at=600.0,
    )


def _fake_frame() -> np.ndarray:
    """A 16x16 BGR uint8 frame — small enough to base64 quickly in tests."""
    return np.zeros((16, 16, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Parser — happy path
# ---------------------------------------------------------------------------
def test_parse_response_valid_returns_full_callout() -> None:
    callout = _parse_response(VALID_RESPONSE)

    assert callout is not None
    assert callout.decision == "Path to bot river, Lee Sin likely ganking top"
    assert callout.category == "pathing"
    assert callout.target_lane == "bot"
    assert callout.confidence == 8
    assert callout.reason.startswith("Lee Sin was last seen")
    assert callout.generated_at > 0


def test_parse_response_trailing_whitespace_parses_cleanly() -> None:
    callout = _parse_response(TRAILING_WHITESPACE)

    assert callout is not None
    assert callout.category == "pathing"
    assert callout.confidence == 8


def test_parse_response_out_of_order_fields_parses_cleanly() -> None:
    callout = _parse_response(VALID_OUT_OF_ORDER)

    assert callout is not None
    assert callout.decision == "Drop a ward in mid bush, enemy mid is roaming"
    assert callout.category == "vision"
    assert callout.target_lane == "mid"
    assert callout.confidence == 7
    assert "Enemy mid is roaming" in callout.reason


# ---------------------------------------------------------------------------
# Parser — malformed paths
# ---------------------------------------------------------------------------
def test_parse_response_missing_category_returns_none() -> None:
    assert _parse_response(MALFORMED_NO_CATEGORY) is None


def test_parse_response_unknown_category_returns_none() -> None:
    assert _parse_response(UNKNOWN_CATEGORY) is None


def test_parse_response_unknown_lane_returns_none() -> None:
    assert _parse_response(UNKNOWN_LANE) is None


def test_parse_response_confidence_out_of_range_returns_none() -> None:
    assert _parse_response(CONFIDENCE_OUT_OF_RANGE) is None


def test_parse_response_confidence_not_integer_returns_none() -> None:
    assert _parse_response(CONFIDENCE_NOT_INTEGER) is None


def test_parse_response_empty_decision_returns_none() -> None:
    assert _parse_response(EMPTY_DECISION) is None


def test_parse_response_garbage_returns_none() -> None:
    assert _parse_response(GARBAGE_RESPONSE) is None


def test_parse_response_empty_string_returns_none() -> None:
    assert _parse_response("") is None


# ---------------------------------------------------------------------------
# Callout dataclass invariants
# ---------------------------------------------------------------------------
def test_callout_is_frozen_dataclass() -> None:
    callout = Callout(
        decision="Recall",
        category="recall",
        target_lane="global",
        confidence=9,
        reason="Low HP, full inventory",
        generated_at=1000.0,
    )
    with pytest.raises((AttributeError, Exception)):
        callout.confidence = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# InferenceEngine.run() — happy path
# ---------------------------------------------------------------------------
def test_run_happy_path_returns_callout_and_sends_keep_alive_minus_one() -> None:
    engine = _make_engine()
    state = _make_state()
    detections = _make_detections()
    frame = _fake_frame()

    with rm_module.Mocker() as m:
        m.post(OLLAMA_CHAT_URL, json=_wrap(VALID_RESPONSE))
        callout = engine.run(frame=frame, state=state, detections=detections)

        assert callout is not None
        assert callout.category == "pathing"
        assert callout.confidence == 8

        request = m.last_request
        assert request is not None
        body = request.json()
        assert body["model"] == "gemma3:4b"
        assert body["keep_alive"] == -1
        assert body["stream"] is False
        # Vision input: the user message must carry the base64 image.
        assert any(
            "images" in msg and msg["images"]
            for msg in body["messages"]
        ), "user message must include base64 image payload"


def test_run_uses_system_prompt_in_request_body() -> None:
    """The /api/chat request body should include both the system prompt
    (defining the response schema) and the user prompt (game state context).
    Without the system prompt, the model would not know the response format.
    """
    engine = _make_engine()
    with rm_module.Mocker() as m:
        m.post(OLLAMA_CHAT_URL, json=_wrap(VALID_RESPONSE))
        engine.run(
            frame=_fake_frame(),
            state=_make_state(),
            detections=_make_detections(),
        )
        body = m.last_request.json()  # type: ignore[union-attr]

    roles = [msg["role"] for msg in body["messages"]]
    assert "system" in roles
    assert "user" in roles


# ---------------------------------------------------------------------------
# InferenceEngine.run() — error paths
# ---------------------------------------------------------------------------
def test_run_http_500_returns_none() -> None:
    engine = _make_engine()
    with rm_module.Mocker() as m:
        m.post(OLLAMA_CHAT_URL, status_code=500, text="internal error")
        result = engine.run(
            frame=_fake_frame(),
            state=_make_state(),
            detections=_make_detections(),
        )
    assert result is None


def test_run_timeout_returns_none() -> None:
    engine = _make_engine(timeout_ms=100)
    with rm_module.Mocker() as m:
        m.post(OLLAMA_CHAT_URL, exc=requests.Timeout)
        result = engine.run(
            frame=_fake_frame(),
            state=_make_state(),
            detections=_make_detections(),
        )
    assert result is None


def test_run_connection_refused_returns_none() -> None:
    engine = _make_engine()
    with rm_module.Mocker() as m:
        m.post(OLLAMA_CHAT_URL, exc=requests.ConnectionError)
        result = engine.run(
            frame=_fake_frame(),
            state=_make_state(),
            detections=_make_detections(),
        )
    assert result is None


def test_run_malformed_model_content_returns_none() -> None:
    """The HTTP call succeeds (200) but the model emits unparseable text.
    The parser layer rejects it; run() must return None gracefully.
    """
    engine = _make_engine()
    with rm_module.Mocker() as m:
        m.post(OLLAMA_CHAT_URL, json=_wrap(MALFORMED_NO_CATEGORY))
        result = engine.run(
            frame=_fake_frame(),
            state=_make_state(),
            detections=_make_detections(),
        )
    assert result is None


def test_run_envelope_missing_message_field_returns_none() -> None:
    """Defensive: a 200 response with an unexpected JSON envelope shape
    (e.g., Ollama version drift) must return None, not crash on KeyError.
    """
    engine = _make_engine()
    with rm_module.Mocker() as m:
        m.post(OLLAMA_CHAT_URL, json={"model": "gemma3:4b", "done": True})
        result = engine.run(
            frame=_fake_frame(),
            state=_make_state(),
            detections=_make_detections(),
        )
    assert result is None


def test_run_invalid_json_response_body_returns_none() -> None:
    """Defensive: a 200 response with a body that isn't valid JSON
    (e.g., the user pointed Ollama at a generic HTTP server) must
    return None, not crash on json.JSONDecodeError.
    """
    engine = _make_engine()
    with rm_module.Mocker() as m:
        m.post(OLLAMA_CHAT_URL, text="<html>not json</html>")
        result = engine.run(
            frame=_fake_frame(),
            state=_make_state(),
            detections=_make_detections(),
        )
    assert result is None


def test_run_generic_request_exception_returns_none() -> None:
    """Defense-in-depth: any ``requests.RequestException`` subtype other
    than the explicitly-handled Timeout/ConnectionError (e.g. ``InvalidURL``,
    ``ChunkedEncodingError``, malformed proxy responses) must collapse to
    None. ``SSLError`` is intentionally not used here because it inherits
    from ``ConnectionError`` and would hit the wrong branch.
    """
    engine = _make_engine()
    with rm_module.Mocker() as m:
        m.post(OLLAMA_CHAT_URL, exc=requests.exceptions.InvalidURL)
        result = engine.run(
            frame=_fake_frame(),
            state=_make_state(),
            detections=_make_detections(),
        )
    assert result is None


# ---------------------------------------------------------------------------
# InferenceEngine.warmup()
# ---------------------------------------------------------------------------
def test_warmup_happy_path_returns_true_with_keep_alive_minus_one() -> None:
    engine = _make_engine()
    with rm_module.Mocker() as m:
        m.post(OLLAMA_CHAT_URL, json=_wrap("ok"))
        result = engine.warmup()

        assert result is True
        body = m.last_request.json()  # type: ignore[union-attr]
        assert body["keep_alive"] == -1
        assert body["model"] == "gemma3:4b"


def test_warmup_http_500_returns_false() -> None:
    engine = _make_engine()
    with rm_module.Mocker() as m:
        m.post(OLLAMA_CHAT_URL, status_code=500)
        assert engine.warmup() is False


def test_warmup_timeout_returns_false() -> None:
    engine = _make_engine(timeout_ms=100)
    with rm_module.Mocker() as m:
        m.post(OLLAMA_CHAT_URL, exc=requests.Timeout)
        assert engine.warmup() is False


def test_warmup_connection_error_returns_false() -> None:
    engine = _make_engine()
    with rm_module.Mocker() as m:
        m.post(OLLAMA_CHAT_URL, exc=requests.ConnectionError)
        assert engine.warmup() is False


def test_warmup_generic_request_exception_returns_false() -> None:
    """Defense-in-depth: ``RequestException`` subtypes other than
    Timeout/ConnectionError on the warmup path (e.g. ``InvalidURL``,
    ``ChunkedEncodingError``) must collapse to False, not propagate.
    """
    engine = _make_engine()
    with rm_module.Mocker() as m:
        m.post(OLLAMA_CHAT_URL, exc=requests.exceptions.InvalidURL)
        assert engine.warmup() is False


# ---------------------------------------------------------------------------
# Single-worker lock — stale-frame drop policy
# ---------------------------------------------------------------------------
def test_run_releases_lock_on_success_so_next_call_can_acquire() -> None:
    """Sanity check: the lock is released in a ``finally`` so two sequential
    calls both succeed. Forgetting the release would deadlock the engine
    after the first ever call — a single-line bug, but a fatal one. The
    ce-work System-Wide Test Check requires verifying lock release explicitly.
    """
    engine = _make_engine()
    with rm_module.Mocker() as m:
        m.post(OLLAMA_CHAT_URL, json=_wrap(VALID_RESPONSE))
        first = engine.run(
            frame=_fake_frame(),
            state=_make_state(),
            detections=_make_detections(),
        )
        second = engine.run(
            frame=_fake_frame(),
            state=_make_state(),
            detections=_make_detections(),
        )
    assert first is not None
    assert second is not None
    # Both calls fired actual HTTP requests — the lock did NOT stay held
    # after the first call.
    assert m.call_count == 2


def test_run_releases_lock_on_error_so_next_call_can_acquire() -> None:
    """Same release invariant on the failure path: a Timeout in the first
    call must not leave the lock held. The fix lives in the ``finally``
    branch of ``run()``.
    """
    engine = _make_engine(timeout_ms=100)
    with rm_module.Mocker() as m:
        # First call raises Timeout; second call succeeds.
        m.post(
            OLLAMA_CHAT_URL,
            [
                {"exc": requests.Timeout},
                {"json": _wrap(VALID_RESPONSE)},
            ],
        )
        first = engine.run(
            frame=_fake_frame(),
            state=_make_state(),
            detections=_make_detections(),
        )
        second = engine.run(
            frame=_fake_frame(),
            state=_make_state(),
            detections=_make_detections(),
        )
    assert first is None
    assert second is not None


def test_concurrent_run_calls_drop_second_frame_via_nonblocking_lock() -> None:
    """Two threads call ``run()`` simultaneously. The first acquires the
    inference lock, makes the HTTP call, and returns a Callout. The
    second sees the lock is held, returns None immediately, and never
    contacts Ollama. Without this drop policy, a second inference would
    queue up on stale data and waste GPU cycles.

    Verified by counting the actual HTTP request count via requests_mock —
    must be exactly 1 even though two threads called run().
    """
    engine = _make_engine()
    state = _make_state()
    detections = _make_detections()
    frame = _fake_frame()

    # Hold the inference lock manually so the first worker thread blocks
    # inside its first call. The second worker arrives, sees the lock
    # held, and returns None immediately.
    engine._lock.acquire()  # noqa: SLF001 — testing internal contract

    results: list[Callout | None] = []
    errors: list[BaseException] = []
    started = threading.Barrier(2)
    second_done = threading.Event()

    def first_worker() -> None:
        try:
            started.wait(timeout=2.0)
            # Wait until the second worker has confirmed its drop, then
            # release the lock so the first call can proceed.
            second_done.wait(timeout=2.0)
            engine._lock.release()  # noqa: SLF001
            with rm_module.Mocker() as m:
                m.post(OLLAMA_CHAT_URL, json=_wrap(VALID_RESPONSE))
                result = engine.run(
                    frame=frame, state=state, detections=detections
                )
                results.append(result)
                # Exactly one HTTP call from this worker.
                assert m.call_count == 1
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def second_worker() -> None:
        try:
            started.wait(timeout=2.0)
            with rm_module.Mocker() as m:
                m.post(OLLAMA_CHAT_URL, json=_wrap(VALID_RESPONSE))
                # Lock is held by the test → run() should drop immediately.
                result = engine.run(
                    frame=frame, state=state, detections=detections
                )
                results.append(result)
                # Verify zero HTTP calls fired — the drop happened before
                # any network activity.
                assert m.call_count == 0, (
                    "second worker must not contact Ollama when lock held"
                )
        finally:
            second_done.set()

    t1 = threading.Thread(target=first_worker)
    t2 = threading.Thread(target=second_worker)
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert errors == [], f"worker threads raised: {errors}"
    # Both workers ran. Exactly one returned a Callout, exactly one returned
    # None — order depends on thread scheduling.
    assert len(results) == 2
    assert results.count(None) == 1
    assert sum(1 for r in results if r is not None) == 1
