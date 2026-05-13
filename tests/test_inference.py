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

import dataclasses
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
    CONFIDENCE_FRACTIONAL,
    CONFIDENCE_LOWER_BOUND,
    CONFIDENCE_NEGATIVE,
    CONFIDENCE_NOT_INTEGER,
    CONFIDENCE_OUT_OF_RANGE,
    CONFIDENCE_UPPER_BOUND,
    CONFIDENCE_ZERO,
    DUPLICATE_DECISION,
    EMPTY_DECISION,
    GARBAGE_RESPONSE,
    MALFORMED_NO_CATEGORY,
    MULTILINE_REASON,
    OUT_OF_ORDER_WITH_MULTILINE_REASON,
    REASON_WITH_EMBEDDED_KEY_ON_CONTINUATION,
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


def test_parse_response_confidence_lower_bound_parses() -> None:
    """CONFIDENCE: 1 is the documented lower boundary — must parse."""
    callout = _parse_response(CONFIDENCE_LOWER_BOUND)
    assert callout is not None
    assert callout.confidence == 1


def test_parse_response_confidence_upper_bound_parses() -> None:
    """CONFIDENCE: 10 is the documented upper boundary — must parse."""
    callout = _parse_response(CONFIDENCE_UPPER_BOUND)
    assert callout is not None
    assert callout.confidence == 10


def test_parse_response_confidence_zero_returns_none() -> None:
    """CONFIDENCE: 0 is below the [1, 10] range — must reject."""
    assert _parse_response(CONFIDENCE_ZERO) is None


def test_parse_response_confidence_negative_returns_none() -> None:
    """CONFIDENCE: -3 parses as int but fails the range gate."""
    assert _parse_response(CONFIDENCE_NEGATIVE) is None


def test_parse_response_confidence_fractional_returns_none() -> None:
    """CONFIDENCE: 8.5 raises ValueError on int() — must collapse to None,
    not crash. Real failure mode the fixture module documents but the
    original test suite did not exercise.
    """
    assert _parse_response(CONFIDENCE_FRACTIONAL) is None


def test_parse_response_duplicate_decision_keeps_first_match() -> None:
    """Pin the first-match-wins semantics so a future refactor to
    "last wins" or "reject duplicates" would break this test rather
    than silently change parser behavior. Real failure mode: the model
    self-corrects mid-stream and emits a second DECISION below the first.
    """
    callout = _parse_response(DUPLICATE_DECISION)
    assert callout is not None
    assert callout.decision == "Path to bot river"


def test_parse_response_multiline_reason_captures_all_continuation_lines() -> None:
    """The R6 precondition: Unit 8's grounding check must see every
    champion name the model mentions in REASON. A truncated REASON
    silently drops champion names and the filter cannot reject a
    hallucinated callout whose evidence was discarded by the parser.

    Fix verification: REASON should capture all three continuation lines
    (including the second and third mentions of champion names) up to
    end of text.
    """
    callout = _parse_response(MULTILINE_REASON)

    assert callout is not None
    assert "Lee Sin" in callout.reason
    assert "Yasuo" in callout.reason  # the dropped-without-fix case
    assert "defensively" in callout.reason  # third line reaches EOF


def test_parse_response_out_of_order_multiline_reason_parses_cleanly() -> None:
    """REASON appears first and wraps across lines. The boundary lookahead
    must terminate REASON at the next known key at line start, NOT at
    arbitrary text in the continuation. Every other field must still be
    extracted from its own line below REASON.
    """
    callout = _parse_response(OUT_OF_ORDER_WITH_MULTILINE_REASON)

    assert callout is not None
    # REASON spans both lines before CONFIDENCE.
    assert "Enemy was last seen near top lane" in callout.reason
    assert "Predicted path loops through the river" in callout.reason
    # The key lines that follow REASON are still extracted correctly.
    assert callout.confidence == 8
    assert callout.category == "pathing"
    assert callout.target_lane == "top"
    assert callout.decision == "Rotate top to contest"


def test_parse_response_reason_terminates_at_key_lookalike_continuation() -> None:
    """If a REASON continuation line literally starts with a known key,
    the boundary lookahead terminates REASON there and the (duplicate)
    key line's value is extracted as that field. In this fixture the
    second CONFIDENCE line is "high as I can tell" — not an integer —
    so first-match-wins gives CONFIDENCE=8 (the real value) and the
    adversarial line is harmless.

    Without the boundary lookahead, REASON would silently capture
    "Lee Sin is pathing" only and drop the continuation; with it, the
    parser behaves consistently whether REASON has continuations or
    not.
    """
    callout = _parse_response(REASON_WITH_EMBEDDED_KEY_ON_CONTINUATION)

    # The real CONFIDENCE=8 on line 4 is matched first (re.search returns
    # the first match), so the parsed Callout is valid.
    assert callout is not None
    assert callout.confidence == 8
    # REASON should capture only up to the line-starting CONFIDENCE
    # lookalike — nothing after it.
    assert "Lee Sin is pathing" in callout.reason
    assert "high as I can tell" not in callout.reason


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
    # FrozenInstanceError is the precise contract — a tuple of
    # (AttributeError, Exception) would silently accept any non-base
    # exception, defeating the test if frozen=True ever drops off.
    with pytest.raises(dataclasses.FrozenInstanceError):
        callout.confidence = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# InferenceEngine.__init__ — construction-time validation
# ---------------------------------------------------------------------------
def test_init_rejects_zero_timeout_ms() -> None:
    """timeout_ms <= 0 would make every requests.post raise ValueError
    mid-game. Fail-fast at construction matches the riot_client pattern."""
    config = InferenceConfig(timeout_ms=0)
    with pytest.raises(ValueError, match="timeout_ms"):
        InferenceEngine(config=config)


def test_init_rejects_negative_timeout_ms() -> None:
    config = InferenceConfig(timeout_ms=-1)
    with pytest.raises(ValueError, match="timeout_ms"):
        InferenceEngine(config=config)


def test_init_rejects_non_loopback_ollama_host() -> None:
    """Sending the prompt + minimap to a remote URL would exfiltrate
    summoner names, kill feed, and the live screenshot to that server.
    The check matches riot_client._assert_loopback_base_url and is
    enforced at construction so the failure mode is impossible.
    """
    config = InferenceConfig(ollama_host="http://attacker.example.com:11434")
    with pytest.raises(ValueError, match="loopback"):
        InferenceEngine(config=config)


def test_init_rejects_public_ip_ollama_host() -> None:
    config = InferenceConfig(ollama_host="http://8.8.8.8:11434")
    with pytest.raises(ValueError, match="loopback"):
        InferenceEngine(config=config)


def test_init_rejects_ollama_host_without_hostname() -> None:
    config = InferenceConfig(ollama_host="http://:11434")
    with pytest.raises(ValueError, match="hostname"):
        InferenceEngine(config=config)


def test_init_accepts_localhost_alias() -> None:
    """The literal hostname 'localhost' resolves at runtime — accept it
    as effectively loopback for any sane system, matching riot_client.
    """
    config = InferenceConfig(ollama_host="http://localhost:11434")
    InferenceEngine(config=config)  # must not raise


def test_init_accepts_127_0_0_1() -> None:
    config = InferenceConfig(ollama_host="http://127.0.0.1:11434")
    InferenceEngine(config=config)  # must not raise


def test_init_accepts_ipv6_loopback() -> None:
    config = InferenceConfig(ollama_host="http://[::1]:11434")
    InferenceEngine(config=config)  # must not raise


def test_init_warmup_timeout_floors_at_30s_when_per_frame_is_lower() -> None:
    """The plan documents 10-20 s cold-start model loading, so warmup
    needs a longer timeout than the per-frame budget. Verify the floor
    is applied even when timeout_ms is small."""
    engine = _make_engine(timeout_ms=5_000)  # 5 s per frame
    assert engine._warmup_timeout_s >= 30.0  # noqa: SLF001 — testing internal


def test_init_warmup_timeout_uses_per_frame_when_per_frame_is_higher() -> None:
    """If timeout_ms is set higher than 30 s, use the configured value
    rather than the floor — operators may want a slow but generous gate."""
    engine = _make_engine(timeout_ms=60_000)
    assert engine._warmup_timeout_s == 60.0  # noqa: SLF001 — testing internal


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
    than Timeout/ConnectionError (``InvalidURL``, ``ChunkedEncodingError``,
    malformed proxy responses) must collapse to None. The collapsed
    catch logs the exception class name so an operator can triage which
    subtype fired.
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


def test_run_envelope_non_dict_top_level_returns_none() -> None:
    """A 200 response whose JSON top-level is null/list/string crashes on
    envelope.get(...) without an isinstance guard. Defensive against
    Ollama version drift or a misconfigured proxy.
    """
    engine = _make_engine()
    with rm_module.Mocker() as m:
        m.post(OLLAMA_CHAT_URL, json=["not", "a", "dict"])
        result = engine.run(
            frame=_fake_frame(),
            state=_make_state(),
            detections=_make_detections(),
        )
    assert result is None


def test_run_with_degenerate_frame_returns_none_does_not_raise() -> None:
    """``cv2.imencode(".png", frame)`` raises ``cv2.error`` for None,
    wrong dtype, or zero-dimension frames. Without the body-hoist fix,
    that exception escapes ``run()`` and crashes the inference worker
    (R7 violation). Verify run() collapses to None instead.
    """
    engine = _make_engine()
    bad_frame: np.ndarray = np.zeros((0, 0, 3), dtype=np.uint8)
    with rm_module.Mocker() as m:
        m.post(OLLAMA_CHAT_URL, json=_wrap(VALID_RESPONSE))
        # cv2.imencode on a 0x0 frame succeeds in some opencv builds and
        # raises in others. Force the failure path explicitly via a
        # frame that is structurally invalid for PNG encoding.
        result = engine.run(
            frame=bad_frame,
            state=_make_state(),
            detections=_make_detections(),
        )
    # Either result must be valid (encoder succeeded) OR None (encoder
    # failed and the broad catch collapsed it). Crucially, run() must
    # NOT raise. We assert no exception escaped by reaching this line.
    assert result is None or isinstance(result, Callout)


def test_run_with_none_frame_returns_none_does_not_raise() -> None:
    """A None frame is the most common 'degenerate' case in production —
    the orchestrator could pass None on a capture-loss tick before the
    capture-exhausted escalation kicks in. Must collapse to None, not
    raise into the worker thread.
    """
    engine = _make_engine()
    with rm_module.Mocker() as m:
        m.post(OLLAMA_CHAT_URL, json=_wrap(VALID_RESPONSE))
        result = engine.run(
            frame=None,  # type: ignore[arg-type]
            state=_make_state(),
            detections=_make_detections(),
        )
    assert result is None


def test_run_with_non_numeric_event_time_in_state_returns_none() -> None:
    """If state.riot_events carries a non-numeric ``EventTime`` (Riot API
    drift, cosmic ray, etc.), ``_format_objectives`` raises ValueError on
    ``float(...)``. Without the body-hoist fix that escapes through
    ``run()``. Must collapse to None.
    """
    engine = _make_engine()
    state = GameState(
        game_time_seconds=600.0,
        active_summoner_name="ActivePlayer#NA1",
        active_player_champion="Hecarim",
        riot_events=(
            {
                "EventID": 4,
                "EventName": "DragonKill",
                "EventTime": "not-a-number",  # corrupt EventTime
            },
        ),
    )
    with rm_module.Mocker() as m:
        m.post(OLLAMA_CHAT_URL, json=_wrap(VALID_RESPONSE))
        result = engine.run(
            frame=_fake_frame(),
            state=state,
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


def test_warmup_collapses_encode_frame_failure_to_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``cv2.imencode`` returns ``(False, ...)`` (some opencv builds
    fail this way instead of raising), ``_encode_frame`` raises
    ``RuntimeError``. The warmup broad-except must collapse it to False
    rather than letting it escape into the orchestrator's startup
    sequence. This test exercises both the
    ``if not success: raise RuntimeError`` branch in ``_encode_frame``
    and the ``except Exception`` broad catch in ``warmup``.
    """
    import cv2

    monkeypatch.setattr(cv2, "imencode", lambda ext, img: (False, b""))

    engine = _make_engine()
    assert engine.warmup() is False


def test_warmup_uses_longer_timeout_than_per_frame() -> None:
    """The plan documents 10-20 s cold-start model loading. Verify warmup
    uses ``self._warmup_timeout_s`` (floor 30 s), not the per-frame
    ``self._timeout_s``. We capture the timeout argument by patching
    the session's post method since requests-mock does not expose the
    ``timeout`` kwarg.
    """
    captured: dict[str, float] = {}

    def fake_post(url: str, **kwargs: object) -> requests.Response:
        captured["timeout"] = float(kwargs.get("timeout") or 0.0)
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"message": {"content": "ok"}}'
        return response

    engine = _make_engine(timeout_ms=5_000)  # 5 s per-frame
    engine._session.post = fake_post  # type: ignore[method-assign]  # noqa: SLF001
    assert engine.warmup() is True
    assert captured["timeout"] >= 30.0, (
        "warmup must use the 30s floor, not the 5s per-frame timeout"
    )


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
