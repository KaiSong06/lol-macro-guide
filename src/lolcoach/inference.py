"""Local Ollama vision client and structured response parser.

The inference engine is the bridge between the deterministic capture/state
pipeline and the language model that does the spatial reasoning. Three jobs:

1. **Format the prompt.** Build the system prompt (constant schema definition)
   and the user prompt (per-frame context block) via :mod:`lolcoach.prompts`.

2. **Send the request to Ollama.** POST to ``/api/chat`` with the user prompt,
   the base64-encoded minimap frame, and ``keep_alive: -1`` to pin the model
   in VRAM. Network/timeout/HTTP errors all collapse to ``None`` — the
   orchestrator (Unit 10) handles retries and escalation, not this module.

3. **Parse the response into a structured Callout.** The expected response
   shape is a 5-line ``KEY: value`` block defined in the system prompt. The
   parser is strict: missing field, unknown enum value, out-of-range
   confidence, or empty decision all collapse to ``None`` plus a log entry.
   Anything that reaches the DecisionFilter (Unit 8) is guaranteed to be a
   well-formed Callout — the filter does not need to defend against parser
   errors.

**Single-worker drop policy.** ``run()`` uses a non-blocking lock acquire so
two concurrent calls cannot both contact Ollama. The second caller returns
``None`` immediately without making a network request — wasting a 5-8s
inference call on a stale frame is worse than producing fewer callouts.

This module never instantiates a :class:`~lolcoach.logging_utils.JsonlLogger`
directly; structured-event logging will be wired in by Unit 10's orchestrator
via stdlib ``logging.Handler``. Until then, the module logs through stdlib
``logging`` like every other module in the project.
"""

from __future__ import annotations

import base64
import ipaddress
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

from lolcoach.config import InferenceConfig
from lolcoach.detector import DetectionResult
from lolcoach.prompts import (
    ALLOWED_CATEGORIES,
    ALLOWED_LANES,
    build_system_prompt,
    build_user_prompt,
)
from lolcoach.state import GameState

logger = logging.getLogger(__name__)

#: Confidence range matches the system prompt rule (1-10 inclusive).
_MIN_CONFIDENCE = 1
_MAX_CONFIDENCE = 10

#: Floor for the warmup timeout in seconds. The plan documents that
#: cold-start model loading can take 10-20 s, so warmup uses
#: ``max(_WARMUP_MIN_TIMEOUT_S, self._timeout_s)`` to give a fresh
#: Ollama process time to load the model on the first call. Per-frame
#: ``run()`` still uses the configured ``timeout_ms`` so live coaching
#: stays tight to the 8 s latency budget.
_WARMUP_MIN_TIMEOUT_S = 30.0

#: Per-field regex — anchored on the field key with optional horizontal
#: whitespace only (``[ \t]*`` not ``\s*``, so the engine cannot eat newlines
#: and slurp the next line into the captured value). ``re.MULTILINE`` makes
#: ``^`` and ``$`` bind to line boundaries instead of string boundaries, so
#: each pattern matches one logical "KEY: value" line independently of order.
_FIELD_RE: dict[str, re.Pattern[str]] = {
    "decision": re.compile(r"^[ \t]*DECISION[ \t]*:[ \t]*(.*)$", re.MULTILINE),
    "category": re.compile(r"^[ \t]*CATEGORY[ \t]*:[ \t]*(.*)$", re.MULTILINE),
    "target_lane": re.compile(
        r"^[ \t]*TARGET_LANE[ \t]*:[ \t]*(.*)$", re.MULTILINE
    ),
    "confidence": re.compile(r"^[ \t]*CONFIDENCE[ \t]*:[ \t]*(.*)$", re.MULTILINE),
    "reason": re.compile(r"^[ \t]*REASON[ \t]*:[ \t]*(.*)$", re.MULTILINE),
}


# ---------------------------------------------------------------------------
# Value type
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Callout:
    """A single parsed coaching callout from the vision LLM.

    Frozen so it can cross thread boundaries from the inference worker into
    the decision filter and TTS worker without any defensive copying.
    """

    decision: str
    category: str
    target_lane: str
    confidence: int
    reason: str
    generated_at: float


# ---------------------------------------------------------------------------
# Response parser — pure function, easy to test
# ---------------------------------------------------------------------------
def _parse_response(text: str) -> Callout | None:
    """Parse a model response into a :class:`Callout`, or return ``None``.

    The parser is strict: any missing or invalid field collapses the whole
    callout to ``None`` plus a structured log line. Order of fields does
    not matter (each field is matched independently).

    Validation rules:

    * All 5 fields must be present and non-empty.
    * ``CATEGORY`` must be in :data:`lolcoach.prompts.ALLOWED_CATEGORIES`.
    * ``TARGET_LANE`` must be in :data:`lolcoach.prompts.ALLOWED_LANES`.
    * ``CONFIDENCE`` must parse as an integer in [1, 10].
    """
    if not text:
        logger.warning("inference_malformed: empty response body")
        return None

    extracted: dict[str, str] = {}
    for field, pattern in _FIELD_RE.items():
        match = pattern.search(text)
        if match is None:
            logger.warning(
                "inference_malformed: missing field=%s raw=%r", field, text
            )
            return None
        value = match.group(1).strip()
        if not value:
            logger.warning(
                "inference_malformed: empty field=%s raw=%r", field, text
            )
            return None
        extracted[field] = value

    category = extracted["category"]
    if category not in ALLOWED_CATEGORIES:
        logger.warning(
            "inference_malformed: unknown category=%r raw=%r", category, text
        )
        return None

    target_lane = extracted["target_lane"]
    if target_lane not in ALLOWED_LANES:
        logger.warning(
            "inference_malformed: unknown lane=%r raw=%r", target_lane, text
        )
        return None

    try:
        confidence = int(extracted["confidence"])
    except (TypeError, ValueError):
        logger.warning(
            "inference_malformed: non-integer confidence=%r raw=%r",
            extracted["confidence"],
            text,
        )
        return None

    if not _MIN_CONFIDENCE <= confidence <= _MAX_CONFIDENCE:
        logger.warning(
            "inference_malformed: out-of-range confidence=%d raw=%r",
            confidence,
            text,
        )
        return None

    return Callout(
        decision=extracted["decision"],
        category=category,
        target_lane=target_lane,
        confidence=confidence,
        reason=extracted["reason"],
        generated_at=time.time(),
    )


# ---------------------------------------------------------------------------
# Construction-time validation helpers
# ---------------------------------------------------------------------------
def _assert_loopback_ollama_host(ollama_host: str) -> None:
    """Reject any ``ollama_host`` whose host is not a loopback address.

    The inference engine forwards the full game-state context block plus
    the base64-encoded minimap PNG to the configured Ollama host on every
    call. If a user (or an attacker who can tamper with ``config.yaml``)
    points ``inference.ollama_host`` at a remote URL, every callout would
    silently exfiltrate summoner names, kill feed entries, and the live
    minimap screenshot to that remote server. This check enforces the
    loopback invariant at construction time so the failure mode is
    impossible, not merely discouraged.

    The same loopback rule lives in :func:`lolcoach.riot_client._assert_loopback_base_url`;
    the helper is duplicated rather than shared to avoid an import cycle
    between ``inference`` and ``riot_client`` (both modules are
    independently consumed by Unit 10's orchestrator).
    """
    parsed = urlparse(ollama_host)
    host = parsed.hostname
    if not host:
        raise ValueError(
            f"InferenceConfig.ollama_host must have a hostname; got {ollama_host!r}"
        )
    # Accept the literal ``localhost`` alias even though it resolves at
    # runtime — DNS resolution against the system's localhost entry is
    # effectively loopback for any sane system.
    if host.lower() == "localhost":
        return
    try:
        ip = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError(
            f"InferenceConfig.ollama_host must be an IP address or 'localhost' "
            f"(got {host!r}); the engine sends prompts and minimap frames to "
            f"this URL on every call so it must point at the loopback interface"
        ) from exc
    if not ip.is_loopback:
        raise ValueError(
            f"InferenceConfig.ollama_host must be a loopback address "
            f"(got {host!r}); the engine sends prompts and minimap frames to "
            f"this URL on every call so it must point at the loopback interface"
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class InferenceEngine:
    """Synchronous Ollama vision client with single-worker drop policy.

    The engine owns a ``requests.Session`` (HTTP keep-alive helps reduce
    per-call latency) and a non-blocking ``threading.Lock`` that enforces
    the single-worker drop rule on :meth:`run`.
    """

    def __init__(self, config: InferenceConfig) -> None:
        # Fail-fast at the boundary on misconfiguration. The orchestrator
        # surfaces the ValueError as a clear startup error rather than
        # crashing the inference worker thread mid-game.
        if config.timeout_ms <= 0:
            raise ValueError(
                f"InferenceConfig.timeout_ms must be > 0 "
                f"(got {config.timeout_ms!r}); a non-positive value would "
                f"make every requests.post raise ValueError mid-game"
            )
        _assert_loopback_ollama_host(config.ollama_host)

        self._config = config
        self._session = requests.Session()
        self._lock = threading.Lock()
        self._chat_url = f"{config.ollama_host.rstrip('/')}/api/chat"
        self._timeout_s = config.timeout_ms / 1000.0
        # The warmup ping has its own (looser) timeout because the plan
        # documents 10-20 s cold-start model loading. Live coaching uses
        # the configured per-frame timeout to stay under the 8 s budget.
        self._warmup_timeout_s = max(_WARMUP_MIN_TIMEOUT_S, self._timeout_s)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def warmup(self) -> bool:
        """Send a small priming request to load the model into VRAM.

        Returns ``True`` if Ollama responded with a 200 (the model is
        loaded and serving). Returns ``False`` on any HTTP error,
        timeout, connection failure, or unexpected exception during body
        construction. Called once at coach startup to avoid the 10-20 s
        cold-start latency on the first real callout.

        Uses :data:`self._warmup_timeout_s` (a floor of 30 s) instead of
        the per-frame ``timeout_s`` so a fresh Ollama process has time to
        load the model on the first call.

        The warmup request carries ``keep_alive: -1`` so the model
        stays pinned in VRAM until the process exits.
        """
        try:
            # A 1x1 black image is enough to satisfy the vision input
            # contract. _encode_frame can raise on degenerate inputs;
            # the broad catch below collapses every body-construction
            # failure to False rather than letting it escape into the
            # orchestrator's startup sequence.
            tiny_frame = np.zeros((1, 1, 3), dtype=np.uint8)
            body = {
                "model": self._config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "warmup",
                        "images": [_encode_frame(tiny_frame)],
                    }
                ],
                "stream": False,
                "keep_alive": self._config.keep_alive,
            }
            response = self._session.post(
                self._chat_url,
                json=body,
                timeout=self._warmup_timeout_s,
            )
        except requests.RequestException as exc:
            logger.warning(
                "inference_warmup_request_error host=%s err_type=%s err=%s",
                self._chat_url,
                type(exc).__name__,
                exc,
            )
            return False
        except Exception as exc:  # noqa: BLE001 - intentional broad catch
            # Defensive: cv2.error, ValueError, TypeError, anything raised
            # during body construction. Collapsing to False here keeps
            # warmup honest with its documented contract.
            logger.warning(
                "inference_warmup_unexpected_error err_type=%s err=%s",
                type(exc).__name__,
                exc,
            )
            return False

        if response.status_code != 200:
            logger.warning(
                "inference_warmup_http_error status=%d host=%s",
                response.status_code,
                self._chat_url,
            )
            return False
        return True

    def run(
        self,
        frame: np.ndarray,
        state: GameState,
        detections: DetectionResult,
    ) -> Callout | None:
        """Send a single inference call and return the parsed Callout.

        Drops the call immediately (returns ``None``) if another inference
        is already in flight on this engine — the stale-frame policy is
        better than queuing because LLM calls take 5-8 s and the next frame
        is more useful than the old one.

        Returns ``None`` on every failure mode and never raises:
        * lock contention (single-worker drop)
        * any ``requests.RequestException`` subtype (Timeout, ConnectionError,
          SSLError, InvalidURL, ChunkedEncodingError, etc.)
        * HTTP non-200
        * malformed response envelope (invalid JSON, non-dict top-level,
          missing or non-string ``message.content``)
        * malformed response content (parser failure)
        * unexpected errors during body construction (``cv2.error`` on a
          degenerate frame, ``ValueError`` from a non-numeric ``EventTime``
          in ``state.riot_events``, etc.)

        The orchestrator (Unit 10) inspects the return value and tracks
        consecutive ``None`` results for the timeout-escalation ladder.
        """
        if not self._lock.acquire(blocking=False):
            logger.debug("inference_dropped_stale_frame")
            return None
        try:
            return self._run_locked(frame=frame, state=state, detections=detections)
        finally:
            self._lock.release()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _run_locked(
        self,
        frame: np.ndarray,
        state: GameState,
        detections: DetectionResult,
    ) -> Callout | None:
        try:
            # Body construction lives inside the try so any error in
            # _encode_frame (cv2.error on degenerate frames) or in
            # build_user_prompt (e.g. non-numeric EventTime in
            # state.riot_events) collapses to None instead of escaping
            # into the orchestrator's worker thread. The docstring
            # contract on run() promises every failure mode returns
            # None — this is what enforces it.
            body = {
                "model": self._config.model,
                "messages": [
                    {"role": "system", "content": build_system_prompt()},
                    {
                        "role": "user",
                        "content": build_user_prompt(state, detections),
                        "images": [_encode_frame(frame)],
                    },
                ],
                "stream": False,
                "keep_alive": self._config.keep_alive,
            }
            response = self._session.post(
                self._chat_url,
                json=body,
                timeout=self._timeout_s,
            )
        except requests.RequestException as exc:
            # Single branch covering Timeout, ConnectionError, SSLError,
            # InvalidURL, ChunkedEncodingError, etc. The exception class
            # name carries the failure mode for log triage.
            logger.warning(
                "inference_request_error host=%s err_type=%s err=%s",
                self._chat_url,
                type(exc).__name__,
                exc,
            )
            return None
        except Exception as exc:  # noqa: BLE001 - intentional broad catch
            # Defensive: cv2.error from _encode_frame, ValueError from
            # _format_objectives/_format_kill_feed on bad EventTime, or
            # anything else that escapes body construction. Returning
            # None keeps the orchestrator's None-counter ladder honest.
            logger.warning(
                "inference_unexpected_error err_type=%s err=%s",
                type(exc).__name__,
                exc,
            )
            return None

        if response.status_code != 200:
            logger.warning(
                "inference_http_error status=%d host=%s",
                response.status_code,
                self._chat_url,
            )
            return None

        try:
            envelope: Any = response.json()
        except ValueError:
            logger.warning("inference_envelope_invalid_json")
            return None

        if not isinstance(envelope, dict):
            logger.warning(
                "inference_envelope_not_dict type=%s",
                type(envelope).__name__,
            )
            return None

        message = envelope.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            logger.warning(
                "inference_envelope_missing_content envelope_keys=%s",
                list(envelope.keys()),
            )
            return None

        return _parse_response(content)


# ---------------------------------------------------------------------------
# Image encoding helper
# ---------------------------------------------------------------------------
def _encode_frame(frame: np.ndarray) -> str:
    """Encode an OpenCV BGR frame as a base64 PNG string for Ollama.

    Ollama's ``/api/chat`` ``images`` field accepts a list of base64
    strings (no ``data:image/...`` prefix). PNG is lossless and small
    enough for a minimap crop; JPEG would compress the icons too aggressively
    and could degrade the model's spatial reasoning.
    """
    success, encoded = cv2.imencode(".png", frame)
    if not success:
        raise RuntimeError("cv2.imencode failed for inference frame")
    return base64.b64encode(encoded.tobytes()).decode("ascii")
