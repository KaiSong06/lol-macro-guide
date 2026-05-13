"""Main orchestration for the lolcoach runtime."""

from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Any

from lolcoach.capture import (
    ROI,
    Capture,
    CaptureError,
    CaptureExhausted,
    FramePacket,
    LatestFrameBuffer,
    load_calibration,
    validate_roi_against_screen,
)
from lolcoach.config import Config, load_config
from lolcoach.detector import Detector
from lolcoach.filter import DecisionFilter
from lolcoach.inference import Callout, InferenceEngine
from lolcoach.logging_utils import JsonlLogger
from lolcoach.riot_client import Callbacks, LifecycleState, RiotClient
from lolcoach.state import StateManager
from lolcoach.template_builder import DEFAULT_OUTPUT_DIR, download_all
from lolcoach.tts import TTS

logger = logging.getLogger(__name__)

INFERENCE_TIMEOUT_LIMIT = 5
INFERENCE_PAUSE_S = 60.0
SYSTEM_CONFIDENCE = 10


class CoachRuntime:
    """Own and coordinate the capture, Riot, inference, filter, and TTS pieces."""

    def __init__(
        self,
        *,
        config: Config,
        capture: Capture | None,
        detector: Detector,
        state: StateManager,
        inference: InferenceEngine,
        decision_filter: DecisionFilter,
        tts: TTS,
        event_logger: JsonlLogger,
        frame_buffer: LatestFrameBuffer | None = None,
        inference_pause_s: float = INFERENCE_PAUSE_S,
    ) -> None:
        self.config = config
        self.capture = capture
        self.detector = detector
        self.state = state
        self.inference = inference
        self.decision_filter = decision_filter
        self.tts = tts
        self.event_logger = event_logger
        self.frame_buffer = frame_buffer or LatestFrameBuffer()
        self.inference_pause_s = inference_pause_s

        self._active = threading.Event()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._consecutive_inference_misses = 0
        self._paused_until_next_game = False

    def callbacks(self) -> Callbacks:
        return Callbacks(
            on_state_change=self.on_state_change,
            on_game_data=self.on_game_data,
            on_role_mismatch=self.on_role_mismatch,
            on_game_end=self.on_game_end,
        )

    def start_workers(self) -> None:
        """Start capture, inference, and TTS workers."""

        self.tts.start()
        if self.capture is not None:
            self._threads.append(
                threading.Thread(target=self._capture_loop, name="capture", daemon=True)
            )
        self._threads.append(
            threading.Thread(target=self._inference_loop, name="inference", daemon=True)
        )
        for thread in self._threads:
            thread.start()

    def stop_workers(self, timeout: float = 2.0) -> None:
        self._stop.set()
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=timeout)
        self.tts.stop(timeout=timeout)
        self.event_logger.close()

    def on_state_change(
        self,
        from_state: LifecycleState,
        to_state: LifecycleState,
        reason: str,
    ) -> None:
        self._log("lifecycle", from_state=from_state.value, to_state=to_state.value, reason=reason)
        if to_state == LifecycleState.STARTING:
            self.event_logger.rotate(game_id=None)
            self._paused_until_next_game = False
            self._consecutive_inference_misses = 0
        if to_state == LifecycleState.ACTIVE:
            self._active.set()
        if to_state == LifecycleState.ENDING:
            self._active.clear()

    def on_game_data(self, data: dict[str, Any]) -> None:
        self.state.update_from_riot(data)
        self._log("riot_game_data", game_time=self.state.snapshot().game_time_seconds)

    def on_role_mismatch(self, role: str) -> None:
        self._active.clear()
        self._system_speak("You're not playing jungle. Coach paused.", "role_mismatch")
        self._log("role_mismatch", role=role)

    def on_game_end(self, reason: str) -> None:
        self._active.clear()
        self.state.reset()
        self._paused_until_next_game = False
        self._consecutive_inference_misses = 0
        self._log("game_end", reason=reason)

    def process_frame_packet(self, packet: FramePacket) -> bool:
        """Run one inference/filter/TTS cycle for a captured frame."""

        if self._paused_until_next_game:
            return False
        snapshot = self.state.snapshot()
        callout = self.inference.run(packet.frame, snapshot, packet.detections)
        if callout is None:
            self._record_inference_miss()
            return False

        self._consecutive_inference_misses = 0
        decision = self.decision_filter.should_speak(callout, snapshot, packet.detections)
        self._log(
            "decision_filter",
            accepted=decision.accepted,
            reason=decision.reason,
            category=callout.category,
            target_lane=callout.target_lane,
            confidence=callout.confidence,
        )
        if not decision.accepted:
            return False
        if self.tts.speak(callout):
            self.decision_filter.record_spoken(callout)
            self._log("callout_spoken", decision=callout.decision, category=callout.category)
            return True
        self._log("callout_dropped_by_tts", decision=callout.decision)
        return False

    def _capture_loop(self) -> None:  # pragma: no cover - thread/hardware path
        assert self.capture is not None
        interval_s = self.config.capture.interval_ms / 1000.0
        while not self._stop.is_set():
            if not self._active.is_set():
                self._stop.wait(interval_s)
                continue
            try:
                frame = self.capture.grab_minimap()
            except CaptureExhausted:
                self._active.clear()
                self._system_speak("Screen capture failed. Coach paused.", "capture_failed")
                self._log("capture_exhausted")
                self._stop.wait(interval_s)
                continue
            except CaptureError:
                self._log("capture_transient_error")
                self._stop.wait(interval_s)
                continue

            snapshot = self.state.snapshot()
            detections = self.detector.detect(
                frame,
                ally_names=snapshot.ally_champions,
                enemy_names=snapshot.enemy_champions,
            )
            self.state.update_from_detector(detections)
            self.frame_buffer.put_latest(
                FramePacket(frame=frame, detections=detections, captured_at=time.time())
            )
            self._stop.wait(interval_s)

    def _inference_loop(self) -> None:  # pragma: no cover - thread loop
        while not self._stop.is_set():
            if not self._active.is_set():
                self._stop.wait(0.1)
                continue
            item = self.frame_buffer.get(timeout=0.5)
            if not isinstance(item, FramePacket):
                continue
            self.process_frame_packet(item)

    def _record_inference_miss(self) -> None:
        self._consecutive_inference_misses += 1
        self._log("inference_miss", consecutive=self._consecutive_inference_misses)
        if self._consecutive_inference_misses == INFERENCE_TIMEOUT_LIMIT:
            self._system_speak(
                "Coach is having trouble with inference. Pausing briefly.",
                "inference_pause",
            )
            self._log("inference_pause", seconds=self.inference_pause_s)
            time.sleep(self.inference_pause_s)
        elif self._consecutive_inference_misses >= INFERENCE_TIMEOUT_LIMIT * 3:
            self._paused_until_next_game = True
            self._system_speak(
                "Coach unavailable. Paused until next game.",
                "inference_unavailable",
            )
            self._log("inference_paused_until_next_game")

    def _system_speak(self, text: str, category: str) -> None:
        self.tts.speak(
            Callout(
                decision=text,
                category=category,
                target_lane="global",
                confidence=SYSTEM_CONFIDENCE,
                reason="system",
                generated_at=time.time(),
            )
        )

    def _log(self, event_type: str, **fields: Any) -> None:
        try:
            self.event_logger.log_event(event_type, **fields)
        except RuntimeError:
            logger.debug("event logger not ready for %s", event_type)


def build_runtime(config: Config) -> tuple[CoachRuntime, RiotClient]:  # pragma: no cover
    """Create the production runtime and Riot client."""

    logging.basicConfig(level=getattr(logging, config.logging.level.upper(), logging.INFO))
    event_logger = JsonlLogger(Path(config.logging.directory))
    event_logger.rotate(game_id=None)

    roi = _resolve_roi(config)
    capture = Capture.create(roi)

    template_dir = DEFAULT_OUTPUT_DIR
    if not template_dir.exists() or not any(template_dir.glob("*.png")):
        download_all(template_dir)
    detector = Detector(template_dir)
    state = StateManager()
    inference = InferenceEngine(config.inference)
    inference.warmup()
    decision_filter = DecisionFilter(
        config.decisions,
        known_champions=detector.known_champions,
    )
    tts = TTS(config.tts)
    runtime = CoachRuntime(
        config=config,
        capture=capture,
        detector=detector,
        state=state,
        inference=inference,
        decision_filter=decision_filter,
        tts=tts,
        event_logger=event_logger,
    )
    riot = RiotClient(config.riot_api, runtime.callbacks())
    return runtime, riot


def _resolve_roi(config: Config) -> ROI:
    raw = config.capture.minimap_roi
    if isinstance(raw, tuple):
        return ROI(*raw)
    roi = load_calibration(Path("calibration.json"))
    if roi is None:
        raise RuntimeError(
            "No minimap calibration found. Run: python -m lolcoach.calibrate "
            "--corners X1,Y1,X2,Y2"
        )
    validate_roi_against_screen(roi, screen_width=10_000, screen_height=10_000)
    return roi


def run(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(prog="python -m lolcoach")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args(argv)

    config = load_config(args.config)
    runtime, riot = build_runtime(config)
    runtime.start_workers()
    riot.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("shutdown requested")
    finally:
        riot.stop()
        runtime.stop_workers()
    return 0
