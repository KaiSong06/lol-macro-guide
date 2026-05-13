"""Integration tests for main orchestration boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lolcoach.capture import FramePacket
from lolcoach.config import Config, DecisionsConfig
from lolcoach.detector import ChampionDetection, DetectionResult
from lolcoach.filter import DecisionFilter
from lolcoach.inference import Callout
from lolcoach.logging_utils import JsonlLogger
from lolcoach.main import CoachRuntime
from lolcoach.riot_client import LifecycleState
from lolcoach.state import StateManager

FIXTURES = Path(__file__).parent / "fixtures"


class FakeInference:
    def __init__(self, results: list[Callout | None]) -> None:
        self.results = list(results)
        self.calls = 0

    def run(self, frame: np.ndarray, state: object, detections: object) -> Callout | None:
        self.calls += 1
        if not self.results:
            return None
        return self.results.pop(0)


class FakeTTS:
    def __init__(self) -> None:
        self.spoken: list[Callout] = []
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self, timeout: float = 2.0) -> None:
        self.stopped = True

    def speak(self, callout: Callout) -> bool:
        self.spoken.append(callout)
        return True


class DummyDetector:
    known_champions = frozenset({"Hecarim", "Jinx", "Ahri", "LeeSin", "Ekko", "Vi"})


def _callout(
    decision: str = "Path to bot river, Lee Sin is missing",
    reason: str = "Lee Sin was last seen bot river.",
    confidence: int = 8,
) -> Callout:
    return Callout(
        decision=decision,
        category="pathing",
        target_lane="bot",
        confidence=confidence,
        reason=reason,
        generated_at=1_000.0,
    )


def _packet() -> FramePacket:
    detections = DetectionResult(
        champions=(
            ChampionDetection(
                name="LeeSin",
                team="enemy",
                position_norm=(0.5, 0.5),
                quadrant="mid",
                confidence=0.9,
            ),
        ),
        detected_at=1_000.0,
    )
    return FramePacket(
        frame=np.zeros((8, 8, 3), dtype=np.uint8),
        detections=detections,
        captured_at=1_000.0,
    )


def _runtime(
    tmp_path: Path,
    *,
    inference_results: list[Callout | None] | None = None,
) -> tuple[CoachRuntime, FakeTTS]:
    state = StateManager()
    tts = FakeTTS()
    event_logger = JsonlLogger(tmp_path / "logs")
    event_logger.rotate(None)
    runtime = CoachRuntime(
        config=Config(
            decisions=DecisionsConfig(
                cooldown_seconds=0,
                confidence_threshold=6,
                dedup_window_seconds=0,
                staleness_threshold_seconds=10_000,
            )
        ),
        capture=None,
        detector=DummyDetector(),  # type: ignore[arg-type]
        state=state,
        inference=FakeInference(inference_results or [_callout()]),  # type: ignore[arg-type]
        decision_filter=DecisionFilter(
            DecisionsConfig(
                cooldown_seconds=0,
                confidence_threshold=6,
                dedup_window_seconds=0,
                staleness_threshold_seconds=10_000,
            ),
            known_champions=DummyDetector.known_champions,
            clock=lambda: 1_000.0,
        ),
        tts=tts,  # type: ignore[arg-type]
        event_logger=event_logger,
        inference_pause_s=0,
    )
    return runtime, tts


def test_single_callout_flows_through_filter_to_tts(tmp_path: Path) -> None:
    runtime, tts = _runtime(tmp_path)
    data = json.loads((FIXTURES / "allgamedata_ingame.json").read_text())
    runtime.on_game_data(data)

    spoken = runtime.process_frame_packet(_packet())

    assert spoken is True
    assert [c.decision for c in tts.spoken] == ["Path to bot river, Lee Sin is missing"]


def test_hallucinated_callout_is_filtered_before_tts(tmp_path: Path) -> None:
    runtime, tts = _runtime(
        tmp_path,
        inference_results=[
            _callout(
                decision="Vi is invading your blue",
                reason="Vi is moving into your jungle.",
            )
        ],
    )
    data = json.loads((FIXTURES / "allgamedata_ingame.json").read_text())
    runtime.on_game_data(data)

    spoken = runtime.process_frame_packet(_packet())

    assert spoken is False
    assert tts.spoken == []


def test_role_mismatch_speaks_pause_once(tmp_path: Path) -> None:
    runtime, tts = _runtime(tmp_path)

    runtime.on_role_mismatch("TOP")

    assert len(tts.spoken) == 1
    assert tts.spoken[0].decision == "You're not playing jungle. Coach paused."


def test_game_end_resets_state(tmp_path: Path) -> None:
    runtime, _tts = _runtime(tmp_path)
    data = json.loads((FIXTURES / "allgamedata_ingame.json").read_text())
    runtime.on_game_data(data)
    assert runtime.state.snapshot().all_champions

    runtime.on_game_end("timeout")

    assert runtime.state.snapshot().all_champions == frozenset()


def test_state_change_active_controls_runtime_activity(tmp_path: Path) -> None:
    runtime, _tts = _runtime(tmp_path)

    runtime.on_state_change(LifecycleState.STARTING, LifecycleState.ACTIVE, "ok")

    assert runtime._active.is_set()  # noqa: SLF001


def test_five_inference_misses_speak_pause_message(tmp_path: Path) -> None:
    runtime, tts = _runtime(tmp_path, inference_results=[None] * 5)
    data = json.loads((FIXTURES / "allgamedata_ingame.json").read_text())
    runtime.on_game_data(data)

    for _ in range(5):
        runtime.process_frame_packet(_packet())

    assert tts.spoken[-1].decision == (
        "Coach is having trouble with inference. Pausing briefly."
    )
