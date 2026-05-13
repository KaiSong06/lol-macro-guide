"""Tests for the TTS worker and interrupt policy."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from lolcoach.config import TtsConfig
from lolcoach.inference import Callout
from lolcoach.tts import TTS, AudioData, PiperSynthesizer, _device_or_none


def _callout(
    *,
    decision: str = "Ward bot river now",
    category: str = "vision",
    confidence: int = 7,
) -> Callout:
    return Callout(
        decision=decision,
        category=category,
        target_lane="bot",
        confidence=confidence,
        reason="Test reason.",
        generated_at=1_000.0,
    )


class FakeOutput:
    def __init__(self, *, block: bool = False) -> None:
        self.block = block
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.plays: list[tuple[AudioData, str | None]] = []
        self.closed = False

    def play(self, audio: AudioData, *, device: str | None = None) -> None:
        self.plays.append((audio, device))
        self.started.set()
        if self.block:
            self.stopped.wait(timeout=2)

    def stop(self) -> None:
        self.stopped.set()

    def close(self) -> None:
        self.closed = True


class FakeSynth:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, text: str) -> AudioData:
        self.texts.append(text)
        return AudioData(samples=np.zeros(4, dtype=np.float32), sample_rate=22_050)


def _tts(output: FakeOutput, synth: FakeSynth | None = None) -> TTS:
    return TTS(
        TtsConfig(
            interrupt_confidence_delta=2,
            interrupt_categories=("gank_warning", "counter_gank"),
        ),
        synthesizer=synth or FakeSynth(),
        output=output,
    )


def test_speak_queues_and_plays_callout() -> None:
    output = FakeOutput()
    synth = FakeSynth()
    tts = _tts(output, synth)
    tts.start()

    assert tts.speak(_callout(decision="Take dragon now")) is True

    deadline = time.time() + 2
    while not output.plays and time.time() < deadline:
        time.sleep(0.01)
    tts.stop()

    assert synth.texts == ["Take dragon now"]
    assert len(output.plays) == 1
    assert output.closed is True


def test_non_priority_callout_drops_during_playback() -> None:
    output = FakeOutput(block=True)
    tts = _tts(output)
    tts.start()
    assert tts.speak(_callout(category="vision", confidence=7)) is True
    assert output.started.wait(timeout=2)

    accepted = tts.speak(_callout(category="pathing", confidence=10))
    tts.stop()

    assert accepted is False


def test_priority_callout_interrupts_when_confidence_delta_is_met() -> None:
    output = FakeOutput(block=True)
    tts = _tts(output)
    tts.start()
    assert tts.speak(_callout(category="vision", confidence=6)) is True
    assert output.started.wait(timeout=2)

    accepted = tts.speak(_callout(category="gank_warning", confidence=8))
    tts.stop()

    assert accepted is True
    assert output.stopped.is_set()


def test_priority_callout_drops_when_confidence_delta_is_too_low() -> None:
    output = FakeOutput(block=True)
    tts = _tts(output)
    tts.start()
    assert tts.speak(_callout(category="vision", confidence=7)) is True
    assert output.started.wait(timeout=2)

    accepted = tts.speak(_callout(category="gank_warning", confidence=8))
    tts.stop()

    assert accepted is False


def test_pending_callout_is_replaced_before_playback() -> None:
    output = FakeOutput()
    synth = FakeSynth()
    tts = _tts(output, synth)

    assert tts.speak(_callout(decision="First")) is True
    assert tts.speak(_callout(decision="Second")) is True
    tts.start()

    deadline = time.time() + 2
    while not synth.texts and time.time() < deadline:
        time.sleep(0.01)
    tts.stop()

    assert synth.texts == ["Second"]


def test_device_or_none_maps_system_default() -> None:
    assert _device_or_none("system_default") is None
    assert _device_or_none("Headphones") == "Headphones"


def test_tts_rejects_negative_interrupt_delta() -> None:
    with pytest.raises(ValueError):
        TTS(
            TtsConfig(interrupt_confidence_delta=-1),
            synthesizer=FakeSynth(),
            output=FakeOutput(),
        )


def test_piper_synthesizer_resolves_paths() -> None:
    synth = PiperSynthesizer(
        TtsConfig(
            piper_path="/opt/piper/piper",
            voice_dir="/opt/piper/voices",
            voice="en_US-test",
        )
    )

    assert synth._piper_path == Path("/opt/piper/piper")  # noqa: SLF001
    assert synth._model_path == Path("/opt/piper/voices/en_US-test.onnx")  # noqa: SLF001
