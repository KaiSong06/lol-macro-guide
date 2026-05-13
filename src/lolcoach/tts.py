"""Piper-backed text-to-speech worker.

The real runtime synthesizes one-sentence callouts with the Piper executable
and plays them through ``sounddevice``. Both dependencies are kept behind
runtime boundaries so tests can inject fakes and this module remains importable
on non-Windows development machines.
"""

from __future__ import annotations

import logging
import queue
import subprocess
import sys
import tempfile
import threading
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from lolcoach.config import TtsConfig
from lolcoach.inference import Callout

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioData:
    """PCM audio samples plus sample rate."""

    samples: np.ndarray
    sample_rate: int


class AudioOutput(Protocol):
    """Minimal playback sink used by ``TTS``."""

    def play(self, audio: AudioData, *, device: str | None = None) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


Synthesizer = Callable[[str], AudioData]


class TTS:
    """Background TTS worker with priority interrupt rules."""

    def __init__(
        self,
        config: TtsConfig,
        *,
        synthesizer: Synthesizer | None = None,
        output: AudioOutput | None = None,
    ) -> None:
        if config.interrupt_confidence_delta < 0:
            raise ValueError("interrupt_confidence_delta must be >= 0")
        self._config = config
        self._synthesizer = synthesizer or PiperSynthesizer(config)
        self._output = output or SoundDeviceOutput()
        self._queue: queue.Queue[Callout] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._currently_playing: Callout | None = None

    @property
    def currently_playing(self) -> Callout | None:
        with self._lock:
            return self._currently_playing

    def start(self) -> None:
        """Start the playback worker. Idempotent."""

        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="tts", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop playback and join the worker."""

        self._stop_event.set()
        self._output.stop()
        self._drain_queue()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        if thread is not None and not thread.is_alive():
            self._thread = None
        self._output.close()

    def speak(self, callout: Callout) -> bool:
        """Queue *callout* for speech.

        Returns ``True`` when the callout was accepted into the TTS pipeline.
        Returns ``False`` when it is dropped because another callout is already
        playing and the new one is not interrupt-eligible.
        """

        with self._lock:
            current = self._currently_playing
            if current is not None:
                if not self._can_interrupt(current, callout):
                    logger.info(
                        "tts_drop_during_playback category=%s confidence=%s",
                        callout.category,
                        callout.confidence,
                    )
                    return False
                self._output.stop()

        self._replace_pending(callout)
        return True

    def _can_interrupt(self, current: Callout, new: Callout) -> bool:
        if new.category not in self._config.interrupt_categories:
            return False
        required = current.confidence + self._config.interrupt_confidence_delta
        return new.confidence >= required

    def _replace_pending(self, callout: Callout) -> None:
        self._drain_queue()
        try:
            self._queue.put_nowait(callout)
        except queue.Full:  # pragma: no cover - drain+put should prevent this
            pass

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _run(self) -> None:  # pragma: no cover - covered through stop/join tests indirectly
        while not self._stop_event.is_set():
            try:
                callout = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._lock:
                self._currently_playing = callout
            try:
                audio = self._synthesizer(callout.decision)
                if not self._stop_event.is_set():
                    self._output.play(audio, device=_device_or_none(self._config.output_device))
            except Exception:  # noqa: BLE001 - TTS failures should not crash runtime
                logger.exception("tts_playback_failed")
            finally:
                with self._lock:
                    if self._currently_playing is callout:
                        self._currently_playing = None
                self._queue.task_done()


class PiperSynthesizer:
    """Subprocess wrapper around the Piper executable."""

    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        self._piper_path = _resolve_piper_path(config.piper_path)
        self._model_path = _resolve_voice_path(config.voice_dir, config.voice)

    def __call__(self, text: str) -> AudioData:  # pragma: no cover - external binary
        with tempfile.TemporaryDirectory(prefix="lolcoach-tts-") as tmp:
            output_path = Path(tmp) / "speech.wav"
            cmd = [
                str(self._piper_path),
                "--model",
                str(self._model_path),
                "--output_file",
                str(output_path),
            ]
            proc = subprocess.run(
                cmd,
                input=text,
                text=True,
                capture_output=True,
                check=False,
                timeout=20,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"piper failed with exit code {proc.returncode}: {proc.stderr}"
                )
            return _read_wav(output_path)


class SoundDeviceOutput:  # pragma: no cover - hardware/audio adapter
    """``sounddevice`` playback adapter, imported lazily."""

    def __init__(self) -> None:
        import sounddevice as sd  # type: ignore[import-not-found]  # pragma: no cover

        self._sd = sd

    def play(self, audio: AudioData, *, device: str | None = None) -> None:
        self._sd.play(audio.samples, audio.sample_rate, device=device)
        self._sd.wait()

    def stop(self) -> None:
        self._sd.stop()

    def close(self) -> None:
        return None


def _read_wav(path: Path) -> AudioData:  # pragma: no cover - exercised via Piper smoke test
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if width == 2:
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        samples = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"unsupported Piper WAV sample width: {width}")

    if channels > 1:
        samples = samples.reshape(-1, channels)
    return AudioData(samples=samples, sample_rate=sample_rate)


def _resolve_piper_path(raw: str) -> Path:
    if raw != "auto":
        return Path(raw)
    exe_name = "piper.exe" if sys.platform == "win32" else "piper"
    return Path("scripts") / "binaries" / "piper" / exe_name


def _resolve_voice_path(raw_dir: str, voice: str) -> Path:
    directory = Path("scripts") / "binaries" / "voices" if raw_dir == "auto" else Path(raw_dir)
    name = voice if voice.endswith(".onnx") else f"{voice}.onnx"
    return directory / name


def _device_or_none(output_device: str) -> str | None:
    if output_device == "system_default":
        return None
    return output_device
