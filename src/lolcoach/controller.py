"""UI-friendly controller around the runtime and Riot client."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from lolcoach.config import Config, load_config
from lolcoach.main import CoachRuntime, build_runtime
from lolcoach.riot_client import RiotClient
from lolcoach.state import GameState


@dataclass(frozen=True)
class CoachStatus:
    """Current desktop-shell status."""

    state: str
    message: str
    is_running: bool
    latest_callouts: tuple[str, ...] = ()
    last_error: str | None = None
    game_state: GameState | None = None


StatusListener = Callable[[CoachStatus], None]
RuntimeBuilder = Callable[[Config], tuple[CoachRuntime, RiotClient]]


@dataclass
class CoachController:
    """Non-blocking start/stop facade for the desktop UI."""

    config_path: Path = Path("config.yaml")
    runtime_builder: RuntimeBuilder = build_runtime
    _runtime: CoachRuntime | None = field(default=None, init=False)
    _riot: RiotClient | None = field(default=None, init=False)
    _state: str = field(default="Ready", init=False)
    _message: str = field(default="Ready to start", init=False)
    _last_error: str | None = field(default=None, init=False)
    _latest_callouts: list[str] = field(default_factory=list, init=False)
    _listeners: list[StatusListener] = field(default_factory=list, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    @property
    def is_running(self) -> bool:
        return self._runtime is not None and self._riot is not None

    def add_listener(self, listener: StatusListener) -> None:
        self._listeners.append(listener)

    def start(self) -> None:
        """Start the coach runtime without blocking the UI event loop."""

        with self._lock:
            if self.is_running:
                return
            try:
                config = load_config(self.config_path)
                runtime, riot = self.runtime_builder(config)
                runtime.start_workers()
                riot.start()
            except Exception as exc:  # noqa: BLE001 - surfaced in UI
                self._state = "Error"
                self._message = "Failed to start coach"
                self._last_error = str(exc)
                self._notify()
                return
            self._runtime = runtime
            self._riot = riot
            self._state = "Waiting for game"
            self._message = "Coach is running"
            self._last_error = None
            self._notify()

    def stop(self) -> None:
        with self._lock:
            runtime = self._runtime
            riot = self._riot
            self._runtime = None
            self._riot = None
        if riot is not None:
            riot.stop()
        if runtime is not None:
            runtime.stop_workers()
        with self._lock:
            self._state = "Ready"
            self._message = "Coach stopped"
            self._notify()

    def record_callout(self, text: str) -> None:
        with self._lock:
            self._latest_callouts.insert(0, text)
            del self._latest_callouts[10:]
            self._notify()

    def status_snapshot(self) -> CoachStatus:
        with self._lock:
            game_state = None
            if self._runtime is not None:
                game_state = self._runtime.state.snapshot()
            return CoachStatus(
                state=self._state,
                message=self._message,
                is_running=self.is_running,
                latest_callouts=tuple(self._latest_callouts),
                last_error=self._last_error,
                game_state=game_state,
            )

    def _notify(self) -> None:
        status = self.status_snapshot()
        for listener in tuple(self._listeners):
            listener(status)
