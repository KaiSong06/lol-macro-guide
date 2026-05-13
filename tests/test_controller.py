"""Tests for the desktop CoachController."""

from __future__ import annotations

from pathlib import Path

from lolcoach.config import Config
from lolcoach.controller import CoachController
from lolcoach.state import StateManager


class FakeRuntime:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.state = StateManager()

    def start_workers(self) -> None:
        self.started = True

    def stop_workers(self) -> None:
        self.stopped = True


class FakeRiot:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def test_controller_start_stop_lifecycle(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    riot = FakeRiot()
    statuses = []

    controller = CoachController(
        config_path=tmp_path / "missing.yaml",
        runtime_builder=lambda _config: (runtime, riot),  # type: ignore[return-value]
    )
    controller.add_listener(statuses.append)

    controller.start()
    started = controller.status_snapshot()
    controller.stop()
    stopped = controller.status_snapshot()

    assert runtime.started is True
    assert riot.started is True
    assert started.is_running is True
    assert stopped.is_running is False
    assert runtime.stopped is True
    assert riot.stopped is True
    assert statuses[-1].state == "Ready"


def test_controller_surfaces_start_error(tmp_path: Path) -> None:
    def boom(_config: Config) -> object:
        raise RuntimeError("no capture")

    controller = CoachController(
        config_path=tmp_path / "missing.yaml",
        runtime_builder=boom,  # type: ignore[arg-type]
    )

    controller.start()
    status = controller.status_snapshot()

    assert status.state == "Error"
    assert status.last_error == "no capture"


def test_controller_records_recent_callouts(tmp_path: Path) -> None:
    controller = CoachController(config_path=tmp_path / "missing.yaml")

    for idx in range(12):
        controller.record_callout(f"callout {idx}")

    status = controller.status_snapshot()
    assert len(status.latest_callouts) == 10
    assert status.latest_callouts[0] == "callout 11"
