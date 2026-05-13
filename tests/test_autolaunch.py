"""Tests for Windows match-start auto-launch support."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from lolcoach import autolaunch


def test_build_monitor_command_source_install(monkeypatch) -> None:
    monkeypatch.setattr(sys, "executable", r"C:\Python311\python.exe")
    monkeypatch.setattr(sys, "frozen", False, raising=False)

    command = autolaunch.build_monitor_command(Path(r"C:\app\config.yaml"))

    assert command == [
        r"C:\Python311\python.exe",
        "-m",
        "lolcoach",
        "--monitor-league",
        "--config",
        r"C:\app\config.yaml",
    ]


def test_build_app_launch_command_frozen_exe(monkeypatch) -> None:
    monkeypatch.setattr(sys, "executable", r"C:\Program Files\lolcoach\lol-macro-guide.exe")
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    command = autolaunch.build_app_launch_command(Path(r"C:\app\config.yaml"))

    assert command == [
        r"C:\Program Files\lolcoach\lol-macro-guide.exe",
        "--minimized",
        "--auto-start",
        "--config",
        r"C:\app\config.yaml",
    ]


def test_register_task_uses_schtasks(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "executable", r"C:\Python311\python.exe")
    monkeypatch.setattr(sys, "frozen", False, raising=False)

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    autolaunch.register_task(Path(r"C:\app\config.yaml"))

    assert calls[0][:5] == ["schtasks", "/Create", "/TN", autolaunch.TASK_NAME, "/TR"]
    assert "lolcoach" in calls[0][5]
    assert "--monitor-league" in calls[0][5]
    assert "/SC" in calls[0]
    assert "ONLOGON" in calls[0]


def test_unregister_task_uses_schtasks_delete(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    autolaunch.unregister_task()

    assert calls == [["schtasks", "/Delete", "/TN", autolaunch.TASK_NAME, "/F"]]


def test_start_monitor_uses_monitor_command(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    class FakePopen:
        def __init__(self, args: list[str], **_kwargs: object) -> None:
            calls.append(args)

    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    autolaunch.start_monitor(tmp_path / "config.yaml")

    assert calls[0][-3:] == ["--monitor-league", "--config", str(tmp_path / "config.yaml")]


def test_read_state_missing_and_invalid_file(tmp_path: Path) -> None:
    assert autolaunch.read_state(tmp_path / "missing.json") == {"enabled": False}

    path = tmp_path / "autolaunch.json"
    path.write_text("{nope", encoding="utf-8")

    assert autolaunch.read_state(path) == {"enabled": False}


def test_get_status_supported_enabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    path = tmp_path / "autolaunch.json"
    autolaunch.write_state(enabled=True, config_path=tmp_path / "config.yaml", path=path)

    status = autolaunch.get_status(path)

    assert status.supported is True
    assert status.enabled is True
    assert status.detail == "Enabled"


def test_set_enabled_registers_and_starts_monitor(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(
        autolaunch,
        "register_task",
        lambda _config_path: calls.append("register"),
    )
    monkeypatch.setattr(
        autolaunch,
        "start_monitor",
        lambda _config_path: calls.append("start"),
    )

    status = autolaunch.set_enabled(tmp_path / "config.yaml", True)

    assert status.enabled is True
    assert calls == ["register", "start"]
    assert autolaunch.read_state()["enabled"] is True


def test_set_enabled_unregisters(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(autolaunch, "unregister_task", lambda: calls.append("unregister"))

    status = autolaunch.set_enabled(tmp_path / "config.yaml", False)

    assert status.enabled is False
    assert calls == ["unregister"]
    assert autolaunch.read_state()["enabled"] is False


def test_monitor_launches_once_per_process_edge(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    autolaunch.write_state(enabled=True, config_path=tmp_path / "config.yaml")
    states = iter([False, True, True, False, True])
    launched: list[list[str]] = []

    autolaunch.monitor_league(
        config_path=tmp_path / "fallback.yaml",
        process_checker=lambda _name: next(states),
        launcher=launched.append,
        sleeper=lambda _seconds: None,
        max_iterations=5,
    )

    assert len(launched) == 2
    assert launched[0][-2:] == ["--config", str(tmp_path / "config.yaml")]


def test_monitor_exits_when_state_is_disabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    autolaunch.write_state(enabled=False, config_path=tmp_path / "config.yaml")

    called = False

    def process_checker(_name: str) -> bool:
        nonlocal called
        called = True
        return True

    result = autolaunch.monitor_league(
        config_path=tmp_path / "config.yaml",
        process_checker=process_checker,
        launcher=lambda _cmd: None,
        sleeper=lambda _seconds: None,
        max_iterations=1,
    )

    assert result == 0
    assert called is False


def test_set_enabled_unsupported_platform(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")

    status = autolaunch.set_enabled(tmp_path / "config.yaml", True)

    assert status.supported is False
    assert status.enabled is False
    assert status.detail == "Unavailable on this platform"


def test_is_process_running_uses_tasklist(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout='"League of Legends.exe","1234"')

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert autolaunch.is_process_running(autolaunch.LEAGUE_PROCESS_NAME) is True
    assert calls[0][:2] == ["tasklist", "/FI"]


def test_claim_main_instance_non_windows(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")

    assert autolaunch.claim_main_instance() is True
