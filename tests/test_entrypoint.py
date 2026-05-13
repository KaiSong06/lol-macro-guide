"""Tests for ``python -m lolcoach`` argument routing."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from lolcoach import __main__, autolaunch


def test_main_routes_monitor_league(monkeypatch, tmp_path: Path) -> None:
    calls: list[Path] = []

    def fake_monitor(*, config_path: Path) -> int:
        calls.append(config_path)
        return 7

    monkeypatch.setattr(autolaunch, "monitor_league", fake_monitor)

    result = __main__.main(["--monitor-league", "--config", str(tmp_path / "config.yaml")])

    assert result == 7
    assert calls == [tmp_path / "config.yaml"]


def test_main_routes_minimized_auto_start_to_ui(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    fake_ui = types.ModuleType("lolcoach.ui.app")

    def fake_run_ui(argv: list[str]) -> int:
        calls.append(argv)
        return 3

    fake_ui.run_ui = fake_run_ui  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lolcoach.ui.app", fake_ui)
    monkeypatch.setattr(autolaunch, "claim_main_instance", lambda: True)

    result = __main__.main(
        ["--minimized", "--auto-start", "--config", str(tmp_path / "config.yaml")]
    )

    assert result == 3
    assert calls == [
        ["--config", str(tmp_path / "config.yaml"), "--minimized", "--auto-start"]
    ]


def test_main_exits_when_main_instance_exists(monkeypatch) -> None:
    monkeypatch.setattr(autolaunch, "claim_main_instance", lambda: False)

    result = __main__.main([])

    assert result == 0
