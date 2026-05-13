"""Smoke tests for the optional PySide6 UI."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("PySide6") is None
    or importlib.util.find_spec("pytestqt") is None,
    reason="PySide6/pytest-qt is not installed in this environment",
)


def test_main_window_pages_render(qtbot: object, tmp_path: Path) -> None:
    from lolcoach.ui.app import MainWindow

    window = MainWindow(config_path=tmp_path / "missing.yaml")
    qtbot.addWidget(window)  # type: ignore[attr-defined]

    assert window.nav.count() == 5
    assert window.pages.count() == 5
    assert window.windowTitle() == "lolcoach"
    assert window.autolaunch_checkbox.text() == "Launch lolcoach when a match starts"
    assert window.autolaunch_status.text()
