"""Shared pytest fixtures for the lolcoach test suite."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Path:
    """A throwaway logs directory for JSONL tests."""
    log_dir = tmp_path / "logs"
    # Intentionally NOT created — tests exercise the auto-mkdir path.
    return log_dir


@pytest.fixture
def tmp_config_file(tmp_path: Path) -> Path:
    """A throwaway config.yaml path for config-loader tests (not yet written)."""
    return tmp_path / "config.yaml"
