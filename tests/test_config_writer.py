"""Tests for UI config read/write round-tripping."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

from lolcoach.config import load_config
from lolcoach.config_writer import load_config_for_edit, save_config


def test_save_config_preserves_unknown_top_level_sections(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
capture:
  interval_ms: 250

custom_plugin:
  enabled: true
""".strip(),
        encoding="utf-8",
    )

    config, raw = load_config_for_edit(path)
    updated = replace(
        config,
        inference=replace(config.inference, model="qwen2.5vl:7b"),
        tts=replace(config.tts, interrupt_categories=("gank_warning",)),
    )
    save_config(path, updated, existing_raw=raw)

    written = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert written["custom_plugin"] == {"enabled": True}
    assert written["inference"]["model"] == "qwen2.5vl:7b"
    assert written["tts"]["interrupt_categories"] == ["gank_warning"]
    assert load_config(path).inference.model == "qwen2.5vl:7b"


def test_load_config_for_edit_missing_file_returns_defaults_and_empty_raw(
    tmp_path: Path,
) -> None:
    config, raw = load_config_for_edit(tmp_path / "missing.yaml")

    assert config.inference.model == "gemma3:4b"
    assert raw == {}
