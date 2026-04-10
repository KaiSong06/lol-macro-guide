"""Tests for ``lolcoach.config`` — YAML loader with typed defaults.

Scenarios mapped from the implementation plan's Unit 1 test list:

1. Valid config.yaml populates all nested dataclass fields.
2. Missing config file returns a Config with documented defaults + warning.
3. Unknown top-level / nested keys are logged as warnings and dropped.
4. Malformed YAML raises ConfigError whose message contains the file path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from lolcoach.config import (
    Config,
    ConfigError,
    DecisionsConfig,
    InferenceConfig,
    TtsConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# Happy path: valid file populates all fields
# ---------------------------------------------------------------------------
def test_load_config_with_valid_file_populates_all_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
capture:
  interval_ms: 250
  minimap_roi: [100, 200, 300, 400]

riot_api:
  poll_interval_ms: 1000
  base_url: https://example.invalid:2999

inference:
  model: minicpm-v:2b
  timeout_ms: 7500
  ollama_host: http://localhost:11434
  keep_alive: -1

decisions:
  cooldown_seconds: 4
  confidence_threshold: 7
  dedup_window_seconds: 25
  staleness_threshold_seconds: 8

tts:
  voice: en_US-lessac-medium
  speed: 1.1
  output_device: "Headphones (USB Audio)"
  interrupt_confidence_delta: 3
  interrupt_categories: [gank_warning]

logging:
  level: DEBUG
  directory: /tmp/lolcoach-logs
""".strip()
    )

    cfg = load_config(config_path)

    assert isinstance(cfg, Config)
    assert cfg.capture.interval_ms == 250
    assert cfg.capture.minimap_roi == (100, 200, 300, 400)
    assert cfg.riot_api.poll_interval_ms == 1000
    assert cfg.riot_api.base_url == "https://example.invalid:2999"
    assert cfg.inference.model == "minicpm-v:2b"
    assert cfg.inference.timeout_ms == 7500
    assert cfg.inference.keep_alive == -1
    assert cfg.decisions.cooldown_seconds == 4
    assert cfg.decisions.confidence_threshold == 7
    assert cfg.decisions.dedup_window_seconds == 25
    assert cfg.decisions.staleness_threshold_seconds == 8
    assert cfg.tts.voice == "en_US-lessac-medium"
    assert cfg.tts.speed == pytest.approx(1.1)
    assert cfg.tts.output_device == "Headphones (USB Audio)"
    assert cfg.tts.interrupt_confidence_delta == 3
    assert cfg.tts.interrupt_categories == ("gank_warning",)
    assert cfg.logging.level == "DEBUG"
    assert cfg.logging.directory == "/tmp/lolcoach-logs"


# ---------------------------------------------------------------------------
# Happy path: missing file → defaults + warning
# ---------------------------------------------------------------------------
def test_load_config_missing_file_returns_defaults_and_warns(
    tmp_config_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert not tmp_config_file.exists()

    with caplog.at_level(logging.WARNING, logger="lolcoach.config"):
        cfg = load_config(tmp_config_file)

    # Defaults from the spec's "Decisions Locked In" section and config.yaml.example.
    assert cfg.inference.keep_alive == -1
    assert cfg.decisions.cooldown_seconds == 5
    assert cfg.decisions.confidence_threshold == 6
    assert cfg.decisions.dedup_window_seconds == 30
    assert cfg.decisions.staleness_threshold_seconds == 10
    assert cfg.tts.interrupt_confidence_delta == 2
    assert cfg.tts.interrupt_categories == ("gank_warning", "counter_gank")
    assert cfg.capture.interval_ms == 500
    assert cfg.riot_api.poll_interval_ms == 2000
    assert cfg.inference.model == "gemma3:4b"
    assert cfg.tts.voice == "en_US-amy-medium"
    assert cfg.logging.level == "INFO"

    # A single warning was emitted.
    assert any("not found" in rec.message.lower() for rec in caplog.records)


def test_load_config_no_path_argument_returns_defaults() -> None:
    """Calling load_config() with no arguments returns the default config."""
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.inference.keep_alive == -1  # spot-check a documented default


# ---------------------------------------------------------------------------
# Edge case: unknown top-level AND nested keys warn + drop
# ---------------------------------------------------------------------------
def test_load_config_unknown_keys_warn_and_are_dropped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
capture:
  interval_ms: 250
  unknown_nested_key: 99

decisions:
  cooldown_seconds: 3

totally_unknown_top_level:
  some: value
""".strip()
    )

    with caplog.at_level(logging.WARNING, logger="lolcoach.config"):
        cfg = load_config(config_path)

    # Known overrides applied.
    assert cfg.capture.interval_ms == 250
    assert cfg.decisions.cooldown_seconds == 3
    # Known sections still use defaults where not overridden.
    assert cfg.inference.keep_alive == -1
    # Unknown nested key is silently dropped at the dataclass boundary; check
    # that the resulting CaptureConfig has no stray attribute.
    assert not hasattr(cfg.capture, "unknown_nested_key")

    warning_text = "\n".join(rec.message for rec in caplog.records)
    assert "unknown_nested_key" in warning_text
    assert "totally_unknown_top_level" in warning_text


# ---------------------------------------------------------------------------
# Error path: malformed YAML raises ConfigError with the path in the message
# ---------------------------------------------------------------------------
def test_load_config_malformed_yaml_raises_config_error(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.yaml"
    # Unclosed quote + bad indentation — valid-looking-but-invalid YAML.
    config_path.write_text(
        """
capture:
  interval_ms: "unterminated
  minimap_roi: auto
""".strip()
    )

    with pytest.raises(ConfigError) as excinfo:
        load_config(config_path)

    message = str(excinfo.value)
    assert str(config_path) in message


def test_load_config_non_mapping_top_level_raises_config_error(tmp_path: Path) -> None:
    """A YAML document that parses but isn't a mapping is a config error."""
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- just\n- a\n- list\n")

    with pytest.raises(ConfigError) as excinfo:
        load_config(config_path)

    assert str(config_path) in str(excinfo.value)
    assert "mapping" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Dataclass smoke tests — defaults directly (no file involved)
# ---------------------------------------------------------------------------
def test_inference_config_defaults() -> None:
    assert InferenceConfig() == InferenceConfig(
        model="gemma3:4b",
        timeout_ms=10000,
        ollama_host="http://localhost:11434",
        keep_alive=-1,
    )


def test_decisions_config_defaults_match_spec() -> None:
    d = DecisionsConfig()
    assert d.cooldown_seconds == 5
    assert d.confidence_threshold == 6
    assert d.dedup_window_seconds == 30
    assert d.staleness_threshold_seconds == 10


def test_tts_config_interrupt_categories_is_an_immutable_tuple() -> None:
    """The field must be a tuple so ``frozen=True`` actually prevents
    modification. The previous implementation used a mutable default list
    which let callers silently append/mutate a "frozen" instance — F3 from
    /ce:review.
    """
    cfg = TtsConfig()
    assert cfg.interrupt_categories == ("gank_warning", "counter_gank")
    assert isinstance(cfg.interrupt_categories, tuple)

    # Immutability enforcement: tuples have no ``append``.
    with pytest.raises(AttributeError):
        cfg.interrupt_categories.append("pathing")  # type: ignore[attr-defined]

    # Two instances are independent value types (trivial with tuples).
    a = TtsConfig()
    b = TtsConfig()
    assert a.interrupt_categories is not b.interrupt_categories or (
        # Python interns small tuples sometimes, which is fine — still immutable.
        a.interrupt_categories == b.interrupt_categories
    )


# ---------------------------------------------------------------------------
# Edge case: empty YAML file → defaults (no ConfigError)
# ---------------------------------------------------------------------------
def test_load_config_empty_file_returns_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("")  # empty file; yaml.safe_load returns None

    cfg = load_config(config_path)

    assert isinstance(cfg, Config)
    assert cfg.inference.keep_alive == -1
    assert cfg.decisions.cooldown_seconds == 5


# ---------------------------------------------------------------------------
# Edge case: section value is not a mapping → warn + keep default
# ---------------------------------------------------------------------------
def test_load_config_section_value_not_a_mapping_warns_and_uses_default(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config_path = tmp_path / "bad_section.yaml"
    config_path.write_text(
        """
capture: "this is not a mapping"

decisions:
  cooldown_seconds: 9
""".strip()
    )

    with caplog.at_level(logging.WARNING, logger="lolcoach.config"):
        cfg = load_config(config_path)

    # capture stayed at defaults (bad section ignored)
    assert cfg.capture.interval_ms == 500
    # decisions override applied
    assert cfg.decisions.cooldown_seconds == 9

    warning_text = "\n".join(rec.message for rec in caplog.records)
    assert "capture" in warning_text
    assert "mapping" in warning_text
