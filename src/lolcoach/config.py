"""Typed configuration loader for lolcoach.

The project uses a YAML file (``config.yaml`` next to the executable) for
user-tunable settings. Every field has a default mirroring
``config.yaml.example`` at the repo root, so the coach runs without any file
present — it simply logs a warning and falls back to defaults.

Unknown top-level sections and unknown nested keys are logged as warnings and
silently dropped. A malformed YAML file (or a document whose root is not a
mapping) raises :class:`ConfigError` with the file path embedded in the
message, so surface errors point at the file the user needs to fix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when a config file exists but cannot be parsed or is ill-shaped."""


# ---------------------------------------------------------------------------
# Nested sub-configs. Defaults mirror ``config.yaml.example``.
# All config dataclasses are frozen — build once, read many times.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CaptureConfig:
    interval_ms: int = 500
    #: Either the sentinel ``"auto"`` (use calibration.json) or a
    #: ``(x, y, w, h)`` tuple. Tuples are immutable so they're safe to
    #: carry on a frozen dataclass. YAML lists are coerced to tuples
    #: during config loading.
    minimap_roi: str | tuple[int, int, int, int] = "auto"


@dataclass(frozen=True)
class RiotApiConfig:
    poll_interval_ms: int = 2000
    base_url: str = "https://127.0.0.1:2999"


@dataclass(frozen=True)
class InferenceConfig:
    model: str = "gemma3:4b"
    timeout_ms: int = 10000
    ollama_host: str = "http://localhost:11434"
    keep_alive: int = -1  # pins the model in VRAM; see spec §12


@dataclass(frozen=True)
class DecisionsConfig:
    cooldown_seconds: int = 5
    confidence_threshold: int = 6
    dedup_window_seconds: int = 30
    staleness_threshold_seconds: int = 10


@dataclass(frozen=True)
class TtsConfig:
    voice: str = "en_US-amy-medium"
    speed: float = 1.0
    output_device: str = "system_default"
    interrupt_confidence_delta: int = 2
    #: Immutable tuple — ``frozen=True`` on the dataclass doesn't protect
    #: a list field from in-place mutation, so we use a tuple instead.
    #: YAML lists are coerced to tuples during config loading.
    interrupt_categories: tuple[str, ...] = ("gank_warning", "counter_gank")


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    directory: str = "./logs"


@dataclass(frozen=True)
class Config:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    riot_api: RiotApiConfig = field(default_factory=RiotApiConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    decisions: DecisionsConfig = field(default_factory=DecisionsConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


_KNOWN_SECTIONS: dict[str, type] = {
    "capture": CaptureConfig,
    "riot_api": RiotApiConfig,
    "inference": InferenceConfig,
    "decisions": DecisionsConfig,
    "tts": TtsConfig,
    "logging": LoggingConfig,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def load_config(path: Path | None = None) -> Config:
    """Load the lolcoach configuration from *path*, returning defaults on miss.

    Behavior:

    * ``path`` is ``None`` or the file does not exist → return a :class:`Config`
      populated entirely with documented defaults and log one warning.
    * The file exists and parses as a YAML mapping → overlay its known keys
      onto the defaults. Unknown keys are logged as warnings and ignored.
    * The file exists but cannot be parsed as YAML → :class:`ConfigError`.
    * The file exists and parses but the root is not a mapping → :class:`ConfigError`.
    """
    if path is None or not path.exists():
        logger.warning("config file not found; using defaults (path=%s)", path)
        return Config()

    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Failed to parse config YAML at {path}: {exc}"
        ) from exc

    if raw is None:
        # Empty file — treat like missing: return defaults, no warning.
        return Config()

    if not isinstance(raw, dict):
        raise ConfigError(
            f"Config at {path} must be a YAML mapping at the top level, got "
            f"{type(raw).__name__}"
        )

    return _build_config(raw, path)


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------
def _build_config(raw: dict[str, Any], path: Path) -> Config:
    for key in raw:
        if key not in _KNOWN_SECTIONS:
            logger.warning(
                "config file %s: unknown top-level section %r, ignoring", path, key
            )

    sections: dict[str, Any] = {}
    for section_name, section_class in _KNOWN_SECTIONS.items():
        if section_name not in raw:
            continue
        sub = raw[section_name]
        if not isinstance(sub, dict):
            logger.warning(
                "config file %s: section %r must be a mapping, ignoring (got %s)",
                path,
                section_name,
                type(sub).__name__,
            )
            continue
        sections[section_name] = _build_section(section_class, sub, path, section_name)
    return Config(**sections)


#: Field names that must be coerced from YAML list -> tuple before being
#: passed to the frozen dataclass constructor. YAML has no tuple primitive,
#: so users write lists; we enforce immutability at the dataclass boundary.
_LIST_TO_TUPLE_FIELDS: frozenset[tuple[str, str]] = frozenset(
    {
        ("capture", "minimap_roi"),
        ("tts", "interrupt_categories"),
    }
)


def _build_section(
    klass: type,
    sub: dict[str, Any],
    path: Path,
    section_name: str,
) -> Any:
    known_field_names = {f.name for f in fields(klass)}
    kwargs: dict[str, Any] = {}
    for key, value in sub.items():
        if key not in known_field_names:
            logger.warning(
                "config file %s: unknown key %r in section %r, ignoring",
                path,
                key,
                section_name,
            )
            continue
        if (section_name, key) in _LIST_TO_TUPLE_FIELDS and isinstance(value, list):
            value = tuple(value)
        kwargs[key] = value
    return klass(**kwargs)
