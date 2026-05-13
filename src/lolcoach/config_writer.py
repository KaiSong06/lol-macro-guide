"""Read and write the user-facing YAML config file.

The runtime loader in :mod:`lolcoach.config` is intentionally permissive: it
drops unknown keys and returns frozen dataclasses. The desktop UI needs the
opposite half of that story: update the known settings while preserving any
unknown sections or comments-adjacent user data as much as a standard YAML
round trip allows.
"""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from lolcoach.config import Config, ConfigError, load_config


def load_config_for_edit(path: Path) -> tuple[Config, dict[str, Any]]:
    """Return the typed config plus the raw YAML mapping for preservation."""

    config = load_config(path)
    if not path.exists():
        return config, {}

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse config YAML at {path}: {exc}") from exc
    if raw is None:
        return config, {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Config at {path} must be a YAML mapping at the top level, got "
            f"{type(raw).__name__}"
        )
    return config, raw


def save_config(path: Path, config: Config, *, existing_raw: dict[str, Any] | None = None) -> None:
    """Write *config* to *path*, preserving unknown top-level sections."""

    raw = dict(existing_raw or {})
    for section in fields(Config):
        value = getattr(config, section.name)
        raw[section.name] = _dataclass_to_yaml_dict(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _dataclass_to_yaml_dict(value: Any) -> Any:
    if not is_dataclass(value):
        return value
    result: dict[str, Any] = {}
    for key, item in asdict(value).items():
        if isinstance(item, tuple):
            item = list(item)
        result[key] = item
    return result
