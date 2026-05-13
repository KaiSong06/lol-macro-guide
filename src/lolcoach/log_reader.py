"""JSONL log discovery and filtering for the desktop UI."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LogRecord:
    """One parsed JSONL record."""

    path: Path
    line_number: int
    event_type: str
    timestamp: float | None
    fields: dict[str, Any]


def list_log_files(directory: Path) -> tuple[Path, ...]:
    if not directory.exists():
        return ()
    return tuple(sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True))


def read_log_file(path: Path) -> tuple[LogRecord, ...]:
    records: list[LogRecord] = []
    if not path.exists():
        return ()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        event_type = str(payload.get("type", "unknown"))
        timestamp_raw = payload.get("t")
        timestamp = float(timestamp_raw) if isinstance(timestamp_raw, int | float) else None
        records.append(
            LogRecord(
                path=path,
                line_number=line_number,
                event_type=event_type,
                timestamp=timestamp,
                fields=payload,
            )
        )
    return tuple(records)


def filter_records(
    records: Iterable[LogRecord],
    *,
    event_types: set[str] | None = None,
    query: str = "",
) -> tuple[LogRecord, ...]:
    needle = query.lower().strip()
    filtered: list[LogRecord] = []
    for record in records:
        if event_types and record.event_type not in event_types:
            continue
        if needle and needle not in json.dumps(record.fields, default=str).lower():
            continue
        filtered.append(record)
    return tuple(filtered)
