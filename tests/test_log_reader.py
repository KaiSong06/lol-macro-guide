"""Tests for JSONL log browsing helpers."""

from __future__ import annotations

import json
from pathlib import Path

from lolcoach.log_reader import filter_records, list_log_files, read_log_file


def test_list_log_files_returns_newest_first(tmp_path: Path) -> None:
    older = tmp_path / "old.jsonl"
    newer = tmp_path / "new.jsonl"
    older.write_text("", encoding="utf-8")
    newer.write_text("", encoding="utf-8")

    files = list_log_files(tmp_path)

    assert files[0] == newer
    assert older in files


def test_read_log_file_skips_invalid_lines(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"t": 1.5, "type": "callout_spoken", "decision": "Ward"}),
                "not json",
                json.dumps(["not", "mapping"]),
            ]
        ),
        encoding="utf-8",
    )

    records = read_log_file(path)

    assert len(records) == 1
    assert records[0].event_type == "callout_spoken"
    assert records[0].timestamp == 1.5


def test_filter_records_by_type_and_query(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"type": "callout_spoken", "decision": "Ward bot"}),
                json.dumps({"type": "decision_filter", "reason": "low_confidence"}),
            ]
        ),
        encoding="utf-8",
    )

    records = filter_records(
        read_log_file(path),
        event_types={"callout_spoken"},
        query="bot",
    )

    assert len(records) == 1
    assert records[0].fields["decision"] == "Ward bot"
