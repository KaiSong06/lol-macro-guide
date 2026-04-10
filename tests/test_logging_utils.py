"""Tests for ``lolcoach.logging_utils`` — thread-safe per-game JSONL logger.

Scenarios mapped from the implementation plan's Unit 1 test list:

1. ``log_event`` writes one valid JSONL line with ``t`` and ``type`` fields.
2. ``rotate("ABC123")`` opens a file named ``session-{iso}-game-ABC123.jsonl``.
3. ``rotate(None)`` opens a benchmark-style file without a game-id suffix.
4. Missing log directory is created on ``rotate``.
5. Concurrent ``log_event`` calls from many threads produce valid JSONL.
6. ``close()`` on an already-closed logger is a no-op.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from lolcoach.logging_utils import JsonlLogger


# ---------------------------------------------------------------------------
# Happy path: basic log_event semantics
# ---------------------------------------------------------------------------
def test_log_event_writes_one_jsonl_line_with_timestamp_and_type(
    tmp_log_dir: Path,
) -> None:
    logger = JsonlLogger(tmp_log_dir)
    path = logger.rotate(game_id=None)

    logger.log_event("capture", latency_ms=48)
    logger.close()

    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["type"] == "capture"
    assert record["latency_ms"] == 48
    assert isinstance(record["t"], float)
    assert record["t"] > 0


def test_log_event_multiple_records_go_to_same_file(tmp_log_dir: Path) -> None:
    logger = JsonlLogger(tmp_log_dir)
    path = logger.rotate(game_id="GAME1")

    logger.log_event("capture", latency_ms=50)
    logger.log_event("inference", latency_ms=3100, decision="Recall")
    logger.log_event("decision", action="spoken", text="Recall")

    logger.close()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["type"] == "capture"
    assert json.loads(lines[1])["type"] == "inference"
    assert json.loads(lines[2])["type"] == "decision"


# ---------------------------------------------------------------------------
# Rotation: per-game filenames
# ---------------------------------------------------------------------------
def test_rotate_with_game_id_opens_per_game_file(tmp_log_dir: Path) -> None:
    logger = JsonlLogger(tmp_log_dir)
    path = logger.rotate(game_id="ABC123")

    assert path.parent == tmp_log_dir
    assert path.name.startswith("session-")
    assert path.name.endswith("-game-ABC123.jsonl")
    logger.close()


def test_rotate_with_none_game_id_opens_benchmark_file(tmp_log_dir: Path) -> None:
    logger = JsonlLogger(tmp_log_dir)
    path = logger.rotate(game_id=None)

    assert path.parent == tmp_log_dir
    assert path.name.startswith("session-")
    assert path.name.endswith(".jsonl")
    assert "game-" not in path.name
    logger.close()


def test_rotate_closes_previous_file_and_opens_new_one(tmp_log_dir: Path) -> None:
    logger = JsonlLogger(tmp_log_dir)

    path1 = logger.rotate(game_id="GAME1")
    logger.log_event("a")
    # Small sleep so the ISO8601 timestamp in the filename rolls forward.
    time.sleep(0.01)
    path2 = logger.rotate(game_id="GAME2")
    logger.log_event("b")
    logger.close()

    assert path1 != path2
    assert path1.read_text(encoding="utf-8").strip() != ""
    assert "GAME1" in path1.name
    assert "GAME2" in path2.name
    # The first file has exactly one record; the second also has exactly one.
    assert len(path1.read_text(encoding="utf-8").strip().splitlines()) == 1
    assert len(path2.read_text(encoding="utf-8").strip().splitlines()) == 1


# ---------------------------------------------------------------------------
# Edge case: log directory does not exist → created
# ---------------------------------------------------------------------------
def test_rotate_creates_missing_log_directory(tmp_path: Path) -> None:
    log_dir = tmp_path / "deeply" / "nested" / "logs"
    assert not log_dir.exists()

    logger = JsonlLogger(log_dir)
    logger.rotate(game_id="G")
    logger.close()

    assert log_dir.exists()
    assert log_dir.is_dir()


# ---------------------------------------------------------------------------
# Thread safety: many writers → all lines valid JSON, no interleaving
# ---------------------------------------------------------------------------
def test_concurrent_log_event_produces_valid_jsonl(tmp_log_dir: Path) -> None:
    logger = JsonlLogger(tmp_log_dir)
    path = logger.rotate(game_id="CONCURRENT")

    num_threads = 10
    events_per_thread = 100
    barrier = threading.Barrier(num_threads)

    def worker(worker_id: int) -> None:
        barrier.wait()
        for i in range(events_per_thread):
            logger.log_event("worker_event", worker=worker_id, seq=i)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    logger.close()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == num_threads * events_per_thread, (
        f"expected {num_threads * events_per_thread} lines, got {len(lines)}"
    )
    # Every line must parse as JSON with the expected fields.
    for line in lines:
        record = json.loads(line)
        assert record["type"] == "worker_event"
        assert 0 <= record["worker"] < num_threads
        assert 0 <= record["seq"] < events_per_thread


# ---------------------------------------------------------------------------
# close() is idempotent
# ---------------------------------------------------------------------------
def test_close_is_idempotent(tmp_log_dir: Path) -> None:
    logger = JsonlLogger(tmp_log_dir)
    logger.rotate(game_id="G")
    logger.log_event("a")

    logger.close()
    # Calling close() a second time must not raise.
    logger.close()
    logger.close()


def test_log_event_before_rotate_raises(tmp_log_dir: Path) -> None:
    """Using the logger without rotating first is a programming error."""
    logger = JsonlLogger(tmp_log_dir)
    with pytest.raises(RuntimeError):
        logger.log_event("oops")
    logger.close()


def test_log_event_after_close_raises(tmp_log_dir: Path) -> None:
    logger = JsonlLogger(tmp_log_dir)
    logger.rotate(game_id=None)
    logger.close()
    with pytest.raises(RuntimeError):
        logger.log_event("oops")
