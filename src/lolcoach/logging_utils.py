"""Thread-safe per-game JSONL logger.

Produces structured event logs under ``logging.directory``, one file per game.
On every ``IDLE -> STARTING`` transition the orchestration layer (Unit 10)
calls :meth:`JsonlLogger.rotate` with the new ``gameId``; the logger closes
any currently-open file and opens a fresh one named
``session-{iso8601}-game-{game_id}.jsonl``. Callers without a game id (e.g.,
the benchmark subcommand in Unit 6) pass ``None`` and get
``session-{iso8601}.jsonl``.

All writes serialize through a single lock so concurrent producers from the
capture, inference, and decision threads cannot interleave partial lines.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO


class JsonlLogger:
    """A JSONL event logger with explicit rotation semantics.

    The logger is owned by the main orchestration loop (Unit 10). It is safe
    to call :meth:`log_event` concurrently from any thread once :meth:`rotate`
    has opened a file.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)
        self._lock = threading.Lock()
        self._file: TextIO | None = None
        self._current_path: Path | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def rotate(self, game_id: str | None) -> Path:
        """Close the current file (if any) and open a new per-game file.

        Returns the path of the newly-opened file so callers can log it.
        Creates the parent directory on demand.
        """
        with self._lock:
            self._close_file_locked()
            self._directory.mkdir(parents=True, exist_ok=True)
            new_path = self._directory / _build_filename(game_id)
            self._file = new_path.open("w", encoding="utf-8")
            self._current_path = new_path
            self._closed = False
            return new_path

    def log_event(self, event_type: str, **fields: Any) -> None:
        """Append one JSONL record with ``t``, ``type``, and caller fields.

        Raises :class:`RuntimeError` if called before :meth:`rotate` or after
        :meth:`close` — both are programming errors the orchestration layer
        should not hit in practice.
        """
        record: dict[str, Any] = {"t": time.time(), "type": event_type}
        record.update(fields)
        line = json.dumps(record, separators=(",", ":"), default=str)

        with self._lock:
            if self._file is None or self._closed:
                raise RuntimeError(
                    "JsonlLogger.log_event called without an active file; "
                    "call rotate() first and do not log after close()"
                )
            self._file.write(line)
            self._file.write("\n")
            self._file.flush()

    def close(self) -> None:
        """Close the current file handle. Idempotent."""
        with self._lock:
            self._close_file_locked()
            self._closed = True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _close_file_locked(self) -> None:
        """Close the current file handle; must be called under ``self._lock``."""
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None


def _build_filename(game_id: str | None) -> str:
    """Build a session filename with a UTC ISO 8601 timestamp.

    The timestamp uses microsecond precision and a compact format so rapid
    successive rotations (e.g., in tests) produce distinct filenames.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    if game_id is None:
        return f"session-{stamp}.jsonl"
    return f"session-{stamp}-game-{game_id}.jsonl"
