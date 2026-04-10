"""Tests for ``lolcoach.benchmark`` — vision LLM latency benchmark.

Scenarios from the implementation plan's Unit 6 test list:

1. Happy: mock Ollama returns N successful responses → stats computed
   correctly (min/max/mean/p50/p95/count).
2. Edge: some calls time out → timeout_count tracked, stats computed over
   successful samples only.
3. Error: Ollama unreachable → main() exits 1 with a helpful message.
4. Edge: n=1 single sample → stats still valid (min == max == mean).
5. Happy: JSONL output file exists and contains the summary line.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests
import requests_mock as rm_module

from lolcoach.benchmark import (
    BenchmarkResult,
    compute_stats,
    main,
    run_benchmark,
)

OLLAMA_URL = "http://localhost:11434/api/chat"


# ---------------------------------------------------------------------------
# Pure stats computation
# ---------------------------------------------------------------------------
def test_compute_stats_with_known_samples() -> None:
    samples = [2.0, 2.1, 2.5, 3.0, 3.1, 3.5, 4.0, 5.0, 6.0, 7.0]
    stats = compute_stats(samples)
    assert stats["count"] == 10
    assert stats["min"] == 2.0
    assert stats["max"] == 7.0
    assert stats["mean"] == pytest.approx(3.82, abs=0.01)
    assert stats["p50"] == pytest.approx(3.3, abs=0.01)
    assert stats["p95"] == pytest.approx(7.0, abs=0.5)


def test_compute_stats_with_single_sample() -> None:
    stats = compute_stats([4.2])
    assert stats["count"] == 1
    assert stats["min"] == 4.2
    assert stats["max"] == 4.2
    assert stats["mean"] == 4.2
    assert stats["p50"] == 4.2
    assert stats["p95"] == 4.2


def test_compute_stats_with_empty_samples() -> None:
    stats = compute_stats([])
    assert stats["count"] == 0
    assert stats["min"] == 0.0
    assert stats["max"] == 0.0
    assert stats["mean"] == 0.0


# ---------------------------------------------------------------------------
# run_benchmark: mocked Ollama end-to-end
# ---------------------------------------------------------------------------
def test_run_benchmark_happy_path(tmp_path: Path) -> None:
    fixture = tmp_path / "frame.png"
    # Any bytes work; the mocked Ollama doesn't care.
    fixture.write_bytes(b"\x89PNG\r\n\x1a\nfake-png-bytes")

    with rm_module.Mocker() as m:
        m.post(
            OLLAMA_URL,
            json={"message": {"content": "DECISION: test"}},
        )
        result = run_benchmark(
            ollama_url=OLLAMA_URL,
            model="fake-model",
            fixture_path=fixture,
            n=5,
            prompt="test prompt",
        )

    assert isinstance(result, BenchmarkResult)
    assert result.timeout_count == 0
    assert result.connection_error_count == 0
    assert result.http_error_count == 0
    assert result.stats["count"] == 5
    assert len(result.samples) == 5
    assert isinstance(result.samples, tuple)


def test_run_benchmark_with_timeouts(tmp_path: Path) -> None:
    fixture = tmp_path / "frame.png"
    fixture.write_bytes(b"pngbytes")

    call_counter = {"n": 0}

    def response_callback(request, context):
        call_counter["n"] += 1
        if call_counter["n"] in (2, 4):
            # Simulate a timeout by raising
            raise requests.Timeout("mocked timeout")
        context.status_code = 200
        return {"message": {"content": "ok"}}

    with rm_module.Mocker() as m:
        m.post(OLLAMA_URL, json=response_callback)
        result = run_benchmark(
            ollama_url=OLLAMA_URL,
            model="fake-model",
            fixture_path=fixture,
            n=5,
            prompt="test",
        )

    assert result.timeout_count == 2
    assert result.connection_error_count == 0
    assert result.http_error_count == 0
    assert result.stats["count"] == 3  # 5 - 2 = 3 successful samples


def test_run_benchmark_counts_http_errors_separately_from_timeouts(
    tmp_path: Path,
) -> None:
    """Non-200 responses are a distinct failure mode from timeouts. The
    benchmark must track them separately so the Week 1 decision tree
    can distinguish "Ollama is slow" (timeouts) from "Ollama returned
    HTTP 503" (server side errors).
    """
    fixture = tmp_path / "frame.png"
    fixture.write_bytes(b"pngbytes")

    call_counter = {"n": 0}

    def response_callback(request, context):
        call_counter["n"] += 1
        if call_counter["n"] in (2, 4):
            context.status_code = 503
            return {"error": "service unavailable"}
        context.status_code = 200
        return {"message": {"content": "ok"}}

    with rm_module.Mocker() as m:
        m.post(OLLAMA_URL, json=response_callback)
        result = run_benchmark(
            ollama_url=OLLAMA_URL,
            model="fake-model",
            fixture_path=fixture,
            n=5,
            prompt="test",
        )

    assert result.timeout_count == 0
    assert result.connection_error_count == 0
    assert result.http_error_count == 2
    assert result.stats["count"] == 3


def test_run_benchmark_counts_mid_run_connection_errors(tmp_path: Path) -> None:
    """If Ollama crashes mid-benchmark, subsequent calls raise
    ConnectionError. These must count as connection_error_count, NOT
    timeout_count — the old implementation lumped them together and
    hid the crash under "slow" in the Week 1 gate.
    """
    fixture = tmp_path / "frame.png"
    fixture.write_bytes(b"pngbytes")

    call_counter = {"n": 0}

    def response_callback(request, context):
        call_counter["n"] += 1
        if call_counter["n"] >= 3:
            raise requests.ConnectionError("crashed mid-run")
        context.status_code = 200
        return {"message": {"content": "ok"}}

    with rm_module.Mocker() as m:
        m.post(OLLAMA_URL, json=response_callback)
        result = run_benchmark(
            ollama_url=OLLAMA_URL,
            model="fake-model",
            fixture_path=fixture,
            n=5,
            prompt="test",
        )

    # First 2 calls succeed, then 3 ConnectionErrors
    assert result.stats["count"] == 2
    assert result.timeout_count == 0
    assert result.connection_error_count == 3
    assert result.http_error_count == 0


def test_run_benchmark_raises_connection_error(tmp_path: Path) -> None:
    fixture = tmp_path / "frame.png"
    fixture.write_bytes(b"pngbytes")

    with rm_module.Mocker() as m:
        m.post(OLLAMA_URL, exc=requests.ConnectionError("refused"))
        with pytest.raises(requests.ConnectionError):
            run_benchmark(
                ollama_url=OLLAMA_URL,
                model="fake-model",
                fixture_path=fixture,
                n=1,
                prompt="test",
            )


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------
def test_main_happy_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fixture = tmp_path / "frame.png"
    fixture.write_bytes(b"pngbytes")
    log_dir = tmp_path / "logs"

    with rm_module.Mocker() as m:
        m.post(OLLAMA_URL, json={"message": {"content": "ok"}})
        exit_code = main(
            [
                "--fixture",
                str(fixture),
                "--model",
                "fake-model",
                "--n",
                "3",
                "--ollama-url",
                OLLAMA_URL,
                "--log-dir",
                str(log_dir),
            ]
        )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Benchmark summary" in captured.out
    assert "count=3" in captured.out
    assert "conn_errors=0" in captured.out
    assert "http_errors=0" in captured.out
    # The JSONL path is echoed to stderr so agents don't have to glob.
    assert "Wrote benchmark log:" in captured.err

    # A JSONL file was written under the log dir.
    jsonl_files = list(log_dir.glob("benchmark-*.jsonl"))
    assert len(jsonl_files) == 1
    # The file contains at least a summary line.
    lines = jsonl_files[0].read_text().strip().splitlines()
    assert len(lines) >= 1
    summary = json.loads(lines[-1])
    assert summary["connection_error_count"] == 0
    assert summary["http_error_count"] == 0
    assert summary["type"] == "benchmark_summary"
    assert summary["count"] == 3


def test_main_exits_nonzero_on_ollama_unreachable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = tmp_path / "frame.png"
    fixture.write_bytes(b"pngbytes")
    log_dir = tmp_path / "logs"

    with rm_module.Mocker() as m:
        m.post(OLLAMA_URL, exc=requests.ConnectionError("refused"))
        exit_code = main(
            [
                "--fixture",
                str(fixture),
                "--model",
                "fake-model",
                "--n",
                "1",
                "--ollama-url",
                OLLAMA_URL,
                "--log-dir",
                str(log_dir),
            ]
        )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Ollama" in captured.err or "ollama" in captured.err


def test_main_missing_fixture_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "does_not_exist.png"
    log_dir = tmp_path / "logs"

    exit_code = main(
        [
            "--fixture",
            str(missing),
            "--model",
            "fake-model",
            "--n",
            "1",
            "--log-dir",
            str(log_dir),
        ]
    )
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "does_not_exist" in captured.err or "not found" in captured.err.lower()
