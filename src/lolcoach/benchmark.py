"""Vision model latency benchmark subcommand.

Invoked as ``python -m lolcoach.benchmark --fixture path/to/frame.png``.
Sends N sequential POST requests to Ollama's ``/api/chat`` with the same
image + prompt, measures each call's wall-clock latency, and reports:

* ``count`` — number of successful samples (excludes timeouts)
* ``timeout_count`` — number of timeouts (separate from count)
* ``min``, ``max``, ``mean``, ``p50``, ``p95`` over the successful samples

Also writes a JSONL report to ``<log_dir>/benchmark-{iso8601}.jsonl`` — one
line per sample plus one final ``benchmark_summary`` line. This is the
Week 1 gate that tells the developer whether to keep the current vision
model, switch to something smaller, reduce capture cadence, or pivot to
text-only coaching. See the plan's §5 decision tree.
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import requests

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "gemma3:4b"
DEFAULT_N = 20
DEFAULT_PROMPT = (
    "You are a League of Legends jungle coach. Briefly describe what you see "
    "on this minimap image in one sentence."
)
DEFAULT_TIMEOUT_S = 30.0


# ---------------------------------------------------------------------------
# Pure stats helper
# ---------------------------------------------------------------------------
def compute_stats(samples: list[float]) -> dict[str, float]:
    """Return min/max/mean/p50/p95/count from a list of latency samples."""
    if not samples:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
    sorted_samples = sorted(samples)
    n = len(sorted_samples)

    def percentile(p: float) -> float:
        if n == 1:
            return sorted_samples[0]
        k = (n - 1) * p
        lo = int(k)
        hi = min(lo + 1, n - 1)
        frac = k - lo
        return sorted_samples[lo] + (sorted_samples[hi] - sorted_samples[lo]) * frac

    return {
        "count": n,
        "min": sorted_samples[0],
        "max": sorted_samples[-1],
        "mean": statistics.fmean(sorted_samples),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
    }


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    stats: dict[str, float]
    timeout_count: int = 0
    samples: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(
    *,
    ollama_url: str,
    model: str,
    fixture_path: Path,
    n: int,
    prompt: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    session: requests.Session | None = None,
) -> BenchmarkResult:
    """Run *n* sequential Ollama calls and return the latency result.

    Raises :class:`requests.ConnectionError` if the very first call fails
    with a connection error (useful for the main() error path).
    """
    image_bytes = fixture_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    owned_session = session is None
    session = session or requests.Session()

    samples: list[float] = []
    timeouts = 0

    payload_template = {
        "model": model,
        "stream": False,
        "keep_alive": -1,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
    }

    try:
        for i in range(n):
            start = time.perf_counter()
            try:
                resp = session.post(ollama_url, json=payload_template, timeout=timeout_s)
            except requests.Timeout:
                timeouts += 1
                continue
            except requests.ConnectionError:
                # If we never connect at all, bubble up so main() can exit.
                if i == 0 and not samples:
                    raise
                timeouts += 1
                continue

            elapsed = time.perf_counter() - start
            if resp.status_code != 200:
                # Treat as a failure (not a timeout) — exclude from stats.
                continue
            samples.append(elapsed)
    finally:
        if owned_session:
            session.close()

    return BenchmarkResult(
        stats=compute_stats(samples),
        timeout_count=timeouts,
        samples=samples,
    )


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lolcoach.benchmark",
        description="Benchmark vision LLM latency for lolcoach.",
    )
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("./logs"),
        help="Directory for the JSONL benchmark report",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="Per-call timeout in seconds",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if not args.fixture.exists():
        print(
            f"--fixture path not found: {args.fixture}",
            file=sys.stderr,
        )
        return 2

    args.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = args.log_dir / f"benchmark-{stamp}.jsonl"

    def _log(event_type: str, **fields: object) -> None:
        with output_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {"t": time.time(), "type": event_type, **fields},
                    separators=(",", ":"),
                    default=str,
                )
            )
            fh.write("\n")

    _log(
        "benchmark_start",
        model=args.model,
        fixture=str(args.fixture),
        n=args.n,
        ollama_url=args.ollama_url,
    )

    try:
        result = run_benchmark(
            ollama_url=args.ollama_url,
            model=args.model,
            fixture_path=args.fixture,
            n=args.n,
            prompt=args.prompt,
            timeout_s=args.timeout_s,
        )
    except requests.ConnectionError as exc:
        print(
            f"Ollama unreachable at {args.ollama_url}: {exc}\n"
            "Is Ollama running? Try: ollama serve",
            file=sys.stderr,
        )
        _log("benchmark_error", error=str(exc))
        return 1

    stats = result.stats
    _log(
        "benchmark_summary",
        count=stats["count"],
        timeout_count=result.timeout_count,
        min=stats["min"],
        max=stats["max"],
        mean=stats["mean"],
        p50=stats["p50"],
        p95=stats["p95"],
        samples=result.samples,
    )

    print(
        "Benchmark summary — count={count} timeouts={timeouts} "
        "min={min:.2f}s mean={mean:.2f}s p50={p50:.2f}s p95={p95:.2f}s max={max:.2f}s".format(
            count=int(stats["count"]),
            timeouts=result.timeout_count,
            min=stats["min"],
            mean=stats["mean"],
            p50=stats["p50"],
            p95=stats["p95"],
            max=stats["max"],
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
