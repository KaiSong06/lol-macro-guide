# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

`lolcoach` — a real-time, local, vision-grounded jungle coach for League of Legends. Windows-only, voice-only MVP. Single Python process, four threads. Pre-alpha, greenfield — no release cut yet. The source of truth for scope, requirements, and unit breakdown is `docs/plans/2026-04-10-001-feat-jungle-coach-mvp-plan.md`; post-MVP backlog is in `TODOS.md`.

## Commands

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run full test suite (includes coverage gate, fails below 90%)
pytest

# Run a single test file / test
pytest tests/test_riot_client.py
pytest tests/test_riot_client.py::test_name -xvs

# Skip the coverage gate while iterating (useful when only a subset is run)
pytest --no-cov tests/test_detector.py

# Lint
ruff check src tests
ruff check --fix src tests   # auto-fix

# Subcommands (modules are directly runnable)
python -m lolcoach                          # entry point stub until Unit 10 lands
python -m lolcoach.calibrate                # interactive minimap ROI picker (Windows)
python -m lolcoach.calibrate --corners X1 Y1 X2 Y2   # scripted (test) mode
python -m lolcoach.template_builder         # download champion portraits from Data Dragon
python -m lolcoach.benchmark --fixture path/to/frame.png   # Week-1 vision latency gate
```

Runtime prerequisites (Windows only): Python 3.11/3.12, Ollama with a vision model (tag TBD from benchmark), an active League client for the Riot Live Client Data API on `https://127.0.0.1:2999`.

## Architecture

The runtime is Approach B "Stateful Coach" from the design doc: four cooperating threads around a single thread-safe state store, with a deterministic CV detector feeding a spatial-reasoning LLM through a structural hallucination filter.

### Threads and data flow

1. **Capture thread** (`capture.py`) — `dxcam`-backed screen grabber. Crops the minimap ROI at `capture.interval_ms` and pushes frames onto a `Queue(maxsize=1)` with drain-then-put so only the newest frame survives. After `MAX_CONSECUTIVE_FAILURES` grabs fail it escalates to `CaptureExhausted`, which the orchestrator turns into a one-shot "Coach paused" TTS line. `dxcam` is imported lazily and the `Capture` class accepts an injected camera so tests run on macOS/Linux.

2. **Riot API thread** (`riot_client.py`) — polls `/liveclientdata/allgamedata` every `riot_api.poll_interval_ms`. Owns the game lifecycle FSM (`IDLE → STARTING → ACTIVE → ENDING`) and fires typed callbacks defined in `Callbacks` (the orchestrator never polls the client). The Live Client Data API does **not** expose `gameData.gameId`, so game identity is derived from `gameTime` — a 200 response whose `gameTime` has jumped backward by more than `GAMETIME_RESET_DELTA_S` triggers a new-game cycle. `base_url` is validated against loopback-only hosts so `verify=False` is scoped to localhost. `urllib3` `InsecureRequestWarning` is silenced once at import.

3. **Inference thread** (orchestrator-owned, Unit 7, not yet in repo) — builds a prompt from the latest `StateManager.snapshot()` plus the latest minimap frame and calls Ollama `/api/chat` with `images: [base64]` and `keep_alive: -1` to pin the model in VRAM. Single worker by design: if a new frame arrives mid-inference, the old one is dropped (stale-frame policy). Cold start for enemy jungler fields returns `None`; the prompt builder **omits** the enemy-jungler line entirely rather than guessing.

4. **TTS thread** (Unit 9, not yet in repo) — Piper subprocess → `sounddevice.OutputStream`. Tracks `currently_playing`; priority interrupts only fire for `tts.interrupt_categories` (default `gank_warning`, `counter_gank`) with `new.confidence ≥ current.confidence + tts.interrupt_confidence_delta`.

### Detection-vs-reasoning split (critical invariant)

This is the single most important architectural constraint and it is enforced structurally, not statistically:

- `detector.py` uses `cv2.matchTemplate` (`TM_CCOEFF_NORMED`) against Riot Data Dragon champion portraits. It is the **primary** enemy-position source — Riot's API gives no enemy coordinates. Templates are filtered by the current Riot roster before `matchTemplate` runs, which is the 16× speedup baked into `d22f202`. Per-champion NMS deduplicates overlapping hits.
- The vision LLM does **spatial reasoning only** — never champion identification.
- `DecisionFilter` (Unit 8, not yet in repo) rejects any callout whose DECISION or REASON text names a champion that is neither in the Riot roster nor was ever seen by the detector. This makes "coach yells about a champion not in the game" architecturally impossible. **Do not add statistical fuzziness to this check.** R6 is a binary guarantee, not a tuned threshold.

### State

`state.py` — `StateManager` aggregates Riot API fields and detector hits behind a single `threading.RLock` (not `Lock` — a writer method can call helper methods without deadlocking itself). Reads go through `snapshot()` which deep-copies under the lock and returns an immutable `GameState` dataclass; the inference worker can then run a multi-second Ollama call on its snapshot without blocking any other thread. The kill feed is a bounded `deque(maxlen=KILL_FEED_MAX)` so a 30-minute game can't grow without bound. `StateManager.reset()` on every `ENDING` transition is load-bearing for the multi-game requirement (R4).

### Config

`config.py` — YAML-backed config with every field defaulted to mirror `config.yaml.example`. All config dataclasses are `frozen=True`. Unknown top-level sections and unknown nested keys are logged and dropped (warning-level). Malformed YAML raises `ConfigError` with the file path embedded so surface errors point at the file to fix. YAML lists (e.g., `minimap_roi: [x, y, w, h]`) are coerced to tuples so frozen dataclass fields stay hashable.

### Jungle heuristics

`jungle_routes.py` — hardcoded jungle paths + `predict_quadrant(last_seen, elapsed_seconds)`. The result is passed to the LLM as a **soft hint** alongside the real minimap image, never as a fact. `StateManager` computes `enemy_jungler_predicted_quadrant` only on snapshot.

### Logging

`logging_utils.JsonlLogger` — one JSONL file per game, rotated on `IDLE → STARTING`. Single serialization lock so capture / inference / decision threads never interleave partial lines. Filenames: `session-{iso8601}-game-{gameId}.jsonl`, or `session-{iso8601}.jsonl` when no game id is available (e.g., the benchmark subcommand). No size-based rotation, no retention policy — the origin spec's "clip-hunting" success signal (R16) needs full history, so `benchmark.py` and `logging_utils.py` preserve the JSONL format.

### Benchmark gate

`benchmark.py` is the Week-1 latency gate described in plan §5. It is **not** a perf micro-benchmark — it reports `min/max/mean/p50/p95` across N sequential Ollama calls and writes a JSONL report. The plan's decision tree says: if measured latency > 12 s under GPU contention, pivot to text-only coaching; 5–8 s is the normal path. Keep the output schema stable — downstream analysis scripts consume it.

## Implementation status (as of this file)

Shipped so far: `capture`, `calibrate`, `detector`, `template_builder`, `riot_client`, `jungle_routes`, `state`, `config`, `logging_utils`, `benchmark`, `prompts`, `inference`, `filter`, `tts`, and main orchestration. Packaging scripts and release docs exist, but the Windows benchmark gate plus clean Windows 10/11 smoke tests still need to be run before cutting a release.

## Conventions that matter in this codebase

- **Frozen dataclasses everywhere** for value types (`GameState`, all `*Config` classes, `ROI`, `DetectionResult`, `JungleCamp`, `Callbacks`). Do not convert them to mutable classes to "simplify" something — the thread-safety guarantees depend on snapshotted value types crossing thread boundaries.
- **Test-first, 90% coverage gate is enforced by `pytest`** (`--cov-fail-under=90` in `pyproject.toml`). Before marking work complete run the full suite; don't just run the file you edited. Hardware-dependent paths (`dxcam.grab`, `cv2.imshow`/`setMouseCallback`, `sounddevice.OutputStream`) are legitimate `pragma: no cover` — the 10-point buffer exists for them, not for ordinary logic.
- **Lazy Windows-only imports.** `dxcam`, `sounddevice`, and similar Windows binaries must be imported inside `if sys.platform == "win32":` branches (with `pragma: no cover`), and the owning class must accept the dependency via constructor injection so tests can pass a fake. Follow the pattern in `capture.py`.
- **Cold-start returns `None`, never a guess.** If the detector has never seen the enemy jungler, `last_seen_position` / `predicted_position` stay `None` and the prompt builder drops the enemy-jungler line entirely. Do not synthesize a default starting position anywhere in the pipeline.
- **`RLock`, not `Lock`.** Reentrancy in `StateManager` is intentional.
- **Loopback-only `verify=False`.** Any code that disables TLS verification must first validate the URL is a loopback IP/host. See the `base_url` check in `riot_client.py` — reuse that helper if you add another local HTTP client.
- **Callback-driven orchestration.** `riot_client.Callbacks` fields are all required; don't add optional callbacks without a default no-op. The orchestrator wires every transition through callbacks — do not poll the client from outside.
- **Never bundle Riot champion art.** Templates are downloaded from Data Dragon by `template_builder.py` at first run; only a tiny hand-picked subset lives under `tests/fixtures/` for unit tests. If you need new visual fixtures, add them there — don't reach for the full roster.

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill tool as your FIRST action. Do NOT answer directly, do NOT use other tools first. The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
- Save progress, checkpoint, resume → invoke checkpoint
- Code quality, health check → invoke health
