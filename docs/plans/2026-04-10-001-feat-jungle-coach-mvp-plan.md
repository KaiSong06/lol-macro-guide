---
title: "feat: LoL Jungle Coach MVP — real-time vision-grounded voice coach"
type: feat
status: active
date: 2026-04-10
origin: external — see Sources & References (gstack project storage, outside repo)
---

# feat: LoL Jungle Coach MVP — real-time vision-grounded voice coach

## Overview

Build a free, local, Windows-only Python desktop app that watches the League of Legends minimap in real time, reasons about it with a local vision LLM, and speaks one-sentence coaching callouts to the jungler. The core magic is **information recovery** — catching telegraphs the player missed (enemy recalls, ward drops, pathing intent) within a ~30 s macro window, not twitch-level predictions. Architecture is the approved "Approach B — Stateful Coach" from the design doc: four threads (capture / Riot API / inference / TTS), a thread-safe `StateManager`, a deterministic OpenCV template-match detector feeding an LLM that does **spatial reasoning only**, and a `DecisionFilter` that structurally prevents hallucinated champion callouts from ever reaching the audio output. MVP ships both a pip source install and a Windows one-click installer. The spec, design doc, and test plan are all locked in (`status: APPROVED`, `ENG CLEARED`) — this plan translates them into dependency-ordered implementation units with file paths and test scenarios.

## Problem Frame

League jungle is the most macro-dependent role: optimal pathing, gank timing, objective control, and enemy-jungler tracking compete with mechanical execution for cognitive bandwidth. Existing tools (Mobalytics, Blitz, Porofessor) provide overlays and post-game analysis; STATUP.GG does real-time voice coaching but is cloud-based, paid, and closed-source. There is no free, local, open-source alternative that feeds the actual minimap to a vision model and reasons about it during a live game. The differentiator is the **detection-vs-reasoning split**: OpenCV template matching finds champion icons (deterministic, no hallucination possible); the vision LLM reasons about what their positions mean. That split is architecturally novel and makes the system robust against the failure mode that would otherwise be fatal for voice output — "coach yells about a champion that isn't even in the game."

See origin document for the full problem framing, cross-model perspective, and why real-time live coaching was chosen over the cheaper replay-narrator alternative. The live prediction moment is the product.

## Requirements Trace

Tagged `R1`–`R16`. Used throughout implementation units.

- **R1.** Desktop Python app runs on Windows 10/11 alongside League (capture + TTS + Ollama HTTP client + single process, four threads).
- **R2.** Voice callout plays during a live game with game-state-aware content (inference → filter → TTS pipeline end-to-end).
- **R3.** First-run calibration succeeds, and a coherent callout plays within 3 minutes of the first game (critical path #1).
- **R4.** Multi-game session (3 games back-to-back) runs without state leakage, memory growth, or crashes — `StateManager.reset()` on every `ENDING` transition is load-bearing.
- **R5.** Capture → speech-start latency ≤ **8 s** typical under GPU contention with League running (macro windows are ~30 s; 5 s is aspirational).
- **R6.** **Zero hallucinated champion callouts** across a 10-game test session. Enforced structurally by `DecisionFilter` grounding check — no statistical tuning, binary guarantee.
- **R7.** Runs crash-free for a full 30-minute game (no unbounded collections, bounded `deque(maxlen=50)` kill feed, proper thread lifecycle).
- **R8.** Test coverage ≥ 90 %, enforced via `pytest --cov-fail-under=90` in the default test command.
- **R9.** Voice-only output for MVP (no overlay, no in-game UI).
- **R10.** Role verification: non-jungle role announces once ("You're not playing jungle. Coach paused.") and halts inference until next game.
- **R11.** Game lifecycle FSM handles `IDLE / STARTING / ACTIVE / ENDING`, including reconnect (same `gameId` within 10 s → stay `ACTIVE`) and remake (new `gameId` → standard `ACTIVE → ENDING → IDLE → STARTING` cycle).
- **R12.** Priority interrupt only fires for `gank_warning` / `counter_gank` with `new.confidence ≥ current.confidence + 2`. All other callouts during playback are dropped.
- **R13.** 5-second callout cooldown, 30-second `(category, target_lane)` dedup window, confidence ≥ 6 gate.
- **R14.** Ollama model pinned via `keep_alive: -1` on every call + `OLLAMA_KEEP_ALIVE=-1` env var documented, plus a warmup ping on coach startup — no first-callout 10–20 s cold-start penalty.
- **R15.** MVP ships **both** distribution paths: `pip install` from source AND Windows one-click installer via PyInstaller + NSIS/Inno.
- **R16.** ≥ 1 "information recovery" callout per play session (enemy recall, ward drop, pathing intent) — this is the clip-worthy success signal from the spec.

## Scope Boundaries

- **Windows 10/11 only.** `dxcam` is Windows-only. No macOS/Linux port.
- **Voice-only output.** No overlay UI, no HUD, no in-game notifications.
- **English only.** Single Piper voice bundled (`en_US-amy-medium`); other voices work via config but are not bundled.
- **Jungle role only.** Non-jungle players get a one-shot pause announcement, not alternate coaching.
- **No cloud fallback.** Everything runs locally — no remote LLM, no telemetry, no analytics.
- **No LLM or Ollama redistribution.** User installs Ollama separately and runs `ollama pull <model>`. The installer bundles only Python runtime + dxcam + OpenCV + Piper binary + default voice.
- **No process/memory injection.** External screen capture via `dxcam` only. Ship with a Riot ToS disclaimer.
- **No CI/CD for MVP.** Manual releases via `scripts/release.sh`; GitHub release is created by hand.
- **No code signing.** Unsigned `.exe`; README documents the "More info → Run anyway" workaround.

### Deferred to Separate Tasks

Tracked in `TODOS.md` at repo root:

- **Auto-detect minimap ROI across resolutions** — MVP uses manual click calibration. Auto-detect is a post-MVP UX improvement.
- **Approach C hybrid rule-based heuristics** — Rule layer for sub-second callouts (dragon timer, enemy jungler spotted) alongside the LLM. Natural Phase 2 once Approach B is validated on real games.
- **Code signing for the Windows installer** — EV cert ($200–400/yr + hardware token). Revisit only if the tool gets meaningful adoption from clip posts.

Additional deferred candidates (not yet in `TODOS.md`): replay-narrator mode, non-jungle role support, overlay UI, non-English voices, Ollama bundling, Linux/macOS port.

## Context & Research

### Relevant Code and Patterns

**This is a greenfield repo.** As of this plan, the repository contains only `CLAUDE.md` (skill routing), `LICENSE` (MIT), and `TODOS.md` (post-MVP backlog). There is no existing Python source, no `docs/` tree, no `AGENTS.md`, no `docs/solutions/`, no prior patterns to follow. Every file in the plan's **Files:** lists is a `Create:` unless explicitly noted.

Consequence: the usual "mirror this existing class" pattern-reuse guidance does not apply. Implementation units must follow **Python idiomatic defaults** (PEP 8, PEP 517/518 `pyproject.toml`, `src/` layout, dataclasses for value types, `threading.RLock` for state, `queue.Queue` for thread communication). The plan assumes the implementer knows these conventions; it does not re-teach them.

### Institutional Learnings

No `docs/solutions/` exists in this repo. No prior learnings file to reference. The design doc's "Cross-Model Perspective" section (lines 45–52 of `kaisong-main-design-20260410-011323.md`) captured the single institutional insight from the office-hours pass: *"This is not a coaching tool project. This is a content creation project. The real product is the clip, not the coaching."* This directly shapes success criterion R16 and justifies preserving the clip-hunting JSONL log format in Unit 1.

### External References

The implementer should read these once before starting each relevant unit. Listed by unit.

- **Unit 2 — Capture.** [`dxcam` README](https://github.com/ra1nty/DXcam) — API shape, capture semantics, known Windows 10 vs 11 quirks. OpenCV `cv2.imshow` + `setMouseCallback` for the interactive calibrate subcommand.
- **Unit 3 — Detector.** [OpenCV `cv2.matchTemplate` docs](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html) — method flags (`TM_CCOEFF_NORMED` is the usual pick for normalized scores). [Riot Data Dragon](https://developer.riotgames.com/docs/lol#data-dragon) — champion portrait URLs (`https://ddragon.leagueoflegends.com/cdn/<patch>/img/champion/<Name>.png`), patch versioning.
- **Unit 4 — Riot client.** [Riot Live Client Data API docs](https://developer.riotgames.com/docs/lol#game-client-api) — endpoints: `/liveclientdata/allgamedata`, `/liveclientdata/activeplayer`, `/liveclientdata/playerlist`. Self-signed localhost cert on `https://127.0.0.1:2999` requires `verify=False`. Response schemas for `allPlayers`, `events`, `activePlayer`.
- **Unit 7 — Inference.** [Ollama `/api/chat` docs](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion) — vision input via `images: [base64]` in message body. `keep_alive: -1` documented at the same URL. Candidate models: `gemma3:4b`, `minicpm-v`, `qwen2.5vl:7b` — verify exact tag names at implementation time (tags shift between Ollama releases).
- **Unit 9 — TTS.** [Piper TTS README](https://github.com/rhasspy/piper) — subprocess invocation (`piper --model <voice.onnx> < text.txt > output.wav`) vs Python binding stability on Windows. [`sounddevice` docs](https://python-sounddevice.readthedocs.io/) — `OutputStream` lifecycle, `.stop()` for interruption, device enumeration for `config.tts.output_device`.
- **Unit 11 — PyInstaller.** [PyInstaller one-file mode docs](https://pyinstaller.org/en/stable/usage.html#cmdoption-F) — `--onefile`, `--add-data`, handling binary dependencies (dxcam DLLs, OpenCV, Piper binary). Known issue: PyInstaller struggles with lazy imports in some OpenCV builds — use `--hidden-import` as needed.
- **Unit 12 — Installer.** [NSIS scripting reference](https://nsis.sourceforge.io/Docs/) OR [Inno Setup docs](https://jrsoftware.org/ishelp/) — choose whichever the implementer is faster with. No meaningful difference for this project.

No parallel Exa / Context7 / framework-docs-researcher dispatch — the design doc already went through cross-model review, and the references above are sufficient targeted entry points.

## Key Technical Decisions

Each decision is carried from the origin spec (§12 "Decisions Locked In") plus a few plan-level choices this document adds. Decisions that were the subject of extensive office-hours / eng-review debate are marked `(see origin)`.

- **Language and runtime: Python 3.11+.** *Why:* `threading.Lock` semantics, `dataclasses`, `queue.Queue(maxsize=1)` + drain-then-put pattern, and PyInstaller compatibility all work cleanly on 3.11. Stay under 3.12 until all deps (dxcam, opencv, piper) are confirmed to support it. *Impact:* pin minimum in `pyproject.toml`.
- **Package layout: `src/lolcoach/`.** *Why:* modern `src/` layout prevents accidental imports from cwd, plays well with pytest + coverage, keeps `tests/` clean. *Alternative considered and rejected:* flat `lolcoach/` at repo root (simpler but invites import-path bugs at release time).
- **Approach B (Stateful Coach) over A or C.** *(see origin §12.)* Approach A lacks state memory. Approach C is strictly better but 2–3 weeks to first demo vs 1–2 weeks for B. Approach C's rule-based layer is already captured in `TODOS.md` as the natural Phase 2.
- **Template matching is the PRIMARY enemy-position source.** *(see origin §12 and design doc §2a.)* Not a fallback. The Riot Live Client Data API does not expose enemy champion coordinates — OpenCV template matching is the **only** path to knowing where they are. This changes how Unit 3 is scoped: the detector is on the critical path, not a "nice to have".
- **Detection deterministic, reasoning flexible — enforced structurally.** *(see origin §12.)* The vision LLM does *not* identify champions; it reasons about spatial intent. The `DecisionFilter` rejects any callout whose DECISION or REASON text mentions a champion the detector never saw (and that isn't in the Riot roster). This makes "coach yells about a champion not in the game" architecturally impossible, not just unlikely. This is the single most important invariant in the plan. Unit 8 is where it lives.
- **8-second latency target, not 5.** *(see origin §12 and spec §5.)* Benchmark-realistic under GPU contention. Jungle macro windows are ~30 s; a 7 s round-trip is still useful ("Lee Sin path predicted, warn in 20 s"). Week 1 benchmark (Unit 6) is a gate: if measured latency exceeds 12 s, pivot to text-only coaching per the spec's decision tree.
- **Cold start returns `None`, not a guess.** *(see origin §12 and design doc §4.)* Until the first confirmed enemy jungler sighting, `last_seen_position` and `predicted_position` stay `None`. The prompt builder (Unit 7) **omits the "Enemy jungler" line entirely** when both are `None`. The vision model sees a clean "no information" signal rather than a hallucinated starting position. Other callouts (objectives, general pathing) still work in this window.
- **Priority interrupt only for gank/counter-gank + 2-point confidence delta.** *(see origin §12.)* Prevents chatter interruptions. The TTS module (Unit 9) tracks `currently_playing` and enforces this via two configurable knobs: `tts.interrupt_confidence_delta` and `tts.interrupt_categories`.
- **Champion templates downloaded on first run, not bundled.** *New plan-level decision.* Bundling Riot portrait assets in an open-source repo is a licensing gray area. Instead, `scripts/build-champion-templates.py` downloads from Riot Data Dragon on first run (idempotent, cached locally under `~/.lolcoach/templates/`). A **tiny subset** of hand-picked templates is checked into `tests/fixtures/champion_templates_subset/` for unit tests only. *Tradeoff accepted:* first-run UX has a one-time ~30 s download; mitigated by a clear progress message.
- **Lifecycle FSM lives inside `riot_client.py`, not a separate module.** *New plan-level decision.* The design doc says "the Riot API Client owns game lifecycle detection". Splitting the FSM into its own module adds an import layer without clarifying responsibilities. Keep FSM transitions as methods on `RiotClient` and test both in `tests/test_riot_client.py` (matches the test plan's 13-test count for that file).
- **JSONL logging writes a new file per game.** *(see origin spec §3 and design doc §3.)* On every `IDLE → STARTING` transition, `logging_utils.rotate_log(game_id)` closes the current file and opens `./logs/session-{iso8601}-game-{gameId}.jsonl`. No size-based rotation, no retention policy. Old files stay forever (small, user deletes manually).
- **Test coverage floor is 90 %, not 100 %.** *(see origin §7.)* Hardware-dependent paths (`dxcam` grab, `sounddevice.OutputStream`) can't be unit-tested without real hardware. The 10-point buffer absorbs these legitimate `pragma: no cover` exclusions.
- **Inference is single-worker.** *(see origin §3 concurrency.)* No inference queue — if a new frame arrives while inference is running, the new frame is dropped (stale frame policy). Simpler than a queue, and correct: there's no point running a second inference on stale data.
- **State Manager uses `RLock`, not `Lock`.** *New plan-level decision.* The design doc said `Lock`. Use `RLock` so a single writer method can call helper methods without deadlocking itself. Negligible perf difference; strictly safer. Unit 5 reflects this.

## Open Questions

### Resolved During Planning

- **Package name?** `lolcoach` (under `src/`). Matches the module entry point (`python -m lolcoach`) already specified in the test plan.
- **Where does `calibration.json` live?** Next to `config.yaml`, i.e., working-directory-relative for source installs and installer-relative for the Windows bundle. Persisted via `config.get_user_data_path()` in Unit 1.
- **Should lifecycle FSM be its own module?** No — keep it inside `riot_client.py`. Reason: design doc explicitly assigns ownership there; splitting adds import layers without clarifying responsibilities. See Key Technical Decisions.
- **Are champion templates redistributed with the repo?** No — downloaded on first run from Riot Data Dragon. A tiny test-only subset is checked in under `tests/fixtures/`. See Key Technical Decisions.
- **Execution posture across units?** Test-first. Origin spec requires ≥90 % coverage; each feature-bearing unit carries `Execution note: Test-first` and enumerated scenarios.
- **Log directory default?** `./logs/` relative to the working directory. Created by `logging_utils.py` if missing. Configurable via `logging.directory`.
- **Quadrant count?** 9 regions (top_lane, top_jungle, top_river, mid, bot_lane, bot_jungle, bot_river, base_blue, base_red). Derived from normalized `[0,1]²` minimap coordinates via fixed thresholds in Unit 3.

### Deferred to Implementation

- **Exact Ollama vision model tag.** The spec lists candidates (`gemma3:4b`, `gemma3:12b`, `minicpm-v`, `qwen2.5vl:7b`) and notes "verify exact tag at implementation time". Unit 6's benchmark subcommand and Unit 7's default config pick the final tag. Deferred because Ollama tags shift between releases — the right answer is "whatever's current on the Ollama model page the day you implement Week 1".
- **Template matching confidence threshold.** Needs real-frame calibration during Unit 3. Start at 0.85 (typical for `cv2.matchTemplate` with `TM_CCOEFF_NORMED`), tune against the fixture set.
- **Prompt phrasing.** The spec locks the response schema (`DECISION / CATEGORY / TARGET_LANE / CONFIDENCE / REASON`) but leaves the exact instruction wording to iterate. Week 2 is where prompts get tuned against recorded game frames.
- **Piper invocation: subprocess or Python binding?** Python binding is newer; subprocess is battle-tested. Test binding on Windows in Unit 9; fall back to subprocess if binding has lifecycle issues under PyInstaller's one-file mode.
- **`httpx` vs `requests` for Riot client.** Both work with `verify=False`. Default to `requests` (already widely known; PyInstaller bundles it without hidden-import quirks). If Unit 4 implementation reveals a concurrency pain point, revisit.
- **Exact PyInstaller hidden-import list.** Impossible to know in advance — emerges during Unit 11 smoke testing on a clean VM. The plan's verification step is "exe runs on clean VM"; the implementer iterates on `--hidden-import` flags until that holds.
- **Installer: NSIS or Inno Setup?** Functionally equivalent for this project. Unit 12 picks whichever the implementer is faster with.
- **Champion name grounding regex edge cases.** Names like `Kai'Sa`, `Vel'Koz`, `Jarvan IV`, `Dr. Mundo`, `Wukong` (in-game ID `MonkeyKing`) need verification against the actual Data Dragon list. Unit 8 test scenarios include these; exact regex emerges during implementation.

## Output Structure

The plan creates an entirely new directory structure. This tree is a scope declaration showing the expected output shape — the implementer may refine the layout during implementation if a better structure emerges. Per-unit `**Files:**` sections are authoritative for what each unit creates.

```
lol-macro-guide/
├── CLAUDE.md                          # (existing) skill routing
├── LICENSE                            # (existing) MIT
├── TODOS.md                           # (existing) post-MVP backlog
├── README.md                          # created Unit 1 (placeholder), expanded Unit 12
├── RELEASE.md                         # created Unit 12
├── pyproject.toml                     # created Unit 1
├── config.yaml.example                # created Unit 1
├── .gitignore                         # created Unit 1
├── docs/
│   └── plans/
│       └── 2026-04-10-001-feat-jungle-coach-mvp-plan.md   # this file
├── src/
│   └── lolcoach/
│       ├── __init__.py                # Unit 1
│       ├── __main__.py                # Unit 1 (stub) → Unit 10 (wired to main.run())
│       ├── config.py                  # Unit 1
│       ├── logging_utils.py           # Unit 1
│       ├── capture.py                 # Unit 2
│       ├── calibrate.py               # Unit 2 (subcommand: python -m lolcoach.calibrate)
│       ├── detector.py                # Unit 3
│       ├── riot_client.py             # Unit 4 (includes lifecycle FSM)
│       ├── jungle_routes.py           # Unit 5
│       ├── state.py                   # Unit 5
│       ├── benchmark.py               # Unit 6 (subcommand: python -m lolcoach.benchmark)
│       ├── prompts.py                 # Unit 7
│       ├── inference.py               # Unit 7
│       ├── filter.py                  # Unit 8
│       ├── tts.py                     # Unit 9
│       └── main.py                    # Unit 10
├── tests/
│   ├── conftest.py                    # Unit 1
│   ├── manual.md                      # Unit 10 (manual E2E checklist)
│   ├── fixtures/
│   │   ├── allgamedata_ingame.json    # Unit 4
│   │   ├── allgamedata_nogame.json    # Unit 4
│   │   ├── allgamedata_remake.json    # Unit 4
│   │   ├── minimap_sample.png         # Unit 2
│   │   ├── ollama_responses.py        # Unit 7
│   │   └── champion_templates_subset/ # Unit 3 (small bundled set for tests)
│   ├── test_config.py                 # Unit 1
│   ├── test_logging_utils.py          # Unit 1
│   ├── test_capture.py                # Unit 2
│   ├── test_detector.py               # Unit 3
│   ├── test_riot_client.py            # Unit 4 (13 tests: poll + lifecycle FSM + role detection)
│   ├── test_jungle_routes.py          # Unit 5
│   ├── test_state.py                  # Unit 5
│   ├── test_benchmark.py              # Unit 6
│   ├── test_inference.py              # Unit 7
│   ├── test_filter.py                 # Unit 8
│   ├── test_tts.py                    # Unit 9
│   └── test_main_integration.py       # Unit 10
└── scripts/
    ├── build-champion-templates.py    # Unit 3 (Data Dragon downloader)
    ├── build-exe.py                   # Unit 11 (PyInstaller driver)
    ├── build.spec                     # Unit 11 (PyInstaller spec file)
    ├── build-installer.nsi            # Unit 12 (or .iss for Inno Setup)
    ├── release.sh                     # Unit 12 (local release automation)
    └── binaries/
        ├── piper/                     # Unit 11 (Piper binary, gitignored or LFS)
        └── voices/                    # Unit 11 (en_US-amy-medium.onnx + .json)
```

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

The design doc and spec already cover the static component diagram and the lifecycle state machine. What a reviewer most benefits from at the plan level is seeing **one full inference cycle across threads** — where locks are held, where deep-copies happen, and where the grounding check gates audio output. This is the non-obvious part of Approach B that the per-unit prose cannot easily communicate on its own.

```mermaid
sequenceDiagram
    participant Cap as Capture Thread
    participant Det as Detector (in Capture)
    participant State as State Manager
    participant Riot as Riot API Thread
    participant Inf as Inference Worker
    participant Ollama
    participant Filt as Decision Filter
    participant TTS as TTS Thread

    loop every 500 ms
        Cap->>Cap: dxcam grab → crop minimap
        Cap->>Det: detect(frame, roster)
        Det-->>Cap: DetectionResult
        Cap->>State: update_from_detector(result) [lock held]
    end

    loop every 2000 ms
        Riot->>Riot: GET /allgamedata (verify=False)
        Riot->>State: update_from_riot(data) [lock held]
        Riot->>Riot: transition FSM if needed
    end

    Note over Inf: triggered on timer (idle → busy)
    Inf->>State: snapshot() [deep-copy under lock → release]
    State-->>Inf: GameState (independent copy)
    Inf->>Inf: build prompt (omit enemy_jungler line if None)
    Inf->>Ollama: POST /api/chat (image + text, keep_alive=-1)
    Note over Ollama: 5–7 s under GPU contention
    Ollama-->>Inf: DECISION / CATEGORY / TARGET_LANE / CONFIDENCE / REASON
    Inf->>Inf: parse; drop if malformed
    Inf->>Filt: accept(callout, snapshot, detections)

    Filt->>Filt: 1. staleness check
    Filt->>Filt: 2. confidence ≥ 6
    Filt->>Filt: 3. grounding — every champ name must be in detections OR roster
    Filt->>Filt: 4. cooldown (5 s since last spoken)
    Filt->>Filt: 5. dedup on (category, target_lane) within 30 s

    alt all checks pass
        Filt->>TTS: speak(callout)
        TTS->>TTS: interrupt rules (confidence delta + category)
        TTS->>TTS: Piper synth → sounddevice stream
        Note over TTS: ~500 ms synth + playback concurrent with next capture
    else any check fails
        Filt->>Filt: log decision event {action: dropped, reason: <which check>}
    end
```

**Critical invariants this diagram encodes:**

1. **The state lock is never held across the Ollama call.** The inference worker deep-copies state under the lock, releases, then runs inference. This is what keeps the 500 ms capture cadence from stalling on a 7 s LLM call.
2. **The `DecisionFilter` sits between inference and TTS.** Every LLM output flows through it. There is no bypass path. This is how R6 ("zero hallucinated champion callouts") is guaranteed *structurally* rather than statistically.
3. **The grounding check queries two sources.** The detector's last-seen champion list AND the Riot API's roster. A champion named in the LLM output must appear in at least one. Neither → reject.
4. **TTS runs concurrently with the next capture cycle.** Playback happens on the TTS thread's `sounddevice.OutputStream` while the Capture thread keeps grabbing frames. Audio latency is *additive* to the budget only for the first word, not for every frame.

## Implementation Units

12 units, grouped into 3 phases. Unit dependencies form a DAG — Units 2, 3, 4 can be implemented in parallel after Unit 1. Unit 5 depends on Unit 4 (state receives Riot data). Unit 6 is the Week 1 gate. Units 7, 8, 9 can be implemented in parallel after Phase 1 lands. Unit 10 wires everything and depends on all earlier units. Units 11 and 12 are sequential packaging work.

**Execution posture for all feature-bearing units:** test-first. Write the failing tests enumerated under `Test scenarios` first, then implement until they pass. Do not expand this into literal `RED/GREEN` substeps — the enumerated scenarios are the contract.

---

### Phase 1 — Foundation & Data Pipeline (Week 1)

- [ ] **Unit 1: Project scaffolding, config loader, JSONL logging**

**Goal:** Create the `src/` layout, tooling config, typed config loader, and thread-safe JSONL logger. All subsequent units depend on these shared primitives. No `lolcoach` module can be meaningfully written until Unit 1 lands.

**Requirements:** R4 (bounded state via logging), R8 (coverage enforcement), R14 (config knobs for `keep_alive`)

**Dependencies:** None

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md` (placeholder; expanded in Unit 12)
- Create: `config.yaml.example`
- Create: `src/lolcoach/__init__.py`
- Create: `src/lolcoach/__main__.py` (stub: prints "use `python -m lolcoach` after Unit 10")
- Create: `src/lolcoach/config.py`
- Create: `src/lolcoach/logging_utils.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Test: `tests/test_config.py`
- Test: `tests/test_logging_utils.py`

**Approach:**
- `pyproject.toml` uses PEP 517/518. Dev deps: `pytest`, `pytest-mock`, `pytest-asyncio`, `pytest-cov`, `requests-mock`, `ruff`. Runtime deps added incrementally per unit (start empty; each subsequent unit adds its own deps). `pytest.ini_options` sets `addopts = "--cov=src/lolcoach --cov-fail-under=90"` as the default so coverage is enforced automatically.
- `src/` layout prevents accidental cwd imports and plays well with coverage.
- `config.py`: loads `config.yaml` via PyYAML; merges with defaults via a `Config` dataclass tree (nested dataclasses: `CaptureConfig`, `RiotApiConfig`, `InferenceConfig`, `DecisionsConfig`, `TtsConfig`, `LoggingConfig`). Unknown keys logged as warnings and dropped. Missing file → use defaults and warn.
- `logging_utils.py`: `JsonlLogger` class with `rotate(game_id: str | None)` for opening a new `./logs/session-{iso8601}-game-{game_id}.jsonl` file, `log_event(event_type: str, **fields)` for appending one JSON object per line (auto-timestamp via `time.time()`), `close()` for flushing and closing. Thread-safe via an internal `threading.Lock` held across the open+write+flush cycle. Creates the log directory if missing.
- `config.yaml.example` mirrors the spec's example config verbatim (minus unnecessary comments), lives at repo root, and is the file users copy to `config.yaml` for customization.

**Execution note:** Test-first. This is shared infrastructure; bugs here cascade into every later unit.

**Patterns to follow:** None (greenfield). External references: PEP 517/518 `pyproject.toml`, standard `src/` layout.

**Test scenarios:**

`tests/test_config.py`:
- *Happy path:* loading a valid `config.yaml` returns a `Config` with all nested dataclass fields populated; typed accessors return expected values.
- *Happy path:* missing `config.yaml` returns a `Config` with default values; the default `inference.keep_alive == -1`, `decisions.cooldown_seconds == 5`, `decisions.confidence_threshold == 6`, `decisions.dedup_window_seconds == 30`, `tts.interrupt_confidence_delta == 2`, `tts.interrupt_categories == ["gank_warning", "counter_gank"]`.
- *Edge case:* `config.yaml` with extra/unknown top-level or nested keys → loaded config drops them, warning is emitted via logger (not via stdout).
- *Error path:* malformed YAML (broken indentation, unclosed quote) raises a `ConfigError` whose message includes the file path.

`tests/test_logging_utils.py`:
- *Happy path:* `log_event("capture", latency_ms=48)` writes exactly one JSONL line containing `{"t": <float>, "type": "capture", "latency_ms": 48}` with a realistic `t`.
- *Happy path:* `rotate("ABC123")` on a logger with an open file closes the old file and opens `./logs/session-*-game-ABC123.jsonl`; subsequent events go to the new file.
- *Edge case:* `rotate(None)` opens a benchmark/session file without a game-id suffix (for Unit 6).
- *Edge case:* log directory does not exist → `rotate()` creates it.
- *Integration / concurrency:* 10 threads each calling `log_event` 100 times → resulting file has 1000 valid JSONL lines, no interleaved partial writes (verified by parsing every line as JSON). Uses `threading.Barrier` to force contention.
- *Edge case:* `close()` on an already-closed logger is a no-op, not an error.

**Verification:** `pytest tests/test_config.py tests/test_logging_utils.py -v` passes; `pytest --cov` reports Unit 1's files at ≥ 95 % line coverage (shared infra must be tightly tested); `python -c "from lolcoach.config import load_config; print(load_config())"` prints a Config dataclass with defaults.

---

- [ ] **Unit 2: Capture module + calibration subcommand**

**Goal:** Implement `dxcam`-based screen capture with minimap ROI cropping and a first-run calibration flow that writes `calibration.json`. Feeds the detection/inference pipeline and satisfies the first-run UX from critical path #1.

**Requirements:** R1 (Windows capture), R3 (first-run calibration), R5 (capture stage of latency budget)

**Dependencies:** Unit 1

**Files:**
- Create: `src/lolcoach/capture.py`
- Create: `src/lolcoach/calibrate.py`
- Create: `tests/fixtures/minimap_sample.png` (real minimap crop from a hand-captured game; ~10 KB)
- Test: `tests/test_capture.py`
- Modify: `pyproject.toml` (add `dxcam` with `sys_platform == "win32"` marker, `opencv-python`, `numpy`)

**Approach:**
- `capture.py` — `Capture` class owns a single `dxcam` instance (dxcam is slow to construct; create once). Methods:
  - `grab_minimap() -> np.ndarray | None` — grabs a full frame via dxcam, crops to the ROI loaded from `calibration.json`, returns the cropped numpy array. Returns `None` on transient failure.
  - `LatestFrameBuffer` helper wrapping `queue.Queue(maxsize=1)` with `put_latest(frame)` that drains the slot first, then puts — semantics: only the newest frame matters.
  - `load_calibration(path: Path) -> ROI | None` — parses `calibration.json` as `{x, y, w, h}`; returns `None` if missing; raises `ValidationError` if present but out-of-bounds for the current screen.
- `calibrate.py` as `python -m lolcoach.calibrate` entry point. Uses `cv2.imshow` + `cv2.setMouseCallback` to present the current screen capture and ask the user to click the minimap top-left, then bottom-right. Writes `calibration.json` next to the config.
- Failure handling: `capture.py` defines `CaptureError` (single transient failure) and `CaptureExhausted` (raised after 10 consecutive `CaptureError`s — caught by `main.py` in Unit 10 and turned into the "Screen capture failed. Coach paused." TTS announcement).
- The `LatestFrameBuffer` lives in `capture.py` and is used by Unit 10 to pass frames from the Capture thread to the Inference worker.

**Execution note:** Test-first for the ROI math, the `LatestFrameBuffer` drain-and-put semantics, and the failure-count escalation. The actual `dxcam.grab()` call is hardware-dependent and covered by `# pragma: no cover` in one narrow branch.

**Patterns to follow:** None (greenfield). External: `dxcam` README for `grab()` semantics, OpenCV `imshow` + `setMouseCallback` for calibration UI.

**Test scenarios:**
- *Happy path:* with a valid mocked `dxcam` returning a known frame and a valid ROI, `grab_minimap()` returns a numpy array with the expected shape.
- *Happy path:* `LatestFrameBuffer.put_latest(frameA); put_latest(frameB); get()` returns `frameB` (not `frameA`) without blocking.
- *Edge case:* `calibration.json` file missing → `load_calibration()` returns `None` (caller prompts the user to run calibrate).
- *Edge case:* `calibration.json` present with valid ROI → returns `ROI(x, y, w, h)` tuple.
- *Edge case:* `calibration.json` exists but is corrupt / invalid JSON → logs a warning and returns `None`.
- *Edge case:* ROI extends past screen bounds (x + w > screen_width) → raises `ValidationError` with an explicit message naming the screen dimensions.
- *Error path:* `dxcam.grab()` returns `None` (transient failure) → `grab_minimap()` raises `CaptureError`.
- *Error path:* 10 consecutive `CaptureError`s in a row → the 11th call raises `CaptureExhausted` instead. Uses a counter-based fake dxcam driver.
- *Integration:* `calibrate.py` with mocked mouse clicks at `(100, 200)` and `(300, 400)` writes `calibration.json` containing `{"x": 100, "y": 200, "w": 200, "h": 200}`.

**Verification:** `pytest tests/test_capture.py -v` passes all 9 scenarios; `python -m lolcoach.calibrate --test-clicks 100,200,300,400` produces the expected `calibration.json` on disk.

---

- [ ] **Unit 3: Deterministic detector + champion template library**

**Goal:** Build the OpenCV template-matching detector that is the **primary** enemy-position source. Without this unit, there is no grounding check, no enemy jungler tracking, and therefore no meaningful coaching. Also build the setup-time Data Dragon downloader script.

**Requirements:** R6 (detection feeds the grounding check), R16 (information recovery requires knowing where enemies are)

**Dependencies:** Unit 1

**Files:**
- Create: `src/lolcoach/detector.py`
- Create: `scripts/build-champion-templates.py`
- Create: `tests/fixtures/champion_templates_subset/` (hand-picked small set: 5 champions — `LeeSin`, `Jinx`, `Hecarim`, `Ekko`, `Ahri` — as 64×64 PNG crops, total ~20 KB)
- Test: `tests/test_detector.py`
- Modify: `pyproject.toml` (verify `opencv-python`, `numpy` already added in Unit 2; add `requests` for the downloader)

**Approach:**
- `detector.py` — `Detector` class:
  - `__init__(template_dir: Path, confidence_threshold: float = 0.85)` — loads all `*.png` templates once on construction.
  - `detect(frame: np.ndarray, ally_names: list[str], enemy_names: list[str]) -> DetectionResult` — returns `DetectionResult(champions=[...], detected_at=time.time())`.
  - Uses `cv2.matchTemplate` with `TM_CCOEFF_NORMED`. Runs each template against the frame, collects all locations above threshold, then applies NMS (non-maximum suppression) to dedupe overlapping hits.
  - For each hit, derives `position_norm = (cx / W, cy / H)` in `[0, 1]²`.
  - `quadrant_of(x: float, y: float) -> str` — returns one of the 9 region names via fixed thresholds. Extracted as a pure function for easy testing.
  - Team assignment: a champion detected with name `"LeeSin"` is assigned `team="ally"` if in `ally_names`, `team="enemy"` if in `enemy_names`, dropped with a logged event if in neither (prevents phantom detections from bleeding into state).
  - `last_seen_champions` property — the most recent `detect()` result's champion names (used by the `DecisionFilter` grounding check in Unit 8).
- `scripts/build-champion-templates.py` — standalone CLI:
  - Fetches current patch from Riot Data Dragon realms: `https://ddragon.leagueoflegends.com/realms/na.json` → `v` field.
  - Downloads each champion's portrait from `https://ddragon.leagueoflegends.com/cdn/{patch}/img/champion/{Name}.png`.
  - Crops to the minimap-icon shape (center circle ~64×64).
  - Writes to `~/.lolcoach/templates/` by default, or a path supplied via `--output`.
  - Idempotent: skips existing files unless `--force` is set.
  - Logs progress: `Downloaded 135/160 champions...`.

**Execution note:** Test-first for the quadrant logic and NMS behavior. Template matching accuracy on a real minimap is a manual E2E concern (Unit 10's `tests/manual.md`), not a unit test.

**Patterns to follow:** None (greenfield). External: OpenCV `matchTemplate` docs, Riot Data Dragon champion portrait URL structure.

**Test scenarios:**
- *Happy path:* given a synthetic 512×512 frame with a single embedded `LeeSin` template at position `(240, 320)`, `detect(frame, enemy_names=["LeeSin"])` returns exactly one champion with `position_norm ≈ (0.47, 0.62)` and `team="enemy"`.
- *Happy path — 9 quadrant thresholds:* `quadrant_of(0.1, 0.1)` → `"top_lane"`; `quadrant_of(0.1, 0.9)` → `"bot_lane"`; `quadrant_of(0.9, 0.1)` → `"top_river"` (or `"base_blue"` per the agreed thresholds); `quadrant_of(0.5, 0.5)` → `"mid"`; plus coverage of the remaining 5 regions. One test method per quadrant, or one parametrized test with 9 cases.
- *Edge case:* overlapping template hits for the same champion → NMS collapses them to the single highest-confidence match.
- *Edge case:* all template matches below the confidence threshold → `detect()` returns an empty `DetectionResult.champions` list (not an error).
- *Edge case:* detected `LeeSin` not in ally or enemy rosters → dropped, `log_event("detection_dropped", ...)` fires, returned list excludes it.
- *Error path:* one template file is corrupt (not a valid PNG) → `__init__` logs the failure and skips that template, does not crash.
- *Edge case:* empty ally and enemy rosters → `detect()` returns empty list (defensive for very early game frames).
- *Integration:* `scripts/build-champion-templates.py --output <tmp> --limit 3` downloads exactly 3 champions, writes 3 PNG files, is idempotent on re-run (no additional downloads).

**Verification:** `pytest tests/test_detector.py -v` passes; `python scripts/build-champion-templates.py --output ~/.lolcoach/templates/ --limit 5` runs and produces 5 PNG files; running a second time is a no-op.

---

- [ ] **Unit 4: Riot API client + lifecycle FSM**

**Goal:** Poll the Riot Live Client Data API, identify the active player, verify role, and drive the `IDLE → STARTING → ACTIVE → ENDING` state machine. Owns game lifecycle detection (per design doc §3).

**Requirements:** R1, R10, R11

**Dependencies:** Unit 1

**Files:**
- Create: `src/lolcoach/riot_client.py`
- Create: `tests/fixtures/allgamedata_ingame.json` (sample of `/liveclientdata/allgamedata` with a jungle active player)
- Create: `tests/fixtures/allgamedata_nogame.json` (sample 404 error response body)
- Create: `tests/fixtures/allgamedata_remake.json` (sample with a different `gameId`)
- Test: `tests/test_riot_client.py`
- Modify: `pyproject.toml` (add `requests`, `urllib3`)

**Approach:**
- `riot_client.py`:
  - `RiotClient` class with `start(callbacks: Callbacks)`, `stop()` lifecycle methods. Runs a polling thread at `config.riot_api.poll_interval_ms` (default 2000 ms).
  - Uses `requests.Session` with `verify=False`. At module load, `urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)` suppresses the one-time cert warning.
  - Polls `https://127.0.0.1:2999/liveclientdata/allgamedata`. On first 200 response, also calls `/liveclientdata/activeplayer` and matches the returned `summonerName` against `data["allPlayers"]` to find the active player's full record.
  - **Role verification:** `position == "JUNGLE"` in the `allPlayers` entry, OR the player's summoner spells include `SummonerSmite` (fallback for older client versions).
  - **Lifecycle FSM** implemented as an enum `LifecycleState` + `_transition(new_state, reason)` method on `RiotClient`. State stored in `self._state` guarded by a lock.
  - Transitions:
    - `IDLE → STARTING` — on first 200 response. Persists `self._game_id = data["gameData"]["gameId"]`. Fires `callbacks.on_state_change(IDLE, STARTING)`.
    - `STARTING → ACTIVE` — after role verification passes. Fires `on_state_change` + `on_game_data(data)` for the first time.
    - `STARTING → IDLE` — on role mismatch. Fires `callbacks.on_role_mismatch(role)`; resets state.
    - `ACTIVE → ENDING` — when 404/error persists for 10 s, OR next 200 has a different `gameId`. Fires `callbacks.on_state_change` + `on_game_end(reason)`.
    - `ENDING → IDLE` — immediate, after the cleanup callback returns.
  - **Reconnect handling:** if `ACTIVE → 404` but returns with the SAME `gameId` within 10 s, stays `ACTIVE` (no `ENDING` transition). Uses a `_404_since: float | None` counter.
  - **Remake handling:** the standard `ACTIVE → ENDING → IDLE → STARTING` cycle with a new `gameId` covers it.
  - Callbacks protocol is a small dataclass: `on_state_change(from_state, to_state)`, `on_game_data(data)`, `on_role_mismatch(role)`, `on_game_end(reason)`.

**Execution note:** Test-first for every FSM transition and every 404/error path. Use `requests-mock` to simulate Riot API responses deterministically. The 13-test count in the test plan maps 1:1 to the scenarios below.

**Patterns to follow:** None (greenfield). External: Riot Live Client Data API docs.

**Test scenarios:**
- *Happy:* first 200 from `/allgamedata` → FSM `IDLE → STARTING`; `self._game_id` matches the `gameData.gameId` in the fixture.
- *Happy:* role verification with `position == "JUNGLE"` → `STARTING → ACTIVE`; `on_game_data` callback fires.
- *Happy:* role verification with `SummonerSmite` present but `position != "JUNGLE"` → `STARTING → ACTIVE` via fallback.
- *Edge:* role verification with `position == "TOP"` and no Smite → `STARTING → IDLE`; `on_role_mismatch("TOP")` callback fires exactly once.
- *Edge:* `ACTIVE` state with 404 responses for 10 consecutive seconds → `ACTIVE → ENDING`; `on_game_end("timeout")` callback fires.
- *Edge:* `ACTIVE` state with next 200 containing a different `gameId` → `ACTIVE → ENDING → IDLE → STARTING` for the new game.
- *Edge (reconnect):* `ACTIVE` → 404 for 5 s → 200 with SAME `gameId` → stays `ACTIVE`; no `ENDING` transition.
- *Edge (remake vs reconnect):* `ACTIVE` → 404 for 5 s → 200 with DIFFERENT `gameId` → `ENDING` (remake path).
- *Error path:* connection refused on `https://127.0.0.1:2999` → poll logs event, returns, retries on next tick; no crash.
- *Error path:* Riot API returns 500 on `/allgamedata` → log, continue polling, no `ENDING` transition.
- *Edge (mirror match):* two `LeeSin` entries in `allPlayers` → active player lookup uses `summonerName`, not champion name, and correctly identifies the active player.
- *Integration:* full mocked lifecycle — `IDLE → STARTING → ACTIVE → ENDING → IDLE` over 30 s of simulated poll ticks; callback sequence matches expectation.
- *Edge (shutdown):* `stop()` called while polling thread is mid-request → thread exits cleanly within the poll interval; no hang.

**Verification:** `pytest tests/test_riot_client.py -v` passes all 13 scenarios; JSONL log from a test run contains expected `lifecycle_transition` events.

---

- [ ] **Unit 5: Jungle routes table + thread-safe StateManager**

**Goal:** Hardcoded jungle-route lookup for enemy jungler position prediction, plus the thread-safe game state store (`StateManager`) that feeds the inference thread and is reset cleanly on every `ENDING` transition.

**Requirements:** R4 (multi-game reset), R11 (FSM-driven reset), R16 (information recovery needs state memory)

**Dependencies:** Unit 4 (StateManager consumes Riot API data)

**Files:**
- Create: `src/lolcoach/jungle_routes.py`
- Create: `src/lolcoach/state.py`
- Test: `tests/test_jungle_routes.py`
- Test: `tests/test_state.py`

**Approach:**
- `jungle_routes.py`:
  - Module-level constant `STANDARD_PATHS: dict[str, list[JungleCamp]]`. Named paths: `blue_top_full_clear`, `red_bot_full_clear`, `blue_top_rush`, `red_bot_rush`, `invade_blue`, `invade_red` (6 paths total per the design doc "~6 standard paths").
  - `JungleCamp(name, quadrant, clear_time_seconds)` dataclass; one ordered list per path.
  - `predict_quadrant(last_seen_quadrant: str | None, elapsed_seconds: float) -> str | None`:
    - If `last_seen_quadrant is None` → returns `None` (cold start).
    - Picks the path whose initial camp is closest to `last_seen_quadrant`.
    - Walks the path summing `clear_time_seconds` until it exceeds `elapsed_seconds`.
    - Returns the current camp's quadrant.
    - If `elapsed_seconds` exceeds the whole path, returns the last camp's quadrant (best-effort tail).
- `state.py`:
  - `GameState` dataclass — value type, no behavior:
    - `game_time_seconds: float`
    - `active_player: ActivePlayer | None` (level, gold, items, summoner spells, position_norm)
    - `allies: list[PlayerEntry]`, `enemies: list[PlayerEntry]`
    - `enemy_jungler_last_seen: tuple[str, float] | None` — (quadrant, timestamp) or `None`
    - `enemy_jungler_predicted_quadrant: str | None` — derived on snapshot
    - `objective_timers: ObjectiveTimers` (dragon_spawn_at, baron_spawn_at, herald_spawn_at, etc.)
    - `kill_feed: list[KillEvent]` — bounded via `collections.deque(maxlen=50)`
    - `all_champions: set[str]` — both teams' champion names, used by the `DecisionFilter` grounding check
  - `StateManager` class with `threading.RLock`:
    - `update_from_riot(data: dict)` — merges `allgamedata` into state; updates `objective_timers` from event feed; updates `kill_feed` from new kills; on an enemy-jungler-dies kill event, sets `enemy_jungler_last_seen = None`.
    - `update_from_detector(result: DetectionResult)` — updates ally/enemy positions; if an enemy with `position == "JUNGLE"` (per Riot roster) is in the detection, sets `enemy_jungler_last_seen = (quadrant, result.detected_at)`.
    - `snapshot() -> GameState` — deep-copies under lock, computes `enemy_jungler_predicted_quadrant` from jungle routes + elapsed time, returns an independent value.
    - `reset()` — clears everything back to a fresh `GameState()`.
  - The computed `enemy_jungler_predicted_quadrant` lives on the snapshot only, not stored — this avoids stale predictions when the inference thread isn't running.

**Execution note:** Test-first with heavy emphasis on thread safety. Every `update_*` method needs a concurrent-writer test that runs under a `threading.Barrier` to force contention.

**Patterns to follow:** None (greenfield). Stdlib `collections.deque(maxlen=...)` for bounded kill feed; `copy.deepcopy` for snapshot.

**Test scenarios:**

`tests/test_jungle_routes.py`:
- *Happy — path 1:* `blue_top_full_clear` with `last_seen="top_jungle"`, `elapsed=30` → returns `"top_jungle"`; with `elapsed=60` → returns next camp's quadrant; with `elapsed=120` → returns mid-path camp.
- *Happy — path 2:* `red_bot_full_clear` symmetric predictions.
- *Happy — paths 3, 4, 5, 6:* one test per remaining standard path, verifying the sequence matches the table.
- *Edge (cold start):* `predict_quadrant(None, 60.0)` returns `None`.
- *Edge (tail overshoot):* `predict_quadrant("top_jungle", 9999.0)` returns the last camp's quadrant (best-effort).

`tests/test_state.py`:
- *Happy:* `update_from_riot(allgamedata_ingame_fixture)` populates `game_time_seconds`, `active_player.level`, `active_player.gold`.
- *Happy:* `update_from_detector` with an enemy `LeeSin` at `("bot_jungle", 1712750000)` → `enemy_jungler_last_seen == ("bot_jungle", 1712750000)`.
- *Happy:* `snapshot()` returns a value that, when mutated by the caller, leaves the internal state unchanged (deep-copy verified via mutation + re-snapshot).
- *Edge:* appending 60 kill events → `kill_feed` retains only the last 50 (deque cap).
- *Edge:* enemy jungler killed in the kill feed → `enemy_jungler_last_seen = None`; next snapshot's `enemy_jungler_predicted_quadrant` is also `None`.
- *Edge (cold start):* fresh `StateManager()` — initial snapshot has `enemy_jungler_last_seen = None` and `enemy_jungler_predicted_quadrant = None`.
- *Edge:* `reset()` clears everything — `game_time_seconds = 0`, empty `kill_feed`, empty `allies` and `enemies`, `enemy_jungler_last_seen = None`.
- *Edge:* `update_from_riot` preserves detector-sourced fields that Riot doesn't provide (e.g., `enemy_jungler_last_seen` is not clobbered when a Riot update arrives).
- *Edge:* dragon kill event in Riot event feed → `objective_timers.dragon_spawn_at` updated to next spawn time.
- *Edge (mirror match):* both teams have `LeeSin`; detector update for an enemy Lee Sin does not overwrite the ally Lee Sin's position.
- *Edge (paused game):* `update_from_riot` with game time unchanged across two updates → no error, state reflects frozen time.
- *Edge:* unknown summoner name in a kill event → event still appended to `kill_feed`, logged as `warning`.
- *Thread safety — writer contention:* two writer threads (one calling `update_from_riot`, one calling `update_from_detector`) in a `threading.Barrier` loop for 1 second each → no deadlocks, no partial reads on a concurrent `snapshot()` thread, all returned snapshots pass an invariant check (internal consistency between `game_time_seconds` and the `allies`/`enemies` lists).
- *Thread safety — reset vs writer:* a thread calling `reset()` concurrently with a writer thread → final state after both finish is a valid (fully-reset or fully-updated) state, never a half-written mix.
- *Edge:* `all_champions` is populated on the first Riot API update and stays populated (used by the grounding check in Unit 8).

**Verification:** `pytest tests/test_jungle_routes.py tests/test_state.py -v` — 6 jungle-route tests + 15 state tests pass. Thread-safety tests pass under `pytest -x`.

---

- [ ] **Unit 6: Vision model benchmark subcommand (Week 1 gate)**

**Goal:** Measure vision LLM latency standalone and under League GPU load. Gate the Week 1 decision tree (< 5 s, 5–8 s, 8–12 s, > 12 s branches). Emits a JSONL report the implementer uses to commit to a final model choice or pivot to text-only coaching.

**Requirements:** R5 (latency budget enforcement)

**Dependencies:** Unit 1 (config, logging), Unit 2 (fixture minimap image — not live capture; the benchmark sends a saved image)

**Files:**
- Create: `src/lolcoach/benchmark.py`
- Test: `tests/test_benchmark.py`
- Modify: `README.md` (add benchmark instructions section referencing the spec's decision tree)

**Approach:**
- `benchmark.py` as `python -m lolcoach.benchmark` entry point with arguments: `--fixture <path>` (image to use), `--model <tag>` (overrides config default), `--n <int>` (number of calls, default 20), `--prompt <path>` (optional prompt override).
- Sends `n` sequential POST requests to Ollama `/api/chat` with the same image + prompt, times each one, collects stats: `min`, `max`, `mean`, `p50`, `p95`, `std`, `timeout_count`.
- Two informal run modes (developer discipline, not code-enforced): `standalone` (League not running) and `under-load` (League running). No automation — the README instructs the developer to alt-tab between them.
- Writes a report to stdout (human-readable table) AND appends a JSONL record to `./logs/benchmark-{timestamp}.jsonl` for archival.
- The decision tree itself lives in the README as prose, not in code. The benchmark's job is to produce the numbers; the developer applies the tree.

**Execution note:** Thin module, mostly a driver. Test the stats math and error handling. The actual Ollama HTTP call is mocked in unit tests.

**Patterns to follow:** None (greenfield). External: Ollama `/api/chat` docs for the request body shape.

**Test scenarios:**
- *Happy:* mock Ollama returns 10 responses with known latencies `[2.0, 2.1, 2.5, 3.0, 3.1, 3.5, 4.0, 5.0, 6.0, 7.0]` → benchmark reports `mean ≈ 3.82`, `p95 = 7.0`, `min = 2.0`, `max = 7.0`.
- *Edge:* Ollama times out on 3 of 10 calls → benchmark reports `timeout_count == 3` and excludes timeouts from latency stats; returns exit code 0 (not an error).
- *Error path:* Ollama unreachable (connection refused) → benchmark exits with non-zero code and a helpful message ("Is Ollama running at http://localhost:11434 ?").
- *Edge:* `--n 1` — single sample still produces a valid report (mean == min == max == that sample).
- *Happy:* JSONL output includes one line per sample + one summary line.

**Verification:** `python -m lolcoach.benchmark --fixture tests/fixtures/minimap_sample.png --n 5` with a running Ollama returns a populated summary and writes a JSONL file. Tests pass under `pytest tests/test_benchmark.py`.

**Phase 1 exit gate:** After Unit 6 runs against a live Ollama + live League, the implementer reviews latency against the spec's decision tree and either (a) commits to the current model, (b) switches to a smaller model, (c) reduces capture cadence to 2000 ms, or (d) pivots to text-only coaching. The pivot is not expected; it's the honest escape hatch if the vision-LLM approach is untenable on the target hardware.

---

### Phase 2 — Intelligence & Voice (Week 2)

- [ ] **Unit 7: Inference engine (Ollama vision + prompt builder)**

**Goal:** Send a minimap frame + structured game state to the local vision LLM and parse its structured response into a `Callout` object. Implements the warmup ping and the `keep_alive: -1` pinning.

**Requirements:** R2, R5, R14

**Dependencies:** Unit 1 (config, logging), Unit 5 (consumes `GameState` snapshot)

**Files:**
- Create: `src/lolcoach/prompts.py`
- Create: `src/lolcoach/inference.py`
- Create: `tests/fixtures/ollama_responses.py` (Python module with `VALID_RESPONSE`, `MALFORMED_NO_CATEGORY`, `UNKNOWN_CATEGORY`, `CONFIDENCE_OUT_OF_RANGE`, `TRAILING_WHITESPACE`, etc. constants)
- Test: `tests/test_inference.py`
- Modify: `pyproject.toml` (add `requests` — already added in Unit 4; verify)

**Approach:**
- `prompts.py`:
  - `build_system_prompt() -> str` — returns the instruction block that defines the response schema, category enum, lane enum, confidence scale, and the "you are a jungle coach" framing.
  - `build_user_prompt(state: GameState, detections: DetectionResult) -> str` — formats the structured context block (game time, own champion, gold, objectives, kill feed, last-seen positions). **Omits the "Enemy jungler: ..." line entirely if `state.enemy_jungler_last_seen is None`** (cold-start rule).
  - All prompt text is loaded from a template string constant; no f-string injection of user data without explicit formatting.
- `inference.py`:
  - `Callout` dataclass: `decision: str`, `category: str`, `target_lane: str`, `confidence: int`, `reason: str`, `generated_at: float`.
  - `InferenceEngine` class:
    - `__init__(config: Config)` — holds the Ollama base URL, model tag, timeout from config; creates a requests session.
    - `warmup() -> bool` — sends a small test payload (tiny image, short prompt) to Ollama with `keep_alive: -1`; returns `True` if the model responded within 30 s, `False` otherwise. Called once at coach startup (Unit 10).
    - `run(frame: np.ndarray, state: GameState, detections: DetectionResult) -> Callout | None` — builds prompt, encodes frame as base64, POSTs to `/api/chat` with `keep_alive: -1`, parses the response via `_parse_response`, returns a `Callout` or `None` (on malformed / timeout / error).
    - `_parse_response(text: str) -> Callout | None` — line-based regex parser for the 5 fields. Returns `None` on any parse failure and logs `{"type": "inference_malformed", "raw": text}`.
    - **Single-worker lock:** `self._lock = threading.Lock()`. `run()` tries `lock.acquire(blocking=False)` and returns `None` immediately if already held (stale-frame drop policy).
    - Timeout handling: wraps the HTTP POST in `timeout=self.config.inference.timeout_ms / 1000.0`; on `requests.Timeout` returns `None` and logs.
    - Network error handling: on `requests.ConnectionError` returns `None` and logs — no retry (decision filter layer handles cooldowns).

**Execution note:** Test-first for prompt builder (golden snapshots for 2–3 representative states) and response parser (fixtures for valid + 5 malformed variants). Mock all HTTP calls with `requests-mock`.

**Patterns to follow:** None (greenfield). External: Ollama `/api/chat` with images.

**Test scenarios:**
- *Happy:* `build_user_prompt(fully_populated_state)` matches a golden snapshot containing all 6 context lines (game_time, champion, gold, objectives, enemy jungler, kill feed).
- *Happy (cold start):* `build_user_prompt(state_with_last_seen=None)` **omits** the "Enemy jungler" line entirely (asserted by substring absence).
- *Happy:* `_parse_response(VALID_RESPONSE)` returns a `Callout(decision="Path to bot river, Lee Sin likely ganking top", category="pathing", target_lane="bot", confidence=8, reason="...")`.
- *Happy:* `_parse_response(TRAILING_WHITESPACE)` (valid response with trailing `\r\n\n`) parses cleanly.
- *Edge:* `_parse_response(MALFORMED_NO_CATEGORY)` (missing `CATEGORY:` line) → returns `None`, logs `inference_malformed`.
- *Edge:* `_parse_response(UNKNOWN_CATEGORY)` (`CATEGORY: taunt`) → returns `None`, logs.
- *Edge:* `_parse_response(CONFIDENCE_OUT_OF_RANGE)` (`CONFIDENCE: 99`) → returns `None`, logs.
- *Error path:* mocked Ollama returns HTTP 500 → `run()` returns `None`, logs `inference_error`.
- *Error path:* mocked Ollama timeout (`requests.Timeout`) → `run()` returns `None`, logs `inference_timeout`.
- *Error path:* mocked Ollama connection refused → `run()` returns `None`, logs `inference_connection_refused`.
- *Happy:* `warmup()` with a small test payload returns `True`; the request body includes `"keep_alive": -1`.
- *Integration — single-worker lock:* two concurrent calls to `run()` from separate threads → first acquires lock and makes HTTP call, second returns `None` immediately without making an HTTP call (verified via `requests-mock` call count).

**Verification:** `pytest tests/test_inference.py -v` passes all 11 scenarios. Prompt golden snapshots stored inline as raw-string constants for easy review.

---

- [ ] **Unit 8: Decision filter — cooldown, dedup, confidence, staleness, and grounding**

**Goal:** Implement the `DecisionFilter` that gates every LLM output before it reaches TTS. This unit carries the most load-bearing invariant in the whole project: the grounding check that makes hallucinated champion callouts structurally impossible.

**Requirements:** R6 (structural hallucination guarantee), R12 (interrupt pre-conditions), R13 (cooldown + dedup + confidence)

**Dependencies:** Unit 5 (consumes `GameState` snapshot), Unit 7 (consumes `Callout`)

**Files:**
- Create: `src/lolcoach/filter.py`
- Test: `tests/test_filter.py`

**Approach:**
- `filter.py`:
  - `FilterResult` dataclass: `accepted: bool`, `reason: str` (when rejected — one of `stale | low_confidence | hallucinated | cooldown | dedup | malformed`).
  - `DecisionFilter` class:
    - `__init__(config: Config, champion_list: set[str])` — the champion list is loaded from the detector's template directory on startup (all known champion names for regex anchoring).
    - `accept(callout: Callout, state: GameState, detections: DetectionResult) -> FilterResult` — applies checks in the fixed order below. Returns on first failure.
    - Checks:
      1. **Staleness:** `time.time() - callout.generated_at > config.decisions.staleness_threshold_seconds` → reject.
      2. **Confidence:** `callout.confidence < config.decisions.confidence_threshold` → reject; logged to JSONL but not surfaced to user.
      3. **Grounding (the critical invariant):**
         - Extract champion-name candidates from `callout.decision + " " + callout.reason` via a single compiled regex built from the full champion list with word boundaries. Handles names with apostrophes (`Kai'Sa`, `Vel'Koz`), spaces (`Lee Sin`, `Jarvan IV`, `Master Yi`), and periods (`Dr. Mundo`). The in-game ID `MonkeyKing` maps to the display name `Wukong` via a small alias table.
         - Every extracted name must appear in `detections.last_seen_champions` OR `state.all_champions` (the Riot roster). If any extracted name is missing from both → reject as `hallucinated`.
         - If no champion names are mentioned → pass grounding (generic callouts about objectives or pathing don't need grounding).
      4. **Cooldown:** `time.time() - self._last_spoken_at < config.decisions.cooldown_seconds` → reject.
      5. **Dedup:** look up the `(callout.category, callout.target_lane)` tuple in `self._recent_spoken` (a small `OrderedDict[tuple, float]` of recent entries); if present AND within `config.decisions.dedup_window_seconds` → reject.
    - `record_spoken(callout)` — called by TTS (Unit 9) after successful playback; updates `self._last_spoken_at` and inserts the tuple into `self._recent_spoken`; evicts entries older than the dedup window on each insert.

**Execution note:** Test-first. This unit is the hallucination guard. Write the "Ekko hallucination" test FIRST and let it fail, then implement the grounding check.

**Patterns to follow:** None (greenfield). Stdlib `re` for regex, `collections.OrderedDict` for recent-spoken tracking.

**Test scenarios:**
- *Happy:* fresh, high-confidence, unique callout with zero champion mentions (e.g., `decision="Recall, buy wards"`) → `accepted=True`.
- *Happy — grounded via detector:* callout mentions "Lee Sin"; detector's `last_seen_champions == {"LeeSin"}` → `accepted=True` (note: regex normalizes space-handling for "Lee Sin" vs "LeeSin").
- *Happy — grounded via Riot roster:* callout mentions "Ekko"; detector never saw Ekko but `state.all_champions` contains "Ekko" (because Ekko is in the game, just not on the last-seen minimap frame) → `accepted=True`.
- *Critical — hallucinated:* callout mentions "Ekko"; detector's `last_seen_champions == {"LeeSin"}` AND `state.all_champions = {"LeeSin", "Jinx", "Hecarim", "Ahri"}` (Ekko not in game) → rejected with `reason="hallucinated"`. **This is the load-bearing test.**
- *Edge — partial hallucination:* callout mentions both "Lee Sin" (valid) and "Ekko" (hallucinated) → rejected (any single invalid name fails the whole callout).
- *Edge — apostrophe champion names:* callout mentions "Kai'Sa"; `all_champions` contains "Kai'Sa" → `accepted=True`.
- *Edge — multi-word champion names:* callout mentions "Lee Sin" with a space; regex correctly matches against `"LeeSin"` in `all_champions` (or vice-versa, depending on normalization choice).
- *Edge — Wukong alias:* callout mentions "Wukong"; `all_champions` contains the in-game ID `"MonkeyKing"` → `accepted=True` via alias table.
- *Cooldown — recent:* `record_spoken` at `t=0`, then `accept` called at `t=3` with a fresh callout → rejected with `reason="cooldown"`.
- *Cooldown — elapsed:* `record_spoken` at `t=0`, then `accept` at `t=6` → `accepted=True`.
- *Dedup — same tuple:* `record_spoken(gank_warning, top)` at `t=0`; new `gank_warning, top` at `t=20` (cooldown has passed) → rejected with `reason="dedup"`.
- *Dedup — different lane:* `record_spoken(gank_warning, top)` at `t=0`; new `gank_warning, bot` at `t=20` → `accepted=True`.
- *Dedup — different category:* `record_spoken(gank_warning, top)` at `t=0`; new `objective_call, global` at `t=20` → `accepted=True`.
- *Dedup — window elapsed:* `record_spoken(gank_warning, top)` at `t=0`; new `gank_warning, top` at `t=35` → `accepted=True` (beyond the 30 s window).
- *Confidence gate:* callout with `confidence=4` → rejected with `reason="low_confidence"`; JSONL log records the drop.
- *Staleness:* callout with `generated_at = time.time() - 15` and staleness threshold = 10 s → rejected with `reason="stale"`.
- *Edge — empty decision text:* callout with `decision=""` → rejected with `reason="malformed"`.

**Verification:** `pytest tests/test_filter.py -v` passes all scenarios. The "Ekko hallucination" test is the critical one — if it fails, R6 is not met and no packaging should proceed.

---

- [ ] **Unit 9: TTS module (Piper + sounddevice + priority interrupt)**

**Goal:** Synthesize callouts via Piper TTS and play them via `sounddevice.OutputStream`, supporting mid-stream interrupt for higher-priority callouts (gank/counter-gank).

**Requirements:** R2, R9 (voice-only), R12 (interrupt rules)

**Dependencies:** Unit 1 (config, logging)

**Files:**
- Create: `src/lolcoach/tts.py`
- Test: `tests/test_tts.py`
- Modify: `pyproject.toml` (add `sounddevice`, `soundfile` for WAV loading, plus either a Piper Python binding or subprocess-only; decision deferred to implementation)

**Approach:**
- `tts.py`:
  - `TTS` class with `start()`, `stop()`, and `speak(callout: Callout)`.
  - Internally runs a single "speaker" worker thread that consumes from a 1-slot callout buffer (new `speak()` calls either interrupt or drop — never queue up).
  - `synthesize(text: str) -> np.ndarray` — invokes Piper via subprocess (initial choice; Python binding as a fallback if the binding is stable under PyInstaller one-file). Returns a numpy array of audio samples.
  - `_play(samples: np.ndarray)` — creates a `sounddevice.OutputStream`, writes samples, waits for playback to finish.
  - **Interrupt logic** in `speak()`:
    - Under `self._lock`, read `self._currently_playing: CalloutState | None`.
    - If nothing playing → set `self._currently_playing = callout`, wake the worker.
    - If something is playing:
      - If `new.confidence >= current.confidence + config.tts.interrupt_confidence_delta` AND `new.category in config.tts.interrupt_categories` → call `self._current_stream.stop()`, replace `self._currently_playing = callout`, wake the worker.
      - Otherwise drop `new` (log `tts_dropped_low_priority`).
  - `_currently_playing` is cleared when playback finishes or is interrupted.
  - Output device routing: if `config.tts.output_device == "system_default"` → use `None` for sounddevice; otherwise look up the device index via `sounddevice.query_devices(name)`.
  - On any Piper failure → log `tts_synth_failed` and skip the callout (the moment has passed; design spec §4).
  - Thread safety: `self._lock` wraps every read/write of `_currently_playing` and `_current_stream`.
  - On `stop()`: sets a shutdown event, stops any active stream, waits for the worker thread to exit with a timeout.

**Execution note:** Test-first for the interrupt logic with a fake synthesizer that returns deterministic WAV data. Actual audio playback is hardware-dependent — smoke-tested manually in `tests/manual.md`.

**Patterns to follow:** None (greenfield). External: Piper TTS README for subprocess invocation, `sounddevice` docs for `OutputStream.stop()`.

**Test scenarios:**
- *Happy:* `speak(callout)` with nothing currently playing → synthesize called once, stream started, `_currently_playing == callout`.
- *Interrupt — valid:* current callout `(confidence=6, category="pathing")`, new callout `(confidence=8, category="gank_warning")` → current stream stopped, new callout plays. Assertion: `stream.stop()` was called exactly once before the second synthesize.
- *Interrupt — wrong category:* current `(confidence=6, category="pathing")`, new `(confidence=9, category="pathing")` → new dropped (category not in `interrupt_categories`). `stream.stop()` NOT called.
- *Interrupt — insufficient delta:* current `(confidence=8, category="gank_warning")`, new `(confidence=9, category="gank_warning")` → new dropped (delta < 2).
- *Interrupt — counter-gank qualifies:* current `(confidence=6, category="pathing")`, new `(confidence=8, category="counter_gank")` → interrupt fires.
- *Error path:* fake Piper raises exception → logged as `tts_synth_failed`, `_currently_playing` cleared, no crash, coach continues running.
- *Edge — shutdown during playback:* `stop()` called while stream is active → stream stopped, worker thread joins within 5 s, no hang.

**Verification:** `pytest tests/test_tts.py -v` passes all 7 scenarios. Manual smoke test: `python -c "from lolcoach.tts import TTS; t = TTS(...); t.speak(fake_callout)"` produces audible speech through the default device (documented in `tests/manual.md`).

---

- [ ] **Unit 10: Main orchestration, thread lifecycle, integration tests**

**Goal:** Wire all components into the 4-thread pipeline. Handle startup (config + calibration + warmup), the ACTIVE-state coaching loop, error escalation, and graceful shutdown. This is the unit that turns the individual modules into "the coach".

**Requirements:** R1, R3, R4, R7, R10

**Dependencies:** Units 2, 3, 4, 5, 7, 8, 9

**Files:**
- Create: `src/lolcoach/main.py`
- Modify: `src/lolcoach/__main__.py` (replace Unit 1's stub with `from lolcoach.main import run; run()`)
- Create: `tests/manual.md` (checklist for the 5 manual E2E flows from the spec)
- Test: `tests/test_main_integration.py`

**Approach:**
- `main.py`:
  - `run()` function, called by `__main__.py`:
    1. Load config via `config.load_config()`.
    2. Initialize the JSONL logger (benchmark-mode filename).
    3. Load calibration; if missing, print a message directing the user to run `python -m lolcoach.calibrate`, then exit.
    4. Create: `Capture`, `Detector`, `StateManager`, `RiotClient`, `InferenceEngine`, `DecisionFilter`, `TTS`.
    5. `inference.warmup()` — blocking; if it fails, print a clear message ("Ollama not reachable or model not installed — run `ollama pull <model>`") and exit with non-zero code.
    6. Wire callbacks on `RiotClient`:
       - `on_state_change(from_, to)` — logs the transition; on `IDLE → STARTING` calls `logger.rotate(game_id)`; on `ENDING` calls `state.reset()` and `tts.stop_current()`.
       - `on_game_data(data)` — calls `state.update_from_riot(data)`.
       - `on_role_mismatch(role)` — calls `tts.speak(one_shot_callout(f"You're not playing {role.lower()}. Coach paused."))` and suspends the inference worker loop until next game.
       - `on_game_end(reason)` — logs; the `ENDING` state_change handler already resets state.
    7. Start threads: Capture (timer), Riot API (timer), Inference worker (reacts to ACTIVE state), TTS (consumes from decision output).
    8. Main thread runs a health-monitor loop: waits on a shutdown event, handles SIGINT.
  - **Inference worker loop** (inside `main.py`, not the inference module — main owns the cadence):
    - On every tick (timer-driven, e.g., 500 ms after the previous inference finished), if lifecycle is `ACTIVE`, call `inference.run(latest_frame, state.snapshot(), detector.last_result)`.
    - If `Callout` returned, pass through `filter.accept(...)`. If accepted, call `tts.speak(callout)`; else log the drop reason.
    - Track consecutive timeout count; on 5 consecutive timeouts, trigger the TTS escalation message and sleep 60 s; on 3 sleep/resume cycles with zero success, transition inference to a permanent `paused_until_next_game` state.
    - `CaptureExhausted` caught at the inference worker level → `tts.speak(one_shot("Screen capture failed. Coach paused."))` and pause capture thread; other threads continue.
  - **Graceful shutdown** (SIGINT handler):
    - Set the shutdown event.
    - Call `riot_client.stop()`, `tts.stop()`, join capture thread, close JSONL logger.
    - Exit with code 0.

**Execution note:** Test-first for integration flows with mocked dxcam, mocked Ollama, mocked Riot API, mocked Piper. The real end-to-end smoke test is a manual step per `tests/manual.md`.

**Patterns to follow:** None (greenfield). Stdlib `signal` for SIGINT, `threading.Event` for shutdown, `concurrent.futures` optional but `threading.Thread` is sufficient.

**Test scenarios:**
- *Integration — single callout end-to-end:* mocked capture returns a fixture frame; mocked detector returns a `LeeSin` detection; mocked Riot returns a jungle active player; mocked Ollama returns a valid response with `confidence=8`; mocked Piper records the spoken text. Assert: exactly one call to `tts.speak`, the spoken text matches the Ollama `DECISION` line, total elapsed wall-time from capture to speak < 8 s (with mocks returning instantly).
- *Integration — multi-game:* simulate game 1 end (`gameId` change) → `state.reset()` called; simulate game 2 start → new JSONL file opened with new game id; assert state is clean (no kill-feed carryover). Runs a 10 s simulated timeline.
- *Integration — non-jungle role at start:* mocked Riot returns `position: TOP`, no Smite → `on_role_mismatch("TOP")` fires; TTS speaks the role-mismatch message exactly once; inference worker never runs; state does not advance to ACTIVE.
- *Integration — 5 consecutive inference timeouts:* mocked Ollama raises `requests.Timeout` 5 times in a row → TTS speaks the escalation message; inference worker sleeps; on the 6th tick after the sleep (mocked clock), inference retries.
- *Integration — graceful shutdown:* raise SIGINT mid-session → all threads join within 5 seconds; JSONL logger `close()` called; no dangling streams.

**Verification:** `pytest tests/test_main_integration.py -v` passes all 5 integration scenarios. Manual E2E flows from `tests/manual.md` (5 flows: first-run calibration, multi-game session, non-jungle role, forced remake, GPU contention) documented and ready for Week 2 end-of-week real-game run.

---

### Phase 3 — Windows Distribution (Week 3)

- [ ] **Unit 11: PyInstaller one-file build**

**Goal:** Produce a standalone `lol-macro-guide.exe` that bundles the Python runtime, all pure-Python dependencies, dxcam DLLs, OpenCV, and the Piper binary + default voice. Does NOT bundle Ollama or the LLM model.

**Requirements:** R1, R15

**Dependencies:** Unit 10 (the coach must run end-to-end in source form first)

**Files:**
- Create: `scripts/build-exe.py` (driver script that invokes PyInstaller with the right args)
- Create: `scripts/build.spec` (PyInstaller spec file — declarative, preferred over long command lines)
- Create: `scripts/binaries/` (directory; contents gitignored for size)
- Create: `scripts/binaries/piper/.gitkeep` (placeholder so the empty dir is tracked)
- Create: `scripts/binaries/voices/.gitkeep`
- Create: `.gitignore` (modify from Unit 1 — add `scripts/binaries/*/*` exclusions except `.gitkeep`, plus `dist/`, `build/`, `*.spec` cache)
- Modify: `pyproject.toml` (add `pyinstaller` as a dev dependency)
- Modify: `README.md` (add build-from-source instructions)

**Approach:**
- `scripts/build.spec`:
  - Entry: `src/lolcoach/__main__.py`.
  - `--onefile` mode.
  - `datas` includes: `scripts/binaries/piper/*`, `scripts/binaries/voices/*`, `config.yaml.example`.
  - `hiddenimports` starts empty; grows iteratively as smoke testing on a clean VM reveals missing lazy imports (OpenCV and sounddevice are known offenders).
  - `excludes` drops `tkinter`, `unittest`, and other stdlib modules the coach doesn't use, to shrink the bundle.
- `scripts/build-exe.py`:
  - Shells out to `pyinstaller scripts/build.spec`.
  - Validates `dist/lol-macro-guide.exe` exists after the build.
  - Prints size in MB.
- Bundling the Piper binary:
  - Piper releases provide a Windows binary at `piper_windows_amd64.zip`. The developer downloads it manually into `scripts/binaries/piper/` before running the build script. Documented in `RELEASE.md` (Unit 12).
  - Default voice: `en_US-amy-medium.onnx` + `en_US-amy-medium.onnx.json` downloaded from the Piper voices repo into `scripts/binaries/voices/`.
- README build section: `python scripts/build-exe.py` (after Piper binary + voice are in place). Output: `dist/lol-macro-guide.exe`.

**Execution note:** Not test-first. Packaging is verified via manual smoke test on a clean Windows 10 AND Windows 11 VM (both documented in `tests/manual.md`). The iterative feedback loop is: build → copy to VM → run → see which import fails → add to `hiddenimports` → rebuild.

**Patterns to follow:** None (greenfield). External: PyInstaller one-file docs.

**Test scenarios:** *Test expectation: none — build and packaging only; verified via manual smoke test on clean Win10/Win11 VMs in `tests/manual.md`.*

**Verification:**
- `python scripts/build-exe.py` completes without error.
- `dist/lol-macro-guide.exe` exists and is between 150 and 400 MB (expected range with Piper voice bundled).
- Running the exe on a clean Windows 10 VM (no Python installed) produces the same behavior as `python -m lolcoach` from source: config loads, calibration prompt appears if no calibration.json, warmup ping succeeds if Ollama is installed.
- Running the exe on a clean Windows 11 VM produces the same behavior.
- Total packaging size documented in `RELEASE.md`.

---

- [ ] **Unit 12: Windows installer, release artifacts, and v0.1.0 cut**

**Goal:** Wrap `dist/lol-macro-guide.exe` in a Windows installer (NSIS or Inno Setup — implementer's choice), write release documentation, and cut the first public release.

**Requirements:** R15

**Dependencies:** Unit 11

**Files:**
- Create: `scripts/build-installer.nsi` (OR `scripts/build-installer.iss` for Inno Setup)
- Create: `scripts/release.sh` (local release automation: pytest → cov check → build-exe → build-installer → print next-step instructions)
- Create: `RELEASE.md` (release checklist)
- Modify: `README.md` (expand to full public-facing docs: install paths, usage, ToS disclaimer, audio routing, troubleshooting)

**Approach:**
- **Installer scope** (whichever tool is used):
  - Wraps `dist/lol-macro-guide.exe`.
  - Installs to `%ProgramFiles%\LoL Macro Guide\`.
  - Creates a Start Menu entry ("LoL Macro Guide").
  - Optional desktop shortcut (user checkbox during install).
  - Uninstaller entry in Add/Remove Programs.
  - **Does NOT bundle Ollama or the LLM.** The installer's "Finish" page links to `https://ollama.com/download` and mentions `ollama pull <model>`.
- `scripts/release.sh` steps (per the spec's release checklist):
  1. Bump version in `pyproject.toml`.
  2. Run `pytest --cov-fail-under=90`.
  3. Run `python scripts/build-exe.py`.
  4. Run the installer build (NSIS or Inno Setup CLI).
  5. Print a reminder to: smoke-test on a clean VM, create a GitHub release, attach both `.exe` and installer `.exe`, draft release notes.
- `RELEASE.md` documents the above as a human-readable checklist with manual verification steps.
- **README expansions:**
  - **Install — Path 1 (pip from source):** `pip install .` from the repo, Ollama setup (`ollama pull <model>`, `OLLAMA_KEEP_ALIVE=-1`), Piper voice download, `config.yaml` copy, first-run calibrate.
  - **Install — Path 2 (Windows installer):** download `setup.exe` from GitHub releases, run it, click through, install Ollama separately, `ollama pull <model>`, run coach from Start Menu.
  - **Audio routing:** default outputs to system default; streamers route via VoiceMeeter or VB-Audio Virtual Cable + `tts.output_device` in config.
  - **ToS disclaimer:** verbatim from the spec — external screen capture only, no injection, gray area, use at your own risk.
  - **Troubleshooting:** Ollama not reachable, calibration issues, low GPU VRAM (switch to smaller model), Windows Defender warning workaround ("More info → Run anyway").

**Execution note:** Not test-first. All verification is manual via `tests/manual.md` and the release checklist in `RELEASE.md`. Run the full install → first callout flow on clean Windows 10 AND Windows 11 VMs before cutting the release.

**Patterns to follow:** None (greenfield). External: NSIS scripting reference OR Inno Setup docs.

**Test scenarios:** *Test expectation: none — packaging and documentation only; verified via manual smoke test on clean Win10/Win11 VMs and release-checklist walkthrough.*

**Verification:**
- `scripts/release.sh` runs clean through steps 1–4 on the developer machine.
- `setup.exe` installs cleanly on Windows 10 VM: Start Menu entry appears, launching the coach produces the first-run calibration prompt.
- `setup.exe` installs cleanly on Windows 11 VM.
- Uninstalling via Add/Remove Programs removes the coach cleanly.
- GitHub release `v0.1.0` is cut with both `lol-macro-guide.exe` and `setup.exe` attached + release notes drafted.
- The first-time setup → first callout flow (critical path #1 from the test plan) succeeds on the developer's actual LoL account with the coach installed via `setup.exe`.

---

## System-Wide Impact

This is a greenfield project, so there is no existing system to impact. But the plan creates a new system with non-trivial internal surface area, so the fields below describe the **internal** system-wide concerns the implementer needs to keep coherent across units.

- **Interaction graph.** Four threads all talk to the `StateManager` via its lock: Capture (writes detector results), Riot API (writes Riot-sourced fields), Inference worker (reads via deep-copy snapshot), TTS (reads nothing from StateManager but consumes filtered callouts). Callbacks flow: `RiotClient → main.py` (lifecycle transitions); `main.py → StateManager.reset()` on ENDING; `main.py → Inference → Filter → TTS` for each coaching tick. The critical invariant: **the StateManager lock is never held across the Ollama HTTP call.** Inference takes a deep-copy snapshot under the lock, releases, then runs inference on the copy. This is what keeps the 500 ms capture cadence from stalling on a 5–7 s LLM call.
- **Error propagation.** Errors are caught at the layer that knows how to handle them and either (a) translated into a user-facing TTS announcement ("Screen capture failed. Coach paused."), (b) logged and dropped (inference failures, TTS synth failures), or (c) escalated up to `main.py` for state-machine transitions (5 consecutive inference timeouts → pause 60 s; `CaptureExhausted` → TTS + pause capture). The escalation ladder is documented in Unit 10's Approach section. No silent failures — every error path writes a JSONL event.
- **State lifecycle risks.** Three state objects require disciplined cleanup on game end: (1) `StateManager.reset()` clears all in-memory state; (2) `logging_utils.rotate(new_game_id)` opens a new file; (3) `tts.stop_current()` drains any in-flight synthesis. All three are wired to the single `on_state_change(ACTIVE → ENDING)` callback in `main.py`. Missing any one of them would cause state leakage across games (R4 failure). Unit 10 integration tests cover the multi-game flow to guarantee this.
- **API surface parity.** The only external API surfaces are: (a) `python -m lolcoach`, (b) `python -m lolcoach.calibrate`, (c) `python -m lolcoach.benchmark`, (d) `config.yaml` schema, (e) `lol-macro-guide.exe` entry point (post-bundling, same behavior as the python module). These are all agent-controllable via CLI. No GUI, no HTTP endpoint.
- **Integration coverage.** Unit tests cover each module in isolation with mocks. The five integration tests in Unit 10 (`test_main_integration.py`) cover cross-layer behavior that mocks alone won't prove: multi-game reset, role mismatch pause, timeout escalation, graceful shutdown, and the single-callout end-to-end path. Manual E2E in `tests/manual.md` covers the parts that require real hardware (dxcam, sounddevice, Ollama, Piper audio output, real League game).
- **Unchanged invariants.** N/A — greenfield.

## Risks & Dependencies

### Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Vision LLM latency > 8 s under GPU contention | Medium | High | Week 1 Unit 6 benchmark is an explicit gate with a 4-branch decision tree (smaller model / 2000 ms capture cadence / text-only coaching / CPU offload via llama.cpp). Pivot to text-only is an honest escape hatch if vision-LLM is untenable. |
| Template matching accuracy on 200×200 minimap below usable threshold | Medium | High | Confidence threshold is configurable; low-confidence detections are dropped rather than reported as phantoms. The grounding check in Unit 8 has a fallback path: even if the detector misses a champion, the Riot roster rescues grounded LLM mentions. Tunable against real fixture frames during Week 1. |
| **Hallucinated champion callout reaches audio output (R6 violation)** | Low | High | **Structural guarantee via `DecisionFilter.grounding_check` (Unit 8).** Every callout is rejected if it mentions any champion not in detector output AND not in Riot roster. The critical test ("LLM says Ekko, detector never saw Ekko, Ekko not in game → reject") is the load-bearing test of the whole project. |
| Riot ToS enforcement against screen-reading tools | Low | High | No process or memory injection; external screen capture only via `dxcam`. README disclaimer copied verbatim from the spec. Community sentiment research is an open question in the spec — revisit if bans materialize. |
| StateManager thread-safety bug | Medium | High | Single `RLock` across all writes; snapshot always deep-copies under the lock then releases; Unit 5 has explicit concurrent-writer tests using `threading.Barrier` to force races. |
| Ollama cold start eats first callout latency (10–20 s model load) | Low | Medium | `keep_alive: -1` on every inference call + `OLLAMA_KEEP_ALIVE=-1` env var documented in README + warmup ping on coach startup (Unit 7's `warmup()` method, called once by Unit 10's `run()`). |
| `dxcam` capture failure on some Win11 builds (driver issues) | Low | Medium | 10-failure threshold before escalating to `CaptureExhausted`; TTS announcement ("Screen capture failed. Coach paused."); coach continues running in the hope of recovery. Manual smoke test on a clean Win11 VM in Unit 11. |
| PyInstaller bundle misses a transitive dependency at runtime | Medium | Medium | Iterative `hiddenimports` refinement during Unit 11 smoke testing on clean VMs. The feedback loop is: build → run → observe which import fails → add to spec → rebuild. OpenCV and sounddevice are known PyInstaller offenders. |
| Piper subprocess lifecycle issues under PyInstaller one-file mode | Low | Medium | Default to subprocess invocation (battle-tested). Python binding is the fallback only if smoke test reveals the subprocess path has issues under the PyInstaller bootstrap loader. |
| Calibration breaks on multi-monitor or ultrawide setups | Medium | Medium | MVP is manual click calibration — handles any resolution as long as the user clicks correctly. Auto-detect ROI is post-MVP (`TODOS.md`). |
| Windows Defender flags unsigned `.exe` | High | Low | README documents the "More info → Run anyway" workaround. Code signing is post-MVP (`TODOS.md`). Expected and accepted for MVP. |
| 90 % test coverage floor blocks a legitimately unrepairable branch (e.g., hardware-only dxcam path) | Medium | Low | `# pragma: no cover` allowed for hardware-only paths with a comment justifying each exclusion. Per-file coverage override in `pyproject.toml` if the overall floor gets too tight. |
| Benchmark reveals vision-LLM approach is untenable on target hardware | Low | High | Decision tree's final branch is text-only coaching (small text model on state text alone, no minimap). Loses the spatial-reasoning differentiator but ships a working coach. Accept this as a pivot; do not hide it behind optimism. |

### External Dependencies & Prerequisites

- **Python 3.11+** — developer workstation and target machines. `sys.version_info` assertion at startup is a cheap safety check.
- **Ollama installed** on the user's machine (not bundled). README links to `https://ollama.com/download`.
- **Ollama model pulled** via `ollama pull <model>` — user runs this once. README specifies the exact tag after the Week 1 benchmark resolves.
- **12–16 GB VRAM GPU** shared with League — documented as a hardware requirement.
- **Piper TTS binary** — bundled in the Windows installer and documented as a manual `pip install`-equivalent step for source installs. Version pinned in `RELEASE.md`.
- **Riot Live Client Data API** — automatically available whenever a League game is running. No API key required. Self-signed cert handled via `verify=False`.
- **Developer tooling:** Git, Python, `pip install -e .[dev]`, PyInstaller (for Unit 11), NSIS or Inno Setup (for Unit 12), a Windows 10 AND Windows 11 VM for smoke testing.

## Documentation / Operational Notes

- **README.md** evolves across units: Unit 1 creates a placeholder; Unit 6 adds the benchmark decision tree; Unit 12 expands to full public-facing docs (install paths, config reference, ToS disclaimer, audio routing, troubleshooting, GPU requirements).
- **RELEASE.md** (Unit 12) is the human-readable release checklist: version bump → tests → build exe → build installer → smoke test VMs → GitHub release → draft notes. Checked into the repo for reproducibility.
- **tests/manual.md** (Unit 10) is the manual E2E checklist: 5 flows from the spec (first-run calibration → first callout; multi-game session; non-jungle role pause; forced remake; GPU contention).
- **Structured JSONL logs** under `./logs/session-{iso8601}-game-{gameId}.jsonl` — one file per game. No size-based rotation; users delete manually. Enables post-session clip hunting: scan for high-confidence predictions, cross-reference with kill-feed events, find the moment the AI called it right.
- **Operational escalation ladder** for inference failures, documented in Unit 10 and mirrored in the coach's runtime behavior: 5 consecutive timeouts → warn + pause 60 s → 3 cycles with zero success → "Coach unavailable" + pause until next game. Honest degradation, no fake fallbacks.
- **Clip-hunting workflow** (for content creation, per origin doc's "information recovery is the product" framing): after each play session, `grep` the JSONL log for `"confidence": [89]` callouts, cross-reference with kill events in the same file, manually clip the matching moments from OBS / ShadowPlay footage.
- **Monitoring** is out of scope — no telemetry, no analytics, no crash reporter. This is a local tool.
- **No rollout plan** — GitHub release is a one-shot artifact drop. Users download and install manually.

## Phased Delivery

### Phase 1 — Foundation & Data Pipeline (Week 1)

Units 1 – 6. Deliverable: a runnable data pipeline that captures the minimap, polls the Riot API, drives the lifecycle FSM, tracks game state in a thread-safe store, and benchmarks the vision LLM end-to-end. No coaching output yet — this is the plumbing.

**Phase exit gate:** Unit 6 benchmark runs against live Ollama + live League. Implementer reviews measured latency against the spec's decision tree and commits to (a) current model, (b) smaller model, (c) reduced capture cadence, or (d) text-only coaching. Phase 2 begins with the committed choice reflected in `config.yaml.example`.

### Phase 2 — Intelligence & Voice (Week 2)

Units 7 – 10. Deliverable: the full coaching pipeline wired end-to-end. Inference turns state + minimap into structured callouts, the DecisionFilter enforces the grounding invariant, TTS speaks the surviving callouts. Main orchestration runs the 4-thread loop and handles all lifecycle transitions.

**Phase exit gate:** Play one real game with the coach running, record the session, and find the clip. This is the "Sun Eve: Play one game, clip the best moment" step from the original office-hours build order. If no coherent callouts happen in that first game, triage: is it a prompt issue (Unit 7)? A filter issue (Unit 8)? A latency issue (Unit 6 pivot)? A grounding issue (Unit 8's hallucination path rejected a legitimate callout)?

### Phase 3 — Windows Distribution (Week 3)

Units 11 – 12. Deliverable: `lol-macro-guide.exe` and `setup.exe` on GitHub releases tagged `v0.1.0`, with README + RELEASE docs, smoke-tested on clean Windows 10 and Windows 11 VMs.

**Phase exit gate:** critical path #1 from the test plan succeeds end-to-end on a clean VM with the installed setup: install → `ollama pull` → calibrate → start a real League game as jungle → hear at least one coherent callout within the first 3 minutes. If this fails, do not cut the release.

## Documentation Plan

- **Unit 1** — placeholder `README.md` with project name and a one-sentence pitch.
- **Unit 6** — benchmark instructions + decision tree appended to `README.md`.
- **Unit 10** — `tests/manual.md` checklist covering the 5 manual E2E flows.
- **Unit 11** — build-from-source instructions appended to `README.md`: `python scripts/build-exe.py` prerequisites and process.
- **Unit 12** — full public-facing `README.md` (install paths, config reference, audio routing, ToS disclaimer, troubleshooting) + `RELEASE.md` release checklist.
- **No separate user manual.** README is the single source of truth for end users. `CLAUDE.md` stays as-is for developer tooling (skill routing).

## Operational / Rollout Notes

- **No monitoring, no telemetry.** This is a local tool. Logs stay on the user's machine.
- **No migration.** Greenfield — no existing users to upgrade.
- **No feature flag.** MVP ships as one release.
- **No canary.** Distribution is GitHub release manual download.
- **Rollback plan:** users who hit issues redownload the previous release from GitHub (or uninstall via Add/Remove Programs). No server-side component to roll back.

## Sources & References

The origin documents live in gstack project storage **outside this repo** at `<gstack-projects-root>/KaiSong06-lol-macro-guide/`. Resolve `<gstack-projects-root>` per your local gstack install (typically a user-scoped directory under your home folder). The per-repo `TODOS.md` is the only cross-reference that lives inside this repo.

- **Origin document (spec):** `kaisong-main-spec-20260410-021300.md` — the condensed tech spec written immediately before this plan; authoritative for product requirements, scope boundaries, and locked-in decisions. Referenced throughout this plan as "the spec" and "origin spec".
- **Full design document:** `kaisong-main-design-20260410-011323.md` — the APPROVED, ENG CLEARED design doc with full architecture, prompt strategy, concurrency model, latency budget, error handling, and build order. The spec is a condensation of this file; read it when a per-unit approach feels under-specified.
- **Eng review test plan:** `kaisong-main-eng-review-test-plan-20260410-014400.md` — the `/plan-eng-review` output with flows, edge cases, critical paths, and manual E2E targets. Per-module test counts in that document map directly to the test scenarios in this plan's implementation units.
- **Post-MVP backlog:** `TODOS.md` at repo root — auto-detect minimap ROI, Approach C hybrid heuristics, code signing. Cross-referenced from "Deferred to Separate Tasks" above. This is the only source doc that lives inside the repo.
- **External API docs:** See "External References" section above for per-unit links to dxcam, OpenCV, Riot Data Dragon, Riot Live Client Data API, Ollama, Piper TTS, sounddevice, PyInstaller, NSIS, Inno Setup.
- **Related PRs / issues:** None. Greenfield.
