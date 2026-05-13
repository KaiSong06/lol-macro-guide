# lol-macro-guide

Real-time, local, vision-grounded jungle coach for League of Legends.
Windows-only and voice-only for the MVP.

The coach watches the minimap, combines deterministic champion-icon detection
with Riot Live Client Data, asks a local Ollama vision model for spatial
reasoning, filters unsafe callouts, and speaks short coaching lines through
Piper TTS. The default app shell is a native desktop control panel for setup,
health checks, start/stop, settings, and logs; the in-game experience remains
voice-only.

## Requirements

- Windows 10 or 11.
- Python 3.11 or 3.12 for source installs.
- League of Legends client in an active game.
- Ollama with a vision model. Default config uses `gemma3:4b`.
- Piper binary plus `en_US-amy-medium` voice for source builds.
- PySide6 for the desktop control panel.

Ollama is not bundled. Install it separately, then run:

```bash
ollama pull gemma3:4b
```

Set `OLLAMA_KEEP_ALIVE=-1` in your shell or Windows environment to keep the
model loaded between calls.

## Source Install

```bash
pip install -e ".[dev]"
python -m lolcoach.template_builder
python -m lolcoach.calibrate --corners X1,Y1,X2,Y2
python -m lolcoach
```

`python -m lolcoach` launches the desktop control panel. For the console-only
runtime, use:

```bash
python -m lolcoach --headless
```

On Windows, the Settings page can enable match-start auto launch. When enabled,
lolcoach registers a per-user Scheduled Task that starts a hidden monitor at
logon. The monitor watches for the in-game `League of Legends.exe` process and
opens lolcoach minimized to the tray with coaching started automatically.

For interactive calibration on Windows, run:

```bash
python -m lolcoach.calibrate
```

Click the minimap top-left and bottom-right corners. This writes
`calibration.json`.

## Configuration

Copy `config.yaml.example` to `config.yaml` and edit values as needed. Useful
fields:

- `capture.minimap_roi`: `auto` uses `calibration.json`; `[x, y, w, h]` uses a manual ROI.
- `inference.model`: Ollama model tag.
- `inference.timeout_ms`: per-call inference timeout.
- `decisions.confidence_threshold`: minimum 1-10 confidence to speak.
- `tts.piper_path`: explicit path to `piper.exe`, or `auto` for packaged layout.
- `tts.voice_dir`: directory containing `<voice>.onnx`, or `auto`.
- `logging.directory`: JSONL output directory.

## Benchmark Gate

Run the Week-1 latency gate on the target Windows machine while League is open:

```bash
python -m lolcoach.benchmark --fixture tests/fixtures/minimap_sample.png --model gemma3:4b --n 20
```

Decision tree:

- `p95 <= 5s`: keep the model.
- `5s < p95 <= 8s`: acceptable for MVP.
- `8s < p95 <= 12s`: try a smaller model or slower capture cadence.
- `p95 > 12s`: pivot away from vision callouts for MVP.

## Build

Place Piper release files manually:

- `scripts/binaries/piper/piper.exe`
- `scripts/binaries/voices/en_US-amy-medium.onnx`
- `scripts/binaries/voices/en_US-amy-medium.onnx.json`

Then build:

```bash
python scripts/build-exe.py
makensis scripts/build-installer.nsi
```

See `RELEASE.md` for the full release checklist.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

## Notes

This project uses external screen capture and Riot's local Live Client Data API.
It does not inject into the League process, modify game memory, or use a cloud
model. The MVP binaries are unsigned, so Windows may show a SmartScreen warning.
