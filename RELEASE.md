# Release Checklist

## Prerequisites

- Windows build machine with Python 3.11 or 3.12.
- PyInstaller installed in the active environment.
- NSIS installed and `makensis` available on `PATH`.
- Piper Windows binary placed under `scripts/binaries/piper/`.
- `en_US-amy-medium.onnx` and its `.json` sidecar placed under
  `scripts/binaries/voices/`.
- Ollama installed separately for manual smoke tests.

## Build

```bash
python -m pytest
python -m ruff check src tests
python scripts/build-exe.py
makensis scripts/build-installer.nsi
```

Or run:

```bash
scripts/release.sh
```

## Manual Verification

Complete every flow in `tests/manual.md` on clean Windows 10 and Windows 11
VMs. Do not cut `v0.1.0` until the installed app produces a coherent jungle
callout within 3 minutes of game start from the desktop control panel.
Verify the Settings auto-launch toggle creates and removes the per-user
`lolcoach-auto-launch` Scheduled Task, and that a match launch opens exactly
one tray instance with coaching started.

## Publish

1. Tag `v0.1.0`.
2. Create a GitHub release.
3. Attach `dist/lol-macro-guide.exe`.
4. Attach `dist/lol-macro-guide-setup.exe`.
5. Include notes that Ollama is a separate install and the binaries are
   unsigned.
