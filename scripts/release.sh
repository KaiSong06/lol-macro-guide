#!/usr/bin/env bash
set -euo pipefail

python -m pytest
python -m ruff check src tests
python scripts/build-exe.py

if command -v makensis >/dev/null 2>&1; then
  makensis scripts/build-installer.nsi
else
  echo "makensis not found; install NSIS and run: makensis scripts/build-installer.nsi" >&2
  exit 1
fi

cat <<'MSG'
Release artifacts built.
Next manual steps:
1. Smoke-test on clean Windows 10 and Windows 11 VMs.
2. Create GitHub release v0.1.0.
3. Attach dist/lol-macro-guide.exe and dist/lol-macro-guide-setup.exe.
MSG
