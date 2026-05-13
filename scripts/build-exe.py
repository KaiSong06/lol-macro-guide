"""Build the lolcoach Windows executable with PyInstaller."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "scripts" / "build.spec"


def main() -> int:
    if not SPEC.exists():
        print(f"Missing PyInstaller spec: {SPEC}", file=sys.stderr)
        return 2
    cmd = [sys.executable, "-m", "PyInstaller", str(SPEC), "--noconfirm"]
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
