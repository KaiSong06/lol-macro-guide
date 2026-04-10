"""Entry point for ``python -m lolcoach``.

Wired to the full orchestration loop in Unit 10. Prior to that, this module
prints a placeholder message so ``python -m lolcoach`` does not crash.
"""

from __future__ import annotations

import sys


def main() -> int:
    sys.stdout.write(
        "lolcoach is under development — full orchestration lands in Unit 10.\n"
        "For now, try the subcommands once they exist: "
        "python -m lolcoach.calibrate or python -m lolcoach.benchmark.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
