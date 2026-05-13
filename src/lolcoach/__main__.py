"""Entry point for ``python -m lolcoach``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="python -m lolcoach")
    parser.add_argument("--headless", action="store_true", help="run the console coach runtime")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument(
        "--monitor-league",
        action="store_true",
        help="run the Windows match monitor",
    )
    parser.add_argument("--minimized", action="store_true", help="start the UI minimized to tray")
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="start coaching after the UI opens",
    )
    args, _unknown = parser.parse_known_args(argv)

    if args.monitor_league:
        from pathlib import Path

        from lolcoach.autolaunch import monitor_league

        return monitor_league(config_path=Path(args.config))

    if args.headless:
        from lolcoach.main import run

        return run(["--config", args.config])

    from lolcoach.autolaunch import claim_main_instance

    if not claim_main_instance():
        return 0

    from lolcoach.ui.app import run_ui

    ui_args = ["--config", args.config]
    if args.minimized:
        ui_args.append("--minimized")
    if args.auto_start:
        ui_args.append("--auto-start")
    return run_ui(ui_args)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
