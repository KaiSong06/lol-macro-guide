"""Windows match-start auto-launch support.

The feature is intentionally Windows-only: a per-user Scheduled Task starts a
tiny monitor at logon, the monitor watches for the in-game League process, and
then launches the main lolcoach UI minimized with coaching started.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TASK_NAME = "lolcoach-auto-launch"
STATE_FILE_NAME = "autolaunch.json"
LEAGUE_PROCESS_NAME = "League of Legends.exe"
MAIN_MUTEX_NAME = "Global\\lolcoach-main-ui"
MONITOR_MUTEX_NAME = "Global\\lolcoach-match-monitor"
ERROR_ALREADY_EXISTS = 183

ProcessChecker = Callable[[str], bool]
Launcher = Callable[[list[str]], None]
Sleeper = Callable[[float], None]

_MAIN_MUTEX_HANDLE: int | None = None
_MONITOR_MUTEX_HANDLE: int | None = None


@dataclass(frozen=True)
class AutoLaunchStatus:
    """UI-facing auto-launch state."""

    supported: bool
    enabled: bool
    detail: str


def is_supported() -> bool:
    """Return whether this platform can register the Windows auto-launch task."""

    return sys.platform == "win32"


def state_path() -> Path:
    """Return the per-user auto-launch state path."""

    root = os.environ.get("LOCALAPPDATA")
    if root:
        return Path(root) / "lolcoach" / STATE_FILE_NAME
    return Path.home() / "AppData" / "Local" / "lolcoach" / STATE_FILE_NAME


def read_state(path: Path | None = None) -> dict[str, Any]:
    """Read persisted auto-launch state, returning disabled on any problem."""

    path = path or state_path()
    if not path.exists():
        return {"enabled": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"enabled": False}
    if not isinstance(payload, dict):
        return {"enabled": False}
    return payload


def write_state(
    *,
    enabled: bool,
    config_path: Path,
    path: Path | None = None,
) -> None:
    """Persist auto-launch state for the logon monitor."""

    path = path or state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"enabled": enabled, "config_path": str(config_path)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_status(path: Path | None = None) -> AutoLaunchStatus:
    """Return current auto-launch status for the Settings UI."""

    if not is_supported():
        return AutoLaunchStatus(False, False, "Unavailable on this platform")
    state = read_state(path)
    enabled = bool(state.get("enabled"))
    return AutoLaunchStatus(True, enabled, "Enabled" if enabled else "Disabled")


def set_enabled(config_path: Path, enabled: bool) -> AutoLaunchStatus:
    """Enable or disable Windows match-start auto-launch."""

    if not is_supported():
        return AutoLaunchStatus(False, False, "Unavailable on this platform")
    if enabled:
        write_state(enabled=True, config_path=config_path)
        register_task(config_path)
        start_monitor(config_path)
        return AutoLaunchStatus(True, True, "Enabled")
    write_state(enabled=False, config_path=config_path)
    unregister_task()
    return AutoLaunchStatus(True, False, "Disabled")


def build_monitor_command(config_path: Path) -> list[str]:
    """Build argv for the background monitor process."""

    command = _base_command()
    command.extend(["--monitor-league", "--config", str(config_path)])
    return command


def build_app_launch_command(config_path: Path) -> list[str]:
    """Build argv for the auto-launched main UI."""

    command = _base_command()
    command.extend(["--minimized", "--auto-start", "--config", str(config_path)])
    return command


def register_task(config_path: Path) -> None:
    """Create or replace the per-user logon Scheduled Task."""

    command = subprocess.list2cmdline(build_monitor_command(config_path))
    subprocess.run(
        [
            "schtasks",
            "/Create",
            "/TN",
            TASK_NAME,
            "/TR",
            command,
            "/SC",
            "ONLOGON",
            "/RL",
            "LIMITED",
            "/F",
        ],
        check=True,
        capture_output=True,
        text=True,
        creationflags=_creationflags(),
    )


def unregister_task() -> None:
    """Delete the per-user logon Scheduled Task if it exists."""

    subprocess.run(
        ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
        check=False,
        capture_output=True,
        text=True,
        creationflags=_creationflags(),
    )


def start_monitor(config_path: Path) -> None:
    """Start the background monitor immediately after the user enables it."""

    subprocess.Popen(  # noqa: S603
        build_monitor_command(config_path),
        close_fds=True,
        creationflags=_creationflags(),
    )


def monitor_league(
    *,
    config_path: Path,
    process_checker: ProcessChecker | None = None,
    launcher: Launcher | None = None,
    sleeper: Sleeper = time.sleep,
    poll_seconds: float = 5.0,
    max_iterations: int | None = None,
) -> int:
    """Run the invisible League match monitor.

    The monitor launches lolcoach on the rising edge of the game process:
    absent -> present launches once, then present -> absent arms the next
    launch. It exits when the persisted state is disabled.
    """

    if is_supported() and not claim_monitor_instance():
        return 0

    process_checker = process_checker or is_process_running
    launcher = launcher or _launch
    launched_for_current_match = False
    iterations = 0

    while True:
        state = read_state()
        if not state.get("enabled", False):
            return 0

        active = process_checker(LEAGUE_PROCESS_NAME)
        if active and not launched_for_current_match:
            launcher(build_app_launch_command(Path(state.get("config_path") or config_path)))
            launched_for_current_match = True
        elif not active:
            launched_for_current_match = False

        iterations += 1
        if max_iterations is not None and iterations >= max_iterations:
            return 0
        sleeper(poll_seconds)


def is_process_running(process_name: str) -> bool:
    """Return whether a Windows process image name is currently running."""

    response = subprocess.run(
        [
            "tasklist",
            "/FI",
            f"IMAGENAME eq {process_name}",
            "/FO",
            "CSV",
            "/NH",
        ],
        check=False,
        capture_output=True,
        text=True,
        creationflags=_creationflags(),
    )
    output = response.stdout.lower()
    return process_name.lower() in output


def claim_main_instance() -> bool:
    """Claim the single main-UI instance mutex on Windows."""

    global _MAIN_MUTEX_HANDLE
    ok, handle = _claim_mutex(MAIN_MUTEX_NAME)
    if ok:
        _MAIN_MUTEX_HANDLE = handle
    return ok


def claim_monitor_instance() -> bool:
    """Claim the single monitor instance mutex on Windows."""

    global _MONITOR_MUTEX_HANDLE
    ok, handle = _claim_mutex(MONITOR_MUTEX_NAME)
    if ok:
        _MONITOR_MUTEX_HANDLE = handle
    return ok


def _base_command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, "-m", "lolcoach"]


def _launch(command: list[str]) -> None:
    subprocess.Popen(command, close_fds=True, creationflags=_creationflags())  # noqa: S603


def _creationflags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _claim_mutex(name: str) -> tuple[bool, int | None]:
    if not is_supported():
        return True, None
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.CreateMutexW(None, False, name)
    if not handle:
        return False, None
    if kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
        kernel32.CloseHandle(handle)
        return False, None
    return True, handle
