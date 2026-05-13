"""Dependency health checks for the desktop control panel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests

from lolcoach.capture import load_calibration
from lolcoach.config import Config
from lolcoach.riot_client import ALLGAMEDATA_PATH
from lolcoach.template_builder import DEFAULT_OUTPUT_DIR
from lolcoach.tts import _resolve_piper_path, _resolve_voice_path


@dataclass(frozen=True)
class HealthResult:
    """One dependency check result."""

    name: str
    ok: bool
    status: str
    detail: str = ""
    remediation: str = ""


def check_ollama(config: Config, *, session: requests.Session | None = None) -> HealthResult:
    session = session or requests.Session()
    url = f"{config.inference.ollama_host.rstrip('/')}/api/tags"
    try:
        response = session.get(url, timeout=2)
    except requests.RequestException as exc:
        return HealthResult(
            "Ollama",
            False,
            "Unavailable",
            str(exc),
            "Start Ollama and pull the configured vision model.",
        )
    if response.status_code != 200:
        return HealthResult("Ollama", False, f"HTTP {response.status_code}", url)
    try:
        payload = response.json()
    except ValueError:
        return HealthResult("Ollama", False, "Invalid response", url)
    models = payload.get("models", []) if isinstance(payload, dict) else []
    names = {str(item.get("name")) for item in models if isinstance(item, dict)}
    if config.inference.model not in names:
        return HealthResult(
            "Ollama",
            False,
            "Model missing",
            config.inference.model,
            f"Run: ollama pull {config.inference.model}",
        )
    return HealthResult("Ollama", True, "Ready", config.inference.model)


def check_riot_api(config: Config, *, session: requests.Session | None = None) -> HealthResult:
    session = session or requests.Session()
    url = f"{config.riot_api.base_url.rstrip('/')}{ALLGAMEDATA_PATH}"
    try:
        response = session.get(url, timeout=2, verify=False)
    except requests.RequestException as exc:
        return HealthResult(
            "League Client",
            False,
            "Unavailable",
            str(exc),
            "Start League and enter a game.",
        )
    if response.status_code == 200:
        return HealthResult("League Client", True, "In game", url)
    if response.status_code == 404:
        return HealthResult(
            "League Client",
            False,
            "Waiting for game",
            url,
            "Enter a live game before starting coaching.",
        )
    return HealthResult("League Client", False, f"HTTP {response.status_code}", url)


def check_templates(template_dir: Path = DEFAULT_OUTPUT_DIR) -> HealthResult:
    count = len(list(template_dir.glob("*.png"))) if template_dir.exists() else 0
    if count == 0:
        return HealthResult(
            "Champion Templates",
            False,
            "Missing",
            str(template_dir),
            "Download templates from Setup.",
        )
    return HealthResult("Champion Templates", True, "Ready", f"{count} templates")


def check_piper(config: Config) -> HealthResult:
    piper_path = _resolve_piper_path(config.tts.piper_path)
    voice_path = _resolve_voice_path(config.tts.voice_dir, config.tts.voice)
    if not piper_path.exists():
        return HealthResult(
            "Piper",
            False,
            "Binary missing",
            str(piper_path),
            "Place piper.exe in scripts/binaries/piper or configure tts.piper_path.",
        )
    if not voice_path.exists():
        return HealthResult(
            "Piper",
            False,
            "Voice missing",
            str(voice_path),
            "Place the voice .onnx file in scripts/binaries/voices or configure tts.voice_dir.",
        )
    return HealthResult("Piper", True, "Ready", f"{piper_path.name} + {voice_path.name}")


def check_calibration(config: Config, *, path: Path = Path("calibration.json")) -> HealthResult:
    if isinstance(config.capture.minimap_roi, tuple):
        return HealthResult("Calibration", True, "Manual ROI", str(config.capture.minimap_roi))
    roi = load_calibration(path)
    if roi is None:
        return HealthResult(
            "Calibration",
            False,
            "Missing",
            str(path),
            "Run calibration from Setup.",
        )
    return HealthResult("Calibration", True, "Ready", str(roi.as_tuple()))


def run_all_checks(config: Config) -> tuple[HealthResult, ...]:
    """Run all UI-visible health checks."""

    return (
        check_calibration(config),
        check_templates(),
        check_ollama(config),
        check_riot_api(config),
        check_piper(config),
    )
