"""Tests for dependency health checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from lolcoach.config import Config, TtsConfig
from lolcoach.health import (
    check_calibration,
    check_ollama,
    check_piper,
    check_riot_api,
    check_templates,
)


class FakeResponse:
    def __init__(self, status_code: int, payload: object | None = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        if self._payload is None:
            raise ValueError("bad json")
        return self._payload


class FakeSession:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self.response = response

    def get(self, *_args: object, **_kwargs: object) -> FakeResponse:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_check_ollama_ready_when_model_present() -> None:
    result = check_ollama(
        Config(),
        session=FakeSession(FakeResponse(200, {"models": [{"name": "gemma3:4b"}]})),  # type: ignore[arg-type]
    )

    assert result.ok is True
    assert result.status == "Ready"


def test_check_ollama_reports_missing_model() -> None:
    result = check_ollama(
        Config(),
        session=FakeSession(FakeResponse(200, {"models": [{"name": "other"}]})),  # type: ignore[arg-type]
    )

    assert result.ok is False
    assert result.status == "Model missing"
    assert "ollama pull" in result.remediation


def test_check_riot_api_treats_404_as_waiting_for_game() -> None:
    result = check_riot_api(Config(), session=FakeSession(FakeResponse(404)))  # type: ignore[arg-type]

    assert result.ok is False
    assert result.status == "Waiting for game"


def test_check_riot_api_reports_request_error() -> None:
    result = check_riot_api(
        Config(),
        session=FakeSession(requests.ConnectionError("nope")),  # type: ignore[arg-type]
    )

    assert result.ok is False
    assert result.status == "Unavailable"


def test_check_templates_counts_pngs(tmp_path: Path) -> None:
    (tmp_path / "LeeSin.png").write_bytes(b"png")

    result = check_templates(tmp_path)

    assert result.ok is True
    assert "1 templates" in result.detail


def test_check_piper_requires_binary_and_voice(tmp_path: Path) -> None:
    piper = tmp_path / "piper.exe"
    voices = tmp_path / "voices"
    voices.mkdir()
    piper.write_text("fake", encoding="utf-8")
    (voices / "en_US-amy-medium.onnx").write_text("fake", encoding="utf-8")

    result = check_piper(
        Config(tts=TtsConfig(piper_path=str(piper), voice_dir=str(voices)))
    )

    assert result.ok is True


def test_check_calibration_reads_saved_roi(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"x": 1, "y": 2, "w": 3, "h": 4}), encoding="utf-8")

    result = check_calibration(Config(), path=path)

    assert result.ok is True
    assert "(1, 2, 3, 4)" in result.detail


def test_check_calibration_reports_missing(tmp_path: Path) -> None:
    result = check_calibration(Config(), path=tmp_path / "missing.json")

    assert result.ok is False
    assert result.status == "Missing"


@pytest.mark.parametrize("status", [500, 503])
def test_check_ollama_reports_http_errors(status: int) -> None:
    result = check_ollama(Config(), session=FakeSession(FakeResponse(status)))  # type: ignore[arg-type]

    assert result.ok is False
    assert result.status == f"HTTP {status}"
