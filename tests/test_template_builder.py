"""Tests for ``lolcoach.template_builder`` — the Data Dragon downloader.

Scenarios:

1. Integration: mocked Data Dragon → ``download_all`` writes N PNG files.
2. Idempotency: second run with the same output dir skips existing files.
3. Force: second run with ``force=True`` re-downloads.
4. HTTP error on a single portrait is logged and skipped, others continue.
5. Decode failure (portrait bytes are not a valid image) is logged and skipped.
6. ``main`` CLI happy path exits 0 and prints a summary.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import requests_mock as rm_module

from lolcoach.template_builder import (
    CHAMPION_DATA_URL_TEMPLATE,
    DATA_DRAGON_REALMS_URL,
    PORTRAIT_URL_TEMPLATE,
    download_all,
    main,
)

PATCH = "14.5.1"
CHAMPIONS = ["Ahri", "LeeSin", "MonkeyKing"]  # 3 sample IDs


def _make_fake_png_bytes(color: int = 100) -> bytes:
    """Return valid PNG-encoded bytes for a tiny solid-color image."""
    img = np.full((120, 120, 3), color, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return bytes(buf.tobytes())


def _register_realms_and_champs(m: rm_module.Mocker) -> None:
    m.get(DATA_DRAGON_REALMS_URL, json={"v": PATCH})
    m.get(
        CHAMPION_DATA_URL_TEMPLATE.format(patch=PATCH),
        json={"data": {name: {"id": name} for name in CHAMPIONS}},
    )


def _register_portrait(m: rm_module.Mocker, name: str, content: bytes) -> None:
    m.get(
        PORTRAIT_URL_TEMPLATE.format(patch=PATCH, name=name),
        content=content,
    )


# ---------------------------------------------------------------------------
# Happy path: downloads N templates to N PNG files
# ---------------------------------------------------------------------------
def test_download_all_writes_expected_png_files(tmp_path: Path) -> None:
    output = tmp_path / "templates"
    with rm_module.Mocker() as m:
        _register_realms_and_champs(m)
        for name in CHAMPIONS:
            _register_portrait(m, name, _make_fake_png_bytes(color=100))

        downloaded, skipped = download_all(output)

    assert downloaded == 3
    assert skipped == 0
    for name in CHAMPIONS:
        p = output / f"{name}.png"
        assert p.exists()
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        assert img is not None
        assert img.shape == (64, 64, 3)  # resized


# ---------------------------------------------------------------------------
# Idempotency: second run skips existing files
# ---------------------------------------------------------------------------
def test_download_all_is_idempotent(tmp_path: Path) -> None:
    output = tmp_path / "templates"
    with rm_module.Mocker() as m:
        _register_realms_and_champs(m)
        for name in CHAMPIONS:
            _register_portrait(m, name, _make_fake_png_bytes(color=100))

        downloaded1, skipped1 = download_all(output)
        downloaded2, skipped2 = download_all(output)

    assert (downloaded1, skipped1) == (3, 0)
    assert (downloaded2, skipped2) == (0, 3)  # all 3 skipped second time


def test_download_all_force_re_downloads(tmp_path: Path) -> None:
    output = tmp_path / "templates"
    with rm_module.Mocker() as m:
        _register_realms_and_champs(m)
        for name in CHAMPIONS:
            _register_portrait(m, name, _make_fake_png_bytes(color=100))

        download_all(output)  # first run — writes 3
        downloaded2, skipped2 = download_all(output, force=True)

    assert downloaded2 == 3
    assert skipped2 == 0


# ---------------------------------------------------------------------------
# HTTP error on a single portrait: log and skip, continue with others
# ---------------------------------------------------------------------------
def test_download_all_skips_portrait_on_http_error(tmp_path: Path) -> None:
    output = tmp_path / "templates"
    with rm_module.Mocker() as m:
        _register_realms_and_champs(m)
        _register_portrait(m, "Ahri", _make_fake_png_bytes(color=50))
        # LeeSin returns 500
        m.get(
            PORTRAIT_URL_TEMPLATE.format(patch=PATCH, name="LeeSin"),
            status_code=500,
        )
        _register_portrait(m, "MonkeyKing", _make_fake_png_bytes(color=200))

        downloaded, skipped = download_all(output)

    assert downloaded == 2
    assert skipped == 1
    assert (output / "Ahri.png").exists()
    assert not (output / "LeeSin.png").exists()
    assert (output / "MonkeyKing.png").exists()


# ---------------------------------------------------------------------------
# Decode failure: portrait bytes are not a valid image
# ---------------------------------------------------------------------------
def test_download_all_skips_undecodable_portrait(tmp_path: Path) -> None:
    output = tmp_path / "templates"
    with rm_module.Mocker() as m:
        _register_realms_and_champs(m)
        _register_portrait(m, "Ahri", _make_fake_png_bytes(color=50))
        # LeeSin returns garbage bytes
        _register_portrait(m, "LeeSin", b"definitely not a PNG")
        _register_portrait(m, "MonkeyKing", _make_fake_png_bytes(color=200))

        downloaded, skipped = download_all(output)

    assert downloaded == 2  # Ahri + MonkeyKing
    assert skipped == 1
    assert (output / "Ahri.png").exists()
    assert not (output / "LeeSin.png").exists()
    assert (output / "MonkeyKing.png").exists()


# ---------------------------------------------------------------------------
# Limit flag
# ---------------------------------------------------------------------------
def test_download_all_with_limit_downloads_subset(tmp_path: Path) -> None:
    output = tmp_path / "templates"
    with rm_module.Mocker() as m:
        _register_realms_and_champs(m)
        # Only register the first two — Limit=2 should never request the third
        _register_portrait(m, "Ahri", _make_fake_png_bytes())
        _register_portrait(m, "LeeSin", _make_fake_png_bytes())

        downloaded, skipped = download_all(output, limit=2)

    assert downloaded == 2
    assert (output / "Ahri.png").exists()
    assert (output / "LeeSin.png").exists()
    assert not (output / "MonkeyKing.png").exists()


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------
def test_main_happy_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "templates"
    with rm_module.Mocker() as m:
        _register_realms_and_champs(m)
        for name in CHAMPIONS:
            _register_portrait(m, name, _make_fake_png_bytes())

        exit_code = main(["--output", str(output), "--limit", "3"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Template build complete" in captured.out
    assert "3 new" in captured.out


def test_main_returns_error_on_realms_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "templates"
    with rm_module.Mocker() as m:
        m.get(DATA_DRAGON_REALMS_URL, status_code=500)
        exit_code = main(["--output", str(output)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Data Dragon" in captured.err
