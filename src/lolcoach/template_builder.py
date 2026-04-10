"""Champion template downloader for the lolcoach detector.

Fetches current-patch champion portraits from Riot Data Dragon, resizes them
to the minimap-icon shape (64×64), and writes them to the user's template
directory. Idempotent by default: templates that already exist on disk are
skipped unless ``--force`` is passed.

Usage::

    python -m lolcoach.template_builder
    python -m lolcoach.template_builder --output ~/.lolcoach/templates
    python -m lolcoach.template_builder --limit 5  # smoke-test with 5 champs

The plan originally placed this at ``scripts/build-champion-templates.py``;
keeping it inside ``lolcoach`` makes it importable and directly testable with
``requests-mock``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

DATA_DRAGON_REALMS_URL = "https://ddragon.leagueoflegends.com/realms/na.json"
CHAMPION_DATA_URL_TEMPLATE = (
    "https://ddragon.leagueoflegends.com/cdn/{patch}/data/en_US/champion.json"
)
PORTRAIT_URL_TEMPLATE = (
    "https://ddragon.leagueoflegends.com/cdn/{patch}/img/champion/{name}.png"
)

#: Target size for minimap-icon templates. Matches the detector's expectation.
TEMPLATE_SIZE = (64, 64)

DEFAULT_OUTPUT_DIR = Path.home() / ".lolcoach" / "templates"


def get_current_patch(session: requests.Session) -> str:
    """Return the current League patch string from Riot's NA realms endpoint."""
    resp = session.get(DATA_DRAGON_REALMS_URL, timeout=10)
    resp.raise_for_status()
    payload: dict[str, Any] = resp.json()
    return str(payload["v"])


def list_champion_ids(session: requests.Session, patch: str) -> list[str]:
    """Return the list of champion data-dragon IDs (e.g., ``LeeSin``, ``MonkeyKing``)."""
    resp = session.get(
        CHAMPION_DATA_URL_TEMPLATE.format(patch=patch), timeout=10
    )
    resp.raise_for_status()
    payload: dict[str, Any] = resp.json()
    return sorted(payload["data"].keys())


def download_template(
    session: requests.Session,
    patch: str,
    name: str,
    output_dir: Path,
    *,
    force: bool = False,
) -> bool:
    """Download one champion portrait and write it to ``output_dir``.

    Returns ``True`` if a new file was written, ``False`` if the file
    already existed (and ``force`` was not set) or if decoding failed.
    """
    out_path = output_dir / f"{name}.png"
    if out_path.exists() and not force:
        return False

    url = PORTRAIT_URL_TEMPLATE.format(patch=patch, name=name)
    resp = session.get(url, timeout=10)
    resp.raise_for_status()

    arr = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("failed to decode portrait for %s from %s", name, url)
        return False

    resized = cv2.resize(img, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), resized)
    return True


def download_all(
    output_dir: Path,
    *,
    limit: int | None = None,
    force: bool = False,
    session: requests.Session | None = None,
) -> tuple[int, int]:
    """Download every champion template. Returns ``(downloaded, skipped)``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    owned_session = session is None
    session = session or requests.Session()
    try:
        patch = get_current_patch(session)
        ids = list_champion_ids(session, patch)
        if limit is not None:
            ids = ids[:limit]

        downloaded = 0
        skipped = 0
        for name in ids:
            try:
                wrote = download_template(
                    session, patch, name, output_dir, force=force
                )
            except requests.RequestException as exc:
                logger.warning("skipped %s due to %s", name, exc)
                skipped += 1
                continue
            if wrote:
                downloaded += 1
            else:
                skipped += 1
        return downloaded, skipped
    finally:
        if owned_session:
            session.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lolcoach.template_builder",
        description="Download League champion portraits for the lolcoach detector.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory (default: ~/.lolcoach/templates)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Download at most N templates (useful for smoke tests)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download templates even if they already exist on disk",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        downloaded, skipped = download_all(
            args.output, limit=args.limit, force=args.force
        )
    except requests.RequestException as exc:
        print(f"Data Dragon request failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Template build complete: {downloaded} new, {skipped} skipped. "
        f"Output dir: {args.output}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
