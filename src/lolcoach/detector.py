"""Deterministic OpenCV template-matching detector for League champions.

The detector is the **primary** enemy-position source — Riot's Live Client
Data API does not expose enemy champion coordinates, so template matching
against a library of champion portraits is the only way to know where
enemies are on the minimap. Output feeds two downstream consumers:

1. The ``StateManager`` (Unit 5) which records ``enemy_jungler_last_seen``
   whenever an enemy with ``position == "JUNGLE"`` is detected.
2. The ``DecisionFilter`` grounding check (Unit 8) which rejects any LLM
   callout naming a champion the detector never saw (and that isn't in the
   Riot roster).

Detection is deterministic: template matching either finds an icon with
confidence above threshold or it doesn't. There is no hallucination path.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Default confidence floor for ``cv2.matchTemplate`` scores (TM_CCOEFF_NORMED).
DEFAULT_CONFIDENCE_THRESHOLD = 0.85

#: Default IoU threshold for per-champion non-maximum suppression.
DEFAULT_NMS_IOU_THRESHOLD = 0.3

# 9-region quadrant grid derived from normalized [0, 1]² minimap coordinates.
# Row-major: row 0 is the top of the minimap, row 2 is the bottom.
_QUADRANT_GRID: tuple[tuple[str, str, str], ...] = (
    ("top_lane", "top_river", "base_red"),
    ("top_jungle", "mid", "bot_jungle"),
    ("base_blue", "bot_river", "bot_lane"),
)


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChampionDetection:
    """A single detected champion on a single frame."""

    name: str
    team: str  # "ally" or "enemy"
    position_norm: tuple[float, float]
    quadrant: str
    confidence: float


@dataclass(frozen=True)
class DetectionResult:
    """The set of detections for one frame plus the wall-clock timestamp."""

    champions: tuple[ChampionDetection, ...]
    detected_at: float


# ---------------------------------------------------------------------------
# Quadrant math (pure function)
# ---------------------------------------------------------------------------
def quadrant_of(x_norm: float, y_norm: float) -> str:
    """Map a normalized minimap position to one of 9 named quadrants.

    Inputs are expected to be in ``[0, 1]`` — values outside this range are
    clamped to the nearest edge region rather than raising.
    """
    col = 0 if x_norm < 1 / 3 else (1 if x_norm < 2 / 3 else 2)
    row = 0 if y_norm < 1 / 3 else (1 if y_norm < 2 / 3 else 2)
    return _QUADRANT_GRID[row][col]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class Detector:
    """Load a directory of champion-portrait PNGs and detect them in frames.

    Template loading happens once at construction. Corrupt templates are
    logged and skipped. The :attr:`known_champions` property exposes the
    set of successfully-loaded template names — useful for the grounding
    check in Unit 8 and for diagnostics.
    """

    def __init__(
        self,
        template_dir: Path,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        nms_iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._nms_iou_threshold = nms_iou_threshold
        self._templates: dict[str, np.ndarray] = {}
        self._last_seen_champions: frozenset[str] = frozenset()
        self._load_templates(Path(template_dir))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def known_champions(self) -> frozenset[str]:
        """Names of all templates successfully loaded at construction."""
        return frozenset(self._templates)

    @property
    def last_seen_champions(self) -> frozenset[str]:
        """Names of champions detected on the most recent frame.

        Used by the DecisionFilter grounding check in Unit 8.
        """
        return self._last_seen_champions

    def detect(
        self,
        frame: np.ndarray,
        ally_names: Collection[str],
        enemy_names: Collection[str],
    ) -> DetectionResult:
        """Run template matching against *frame* and return grounded detections.

        Champions whose templates aren't loaded, whose matches are below the
        confidence threshold, or whose names are in neither ally nor enemy
        rosters are dropped. NMS collapses overlapping hits per champion.
        """
        ally_set = set(ally_names)
        enemy_set = set(enemy_names)

        raw_hits_by_name: dict[str, list[dict[str, float | int]]] = {}
        for name, template in self._templates.items():
            hits = self._match_template(frame, template)
            if hits:
                raw_hits_by_name[name] = hits

        detections: list[ChampionDetection] = []
        frame_h, frame_w = frame.shape[:2]
        for name, hits in raw_hits_by_name.items():
            if name in ally_set:
                team = "ally"
            elif name in enemy_set:
                team = "enemy"
            else:
                logger.info(
                    "detector dropping unknown champion %r not in any roster",
                    name,
                )
                continue

            for hit in _nms(hits, self._nms_iou_threshold):
                cx = float(hit["x"]) + float(hit["w"]) / 2.0
                cy = float(hit["y"]) + float(hit["h"]) / 2.0
                x_norm = cx / frame_w
                y_norm = cy / frame_h
                detections.append(
                    ChampionDetection(
                        name=name,
                        team=team,
                        position_norm=(x_norm, y_norm),
                        quadrant=quadrant_of(x_norm, y_norm),
                        confidence=float(hit["confidence"]),
                    )
                )

        self._last_seen_champions = frozenset(d.name for d in detections)
        return DetectionResult(
            champions=tuple(detections),
            detected_at=time.time(),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _load_templates(self, template_dir: Path) -> None:
        if not template_dir.exists():
            logger.warning("detector template directory %s does not exist", template_dir)
            return
        for path in sorted(template_dir.glob("*.png")):
            name = path.stem
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                logger.warning("detector could not read template %s", path)
                continue
            self._templates[name] = img

    def _match_template(
        self, frame: np.ndarray, template: np.ndarray
    ) -> list[dict[str, float | int]]:
        """Return raw hits above the confidence threshold for one template."""
        frame_h, frame_w = frame.shape[:2]
        tmpl_h, tmpl_w = template.shape[:2]
        if tmpl_h > frame_h or tmpl_w > frame_w:
            return []

        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= self._confidence_threshold)
        hits: list[dict[str, float | int]] = []
        for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
            hits.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "w": int(tmpl_w),
                    "h": int(tmpl_h),
                    "confidence": float(result[y, x]),
                }
            )
        return hits


# ---------------------------------------------------------------------------
# Non-maximum suppression helpers
# ---------------------------------------------------------------------------
def _nms(
    hits: list[dict[str, float | int]],
    iou_threshold: float,
) -> list[dict[str, float | int]]:
    if not hits:
        return []
    sorted_hits = sorted(hits, key=lambda h: float(h["confidence"]), reverse=True)
    kept: list[dict[str, float | int]] = []
    for hit in sorted_hits:
        if any(_iou(hit, k) > iou_threshold for k in kept):
            continue
        kept.append(hit)
    return kept


def _iou(a: dict[str, float | int], b: dict[str, float | int]) -> float:
    ax1, ay1 = int(a["x"]), int(a["y"])
    ax2, ay2 = ax1 + int(a["w"]), ay1 + int(a["h"])
    bx1, by1 = int(b["x"]), int(b["y"])
    bx2, by2 = bx1 + int(b["w"]), by1 + int(b["h"])

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = int(a["w"]) * int(a["h"])
    b_area = int(b["w"]) * int(b["h"])
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union
