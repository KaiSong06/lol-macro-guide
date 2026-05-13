"""Decision filter for LLM callouts before they reach TTS.

The inference parser guarantees shape: anything passed here is a well-formed
``Callout``. This module enforces product/runtime policy:

* stale inference results are dropped
* low-confidence callouts are dropped
* global cooldown and per ``(category, target_lane)`` dedup are enforced
* champion names mentioned by the LLM must be grounded in the live roster or
  in the detector result for the frame the model saw

The champion grounding check is intentionally binary. It does not use fuzzy
matching or thresholds because R6 is a structural guarantee: a champion that
is not in the game must never reach audio output.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass

from lolcoach.config import DecisionsConfig
from lolcoach.detector import DetectionResult
from lolcoach.inference import Callout
from lolcoach.state import GameState


@dataclass(frozen=True)
class FilterDecision:
    """Result of applying the decision filter to one callout."""

    accepted: bool
    reason: str


class DecisionFilter:
    """Gate LLM callouts before they reach TTS."""

    def __init__(
        self,
        config: DecisionsConfig,
        *,
        known_champions: Collection[str] = (),
        clock: Callable[[], float] | None = None,
    ) -> None:
        if config.confidence_threshold < 1 or config.confidence_threshold > 10:
            raise ValueError("confidence_threshold must be in [1, 10]")
        if config.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0")
        if config.dedup_window_seconds < 0:
            raise ValueError("dedup_window_seconds must be >= 0")
        if config.staleness_threshold_seconds < 0:
            raise ValueError("staleness_threshold_seconds must be >= 0")

        self._config = config
        self._clock = clock or time.time
        self._known_champions = frozenset(known_champions)
        self._last_spoken_at: float | None = None
        self._recent_spoken: dict[tuple[str, str], float] = {}

    def should_speak(
        self,
        callout: Callout,
        state: GameState,
        detections: DetectionResult,
    ) -> FilterDecision:
        """Return whether *callout* may be sent to TTS."""

        now = self._clock()
        self._evict_recent(now)

        age = now - callout.generated_at
        if age > self._config.staleness_threshold_seconds:
            return FilterDecision(False, "stale")

        if callout.confidence < self._config.confidence_threshold:
            return FilterDecision(False, "low_confidence")

        if (
            self._last_spoken_at is not None
            and now - self._last_spoken_at < self._config.cooldown_seconds
        ):
            return FilterDecision(False, "cooldown")

        key = (callout.category, callout.target_lane)
        if key in self._recent_spoken:
            return FilterDecision(False, "dedup")

        ungrounded = self._ungrounded_mentions(callout, state, detections)
        if ungrounded:
            names = ",".join(sorted(ungrounded))
            return FilterDecision(False, f"ungrounded_champion:{names}")

        return FilterDecision(True, "accepted")

    def record_spoken(self, callout: Callout) -> None:
        """Record that *callout* was accepted by TTS and began playback."""

        now = self._clock()
        self._last_spoken_at = now
        self._recent_spoken[(callout.category, callout.target_lane)] = now
        self._evict_recent(now)

    def _evict_recent(self, now: float) -> None:
        window = self._config.dedup_window_seconds
        expired = [
            key for key, spoken_at in self._recent_spoken.items() if now - spoken_at >= window
        ]
        for key in expired:
            del self._recent_spoken[key]

    def _ungrounded_mentions(
        self,
        callout: Callout,
        state: GameState,
        detections: DetectionResult,
    ) -> frozenset[str]:
        allowed = _canonical_set(state.all_champions) | _canonical_set(
            detections.last_seen_champions
        )
        lexicon = (
            self._known_champions
            | state.all_champions
            | detections.last_seen_champions
        )
        mentions = find_champion_mentions(
            f"{callout.decision}\n{callout.reason}",
            lexicon,
        )
        return frozenset(name for name in mentions if name not in allowed)


_SPECIAL_ALIASES: dict[str, tuple[str, ...]] = {
    "AurelionSol": ("Aurelion Sol",),
    "Belveth": ("Bel'Veth", "Bel Veth"),
    "Chogath": ("Cho'Gath", "Cho Gath"),
    "DrMundo": ("Dr. Mundo", "Dr Mundo"),
    "JarvanIV": ("Jarvan IV", "Jarvan 4"),
    "Kaisa": ("Kai'Sa", "Kai Sa"),
    "Khazix": ("Kha'Zix", "Kha Zix"),
    "KogMaw": ("Kog'Maw", "Kog Maw"),
    "KSante": ("K'Sante", "K Sante"),
    "LeeSin": ("Lee Sin",),
    "MasterYi": ("Master Yi",),
    "MissFortune": ("Miss Fortune",),
    "MonkeyKing": ("Wukong",),
    "Nunu": ("Nunu & Willump", "Nunu and Willump"),
    "RekSai": ("Rek'Sai", "Rek Sai"),
    "TahmKench": ("Tahm Kench",),
    "TwistedFate": ("Twisted Fate",),
    "Velkoz": ("Vel'Koz", "Vel Koz"),
    "XinZhao": ("Xin Zhao",),
}


def find_champion_mentions(
    text: str,
    champion_names: Collection[str],
) -> frozenset[str]:
    """Return canonical champion names from *champion_names* mentioned in text.

    Canonical names are normalized to lowercase alphanumeric strings, so
    ``LeeSin`` and ``Lee Sin`` both become ``leesin``. Matching uses
    case-insensitive token boundaries and intentionally avoids substring
    scans, which prevents short names like ``Vi`` from matching words such as
    ``vision``.
    """

    mentions: set[str] = set()
    for canonical, aliases in _alias_map(champion_names).items():
        for alias in aliases:
            if _contains_alias(text, alias):
                mentions.add(canonical)
                break
    return frozenset(mentions)


def _canonical_set(names: Collection[str]) -> frozenset[str]:
    return frozenset(_normalize_name(name) for name in names)


def _alias_map(champion_names: Collection[str]) -> dict[str, frozenset[str]]:
    mapping: dict[str, set[str]] = {}
    for name in champion_names:
        canonical = _normalize_name(name)
        aliases = mapping.setdefault(canonical, set())
        aliases.add(name)
        split = _split_camel(name)
        if split != name:
            aliases.add(split)
        for alias in _SPECIAL_ALIASES.get(name, ()):
            aliases.add(alias)
    return {k: frozenset(v) for k, v in mapping.items()}


def _contains_alias(text: str, alias: str) -> bool:
    words = re.findall(r"[A-Za-z0-9]+", alias)
    if not words:
        return False
    pattern = r"(?<![A-Za-z0-9])" + r"[\W_]*".join(map(re.escape, words)) + r"(?![A-Za-z0-9])"
    return re.search(pattern, text, re.IGNORECASE) is not None


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _split_camel(name: str) -> str:
    # Data Dragon IDs are mostly CamelCase without spaces. Preserve all-caps
    # runs such as "IV" in JarvanIV.
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced)
    return spaced
