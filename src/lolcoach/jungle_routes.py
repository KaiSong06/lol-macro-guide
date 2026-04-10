"""Hardcoded jungle-route lookup tables + enemy jungler position prediction.

The prediction is a crude but useful heuristic: given the last-seen quadrant
and elapsed seconds, follow one of six standard jungle paths forward and
return the most likely current quadrant. The vision LLM (Unit 7) uses this
as *a hint*, not a fact — the prompt includes ``enemy_jungler_predicted_quadrant``
only as soft context alongside the real minimap image.

Timings are approximate level-1 clear + travel times. They don't need to be
perfect; the point is to tell the LLM "enemy was bot jungle 60s ago, they're
probably around mid now" so it can reason about gank setups, not to catch
them in the act.
"""

from __future__ import annotations

from dataclasses import dataclass


#: A single jungle camp on a scripted path.
@dataclass(frozen=True)
class JungleCamp:
    name: str
    quadrant: str
    clear_time_seconds: float


#: Six standard jungle paths. Each entry is ordered from path start to end.
#: ``clear_time_seconds`` on the first camp is 0 (you're already there when
#: the path starts) — subsequent camps include both the travel time and the
#: clear time, so cumulative elapsed time walks cleanly from one to the next.
STANDARD_PATHS: dict[str, tuple[JungleCamp, ...]] = {
    "blue_top": (
        JungleCamp("BlueBuff", "top_jungle", 0.0),
        JungleCamp("Gromp", "top_jungle", 25.0),
        JungleCamp("Wolves", "top_jungle", 25.0),
        JungleCamp("Raptors", "mid", 30.0),
        JungleCamp("RedBuff", "bot_jungle", 35.0),
        JungleCamp("Krugs", "bot_jungle", 40.0),
    ),
    "blue_bot": (
        JungleCamp("BlueBuff", "top_jungle", 0.0),
        JungleCamp("Wolves", "top_jungle", 25.0),
        JungleCamp("Gromp", "top_jungle", 25.0),
        JungleCamp("Raptors", "mid", 30.0),
        JungleCamp("RedBuff", "bot_jungle", 35.0),
        JungleCamp("Krugs", "bot_jungle", 40.0),
    ),
    "red_bot": (
        JungleCamp("RedBuff", "bot_jungle", 0.0),
        JungleCamp("Krugs", "bot_jungle", 25.0),
        JungleCamp("Raptors", "mid", 30.0),
        JungleCamp("Wolves", "top_jungle", 35.0),
        JungleCamp("Gromp", "top_jungle", 25.0),
        JungleCamp("BlueBuff", "top_jungle", 30.0),
    ),
    "red_top": (
        JungleCamp("RedBuff", "bot_jungle", 0.0),
        JungleCamp("Raptors", "mid", 25.0),
        JungleCamp("Krugs", "bot_jungle", 25.0),
        JungleCamp("BlueBuff", "top_jungle", 40.0),
        JungleCamp("Gromp", "top_jungle", 25.0),
        JungleCamp("Wolves", "top_jungle", 25.0),
    ),
    "invade_blue": (
        JungleCamp("RedBuff", "bot_jungle", 0.0),
        JungleCamp("InvadeBlueBuff", "top_jungle", 50.0),
        JungleCamp("Gromp", "top_jungle", 25.0),
    ),
    "invade_red": (
        JungleCamp("BlueBuff", "top_jungle", 0.0),
        JungleCamp("InvadeRedBuff", "bot_jungle", 50.0),
        JungleCamp("Krugs", "bot_jungle", 25.0),
    ),
}


def predict_along_path(path_name: str, elapsed_seconds: float) -> str:
    """Walk a named path forward by *elapsed_seconds* and return the current quadrant.

    Raises :class:`KeyError` for unknown path names.
    """
    path = STANDARD_PATHS[path_name]
    if not path:
        raise ValueError(f"path {path_name!r} is empty")

    remaining = float(elapsed_seconds)
    current_idx = 0
    while current_idx + 1 < len(path):
        next_travel = path[current_idx + 1].clear_time_seconds
        if remaining < next_travel:
            break
        remaining -= next_travel
        current_idx += 1

    return path[current_idx].quadrant


def predict_quadrant(
    last_seen: str | None,
    elapsed_seconds: float,
) -> str | None:
    """Best-effort guess for where the enemy jungler is now.

    * Cold start (``last_seen is None``) → ``None`` — the caller should omit
      any predicted-position text from the LLM prompt entirely.
    * Under 30 seconds elapsed → return ``last_seen`` (they're probably still
      there).
    * Longer elapsed times → walk a standard path forward. We use ``blue_top``
      or ``red_bot`` depending on which side the last-seen quadrant is on.
    """
    if last_seen is None:
        return None

    if elapsed_seconds < 30.0:
        return last_seen

    # Pick a canonical path based on which jungle half we saw them in. This
    # is a crude heuristic — the LLM is the real reasoner.
    if last_seen in {"top_jungle", "top_lane", "top_river"}:
        path_name = "blue_top"
    elif last_seen in {"bot_jungle", "bot_lane", "bot_river"}:
        path_name = "red_bot"
    else:
        # mid, base_blue, base_red — we don't have a confident walk, stay put.
        return last_seen

    return predict_along_path(path_name, elapsed_seconds)
