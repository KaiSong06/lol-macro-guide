# TODOs

Captured during /plan-eng-review on 2026-04-10.

---

## Auto-detect minimap ROI across resolutions

**What:** Detect the minimap bounds automatically using template matching against the minimap border graphic, instead of requiring manual calibration on first run.

**Why:** First-run UX improvement. Manual calibration works but adds friction for non-technical users. Also handles mid-session HUD changes (resolution change, HUD scale adjustment).

**Pros:**
- Smoother onboarding — no need to click minimap corners
- Supports HUD scale changes mid-session
- Works across 1080p, 1440p, 4K without per-resolution config

**Cons:**
- Adds complexity to `capture.py`
- May fail on non-standard HUDs (custom layouts, streamer mode)
- Manual calibration remains as the reliable fallback

**Context:** The MVP plan requires user to click minimap corners on first run, saving the ROI to `calibration.json`. This works reliably but feels clunky. A template-match approach would use League's minimap border graphic (the square outline) as a detection template. Run once at coach startup, save result, fall back to manual if confidence is low.

**Depends on:** `capture.py` shipped and working with manual calibration.

---

## Rule-based fallback heuristics (Approach C)

**What:** Add a fast rule-based layer that runs alongside the LLM vision pipeline. Provides sub-second callouts for obvious events (dragon timer warnings, ward timing, enemy jungler spotted via template match alone).

**Why:** The LLM inference path takes 5-8 seconds. For obvious events (dragon spawns in 30s, Smite is off cooldown, enemy jungler just appeared on minimap), the user shouldn't have to wait for LLM reasoning. This is the Approach C architecture that office-hours considered and deferred.

**Pros:**
- Sub-second callouts for simple events
- Reduces LLM inference load (fewer calls for events that don't need reasoning)
- More responsive coach UX
- Lower GPU contention

**Cons:**
- Two coaching code paths to maintain (rules + LLM)
- Need to design rule priority vs LLM priority (which speaks first?)
- Potential for rules and LLM to contradict each other on the same event

**Context:** Office-hours compared three approaches. A (Lean Pipeline, 5/10 completeness), B (Stateful Coach, 7/10, chosen), C (Hybrid Intelligence, 8/10, deferred). The hybrid design is strictly better once the base coach is working. This is a natural Phase 2.

**Depends on:** Base Stateful Coach (Approach B) shipped and validated on real games.

---

## Code signing for Windows installer

**What:** Obtain an EV Code Signing certificate and sign the `.exe` and installer so Windows Defender doesn't flag them.

**Why:** Unsigned binaries trigger "Windows protected your PC" warnings that scare away non-technical users. A signed binary gets trusted immediately.

**Pros:**
- No scary warnings on install
- Better trust signal for users discovering the tool via Reddit/YouTube
- Required for serious distribution at scale

**Cons:**
- EV Code Signing certs cost $200-400/year
- Requires a hardware token in most cases
- Overkill for a hobby project until it has actual users

**Context:** MVP ships unsigned. README documents the workaround ("click More info → Run anyway"). If the tool starts getting meaningful adoption from clip posts, revisit.

**Depends on:** MVP shipped + evidence of real users downloading the installer.
