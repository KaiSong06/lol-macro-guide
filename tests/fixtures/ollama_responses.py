"""Fixture Ollama response bodies for the inference parser tests.

These constants represent the **text content** the local vision model emits —
specifically, the value of ``response["message"]["content"]`` from Ollama's
``/api/chat`` JSON envelope. The HTTP-layer tests wrap these constants in a
synthetic JSON envelope when registering routes with ``requests-mock``.

The expected response shape is a 5-line block, one ``KEY: value`` per line:

    DECISION: <one sentence the coach will speak>
    CATEGORY: <one of the known coach categories>
    TARGET_LANE: <top|mid|bot|jungle|global>
    CONFIDENCE: <integer 1-10>
    REASON: <one sentence justification, used for grounding check>

The parser must accept ``VALID_RESPONSE`` and ``TRAILING_WHITESPACE`` and
must reject every other constant in this module by returning ``None`` and
logging an ``inference_malformed`` event.

Field-order independence is intentional: ``VALID_OUT_OF_ORDER`` exercises
the parser's per-field regex matching, since real LLM output occasionally
reorders fields when the model self-corrects mid-stream.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Valid responses — the parser must accept these.
# ---------------------------------------------------------------------------

VALID_RESPONSE = (
    "DECISION: Path to bot river, Lee Sin likely ganking top\n"
    "CATEGORY: pathing\n"
    "TARGET_LANE: bot\n"
    "CONFIDENCE: 8\n"
    "REASON: Lee Sin was last seen near bot river 30s ago,"
    " predicted path heads through mid jungle.\n"
)

#: Same valid payload, but with extra ``\r\n`` and trailing whitespace
#: from the model's terminal newline. The parser must strip cleanly.
TRAILING_WHITESPACE = VALID_RESPONSE + "\r\n\n   \n"

#: Same fields as VALID_RESPONSE but in a different order. Parser uses
#: per-field regex so order should not matter.
VALID_OUT_OF_ORDER = (
    "REASON: Enemy mid is roaming, ward the bush.\n"
    "CONFIDENCE: 7\n"
    "TARGET_LANE: mid\n"
    "CATEGORY: vision\n"
    "DECISION: Drop a ward in mid bush, enemy mid is roaming\n"
)


# ---------------------------------------------------------------------------
# Malformed responses — the parser must reject these and return None.
# ---------------------------------------------------------------------------

#: Missing the CATEGORY line entirely. Real failure mode: model truncates
#: output mid-stream because of a token-budget cap.
MALFORMED_NO_CATEGORY = (
    "DECISION: Path to bot river\n"
    "TARGET_LANE: bot\n"
    "CONFIDENCE: 8\n"
    "REASON: Some reason text.\n"
)

#: CATEGORY value not in the allowed coach enum. Real failure mode: the
#: model invents a category that sounds plausible ("taunt", "joke",
#: "pinged"). Filter must reject so unknown categories never reach TTS.
UNKNOWN_CATEGORY = (
    "DECISION: Path to bot river\n"
    "CATEGORY: taunt\n"
    "TARGET_LANE: bot\n"
    "CONFIDENCE: 8\n"
    "REASON: Some reason text.\n"
)

#: TARGET_LANE value not in the allowed coach enum. Real failure mode: the
#: model spells out the lane in natural language ("bottom lane") instead
#: of using the enum keyword.
UNKNOWN_LANE = (
    "DECISION: Recall now\n"
    "CATEGORY: recall\n"
    "TARGET_LANE: bottom_lane\n"
    "CONFIDENCE: 9\n"
    "REASON: Low HP and full inventory slot.\n"
)

#: CONFIDENCE value outside [1, 10]. Real failure mode: the model writes
#: a percentage ("88") or a decibel-style integer ("100"). Anything outside
#: the documented 1-10 range fails grounding.
CONFIDENCE_OUT_OF_RANGE = (
    "DECISION: Path to bot river\n"
    "CATEGORY: pathing\n"
    "TARGET_LANE: bot\n"
    "CONFIDENCE: 99\n"
    "REASON: Some reason text.\n"
)

#: CONFIDENCE value that is not an integer. Real failure mode: the model
#: writes "high" or "8.5" instead of an integer. The parser must catch
#: int() conversion errors without crashing.
CONFIDENCE_NOT_INTEGER = (
    "DECISION: Path to bot river\n"
    "CATEGORY: pathing\n"
    "TARGET_LANE: bot\n"
    "CONFIDENCE: high\n"
    "REASON: Some reason text.\n"
)

#: CONFIDENCE value at the lower boundary (1). Must parse as a valid
#: Callout — the schema documents 1-10 inclusive.
CONFIDENCE_LOWER_BOUND = (
    "DECISION: Stand back, let the wave push\n"
    "CATEGORY: pathing\n"
    "TARGET_LANE: top\n"
    "CONFIDENCE: 1\n"
    "REASON: Low certainty about enemy intent.\n"
)

#: CONFIDENCE value at the upper boundary (10). Must parse as a valid
#: Callout.
CONFIDENCE_UPPER_BOUND = (
    "DECISION: Smite Baron now\n"
    "CATEGORY: objective_call\n"
    "TARGET_LANE: global\n"
    "CONFIDENCE: 10\n"
    "REASON: Baron is at execute range and enemy team is dead.\n"
)

#: CONFIDENCE value of zero — below the [1, 10] range. Must reject.
CONFIDENCE_ZERO = (
    "DECISION: Path to bot river\n"
    "CATEGORY: pathing\n"
    "TARGET_LANE: bot\n"
    "CONFIDENCE: 0\n"
    "REASON: Some reason text.\n"
)

#: Negative CONFIDENCE value. int() parses negative integers cleanly so
#: the range gate is the only thing that catches this. Must reject.
CONFIDENCE_NEGATIVE = (
    "DECISION: Path to bot river\n"
    "CATEGORY: pathing\n"
    "TARGET_LANE: bot\n"
    "CONFIDENCE: -3\n"
    "REASON: Some reason text.\n"
)

#: Fractional CONFIDENCE value (8.5). int('8.5') raises ValueError, so the
#: parser must collapse to None rather than crashing. Real failure mode:
#: the model writes a decimal number to express partial confidence.
CONFIDENCE_FRACTIONAL = (
    "DECISION: Path to bot river\n"
    "CATEGORY: pathing\n"
    "TARGET_LANE: bot\n"
    "CONFIDENCE: 8.5\n"
    "REASON: Some reason text.\n"
)

#: Two DECISION lines in the same response. The parser uses
#: ``re.search`` which returns the FIRST match — pin the contract so a
#: future "last wins" refactor would break this test rather than
#: silently change behavior. Real failure mode: a model that
#: self-corrects mid-stream and emits a second DECISION line below
#: the first.
DUPLICATE_DECISION = (
    "DECISION: Path to bot river\n"
    "CATEGORY: pathing\n"
    "TARGET_LANE: bot\n"
    "CONFIDENCE: 8\n"
    "REASON: Lee Sin pathing through bot river.\n"
    "DECISION: Actually wait at top river\n"
)

#: REASON wraps across multiple physical lines. The parser must capture
#: all of them (up to the next known key or end of text) so Unit 8's
#: grounding check sees every champion name the model mentioned. A
#: truncated REASON is an R6 precondition violation: the filter can't
#: reject a hallucinated champion whose name was silently dropped by
#: the parser. Real failure mode: small 4B-scale vision models emit
#: wrapped REASON text when the thought spans more than one clause.
MULTILINE_REASON = (
    "DECISION: Path to bot river, Lee Sin likely ganking top\n"
    "CATEGORY: pathing\n"
    "TARGET_LANE: bot\n"
    "CONFIDENCE: 8\n"
    "REASON: Lee Sin was last seen near bot river 30s ago.\n"
    "Yasuo is pushing top and could dive.\n"
    "Both are reasons to path defensively.\n"
)

#: Out-of-order response (REASON first) where REASON wraps across multiple
#: lines. None of the continuation lines start with a known key, so the
#: parser should capture the full wrapped text for REASON and still pull
#: each of the other fields correctly when their key lines appear below.
#: This verifies the boundary lookahead does not false-positive on
#: arbitrary text inside a multi-line value.
OUT_OF_ORDER_WITH_MULTILINE_REASON = (
    "REASON: Enemy was last seen near top lane.\n"
    "Predicted path loops through the river.\n"
    "CONFIDENCE: 8\n"
    "CATEGORY: pathing\n"
    "TARGET_LANE: top\n"
    "DECISION: Rotate top to contest\n"
)

#: Adversarial regression: a REASON that wraps across multiple lines AND
#: one of the continuation lines happens to start with a literal known
#: key. The boundary lookahead terminates REASON at that continuation
#: line, and the (duplicate) key line's value gets extracted as a
#: separate field. In this fixture the duplicate CONFIDENCE is a
#: non-integer, which fails the int() conversion — parser returns None
#: instead of silently accepting a wrong value. Real failure mode:
#: model self-narration leaks field-key-lookalike lines mid-reason.
REASON_WITH_EMBEDDED_KEY_ON_CONTINUATION = (
    "DECISION: Path to bot\n"
    "CATEGORY: pathing\n"
    "TARGET_LANE: bot\n"
    "CONFIDENCE: 8\n"
    "REASON: Lee Sin is pathing\n"
    "CONFIDENCE: high as I can tell\n"
)

#: Empty DECISION line. Even if every other field parses, an empty
#: decision is unspeakable — TTS would broadcast silence. Reject.
EMPTY_DECISION = (
    "DECISION:\n"
    "CATEGORY: pathing\n"
    "TARGET_LANE: bot\n"
    "CONFIDENCE: 8\n"
    "REASON: Some reason text.\n"
)

#: Completely unparseable garbage from the model. Real failure mode: the
#: model dumps raw chain-of-thought, an apology, or model-card boilerplate
#: instead of the structured response. Reject hard.
GARBAGE_RESPONSE = (
    "I'm sorry, I cannot analyze League of Legends gameplay because "
    "I do not have access to real-time game state. Please consult a "
    "professional coach for personalized advice.\n"
)
