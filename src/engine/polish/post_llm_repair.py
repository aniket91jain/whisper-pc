"""Post-LLM polish output repair + safety checks.

Mirrors whisper-mobile's PostLlmRepair.kt for cross-platform parity. Catches
preamble leakage, surrounding quote wrappers, runaway hallucination, the
__EMPTY__ sentinel emitted by the v3b polish prompt, and reasoning-mode
<thinking>/<reasoning>/<analysis> tag leakage. Falls back to the raw
transcript when the polish output is rejected.

See plans/whisper-polish-deep-dive.md Section 11.9 for the design rationale
and Section 6 for the cross-app research that informed the rejection rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


EMPTY_SENTINEL = "__EMPTY__"

# Tunable: if rapidfuzz.fuzz.ratio (0-100) divided by 100 is below this, the
# polish output is rejected as having wandered too far from raw. Plan
# recommends starting at 0.5 and tuning empirically after a week of normal use.
LEVENSHTEIN_REJECT_THRESHOLD = 0.5

# Minimum raw-transcript length for the Levenshtein check to apply. Below
# this, small absolute edits (case fix, period add) inflate the ratio unfairly.
LEVENSHTEIN_MIN_RAW_CHARS = 20

# Word-overlap floor as a secondary check (alongside Levenshtein). Kept as
# belt-and-suspenders per user feedback — layer the new checks on top of the
# battle-tested existing ones rather than replacing wholesale.
WORD_OVERLAP_FLOOR = 0.30
WORD_OVERLAP_MIN_RAW_WORDS = 5

# VoiceInk-style reasoning-tag stripper. Some Groq/OpenAI models emit
# chain-of-thought in these wrappers despite reasoning_effort=low.
_THINKING_TAG_PATTERNS = [
    re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<analysis>.*?</analysis>", re.DOTALL | re.IGNORECASE),
]

# Wrap-around tags the polish prompt instructs the model to omit but
# sometimes echoes anyway.
_TRANSCRIPT_WRAPPER_TAGS = ("[TRANSCRIPT]", "[/TRANSCRIPT]")

# Tail-marker emitted by the polish prompt's SPELLING (dictionary add) rule
# when the explicit "spelled" trigger fires. Format: <<DICT_ADD: word1, word2>>
# at end-of-output. Extracted into RepairResult.dict_additions and stripped
# from the visible text. End-anchored so a stray marker mid-text doesn't get
# silently consumed.
_DICT_ADD_TAIL_PATTERN = re.compile(r"<<DICT_ADD:\s*([^>]+?)\s*>>\s*$")

# Preamble regexes — strip leading "Here's the cleaned text:" /
# "Sure, let me ..." style openers that slip past the prompt's IDENTITY
# EXAMPLES.
_PREAMBLE_PATTERNS = [
    re.compile(r"^(here'?s|here is|the cleaned)\b.*?[:\.]\s*", re.IGNORECASE),
    re.compile(r"^(sure,?|let me|i'?ve|i'?ll|okay,?|alright,?)\b.*?[:\.]\s*",
               re.IGNORECASE),
]

# Bad-first-token list — catches preambles the regex missed. If polish output
# starts (case-insensitively) with any of these AND the same phrase wasn't in
# the raw transcript, reject as preamble.
_BAD_FIRST_TOKENS = (
    "sure,", "sure!", "of course,", "of course!",
    "here is", "here's", "here are",
    "i've", "i have", "i'll", "i will",
    "let me", "okay", "ok,", "alright", "great!",
    "actually,", "well,", "i think", "i understand",
    "it sounds like", "thanks!", "thank you for",
)


@dataclass
class RepairResult:
    """Outcome of a polish-output repair pass."""

    # The text to paste. May be the cleaned polish, the raw transcript (on
    # rejection), or an empty string (on __EMPTY__ sentinel).
    final_text: str

    # Whether the polish output was rejected and we fell back to raw.
    polish_rejected: bool

    # Human-readable reason for rejection (None when accepted).
    rejection_reason: Optional[str]

    # Words the polish prompt's SPELLING (dictionary add) rule extracted from
    # a "<word> spelled <letters>" pattern. Populated by parsing the
    # <<DICT_ADD: word1, word2>> tail marker that the polish model emits when
    # the explicit "spelled" trigger fires. The marker is stripped from
    # final_text. The caller decides whether to actually persist these (gated
    # by llm_polish.enable_dict_autoadd_from_spelling).
    dict_additions: List[str] = field(default_factory=list)


def _strip_wrapper_tags(text: str) -> str:
    for tag in _TRANSCRIPT_WRAPPER_TAGS:
        text = text.replace(tag, "")
    return text.strip()


def _strip_thinking_tags(text: str) -> str:
    for pattern in _THINKING_TAG_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()


def _strip_wrapping_quotes(text: str) -> str:
    pairs = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"))
    for open_q, close_q in pairs:
        if text.startswith(open_q) and text.endswith(close_q) and len(text) >= 2:
            return text[1:-1].strip()
    return text


def _strip_preamble(text: str) -> str:
    for pattern in _PREAMBLE_PATTERNS:
        text = pattern.sub("", text).strip()
    return text


def _starts_with_bad_token(text: str) -> Optional[str]:
    lower = text[:50].lower()
    for token in _BAD_FIRST_TOKENS:
        if lower.startswith(token):
            return token
    return None


def _extract_dict_additions(text: str) -> tuple[str, List[str]]:
    """Find a trailing <<DICT_ADD: ...>> marker, return (stripped_text, words).

    If no marker is present, returns the text unchanged and an empty list. The
    marker is end-anchored so a malformed mid-text occurrence is left alone
    (rather than silently consumed). Words are split on comma, trimmed, and
    de-duplicated case-insensitively in the order first seen.
    """
    match = _DICT_ADD_TAIL_PATTERN.search(text)
    if not match:
        return text, []
    raw_words = match.group(1).split(",")
    seen_lower = set()
    words: List[str] = []
    for w in raw_words:
        cleaned = w.strip()
        if not cleaned:
            continue
        lower = cleaned.lower()
        if lower in seen_lower:
            continue
        seen_lower.add(lower)
        words.append(cleaned)
    stripped = text[: match.start()].rstrip()
    return stripped, words


def _word_overlap_ratio(raw: str, polished: str) -> float:
    raw_words = {w for w in raw.lower().split() if w}
    if len(raw_words) < WORD_OVERLAP_MIN_RAW_WORDS:
        return 1.0  # too few words for meaningful overlap; pass
    polished_words = {w for w in polished.lower().split() if w}
    if not polished_words:
        return 0.0
    return len(raw_words & polished_words) / len(raw_words)


def _levenshtein_similarity(raw: str, polished: str) -> float:
    """Rapidfuzz fuzz.ratio normalised to 0-1. Returns 1.0 (pass) if
    rapidfuzz is not importable so a missing dependency does not hard-fail
    the polish path."""
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return 1.0
    return fuzz.ratio(raw.lower(), polished.lower()) / 100.0


def apply(raw: str, polished: Optional[str]) -> RepairResult:
    """Run the full post-LLM repair pipeline.

    Order of operations:
      1. Strip [TRANSCRIPT] wrapper tags (model echo).
      2. Strip <thinking>/<reasoning>/<analysis> tags (VoiceInk pattern).
      3. Extract + strip trailing <<DICT_ADD: ...>> marker (auto-add feature).
      4. EMPTY sentinel -> return "" (clean no-paste).
      5. Strip wrapping quotes.
      6. Strip leading preamble via regex.
      7. Empty after stripping -> reject, return raw.
      8. Bad-first-token -> reject, return raw.
      9. Levenshtein similarity -> reject if too low.
      10. Word overlap floor -> reject if too low.
    """
    if polished is None:
        return RepairResult(final_text=raw, polish_rejected=True,
                            rejection_reason="Polish output was None")

    p = polished.strip()
    p = _strip_wrapper_tags(p)
    p = _strip_thinking_tags(p)

    # Extract + strip the auto-add tail marker before any other text checks so
    # the marker text itself doesn't influence Levenshtein / word-overlap or
    # leak into bad-first-token false positives. Empty list when absent.
    p, dict_additions = _extract_dict_additions(p)

    # EMPTY sentinel -> intentional empty output, paste nothing.
    if p == EMPTY_SENTINEL:
        return RepairResult(final_text="", polish_rejected=False,
                            rejection_reason=None,
                            dict_additions=dict_additions)

    p = _strip_wrapping_quotes(p)
    p = _strip_preamble(p)

    if not p:
        return RepairResult(
            final_text=raw, polish_rejected=True,
            rejection_reason="Polish empty after stripping wrappers and preamble",
            dict_additions=dict_additions)

    bad_token = _starts_with_bad_token(p)
    if bad_token is not None and bad_token.lower() not in raw.lower():
        return RepairResult(
            final_text=raw, polish_rejected=True,
            rejection_reason=f"Polish starts with bad token: {bad_token!r}",
            dict_additions=dict_additions)

    # Skip similarity/overlap checks when a well-formed <<DICT_ADD>> marker
    # was emitted: the SPELLING (dict-add) rule deletes the spoken attempt,
    # the word "spelled", and the letter sequence, which legitimately
    # collapses the output far below the similarity floor. The marker's
    # presence is evidence polish did its job, not that it wandered.
    if not dict_additions:
        if len(raw.strip()) >= LEVENSHTEIN_MIN_RAW_CHARS:
            similarity = _levenshtein_similarity(raw, p)
            if similarity < LEVENSHTEIN_REJECT_THRESHOLD:
                return RepairResult(
                    final_text=raw, polish_rejected=True,
                    rejection_reason=(
                        f"Levenshtein similarity {similarity:.2f} < "
                        f"{LEVENSHTEIN_REJECT_THRESHOLD}"))

        overlap = _word_overlap_ratio(raw, p)
        if overlap < WORD_OVERLAP_FLOOR:
            return RepairResult(
                final_text=raw, polish_rejected=True,
                rejection_reason=f"Word overlap {overlap:.0%} < {WORD_OVERLAP_FLOOR:.0%}")

    return RepairResult(final_text=p, polish_rejected=False,
                        rejection_reason=None,
                        dict_additions=dict_additions)
