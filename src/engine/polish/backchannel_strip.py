"""Strip standalone backchannel utterances ("Mm-hmm.", "Hmm.", "Uh-huh.")
that ASR engines — Scribe RT especially — inject when there's a long pause
in the input audio.

Conservative by design: only strips a backchannel when it appears as its
own standalone "sentence" — i.e., at the start of the transcript, between
two sentence terminators, or at the end of the transcript. Never strips
when the token is mid-sentence ("she said mm-hmm and walked off") because
that could be a legitimate quoted use.

Idempotent. Loops up to 8 passes so chains like "Mm-hmm. Hmm. Real text."
are fully cleaned.
"""

from __future__ import annotations

import re


# Patterns we strip. All match case-insensitively as standalone tokens.
#   mm-?hmm+   → "mm-hmm", "mmhmm", "mm-hmmm", ...
#   mhmm*      → "mhm", "mhmm", "mhmmm", ...
#   hmm+       → "hmm", "hmmm", ...
#   uh-?huh    → "uh-huh", "uhhuh"
#   mmm+       → "mmm", "mmmm", ... (3+ m's; bare "mm" intentionally
#                not matched — too ambiguous with real abbreviations)
_BACKCHANNEL_TOKEN = r"(?:mm-?hmm+|mhmm*|hmm+|uh-?huh|mmm+)"

# Position 1: leading. "Mm-hmm. <rest>" → "<rest>".
_LEADING = re.compile(
    rf"^\s*\b{_BACKCHANNEL_TOKEN}\b[\s.!?,]*",
    re.IGNORECASE,
)

# Position 2: between two sentences. "<a>. Mm-hmm. <b>" → "<a>. <b>".
# Keeps the preceding terminator so the sentence boundary is preserved.
_INLINE = re.compile(
    rf"([.!?])\s+\b{_BACKCHANNEL_TOKEN}\b[\s.!?,]*",
    re.IGNORECASE,
)

# Position 3: trailing. "<rest>. Mm-hmm." → "<rest>.".
# Requires a sentence-terminator BEFORE the backchannel so we don't
# accidentally strip a real "...mm-hmm" ending (e.g., "she said mm-hmm").
_TRAILING = re.compile(
    rf"([.!?])\s+\b{_BACKCHANNEL_TOKEN}\b\s*[.!?,]*\s*$",
    re.IGNORECASE,
)


def apply(s: str) -> str:
    """Return `s` with standalone backchannels removed. Safe on empty."""
    if not s:
        return s
    # Order matters: trailing first (most specific — only matches at $),
    # then leading (also boundary-anchored), then inline (matches mid-text
    # and would otherwise eat the same token before trailing got a chance,
    # leaving a stray space).
    for _ in range(8):  # bounded; chains converge in 2-3 iterations in practice
        before = s
        s = _TRAILING.sub(r"\1", s)
        s = _LEADING.sub("", s)
        s = _INLINE.sub(r"\1 ", s)
        if s == before:
            break
    return s
