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

# Mid-sentence pause-acknowledgment hallucination, observed >14× in 23 days of
# Scribe v2 RT output: "...is a need- Mm-hmm ... which is better...". The
# engine injects an acknowledgment token at pause/hesitation points where the
# speaker actually paused silently. Signature is `<word>- <filler> <ellipsis>
# <continuation>`. Strip the injected token, keep the ellipsis as the natural
# pause indicator. Widened to "okay"/"yes" — confirmed by the user as observed
# hallucinations in this exact shape (e.g. "hundreds of- Okay ... notes").
_DASH_FILLER_TOKEN = r"(?:mm-?hmm+|mhmm*|hmm+|uh-?huh|mmm+|okay|yes)"
_DASH_FILLER = re.compile(
    rf"(\w)-\s+\b{_DASH_FILLER_TOKEN}\b\s*\.{{2,}}\s*",
    re.IGNORECASE,
)

# Stuck-decode repetition: "...folder? Okay. Okay. Okay. Okay. Okay. Okay."
# Collapse 3+ consecutive identical short sentences to one. Length cap (≤8
# chars) keeps the rule from touching legitimate emphatic repetition of real
# content words ("Stop. Stop. Stop." would collapse, "Maybe. Maybe. Maybe."
# would too — both judged acceptable for the bug we're fixing).
_STUCK_REPEAT = re.compile(
    r"\b(\w{1,8})([.!?])(?:\s+\1[.!?])+(?=\s|$)",
    re.IGNORECASE,
)

# Whole-utterance backchannel: the entire transcript is just a single
# acknowledgment token. Wider set than _BACKCHANNEL_TOKEN — also includes
# "okay" (user-confirmed hallucination as a standalone utterance). "Yes" is
# intentionally excluded here: a brief "Yes." reply is a legitimate use and
# stripping it would be destructive. The dash-filler rule still catches
# "Yes" in its narrow mid-sentence shape.
_WHOLE_UTTERANCE = re.compile(
    r"^\s*\b(?:mm-?hmm+|mhmm*|hmm+|uh-?huh|mmm+|okay)\b\s*[.!?,]*\s*$",
    re.IGNORECASE,
)


def apply(s: str) -> str:
    """Return `s` with standalone backchannels removed. Safe on empty."""
    if not s:
        return s
    # Order matters: trailing first (most specific — only matches at $),
    # then leading (also boundary-anchored), then inline (matches mid-text
    # and would otherwise eat the same token before trailing got a chance,
    # leaving a stray space). Dash-filler and stuck-repeat run before the
    # boundary-anchored passes so the cleanup is visible to them.
    for _ in range(8):  # bounded; chains converge in 2-3 iterations in practice
        before = s
        s = _DASH_FILLER.sub(r"\1... ", s)
        s = _STUCK_REPEAT.sub(r"\1\2", s)
        s = _TRAILING.sub(r"\1", s)
        s = _LEADING.sub("", s)
        s = _INLINE.sub(r"\1 ", s)
        s = _WHOLE_UTTERANCE.sub("", s)
        if s == before:
            break
    return s
