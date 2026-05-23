"""Client-side regex post-processor for the v0.3 ElevenLabs Realtime path.

Python equivalent of mobile's `RegexPolish.kt`. Replaces the Groq LLM polish
step entirely. Runs in sub-millisecond on typical dictation.

Pipeline (each step receives previous step's output):

  1. proper_noun_fixes        — "owner res" → "OwnerRez", "asa gaon" → "Assagao"
  2. spoken punctuation       — "comma" → ",", "new paragraph" → "[blank line]"
                                (reuses _normalize_spoken_symbols from transcription.py)
  3. alphanumeric NATO        — "alpha bravo 1 2 3" → "AB123"
  4. email shorthand          — "aniket at gmail dot com" → "aniket@gmail.com"
  5. spelling capture         — "Aniket spelled A-N-I-K-E-T" → "Aniket" + DICT_ADD
  6. scratch handler          — "<prev> scratch that <new>" → "<new>"
  7. voice format expansion   — "[blank line]" → "\\n\\n", "[newline]" → "\\n"
  8. final cleanup            — collapse spaces, trim ends, terminal punctuation

Returns (final_text, dict_additions) so the caller can persist any DICT_ADD
words to the structured proper_nouns config (which feeds keyterms).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import (
    alphanumeric_nato,
    backchannel_strip,
    email_shorthand,
    proper_noun_fixes,
    scratch_handler,
    spelling_capture,
)


@dataclass
class Result:
    final_text: str
    dict_additions: list[str] = field(default_factory=list)


@dataclass
class Toggles:
    """v0.3.1: per-sub-module enable flags. All default True; callers pass
    `Toggles.from_config()` to honour user settings."""
    proper_nouns: bool = True
    spoken_punctuation: bool = True
    alphanumeric: bool = True
    email: bool = True
    spelling: bool = True
    scratch: bool = True

    @classmethod
    def from_config(cls) -> "Toggles":
        from utils import ConfigManager
        def get(k: str) -> bool:
            v = ConfigManager.get_config_value("regex_polish", k)
            return True if v is None else bool(v)
        return cls(
            proper_nouns=get("proper_nouns"),
            spoken_punctuation=get("spoken_punctuation"),
            alphanumeric=get("alphanumeric"),
            email=get("email"),
            spelling=get("spelling"),
            scratch=get("scratch"),
        )


def apply(raw_from_stt: str, toggles: Toggles | None = None) -> Result:
    if not raw_from_stt or not raw_from_stt.strip():
        return Result("", [])
    if toggles is None:
        toggles = Toggles()

    s = raw_from_stt

    # Step 1: proper-noun mishears.
    if toggles.proper_nouns:
        s = proper_noun_fixes.apply(s)

    # Step 2: spoken punctuation — reuses the existing PC-side normalizer.
    if toggles.spoken_punctuation:
        from transcription import _normalize_spoken_symbols
        s = _normalize_spoken_symbols(s)

        # Step 2.5: post-sentence-terminator voice formatting that the spoken-
        # symbols pass misses. E.g. "First thought. new paragraph Second."
        s = re.sub(
            r"([.!?])\s+new\s+paragraph\s+",
            lambda m: m.group(1) + "[blank line]",
            s, flags=re.IGNORECASE,
        )
        s = re.sub(
            r"([.!?])\s+new\s+line\s+",
            lambda m: m.group(1) + "[newline]",
            s, flags=re.IGNORECASE,
        )

    # Step 3: alphanumeric / NATO collapse.
    if toggles.alphanumeric:
        s = alphanumeric_nato.normalize(s)

    # Step 4: email shorthand.
    if toggles.email:
        s = email_shorthand.apply(s)

    # Step 5: spelling capture — may emit DICT_ADD words.
    if toggles.spelling:
        spelling_result = spelling_capture.apply(s)
        s = spelling_result.cleaned
        dict_words = spelling_result.new_dict_words
    else:
        dict_words = []

    # Step 6: scratch handler.
    if toggles.scratch:
        s = scratch_handler.apply(s)

    # Step 7: expand voice-format placeholders to real newlines.
    s = re.sub(r"\s*\[blank\s*line\]\s*", "\n\n", s)
    s = re.sub(r"\s*\[newline\]\s*", "\n", s)

    # Step 7.5: strip standalone backchannels ("Mm-hmm.", "Hmm.", "Uh-huh.")
    # that Scribe RT injects during long pauses. Done after scratch (so any
    # "Mm-hmm. Scratch that. Real" survives the scratch transform) and
    # after newline expansion (so backchannels in their own paragraphs
    # are also caught at line boundaries).
    s = backchannel_strip.apply(s)

    # Step 8: final cleanup — same idempotent passes used by the existing
    # RegexPrePass mobile equivalent (capitalization, terminal punctuation,
    # double-space collapse, space-before-punct).
    s = _final_cleanup(s)

    return Result(final_text=s, dict_additions=dict_words)


def _final_cleanup(s: str) -> str:
    # Collapse double spaces (but preserve newlines)
    s = re.sub(r"[ \t]+", " ", s)
    # Remove space before terminal punctuation
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    # Strip leading whitespace on each line
    s = "\n".join(line.lstrip() for line in s.splitlines())
    # Ensure terminal punctuation
    s = s.strip()
    if s and s[-1] not in ".!?":
        s += "."
    # Capitalize first letter
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s
