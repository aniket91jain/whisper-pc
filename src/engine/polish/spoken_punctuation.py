"""Spoken-punctuation normalizer — convert dictation command words to symbols.

Extracted from ``transcription.py`` (2026-06) so it can be unit-tested in
isolation (no numpy/openai import) and shared by both transcription paths:

  * Groq/Whisper path — ``transcription.post_process_transcription`` calls this
    before LLM polish.
  * ElevenLabs path — ``engine.polish.regex_polish`` calls this as pipeline
    step 2.

Mobile mirror: ``SpokenPunctuation.kt``. Keep the two in sync — the parity
fixtures in ``tests/test_spoken_punctuation.py`` and the Kotlin test exercise
the same phrase set.

Why the anchors are tolerant of surrounding punctuation/caps
------------------------------------------------------------
ElevenLabs Scribe auto-capitalizes and auto-punctuates, so a spoken command
rarely arrives as the bare ``word command word`` shape Whisper produced. It
shows up as ``Open bracket``, ``... five, close bracket.``, ``First. New
paragraph ...``. The patterns below therefore (a) fire at start-of-string and
after a sentence terminator, not only after a word char, and (b) absorb a
stray comma/period ElevenLabs glues to the command token.

Safety model (unchanged in spirit):
  * Multi-word, unambiguous phrases ("open bracket", "new paragraph") are
    ``safe=True`` and fire inline / at utterance edges.
  * Single risky words that are common content ("period", "colon", "dash",
    "quote") are ``safe=False`` and fire ONLY on a model "double-render"
    (symbol already present on both sides) — which real content never has.
  * Paired ``quote ... unquote`` is handled separately and IS safe, because
    requiring a matching open+close makes false positives vanishingly rare.
"""

from __future__ import annotations

import re

# Order matters: longer/more-specific phrases first so "exclamation mark"
# wins over a hypothetical "exclamation", and "open parenthesis" wins over
# "open paren".
#
# Tuple: (phrase_regex, symbol, side, safe_inline)
#   side "L" — opening symbol, attaches to the FOLLOWING word:  "( x"  -> "(x"
#   side "R" — closing symbol, attaches to the PRECEDING word:  "x ,"  -> "x,"
#   side "B" — both sides closed up:                             "x # y" -> "x#y"
_SPOKEN_PUNCT = [
    (r"new\s+paragraph",                            "[blank line]", "B", True),
    (r"new\s+line",                                 "[newline]",    "B", True),
    (r"exclamation\s+(?:mark|point)",               "!",            "R", True),
    (r"question\s+mark",                            "?",            "R", True),
    (r"open\s+parenthesis|open\s+paren",            "(",            "L", True),
    (r"close\s+parenthesis|close\s+paren",          ")",            "R", True),
    # "bracket" defaults to ROUND brackets per user pref (2026-07-05): saying
    # "bracket" gives ( ); square brackets require the explicit word "square".
    # Square variants MUST precede the bare "bracket" rules so they win the
    # longest-match race.
    (r"open\s+square\s+bracket",                    "[",            "L", True),
    (r"close\s+square\s+bracket",                   "]",            "R", True),
    (r"open\s+bracket",                             "(",            "L", True),
    (r"close\s+bracket",                            ")",            "R", True),
    (r"open\s+curly(?:\s+brace)?|open\s+brace",     "{",            "L", True),
    (r"close\s+curly(?:\s+brace)?|close\s+brace",   "}",            "R", True),
    # "unquote" / "close quote" are synonyms of "end quote" for a lone closing
    # mark. A full "quote ... unquote" pair is handled by _QUOTE_PAIR_RE below
    # (which runs first), so these only fire when an opening command is absent.
    (r"unquote|end\s+quote|close\s+quote",          '"',            "R", True),
    (r"semi[\s-]?colon",                            ";",            "R", True),
    (r"forward\s+slash",                            "/",            "B", True),
    (r"back[\s-]?slash",                            "\\",           "B", True),
    (r"at\s+(?:sign|symbol)",                       "@",            "B", True),
    (r"hash\s+(?:sign|tag)|hashtag",                "#",            "B", True),
    (r"equals\s+sign",                              "=",            "B", True),
    (r"ellipsis",                                   "...",          "R", True),
    (r"asterisk",                                   "*",            "B", True),
    (r"hyphen",                                     "-",            "B", True),
    (r"comma",                                      ",",            "R", True),
    (r"full[\s-]?stop",                             ".",            "R", True),
    # Risky inline matches — these spoken words are commonly legitimate content
    # ("period of rest", "made a dash", "colon cancer"). Only fire on a model
    # double-render, which a transcriber would never produce around real content.
    (r"period",                                     ".",            "R", False),
    (r"colon",                                      ":",            "R", False),
    (r"dash",                                       "-",            "B", False),
    # Plain "slash" -> / always, per user pref (2026-07-05): the user treats
    # "slash" as the character, not the verb. safe=True so it fires inline.
    # Positioned AFTER "back slash" above so "back slash" -> \ still wins.
    # Accepted trade-off: "slash" used as a verb ("slash costs") also becomes "/".
    (r"slash",                                      "/",            "B", True),
    (r"hash",                                       "#",            "B", False),
    (r"equals",                                     "=",            "B", False),
    (r"quote",                                      '"',            "L", False),
    (r"star",                                       "*",            "B", False),
]

# Paired spoken quotes: "quote ... unquote" (and "open/begin quote ...
# close/end quote") -> "...". Pair-matching makes this safe despite "quote"
# being a content word. IGNORECASE for ElevenLabs' auto-capital; [\s,]* absorbs
# a comma it tends to glue to the markers ("Quote, hello, unquote.").
_QUOTE_PAIR_RE = re.compile(
    r"\b(?:open\s+quote|begin\s+quote|quote)\b[\s,]*"
    r"(.+?)"
    r"[\s,]*\b(?:unquote|end\s+quote|close\s+quote)\b",
    re.IGNORECASE | re.DOTALL,
)


def normalize(text: str) -> str:
    """Convert spoken punctuation commands to their symbols.

    Uses callable replacements throughout: literal replacement strings would
    interpret backreferences (``\\1``, ``\\``), which collides with symbols
    like ``\\`` and placeholder tokens like ``[newline]``.
    """
    if not text:
        return text

    # 0. Paired spoken quotes first, so the standalone "unquote"/"end quote"
    #    table entries don't pre-consume the closing marker.
    text = _QUOTE_PAIR_RE.sub(lambda m: '"' + m.group(1).strip() + '"', text)

    for phrase, sym, side, safe in _SPOKEN_PUNCT:
        sym_esc = re.escape(sym)
        p = rf"\b(?:{phrase})\b"

        # 1. Double-render: symbol already on both sides of the spoken word.
        #    Always safe — a transcriber never wraps real content this way.
        text = re.sub(
            rf"{sym_esc}\s*{p}\s*{sym_esc}",
            lambda m, s=sym: s,
            text,
            flags=re.IGNORECASE,
        )

        if not safe:
            continue

        if side == "L":
            # Opening symbol. Fire at start-of-string or after any whitespace
            # (so it works after a sentence terminator too — the terminator sits
            # before the captured space and is preserved). Absorb a trailing
            # ElevenLabs comma; attach the symbol to the following word.
            text = re.sub(
                rf"(^|\s)\s*{p}[\s,]*(?=\w)",
                lambda m, s=sym: m.group(1) + s,
                text,
                flags=re.IGNORECASE,
            )
        elif side == "R":
            # Closing symbol. Attach to the preceding word; absorb stray commas
            # ElevenLabs inserts before/after the command. Inline keeps the
            # following space; the end pattern absorbs a trailing terminator.
            text = re.sub(
                rf"(?<=\w)[\s,]+{p}[,]?(?=\s+\w)",
                lambda m, s=sym: s,
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                rf"(?<=\w)[\s,]+{p}\s*[.!?]?\s*$",
                lambda m, s=sym: s,
                text,
                flags=re.IGNORECASE,
            )
        else:  # "B"
            text = re.sub(
                rf"(?<=\w)[\s,]+{p}[\s,]*(?=\w)",
                lambda m, s=sym: s,
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                rf"(?<=\w)[\s,]+{p}\s*[.!?]?\s*$",
                lambda m, s=sym: s,
                text,
                flags=re.IGNORECASE,
            )

    return text
