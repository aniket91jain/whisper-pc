"""Extract spelled-out proper nouns from a transcript.

Python port of mobile's `SpellingCapture.kt`.

Two patterns:

  A) Light correction (no trigger word): "Anikat A-N-I-K-E-T" → "Aniket"
     Replaces the spoken attempt + letters with the correctly-spelled word.
     Does NOT add to the dictionary.

  B) Dictionary add ("spelled" trigger): "Aniket spelled A-N-I-K-E-T"
     → "Aniket"; also returns the captured words so the caller can persist
     them to the structured proper_nouns list (which feeds keyterms).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Result:
    cleaned: str
    new_dict_words: list[str] = field(default_factory=list)


# Letter sequence: ≥2 letters separated by - . _ or whitespace.
_LETTER_SEQ = r"[A-Za-z](?:[-.\s_]+[A-Za-z]){1,}"

# Form B: explicit "spelled" trigger
_SPELLED_TRIGGER = re.compile(
    rf"(\S+)\s+spelled(?:\s+(?:as|out))?\s+({_LETTER_SEQ})",
    re.IGNORECASE,
)

# Form A: implicit — word followed by ≥3-letter sequence with matching first letter
_IMPLICIT = re.compile(
    rf"(\b[A-Za-z]{{2,}}\b)\s+([A-Za-z](?:[-.\s_]+[A-Za-z]){{2,}})(?=[\s.,;:!?)\]]|$)",
    re.IGNORECASE,
)


def apply(text: str) -> Result:
    new_words: list[str] = []
    s = text

    # Form B: explicit "spelled" trigger — collapse + persist.
    def b_repl(m: re.Match) -> str:
        letters = _extract_letters(m.group(2))
        word = _capitalize(letters)
        if word:
            new_words.append(word)
            return word
        return m.group(0)

    s = _SPELLED_TRIGGER.sub(b_repl, s)

    # Form A: implicit — collapse only, don't persist.
    def a_repl(m: re.Match) -> str:
        attempt = m.group(1)
        letters = _extract_letters(m.group(2))
        if (len(letters) >= 3
                and attempt[0].lower() == letters[0].lower()):
            return _capitalize(letters)
        return m.group(0)

    s = _IMPLICIT.sub(a_repl, s)

    return Result(cleaned=s, new_dict_words=new_words)


def _extract_letters(raw: str) -> str:
    return re.sub(r"[^A-Za-z]", "", raw)


def _capitalize(letters: str) -> str:
    if not letters:
        return ""
    return letters[0].upper() + letters[1:].lower()
