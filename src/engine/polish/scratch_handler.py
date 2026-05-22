"""Handle 'scratch that' / 'delete that' voice commands.

Python port of mobile's `ScratchHandler.kt`. Within each sentence, delete
everything before "scratch that" and keep what came after. If "scratch
that" is at the end of a sentence, delete the whole sentence.
"""

from __future__ import annotations

import re

_SCRATCH = re.compile(r"\b(?:scratch|delete)\s+that\b", re.IGNORECASE)
_TERMINAL = re.compile(r"[.!?]+\s*$")


def apply(text: str) -> str:
    if not _SCRATCH.search(text):
        return text
    sentences = _split_sentences(text)
    processed = [_process_one(s) for s in sentences]
    return "".join(processed).lstrip()


def _process_one(sentence: str) -> str:
    m = _SCRATCH.search(sentence)
    if not m:
        return sentence
    after = sentence[m.end():].lstrip()
    if not after or _TERMINAL.fullmatch(after):
        # Trailing erase — drop whole sentence, preserve any trailing whitespace
        # for boundary safety.
        return _trailing_ws(sentence)
    return _capitalize_first(after)


def _split_sentences(text: str) -> list[str]:
    out: list[str] = []
    start = 0
    for m in re.finditer(r"[.!?]+\s+", text):
        out.append(text[start:m.end()])
        start = m.end()
    if start < len(text):
        out.append(text[start:])
    return out


def _trailing_ws(s: str) -> str:
    i = len(s)
    while i > 0 and s[i - 1].isspace():
        i -= 1
    return s[i:]


def _capitalize_first(s: str) -> str:
    if not s or s[0].isupper():
        return s
    return s[0].upper() + s[1:]
