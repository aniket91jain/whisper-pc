"""Collapse NATO phonetic + digit sequences into proper codes.

Python port of mobile's `AlphanumericNato.kt`.

Examples:
  "MH zero one alpha bravo nine eight seven six"  → "MH01AB9876"
  "alpha bravo charlie 1 2 3"                     → "ABC123"
  "9 9 5 8 2 double 1 2 double 1"                 → "9958211211"

A token is a code candidate if it's one of:
  - A NATO phonetic word (alpha → A, ..., zulu → Z)
  - A digit word (zero → 0, ..., nine → 9, oh → 0)
  - A single uppercase letter (A-Z)
  - A digit run (\\d+)
  - A short uppercase string (2-4 chars)
  - "double" / "triple" followed by one of the above (repeat the code unit)

We collapse runs of ≥2 adjacent candidates (separated by whitespace) into a
single concatenated token. Standalone candidates are left alone — "chapter 5"
stays "chapter 5", not "5".
"""

from __future__ import annotations

import re


_NATO_TO_LETTER: dict[str, str] = {
    "alpha": "A", "bravo": "B", "charlie": "C", "delta": "D", "echo": "E",
    "foxtrot": "F", "golf": "G", "hotel": "H", "india": "I", "juliet": "J",
    "juliett": "J", "kilo": "K", "lima": "L", "mike": "M", "november": "N",
    "oscar": "O", "papa": "P", "quebec": "Q", "romeo": "R", "sierra": "S",
    "tango": "T", "uniform": "U", "victor": "V", "whiskey": "W", "whisky": "W",
    "x-ray": "X", "xray": "X", "yankee": "Y", "zulu": "Z",
}

_WORD_TO_DIGIT: dict[str, str] = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

_SINGLE_UPPER = re.compile(r"^[A-Z]$")
_DIGIT_RUN = re.compile(r"^\d+$")
_SHORT_UPPER = re.compile(r"^[A-Z]{2,4}$")
_TOKEN_PATTERN = re.compile(r"(\s+)|(\S+)")


def _code_unit_for(raw_token: str) -> str | None:
    cleaned = raw_token.strip(",.;!?").lower()
    if not cleaned:
        return None
    if cleaned in _NATO_TO_LETTER:
        return _NATO_TO_LETTER[cleaned]
    if cleaned in _WORD_TO_DIGIT:
        return _WORD_TO_DIGIT[cleaned]
    if _SINGLE_UPPER.match(raw_token):
        return raw_token.upper()
    if _DIGIT_RUN.match(cleaned):
        return cleaned
    if _SHORT_UPPER.match(raw_token):
        return raw_token
    return None


def normalize(text: str) -> str:
    if not text:
        return text
    tokens = [(m.group(0), m.group(1) is not None) for m in _TOKEN_PATTERN.finditer(text)]
    if len(tokens) < 2:
        return text

    out_parts: list[str] = []
    i = 0
    while i < len(tokens):
        raw, is_ws = tokens[i]
        if is_ws:
            out_parts.append(raw)
            i += 1
            continue
        collapsed, advance = _try_collapse(tokens, i)
        if collapsed is not None:
            out_parts.append(collapsed)
            i += advance
        else:
            out_parts.append(raw)
            i += 1
    return "".join(out_parts)


def _try_collapse(tokens: list[tuple[str, bool]], start_idx: int) -> tuple[str | None, int]:
    """Try to collapse a code sequence starting at start_idx.

    Returns (collapsed_string, num_tokens_consumed) or (None, 0) if not a code.
    """
    candidates: list[str] = []
    consumed: list[int] = []
    i = start_idx

    while i < len(tokens):
        raw, is_ws = tokens[i]
        if is_ws:
            if not candidates:
                break
            i += 1
            continue

        cleaned = raw.strip(",.;!?").lower()
        if not cleaned:
            break

        # "double" / "triple" — look ahead for a code unit to repeat.
        if cleaned in ("double", "triple"):
            mult = 2 if cleaned == "double" else 3
            next_idx = _next_non_ws(tokens, i + 1)
            if next_idx is None:
                break
            next_raw, _ = tokens[next_idx]
            unit = _code_unit_for(next_raw)
            if unit is None:
                if not candidates:
                    return None, 0
                break
            candidates.append(unit * mult)
            consumed.append(i)
            consumed.append(next_idx)
            i = next_idx + 1
            continue

        unit = _code_unit_for(raw)
        if unit is None:
            break
        candidates.append(unit)
        consumed.append(i)
        i += 1

    if len(candidates) < 2:
        return None, 0
    advance = consumed[-1] - start_idx + 1
    return "".join(candidates), advance


def _next_non_ws(tokens: list[tuple[str, bool]], from_idx: int) -> int | None:
    i = from_idx
    while i < len(tokens) and tokens[i][1]:
        i += 1
    return i if i < len(tokens) else None
