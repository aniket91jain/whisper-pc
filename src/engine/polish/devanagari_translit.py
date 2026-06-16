"""Devanagari → Latin transliteration for the ElevenLabs STT path.

Belt-and-suspenders safety net for Hindi loanwords. We pin ElevenLabs to
`language_code=eng` which should romanize Hindi as it transcribes, but if any
Devanagari slips through (mid-sentence code-switch the language pin missed,
or future model changes) this pass converts those spans to Latin (ITRANS-ish)
locally so the user never sees Devanagari in the pasted text.

Cheap and additive: regex-detects Devanagari spans (U+0900–U+097F), feeds
each span through `indic_transliteration.sanscript` (Devanagari → ITRANS),
splices the Latin back in. Untouched if the transcript contains no
Devanagari (the dominant case).

Scheme choice: VELTHUIS produces lowercase ASCII with doubled vowels for
length (`namaste`, `dhanyavaada`) — matches how Indians informally write
Hindi in Latin (texting style). ITRANS / HK use uppercase as quantity
markers (`dhanyavAda`, `maiM`) which read as ugly mixed-case. IAST uses
diacritics which aren't ASCII.

After VELTHUIS we strip its anusvara/visarga markers (`.m`, `~m`, `.h`)
which are technically informative but visually noisy and unnecessary for
the dictation use case — `nahii.m` → `nahiin`, `haa~m` → `haan`.
"""

from __future__ import annotations

import re


# Devanagari code block (excludes Devanagari Extended at U+A8E0-U+A8FF; rare,
# add later if it shows up).
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]+(?:[\s‌‍][ऀ-ॿ]+)*")

# Post-VELTHUIS cleanup: turn its dot/tilde-prefixed nasal markers into plain
# letters that read naturally in Latin script.
#   `.m` / `~m`  → `n`  (anusvara / chandrabindu — both render as a nasal `n`
#                         in informal romanization, e.g. `nahii.m` → `nahiin`)
#   `.h`         → `h`  (visarga — rare in modern Hindi)
#   `.t` `.d` `.n` `.s` `.r` → `t d n s r` (retroflex markers — informal
#                                            Latin-Hindi drops the distinction)
_VELTHUIS_CLEANUP = [
    (re.compile(r"\.m\b"), "n"),
    (re.compile(r"~m\b"), "n"),
    (re.compile(r"\.h\b"), "h"),
    (re.compile(r"\.([tdnsr])"), r"\1"),
]


def has_devanagari(text: str) -> bool:
    """Cheap O(n) scan for any Devanagari codepoint — saves the import cost
    when there's nothing to transliterate (the dominant case)."""
    if not text:
        return False
    for ch in text:
        if "ऀ" <= ch <= "ॿ":
            return True
    return False


def transliterate(text: str) -> str:
    """Convert any Devanagari spans in `text` to ITRANS Latin. Leaves the
    rest of the text untouched.

    Safe to call on every transcript: skips the import + work when the input
    contains no Devanagari at all.
    """
    if not has_devanagari(text):
        return text

    # Defer the import — only pay the cost when we actually have work to do.
    from indic_transliteration import sanscript

    def _convert(m: re.Match[str]) -> str:
        try:
            out = sanscript.transliterate(m.group(0), sanscript.DEVANAGARI, sanscript.VELTHUIS)
        except Exception:
            # If the library chokes on something exotic, leave the original
            # Devanagari in place rather than dropping the span. A bad pass
            # shouldn't lose content.
            return m.group(0)
        for pat, repl in _VELTHUIS_CLEANUP:
            out = pat.sub(repl, out)
        return out

    return _DEVANAGARI_RE.sub(_convert, text)
