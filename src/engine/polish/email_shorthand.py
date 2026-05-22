"""Convert 'aniket at gmail dot com' to 'aniket@gmail.com'.

Python port of mobile's `EmailShorthand.kt`.

Conservative: only fires on a recognisable email shape — `<word> at <word>
dot <word>[ dot <word>]`. Won't fire on "meet me at the office".
"""

from __future__ import annotations

import re

_EMAIL_PATTERN = re.compile(
    r"([A-Za-z0-9._-]+)\s+at\s+([A-Za-z0-9_-]+)((?:\s+dot\s+[A-Za-z0-9_-]+){1,3})",
    re.IGNORECASE,
)
_DOT_SUB = re.compile(r"\s+dot\s+", re.IGNORECASE)


def apply(text: str) -> str:
    def repl(m: re.Match) -> str:
        local = m.group(1)
        domain = m.group(2)
        tld_chain = m.group(3)  # " dot com" or " dot co dot uk"
        tld_formatted = _DOT_SUB.sub(".", tld_chain)
        return f"{local}@{domain}{tld_formatted}"

    return _EMAIL_PATTERN.sub(repl, text)
