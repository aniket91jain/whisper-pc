"""Common ElevenLabs / generic-STT mishears for the user's proper nouns.

Python port of mobile's `RegexPrePass.PROPER_NOUN_FIXES` (Kotlin). The two
should be kept in sync — when one gets a new mishear → correct mapping,
so should the other.

Applied as case-insensitive whole-word regex substitution before any other
polish step. Idempotent: running twice produces the same result.

To add a new mapping, just append to PROPER_NOUN_FIXES. Order matters when
one mishear is a prefix of another — keep the more specific mapping first.
"""

from __future__ import annotations

import re


# Order matters when one mishear is a prefix of another.
PROPER_NOUN_FIXES: list[tuple[str, str]] = [
    # Products
    ("owner ras", "OwnerRez"),
    ("owner res", "OwnerRez"),
    ("owner raz", "OwnerRez"),
    ("ownerres", "OwnerRez"),
    ("ownerraz", "OwnerRez"),
    ("own arrays", "OwnerRez"),
    ("price labs", "PriceLabs"),
    ("price laps", "PriceLabs"),
    ("price letters", "PriceLabs"),
    ("pricelabs", "PriceLabs"),
    ("plowed", "Plaud"),
    ("plod", "Plaud"),
    ("sonic ox", "Soniox"),
    ("sonics", "Soniox"),
    ("sony ox", "Soniox"),
    # Locations
    ("asa gaon", "Assagao"),
    ("asagao", "Assagao"),
    ("majoreda", "Majorda"),
    ("majoredo", "Majorda"),
    # Business
    ("jasmine journey", "Jasmine Journeys"),
    ("j j", "JJ"),
    # People (extend as observed)
    ("joppen chetta", "Joppan chetta"),
    ("joppen", "Joppan"),
    ("preksha sha", "Preksha Shah"),
    ("vinu daniels", "Vinu Daniel"),
]


def apply(text: str) -> str:
    """Replace common mishears with correct spellings. Whole-word, case-insensitive."""
    out = text
    for wrong, right in PROPER_NOUN_FIXES:
        out = re.sub(r"\b" + re.escape(wrong) + r"\b", right, out, flags=re.IGNORECASE)
    return out
