"""Render llm_polish.proper_nouns into the text block that substitutes the
{{PROPER_NOUNS_BLOCK}} placeholder in the polish system prompt.

Output shape matches the pre-refactor hardcoded block byte-for-byte when the
structured config is seeded from the original list — see Phase A regression
check in plans/whisper-vocab-1.md.
"""

PLACEHOLDER = "{{PROPER_NOUNS_BLOCK}}"

_HEADER = (
    "PROPER NOUNS (correct STT errors for these names only, no others). "
    "HIGH PRIORITY: when ANY of the listed misheard variants appears in the "
    "input, ALWAYS replace it with the correct spelling, even if the "
    "surrounding context is ambiguous."
)


def _format_entry(entry):
    word = (entry.get("word") or "").strip()
    misheard_raw = entry.get("misheard") or []
    misheard = [m.strip() for m in misheard_raw if isinstance(m, str) and m.strip()]
    if misheard:
        return f"{word} ({' / '.join(misheard)})"
    return word


def _format_category(label, entries):
    words = [_format_entry(e) for e in (entries or []) if isinstance(e, dict) and e.get("word")]
    return f"  {label}: {', '.join(words)}."


def render(proper_nouns):
    """Build the multi-line PROPER NOUNS block from structured config.

    proper_nouns is the dict at config.llm_polish.proper_nouns:
        {locations: [{word, misheard}, ...], people: [...], products: [...]}
    """
    if not isinstance(proper_nouns, dict):
        return _HEADER
    return "\n".join([
        _HEADER,
        _format_category("Locations", proper_nouns.get("locations")),
        _format_category("People",    proper_nouns.get("people")),
        _format_category("Products",  proper_nouns.get("products")),
    ])


def substitute(system_prompt, proper_nouns):
    """Replace the {{PROPER_NOUNS_BLOCK}} placeholder in the prompt with the rendered block.

    If the placeholder is absent (legacy hardcoded prompts), returns the prompt unchanged.
    """
    if not system_prompt or PLACEHOLDER not in system_prompt:
        return system_prompt
    return system_prompt.replace(PLACEHOLDER, render(proper_nouns))
