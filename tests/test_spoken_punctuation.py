"""Parity + regression fixtures for the spoken-punctuation normalizer.

Run:  python tests/test_spoken_punctuation.py   (from whisper-pc/)
No third-party deps — imports only engine.polish.spoken_punctuation (re-only).

The CASES list is the shared contract with the mobile SpokenPunctuation.kt
port; keep the Kotlin test in sync with this fixture.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.polish.spoken_punctuation import normalize  # noqa: E402

# (input, expected) — expected is the normalizer output BEFORE token expansion,
# so "[blank line]" / "[newline]" appear literally here (regex_polish step 7
# turns them into real newlines downstream).
CASES = [
    # --- the user's headline commands, ElevenLabs-style (capitalized/punctuated) ---
    ("Quote hello world unquote", '"hello world"'),
    ("Quote, hello world, unquote.", '"hello world".'),
    ("open quote keep it simple close quote", '"keep it simple"'),
    # "bracket" defaults to ROUND brackets (user pref 2026-07-05); square
    # brackets need the explicit word "square".
    ("Open bracket five close bracket", "(five)"),
    ("Open bracket five, close bracket.", "(five)"),
    ("close the door open bracket gently close bracket now",
     "close the door (gently) now"),
    ("open square bracket five close square bracket", "[five]"),
    # plain "slash" -> / (user pref 2026-07-05)
    ("the file is a slash b", "the file is a/b"),
    ("the path is foo slash bar slash baz", "the path is foo/bar/baz"),
    ("first thought new paragraph second thought",
     "first thought[blank line]second thought"),

    # --- Whisper-style regression (lowercase, no auto punctuation) ---
    ("say hello comma then go", "say hello, then go"),
    ("open paren note close paren", "(note)"),
    ("the meeting is at noon full stop", "the meeting is at noon."),
    ("send it to me at sign now", "send it to me@now"),
    ("question one new line question two",
     "question one[newline]question two"),

    # --- start / end of utterance ---
    ("open bracket alpha close bracket", "(alpha)"),
    ("say hello comma", "say hello,"),

    # --- risky words must NOT fire as content ---
    ("during the grace period of rest", "during the grace period of rest"),
    ("he made a dash for the exit", "he made a dash for the exit"),
    ("colon cancer is treatable", "colon cancer is treatable"),
    ("a quote from the book", "a quote from the book"),

    # --- safe multi-word still fires after a comma ElevenLabs inserts ---
    ("the list is one, comma two", "the list is one, two"),
]


def main() -> int:
    failures = []
    for src, expected in CASES:
        got = normalize(src)
        status = "ok " if got == expected else "FAIL"
        if got != expected:
            failures.append((src, expected, got))
        print(f"[{status}] {src!r}\n        -> {got!r}")
    print()
    if failures:
        print(f"{len(failures)}/{len(CASES)} FAILED:")
        for src, expected, got in failures:
            print(f"  input:    {src!r}")
            print(f"  expected: {expected!r}")
            print(f"  got:      {got!r}")
        return 1
    print(f"All {len(CASES)} cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
