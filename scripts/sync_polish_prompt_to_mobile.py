#!/usr/bin/env python3
"""Sync PC's polish prompt to Mobile's PolishPromptLoader.kt.

Source of truth: this repo's `src/config.yaml` → `llm_polish.system_prompt`.

This script rewrites the block between the BEGIN-SYNC and END-SYNC marker
comments in whisper-mobile's PolishPromptLoader.kt with the current PC value,
so the two apps' polish prompts stay byte-identical.

Usage (run from the whisper-pc repo root):
    python scripts/sync_polish_prompt_to_mobile.py [--mobile-repo PATH] [--check]

Flags:
    --mobile-repo PATH   Override the default sibling-checkout path
                         (../whisper-mobile).
    --check              Don't write; exit 1 if Mobile is out of sync.
                         Useful as a pre-commit / CI gate.

Edit workflow:
    1. Edit src/config.yaml (and src/config_schema.yaml for the schema default).
    2. Run this script.
    3. Commit BOTH repos.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MOBILE_REPO = REPO_ROOT.parent / "whisper-mobile"
MOBILE_LOADER_REL = (
    "app/src/main/kotlin/com/aniket91jain/whispermobile/"
    "engine/polish/PolishPromptLoader.kt"
)

BEGIN_MARKER = "// BEGIN-SYNC: polish prompt body — auto-replaced by sync_polish_prompt_to_mobile.py"
END_MARKER = "// END-SYNC: polish prompt body"


def load_pc_prompt() -> str:
    """Read the canonical polish prompt from PC's config.yaml."""
    config_path = REPO_ROOT / "src" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prompt = cfg.get("llm_polish", {}).get("system_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise SystemExit(
            f"Could not read llm_polish.system_prompt from {config_path}"
        )
    return prompt


def render_kotlin_block(prompt_body: str) -> str:
    """Render the PC prompt as Kotlin source between the BEGIN/END markers.

    The block is exactly:
        // BEGIN-SYNC: ...
            private val BASE = ""\"
                <lines indented by 8 spaces so trimIndent() yields the
                 same content as the PC YAML literal block>
            ""\".trimIndent()
        // END-SYNC: ...

    Kotlin's trimIndent() strips the common minimum leading whitespace from
    all non-blank lines. We emit lines with 8 leading spaces so trimIndent
    yields exactly the prompt body (with no leading whitespace).

    The prompt's example lines use a literal `\\n` in the rendered output
    (LLM-facing escape). Both PC YAML and Kotlin raw strings preserve `\\n`
    as two characters by default, so no further escaping is needed.
    """
    indented_lines = [
        ("        " + line) if line.strip() else ""
        for line in prompt_body.splitlines()
    ]
    indented_body = "\n".join(indented_lines)
    return (
        f"{BEGIN_MARKER}\n"
        f"    private val BASE = \"\"\"\n"
        f"{indented_body}\n"
        f"    \"\"\".trimIndent()\n"
        f"    {END_MARKER}"
    )


def replace_marked_block(file_text: str, new_block: str) -> str:
    """Replace whatever sits between BEGIN_MARKER and END_MARKER (inclusive
    of both markers) with new_block. Raises if either marker is missing."""
    # Match the begin marker line, everything up through the end marker line.
    # The pattern uses [\s\S]*? (lazy) so it doesn't span past the first
    # END-SYNC found after BEGIN-SYNC.
    pattern = re.compile(
        re.escape(BEGIN_MARKER) + r"[\s\S]*?" + re.escape(END_MARKER)
    )
    if not pattern.search(file_text):
        raise SystemExit(
            "Could not locate BEGIN-SYNC / END-SYNC markers in Mobile loader. "
            "Make sure both marker comments are present and spelled exactly."
        )
    # Use a function replacement so re.sub does NOT interpret backslash
    # escapes in new_block (the prompt body contains literal `\n` sequences
    # in its examples; with a string replacement, re.sub would convert them
    # to actual newlines).
    return pattern.sub(lambda _m: new_block, file_text, count=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mobile-repo",
        type=Path,
        default=DEFAULT_MOBILE_REPO,
        help="Path to the whisper-mobile repo (default: ../whisper-mobile).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Don't write; exit 1 if Mobile is out of sync.",
    )
    args = parser.parse_args()

    mobile_loader = args.mobile_repo / MOBILE_LOADER_REL
    if not mobile_loader.is_file():
        raise SystemExit(f"Mobile loader not found at {mobile_loader}")

    pc_prompt = load_pc_prompt()
    new_block = render_kotlin_block(pc_prompt)

    current_text = mobile_loader.read_text(encoding="utf-8")
    new_text = replace_marked_block(current_text, new_block)

    if new_text == current_text:
        print(f"In sync: {mobile_loader}")
        return 0

    if args.check:
        print(f"OUT OF SYNC: {mobile_loader}", file=sys.stderr)
        print(
            "Run `python scripts/sync_polish_prompt_to_mobile.py` from the "
            "whisper-pc repo root to update.",
            file=sys.stderr,
        )
        return 1

    mobile_loader.write_text(new_text, encoding="utf-8")
    print(f"Updated: {mobile_loader}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
