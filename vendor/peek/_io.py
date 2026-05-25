from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from an LLM response."""
    s = text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    for block in _scan_fenced_blocks(text):
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue

    for blob in _scan_balanced_braces(text):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            continue
    return None


def _scan_fenced_blocks(text: str) -> list[str]:
    """Return the contents of every ```...``` fenced block in ``text``.

    peek-patch BUG-FIX: the upstream implementation used
    ``re.findall(r'```(?:json)?\\s*(.*?)\\s*```', text, re.DOTALL | re.IGNORECASE)``
    which can exhibit catastrophic backtracking when the LLM emits an opening
    fence without a matching closing fence (a real failure mode observed in
    a 45-minute hung benchmark run). This O(n) state machine is bulletproof
    against malformed input: unclosed fences are ignored gracefully instead
    of triggering exponential regex search.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        start = text.find("```", i)
        if start < 0:
            return out
        # Skip the fence marker; also skip an optional "json" tag and any
        # trailing whitespace on that line up to the first newline.
        after = start + 3
        nl = text.find("\n", after)
        if nl < 0:
            return out
        end = text.find("```", nl + 1)
        if end < 0:
            # Unclosed fence: bail out instead of treating the rest of the
            # document as the (possibly enormous) match body.
            return out
        out.append(text[nl + 1 : end])
        i = end + 3
    return out


def _scan_balanced_braces(text: str) -> list[str]:
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, start = 1, i
        i += 1
        while i < n and depth > 0:
            c = text[i]
            if c == '"':
                i += 1
                while i < n and text[i] != '"':
                    i += 2 if text[i] == "\\" else 1
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        if depth == 0:
            out.append(text[start:i])
    return out
