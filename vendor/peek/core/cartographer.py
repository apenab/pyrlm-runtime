from __future__ import annotations

from peek._io import extract_json, load_prompt
from peek.core.context_map import normalize_section
from peek.core.types import SECTIONS, CartographerOutput, Operation
from peek.llm.base import LMClient

_ALLOWED_SECTIONS: frozenset[str] = frozenset(SECTIONS)


class Cartographer:
    """Translates Distiller output into structured edits against the context map.

    Edits are validated: only ADD/DELETE/REPLACE operations targeting whitelisted
    sections are returned. The operations are intended to be applied via
    :meth:`peek.core.context_map.ContextMap.apply`.
    """

    def __init__(self, client: LMClient, prompt: str | None = None):
        self.client = client
        self.prompt = prompt or load_prompt("cartographer.txt")

    def __call__(
        self,
        *,
        reflection: str,
        current_map: str,
        question: str,
        token_budget: int,
        current_tokens: int,
    ) -> CartographerOutput:
        content = self.prompt.format(
            reflection=reflection,
            current_playbook=current_map,
            question_context=question,
            token_budget=token_budget,
            current_tokens=current_tokens,
        )
        raw = self.client.completion([{"role": "user", "content": content}]) or ""
        usage = self.client.last_usage()
        return CartographerOutput(raw=raw, operations=_parse_ops(raw), usage=usage)


def _parse_ops(raw: str) -> list[Operation]:
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        return []
    if not isinstance(parsed.get("reasoning"), str):
        return []
    raw_ops = parsed.get("operations")
    if not isinstance(raw_ops, list):
        return []

    out: list[Operation] = []
    for op in raw_ops:
        if not isinstance(op, dict):
            continue
        kind = op.get("type")
        if kind == "ADD":
            section = op.get("section")
            content = op.get("content")
            if not isinstance(section, str) or not isinstance(content, str) or not content:
                continue
            section_norm = normalize_section(section)
            if section_norm not in _ALLOWED_SECTIONS:
                continue
            out.append(Operation(type="ADD", section=section_norm, content=content))
        elif kind == "DELETE":
            item_id = op.get("item_id") or op.get("bullet_id")
            if isinstance(item_id, str) and item_id:
                out.append(Operation(type="DELETE", item_id=item_id))
        elif kind == "REPLACE":
            item_id = op.get("item_id") or op.get("bullet_id")
            content = op.get("content")
            if isinstance(item_id, str) and item_id and isinstance(content, str) and content:
                out.append(Operation(type="REPLACE", item_id=item_id, content=content))
    return out
