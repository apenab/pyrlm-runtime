from __future__ import annotations

from peek._io import extract_json, load_prompt
from peek.core.types import DistillerOutput, ItemTag, Usage
from peek.llm.base import LMClient

_VALID_TAGS = {"helpful", "harmful", "neutral", "stale"}


class Distiller:
    """Extracts transferable contextual knowledge from an agent trajectory.

    Produces a diagnosis, per-item tags for the current map, and a set of
    cache candidates. Uses the no-ground-truth prompt that shipped with the
    paper experiments by default.
    """

    def __init__(self, client: LMClient, prompt: str | None = None):
        self.client = client
        self.prompt = prompt or load_prompt("distiller.txt")

    def __call__(
        self,
        trajectory: str,
        context_map: str,
        *,
        question: str = "",
    ) -> DistillerOutput:
        content = self.prompt.format(
            playbook=context_map or "N/A",
            trace_history=trajectory,
        )
        if question:
            content += f"\n\n- Task context (the question the agent was answering):\n{question}\n"

        raw = self.client.completion([{"role": "user", "content": content}]) or ""
        usage = self.client.last_usage()
        return _parse(raw, usage)


def _parse(raw: str, usage: Usage) -> DistillerOutput:
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        return DistillerOutput(raw=raw, diagnosis=raw, usage=usage)

    tags_in = parsed.get("item_tags") or parsed.get("bullet_tags") or {}
    tags: dict[str, ItemTag] = {}
    if isinstance(tags_in, dict):
        for k, v in tags_in.items():
            if isinstance(v, str) and v in _VALID_TAGS:
                tags[str(k)] = v  # type: ignore[assignment]

    candidates = parsed.get("cache_candidates") or []
    if not isinstance(candidates, list):
        candidates = []

    return DistillerOutput(
        raw=raw,
        diagnosis=str(parsed.get("diagnosis", "")),
        item_tags=tags,
        cache_candidates=candidates,
        usage=usage,
    )
