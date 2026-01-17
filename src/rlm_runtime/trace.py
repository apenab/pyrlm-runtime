from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from typing import Literal

from .adapters.base import Usage


@dataclass
class TraceStep:
    step_id: int
    kind: Literal["root_call", "repl_exec", "subcall"]
    depth: int
    prompt_summary: str | None = None
    code: str | None = None
    stdout: str | None = None
    error: str | None = None
    usage: Usage | None = None
    cache_hit: bool = False
    input_hash: str | None = None
    output_hash: str | None = None
    cache_key: str | None = None


@dataclass
class Trace:
    steps: list[TraceStep]

    def add(self, step: TraceStep) -> None:
        self.steps.append(step)

    def to_json(self) -> str:
        def serialize(step: TraceStep) -> dict:
            data = asdict(step)
            if step.usage:
                data["usage"] = step.usage.to_dict()
            return data

        payload = [serialize(step) for step in self.steps]
        return json.dumps(payload, ensure_ascii=True, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "Trace":
        items = json.loads(raw)
        steps: list[TraceStep] = []
        for item in items:
            usage_data = item.get("usage")
            usage = Usage.from_dict(usage_data) if usage_data else None
            steps.append(
                TraceStep(
                    step_id=item["step_id"],
                    kind=item["kind"],
                    depth=item.get("depth", 0),
                    prompt_summary=item.get("prompt_summary"),
                    code=item.get("code"),
                    stdout=item.get("stdout"),
                    error=item.get("error"),
                    usage=usage,
                    cache_hit=item.get("cache_hit", False),
                    input_hash=item.get("input_hash"),
                    output_hash=item.get("output_hash"),
                    cache_key=item.get("cache_key"),
                )
            )
        return cls(steps=steps)
