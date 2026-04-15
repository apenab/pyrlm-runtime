from __future__ import annotations

import threading
from dataclasses import dataclass, asdict, field
import json
from typing import Literal

from .adapters.base import Usage


@dataclass
class TraceStep:
    step_id: int
    # Extended to support recursive subcalls and nested step types
    kind: Literal[
        "root_call",
        "repl_exec",
        "subcall",
        "recursive_subcall",
        "sub_root_call",
        "sub_repl_exec",
        "sub_subcall",
        "baseline_call",
    ]
    depth: int
    prompt_summary: str | None = None
    code: str | None = None
    output: str | None = None
    stdout: str | None = None
    error: str | None = None
    usage: Usage | None = None
    model: str | None = None
    elapsed: float | None = None
    cache_hit: bool = False
    input_hash: str | None = None
    output_hash: str | None = None
    cache_key: str | None = None
    parallel_group_id: str | None = None
    parallel_index: int | None = None
    parallel_total: int | None = None


@dataclass
class Trace:
    steps: list[TraceStep]
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, step: TraceStep) -> None:
        with self._lock:
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
                    output=item.get("output"),
                    stdout=item.get("stdout"),
                    error=item.get("error"),
                    usage=usage,
                    model=item.get("model"),
                    elapsed=item.get("elapsed"),
                    cache_hit=item.get("cache_hit", False),
                    input_hash=item.get("input_hash"),
                    output_hash=item.get("output_hash"),
                    cache_key=item.get("cache_key"),
                    parallel_group_id=item.get("parallel_group_id"),
                    parallel_index=item.get("parallel_index"),
                    parallel_total=item.get("parallel_total"),
                )
            )
        return cls(steps=steps)
