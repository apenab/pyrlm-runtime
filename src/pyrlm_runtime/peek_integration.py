"""Optional integration with peek-ai (https://github.com/zhuohangu/peek).

Adds PEEK orientation-cache support to pyrlm_runtime: a small, constant-sized
context map injected into the system prompt that accumulates reusable structural
knowledge about a recurring external context across multiple RLM runs.

Requires the optional `peek` extra::

    pip install pyrlm-runtime[peek]   # or: pip install peek-ai

Usage::

    from pyrlm_runtime.peek_integration import PeekSession

    session = PeekSession.create(adapter=my_adapter)
    for query in queries:
        rlm = RLM(adapter=my_adapter,
                  system_prompt_supplement=session.system_prompt_supplement)
        answer, trace = rlm.run(query, context)
        session.update_from_run(trace, query)

    session.save("corpus.peek.json")
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .adapters.base import ModelAdapter
from .trace import Trace

if TYPE_CHECKING:
    from peek import CachePolicy, UpdateResult
    from peek.core.types import Usage as PeekUsage


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_jsonable(v) for v in obj]
    return obj


def _require_peek() -> None:
    try:
        import peek  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "peek-ai is required for PEEK integration. "
            "Install with: pip install pyrlm-runtime[peek]  or  pip install peek-ai"
        ) from exc


class _PeekLMClientAdapter:
    """Bridges pyrlm_runtime's ModelAdapter to the peek.LMClient protocol."""

    def __init__(self, adapter: ModelAdapter, *, max_tokens: int = 2048) -> None:
        self._adapter = adapter
        self._max_tokens = max_tokens
        self._last_input: int = 0
        self._last_output: int = 0

    def completion(self, messages: list[dict[str, Any]]) -> str:
        response = self._adapter.complete(messages, max_tokens=self._max_tokens)
        self._last_input = response.usage.prompt_tokens
        self._last_output = response.usage.completion_tokens
        return response.text

    def last_usage(self) -> PeekUsage:
        from peek.core.types import Usage as PeekUsage

        return PeekUsage(input_tokens=self._last_input, output_tokens=self._last_output)


def trace_to_peek_trajectory(trace: Trace, query: str = "") -> str:
    """Serialize a Trace into the plain-string trajectory format that peek's Distiller expects.

    Root-level (depth=0) steps are rendered with full code/output. Nested subcall
    steps (depth>0) are included with a depth marker so the Distiller can see
    recursion but stays focused on root-level orientation work.
    """
    parts: list[str] = []
    if query:
        parts.append(f"Question: {query}\n")

    for step in trace.steps:
        prefix = "  " * step.depth
        header = f"{prefix}--- STEP {step.step_id} [{step.kind}]"
        if step.depth > 0:
            header += f" (depth={step.depth})"
        parts.append(header)

        if step.kind in ("root_call", "sub_root_call"):
            if step.output:
                for line in step.output.splitlines():
                    parts.append(f"{prefix}{line}")
        elif step.kind in ("repl_exec", "sub_repl_exec"):
            if step.code:
                parts.append(f"{prefix}Code:")
                parts.append(f"{prefix}```python")
                for line in step.code.splitlines():
                    parts.append(f"{prefix}{line}")
                parts.append(f"{prefix}```")
            if step.stdout:
                parts.append(f"{prefix}Output:")
                for line in step.stdout.splitlines():
                    parts.append(f"{prefix}{line}")
            if step.error:
                parts.append(f"{prefix}Error: {step.error}")
        elif step.kind in ("subcall", "recursive_subcall", "sub_subcall"):
            if step.output:
                truncated = step.output[:500]
                if len(step.output) > 500:
                    truncated += " [truncated]"
                parts.append(f"{prefix}Result: {truncated}")

    return "\n".join(parts)


class PeekSession:
    """Maintains a PEEK context map across RLM runs on a recurring external context.

    The map is injected via ``system_prompt_supplement`` and updated after each
    run via ``update_from_run``. Evolution freezes after ``evolve_steps`` updates
    (``None`` = evolve indefinitely). The map persists to disk via ``save``/``load``.

    When ``trace_dir`` is set, a per-update JSON snapshot is written to
    ``trace_dir/q{NN}.json`` for offline diagnostic analysis. See
    ``examples/peek_bench/analyze_peek_trace.py``.
    """

    def __init__(self, policy: CachePolicy, *, trace_dir: Path | None = None) -> None:
        self._policy = policy
        self._trace_dir = Path(trace_dir) if trace_dir is not None else None
        if self._trace_dir is not None:
            self._trace_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        adapter: ModelAdapter,
        *,
        token_budget: int = 1024,
        evolve_steps: int | None = None,
        max_tokens: int = 2048,
        trace_dir: str | Path | None = None,
    ) -> PeekSession:
        """Create a fresh session backed by the given ModelAdapter.

        Args:
            adapter: ModelAdapter used for Distiller and Cartographer LLM calls.
            token_budget: Hard token cap for the context map (B in the paper). Default: 1024.
            evolve_steps: How many updates before the map is frozen. None = always evolve.
            max_tokens: Max completion tokens for Distiller/Cartographer calls.
            trace_dir: If set, write a JSON trace per ``update_from_run`` call.
        """
        _require_peek()
        from peek import CachePolicy

        lm_client = _PeekLMClientAdapter(adapter, max_tokens=max_tokens)
        policy = CachePolicy(
            client=lm_client,
            token_budget=token_budget,
            evolve_steps=evolve_steps,
        )
        return cls(policy, trace_dir=trace_dir)

    @property
    def system_prompt_supplement(self) -> str:
        """Context map text to prepend to the RLM system prompt.

        Returns an empty string until the map has at least one entry, so the
        first run (with an empty map) doesn't inject noise.
        """
        from peek import ContextMap

        cmap = ContextMap(self._policy.current_map_text)
        if not cmap.items():
            return ""
        return f"\n\n## Context Map (Orientation Cache)\n{self._policy.current_map_text}"

    def update_from_run(self, trace: Trace, query: str = "") -> UpdateResult | None:
        """Update the context map from the completed RLM run.

        Returns the UpdateResult (with per-step usage) or None if evolution is frozen.
        When ``trace_dir`` is set, also writes a JSON snapshot of the update.
        """
        trajectory = trace_to_peek_trajectory(trace, query=query)

        if self._trace_dir is None:
            return self._policy.update(trajectory=trajectory, question=query)

        snapshot_before = self._snapshot_map()
        step_idx = self._policy.steps
        result = self._policy.update(trajectory=trajectory, question=query)
        snapshot_after = self._snapshot_map()

        self._write_trace(
            step_idx=step_idx,
            query=query,
            trajectory=trajectory,
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
            result=result,
        )
        return result

    def _snapshot_map(self) -> dict[str, Any]:
        from peek import ContextMap

        cmap = ContextMap(self._policy.current_map_text)
        items = [
            {
                "id": it.id,
                "section": it.section,
                "content": it.content,
                "score": float(self._policy.scores.get(it.id, 0.0)),
            }
            for it in cmap.items()
        ]
        return {
            "text": self._policy.current_map_text,
            "items": items,
            "scores": {k: float(v) for k, v in self._policy.scores.items()},
        }

    def _write_trace(
        self,
        *,
        step_idx: int,
        query: str,
        trajectory: str,
        snapshot_before: dict[str, Any],
        snapshot_after: dict[str, Any],
        result: UpdateResult | None,
    ) -> None:
        assert self._trace_dir is not None

        before_ids = {it["id"] for it in snapshot_before["items"]}
        after_ids = {it["id"] for it in snapshot_after["items"]}
        before_by_id = {it["id"]: it for it in snapshot_before["items"]}
        after_by_id = {it["id"]: it for it in snapshot_after["items"]}

        evicted = [
            {**before_by_id[i], "reason": "evicted_or_deleted"}
            for i in (before_ids - after_ids)
        ]
        added = [after_by_id[i] for i in (after_ids - before_ids)]
        modified = [
            {
                "id": i,
                "before": before_by_id[i]["content"],
                "after": after_by_id[i]["content"],
                "score": after_by_id[i]["score"],
            }
            for i in (before_ids & after_ids)
            if before_by_id[i]["content"] != after_by_id[i]["content"]
        ]

        payload: dict[str, Any] = {
            "step_idx": step_idx,
            "query": query,
            "trajectory": trajectory,
            "map_before": snapshot_before,
            "map_after": snapshot_after,
            "items_evicted": evicted,
            "items_added": added,
            "items_modified": modified,
            "evolving": self._policy.evolving,
        }
        if result is None:
            payload["result"] = None
        else:
            payload["result"] = {
                "distiller": _to_jsonable(result.distiller),
                "cartographer_raw": result.cartographer_raw,
                "operations_applied": result.operations_applied,
                "usage": _to_jsonable(result.usage),
            }

        path = self._trace_dir / f"q{step_idx:02d}.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    @property
    def steps(self) -> int:
        """Number of update calls so far."""
        return self._policy.steps

    @property
    def evolving(self) -> bool:
        """True while the map is still being updated."""
        return self._policy.evolving

    def save(self, path: str | Path) -> None:
        """Persist the map and scores to a JSON file."""
        self._policy.save(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        adapter: ModelAdapter,
        *,
        max_tokens: int = 2048,
    ) -> PeekSession:
        """Load a previously saved session.

        Args:
            path: Path to the JSON file created by ``save``.
            adapter: ModelAdapter for future Distiller/Cartographer calls.
            max_tokens: Max completion tokens for those calls.
        """
        _require_peek()
        from peek import CachePolicy

        lm_client = _PeekLMClientAdapter(adapter, max_tokens=max_tokens)
        policy = CachePolicy.load(path, client=lm_client)
        return cls(policy)
