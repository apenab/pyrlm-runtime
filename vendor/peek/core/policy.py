"""Programmable cache policy implementing PEEK Algorithm 1.

The policy wraps a single context map and an LM-backed Distiller and
Cartographer. After each agent run on a recurring external context, the caller
hands the trajectory to :meth:`CachePolicy.update`; for the first
``evolve_steps`` calls the map is updated, otherwise it is reused as-is.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from peek.core.cartographer import Cartographer
from peek.core.context_map import ContextMap
from peek.core.distiller import Distiller
from peek.core.evictor import evict, update_scores
from peek.core.types import DistillerOutput, ItemTag, Usage
from peek.llm.base import LMClient

TokenCounter = Callable[[str], int]


def _default_tokenizer() -> TokenCounter:
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    return lambda s: len(enc.encode(s))


@dataclass
class UpdateResult:
    distiller: DistillerOutput
    cartographer_raw: str
    operations_applied: int
    map_text: str
    usage: Usage


@dataclass
class CachePolicy:
    """Maintains a single context map for a recurring external context.

    Parameters
    ----------
    client : LMClient
        Language-model client used by both Distiller and Cartographer.
    token_budget : int
        Hard token budget enforced by the Evictor after each update.
    evolve_steps : int | None
        Number of update calls during which the map is allowed to evolve.
        ``None`` means evolve indefinitely (m = n in the paper).
    cmap : ContextMap | None
        Starting map. Defaults to the paper's initial context map.
    token_counter : callable | None
        ``str -> int`` token counter. Defaults to ``tiktoken`` ``o200k_base``.
    """

    client: LMClient
    token_budget: int = 1024
    evolve_steps: int | None = None
    cmap: ContextMap = field(default_factory=ContextMap.initial)
    token_counter: TokenCounter | None = None
    # peek-patch C3 — scores carry float values after score-decay was introduced.
    scores: dict[str, float] = field(default_factory=dict)
    steps: int = 0

    def __post_init__(self) -> None:
        self._distiller = Distiller(self.client)
        self._cartographer = Cartographer(self.client)
        if self.token_counter is None:
            self.token_counter = _default_tokenizer()

    @property
    def current_map_text(self) -> str:
        return self.cmap.text

    @property
    def evolving(self) -> bool:
        return self.evolve_steps is None or self.steps < self.evolve_steps

    def update(
        self,
        *,
        trajectory: str,
        question: str = "",
    ) -> UpdateResult | None:
        """Run one cache-policy step. Returns ``None`` when evolution is frozen."""
        if not self.evolving:
            self.steps += 1
            return None

        distilled = self._distiller(
            trajectory,
            self.cmap.text,
            question=question,
        )
        self.scores = update_scores(self.scores, distilled.item_tags)

        assert self.token_counter is not None
        edits = self._cartographer(
            reflection=distilled.raw,
            current_map=self.cmap.text,
            question=question,
            token_budget=self.token_budget,
            current_tokens=self.token_counter(self.cmap.text),
        )
        if edits.operations:
            self.cmap = self.cmap.apply(edits.operations)
        self.cmap = evict(self.cmap, self.scores, self.token_budget, self.token_counter)
        self.steps += 1

        return UpdateResult(
            distiller=distilled,
            cartographer_raw=edits.raw,
            operations_applied=len(edits.operations),
            map_text=self.cmap.text,
            usage=distilled.usage + edits.usage,
        )

    def tag(self, item_id: str, tag: ItemTag) -> None:
        """Apply a manual Distiller-equivalent tag to a single item."""
        self.scores = update_scores(self.scores, {item_id: tag})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "map_text": self.cmap.text,
            "scores": self.scores,
            "steps": self.steps,
            "token_budget": self.token_budget,
            "evolve_steps": self.evolve_steps,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        client: LMClient,
        token_counter: TokenCounter | None = None,
    ) -> CachePolicy:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            client=client,
            token_budget=int(payload.get("token_budget", 1024)),
            evolve_steps=payload.get("evolve_steps"),
            cmap=ContextMap(payload["map_text"]),
            token_counter=token_counter,
            scores=dict(payload.get("scores", {})),
            steps=int(payload.get("steps", 0)),
        )
