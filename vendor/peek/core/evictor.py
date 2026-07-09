"""Priority-based eviction enforcing a hard token budget on the context map.

Items are evicted in ascending order of their accumulated Distiller score,
ties broken by item age (older IDs evicted first). See §3.2 of the PEEK paper.

peek-patch C3 (score decay):
existing scores are multiplied by ``SCORE_DECAY`` before each tagging pass,
and ``neutral`` tags contribute ``NEUTRAL_PENALTY`` (≥ 0) instead of 0.
Setting both to ``1.0`` / ``0.0`` recovers upstream behaviour exactly.
This removes the score-zero stickiness failure mode documented in
``docs/peek-bench/PEEK-DIAGNOSIS.md`` — items the Distiller cannot
positively endorse drift toward eviction over time, freeing slots for
newer evidence. Validated empirically: +5.7pp aggregate vs upstream on
a 5-context oolong-synth in-session A/B (see ``PEEK-EXPERIMENTS.md``
Phase 4.C.1).
"""

from __future__ import annotations

import re
from collections.abc import Callable

from peek.core.context_map import ContextMap
from peek.core.types import ItemTag

_NUMERIC_TAIL = re.compile(r"-(\d+)$")

# peek-patch C3. Upstream behaviour: SCORE_DECAY=1.0, NEUTRAL_PENALTY=0.0.
SCORE_DECAY: float = 0.85
NEUTRAL_PENALTY: float = 0.5


def update_scores(
    scores: dict[str, float],
    tags: dict[str, ItemTag],
    *,
    decay: float = SCORE_DECAY,
    neutral_penalty: float = NEUTRAL_PENALTY,
) -> dict[str, float]:
    """Apply decay to ``scores`` then add Distiller tag contributions."""
    out: dict[str, float] = {k: float(v) * decay for k, v in scores.items()}
    for item_id, tag in tags.items():
        if tag == "helpful":
            out[item_id] = out.get(item_id, 0.0) + 1.0
        elif tag in ("harmful", "stale"):
            out[item_id] = out.get(item_id, 0.0) - 1.0
        elif tag == "neutral":
            out[item_id] = out.get(item_id, 0.0) - neutral_penalty
        else:
            out.setdefault(item_id, 0.0)
    return out


def evict(
    cmap: ContextMap,
    scores: dict[str, float],
    token_budget: int,
    token_counter: Callable[[str], int],
) -> ContextMap:
    if token_counter(cmap.text) <= token_budget:
        return cmap

    ordered_ids = [it.id for it in cmap.items()]
    ordered_ids.sort(key=lambda bid: (scores.get(bid, 0), _id_age(bid)))

    removed: set[str] = set()
    for bid in ordered_ids:
        removed.add(bid)
        trial = _strip_items(cmap.text, removed)
        if token_counter(trial) <= token_budget:
            return ContextMap(trial + "\n" if not trial.endswith("\n") else trial)
    return ContextMap(_strip_items(cmap.text, set(ordered_ids)))


def _id_age(item_id: str) -> int:
    m = _NUMERIC_TAIL.search(item_id)
    return int(m.group(1)) if m else 0


def _strip_items(text: str, ids: set[str]) -> str:
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and "]" in stripped:
            bid = stripped[1 : stripped.index("]")]
            if bid in ids:
                continue
        out.append(line)
    return "\n".join(out)
