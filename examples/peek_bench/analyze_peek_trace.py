#!/usr/bin/env python3
"""Diagnostic analysis of PEEK trace files.

Reads the per-update JSON traces (q{NN}.json) written by `PeekSession` when
constructed with a `trace_dir`, and reports:

  - per-query churn (ops applied, items added/evicted/modified, map size, duplicate rate)
  - per-query Distiller behaviour (helpful / harmful / neutral tag distribution)
  - per-item lifetime (birth query, score timeline, death query, usage proxy)
  - side-by-side comparison across multiple contexts

Usage:
    uv run python examples/peek_bench/analyze_peek_trace.py \
        docs/peek-bench/runs/run_XXXX/traces

Optional args narrow the report:
    --ctx 30036                   # single context_window_id
    --top-items 10                # show this many top-/bottom-scoring items
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_context_traces(ctx_dir: Path) -> list[dict[str, Any]]:
    """Load all q{NN}.json files for a context, sorted by step_idx."""
    files = sorted(ctx_dir.glob("q*.json"))
    return [json.loads(f.read_text()) for f in files]


def discover_contexts(traces_root: Path) -> dict[int, Path]:
    """Return {context_window_id: directory} for each ctx_<id>/ under root."""
    out: dict[int, Path] = {}
    for sub in sorted(traces_root.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("ctx_"):
            continue
        try:
            cid = int(sub.name.removeprefix("ctx_"))
        except ValueError:
            continue
        out[cid] = sub
    return out


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------


def jaccard(a: str, b: str) -> float:
    """Token-level Jaccard over whitespace-split lowercase words."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def duplicate_pairs(items: list[dict[str, Any]], threshold: float = 0.7) -> int:
    """Count pairs of items in the same section with content Jaccard >= threshold."""
    by_section: dict[str, list[str]] = defaultdict(list)
    for it in items:
        by_section[it["section"]].append(it["content"])
    pairs = 0
    for contents in by_section.values():
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                if jaccard(contents[i], contents[j]) >= threshold:
                    pairs += 1
    return pairs


def _extract_json(raw: str) -> Any:
    """Best-effort copy of peek._io.extract_json — direct, fence, or balanced.

    The fence scan uses an O(n) state machine, not a regex with ``.*?`` plus
    ``re.DOTALL`` — that pattern can catastrophically backtrack on LLM output
    with an unclosed code fence. See vendor/peek/_io.py for the same fix.
    """
    try:
        return json.loads(raw)
    except Exception:
        pass
    # fenced-block scan (state machine, no regex)
    i, n = 0, len(raw)
    while i < n:
        start = raw.find("```", i)
        if start < 0:
            break
        nl = raw.find("\n", start + 3)
        if nl < 0:
            break
        end = raw.find("```", nl + 1)
        if end < 0:
            break
        try:
            return json.loads(raw[nl + 1 : end].strip())
        except Exception:
            pass
        i = end + 3
    # balanced braces fallback
    depth = 0
    start = -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(raw[start : i + 1])
                except Exception:
                    pass
    return None


def cartographer_ops_breakdown(t: dict[str, Any]) -> dict[str, int]:
    """Parse cartographer_raw and count operations vs silent failures.

    A DELETE/REPLACE op silently fails when its item_id is not in map_before.
    """
    result = t.get("result") or {}
    raw = result.get("cartographer_raw") or ""
    parsed = _extract_json(raw)
    ops = (parsed or {}).get("operations") if isinstance(parsed, dict) else None
    if not isinstance(ops, list):
        return {"emit_add": 0, "emit_del": 0, "emit_repl": 0, "fail_del": 0, "fail_repl": 0}
    before_ids = {it["id"] for it in t["map_before"]["items"]}
    n_add = n_del = n_repl = n_fail_del = n_fail_repl = 0
    for op in ops:
        if not isinstance(op, dict):
            continue
        kind = op.get("type")
        item_id = op.get("item_id") or op.get("bullet_id")
        if kind == "ADD":
            n_add += 1
        elif kind == "DELETE":
            n_del += 1
            if isinstance(item_id, str) and item_id not in before_ids:
                n_fail_del += 1
        elif kind == "REPLACE":
            n_repl += 1
            if isinstance(item_id, str) and item_id not in before_ids:
                n_fail_repl += 1
    return {
        "emit_add": n_add,
        "emit_del": n_del,
        "emit_repl": n_repl,
        "fail_del": n_fail_del,
        "fail_repl": n_fail_repl,
    }


def per_query_row(t: dict[str, Any]) -> dict[str, Any]:
    """Extract one row of per-query metrics from a trace."""
    result = t.get("result") or {}
    distiller = result.get("distiller") or {}
    tags = distiller.get("item_tags") or {}
    n_helpful = sum(1 for v in tags.values() if v == "helpful")
    n_harmful = sum(1 for v in tags.values() if v == "harmful")
    n_stale = sum(1 for v in tags.values() if v == "stale")
    n_neutral = sum(1 for v in tags.values() if v == "neutral")
    map_after_items = t["map_after"]["items"]
    ops_breakdown = cartographer_ops_breakdown(t)
    return {
        "q": t["step_idx"],
        "evolving": t["evolving"],
        "items_before": len(t["map_before"]["items"]),
        "items_after": len(map_after_items),
        "added": len(t["items_added"]),
        "evicted": len(t["items_evicted"]),
        "modified": len(t["items_modified"]),
        "ops_applied": result.get("operations_applied", 0),
        "tag_helpful": n_helpful,
        "tag_harmful": n_harmful,
        "tag_stale": n_stale,
        "tag_neutral": n_neutral,
        "n_cache_candidates": len(distiller.get("cache_candidates") or []),
        "duplicate_pairs": duplicate_pairs(map_after_items),
        **ops_breakdown,
    }


# ---------------------------------------------------------------------------
# Per-item lifetime
# ---------------------------------------------------------------------------


_ID_RE = re.compile(r"\[([a-z]+-\d+)\]")


def item_lifetimes(traces: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Track each item id across the query sequence.

    For each id we record: birth query, death query (if evicted), final
    content + section, score history, and a 'used' counter computed as the
    number of subsequent trajectories that mention the id verbatim
    (substring of the literal '[id]' marker).
    """
    by_id: dict[str, dict[str, Any]] = {}
    for t in traces:
        q = t["step_idx"]
        # Birth detection: items present in map_after but not map_before
        for added in t["items_added"]:
            by_id.setdefault(
                added["id"],
                {
                    "birth_q": q,
                    "section": added["section"],
                    "content_first": added["content"],
                    "score_history": [],
                    "death_q": None,
                    "death_reason": None,
                    "used_in_trajectories": 0,
                },
            )
        # Score updates: record score per query while item alive
        scores = t["map_after"].get("scores", {})
        for item in t["map_after"]["items"]:
            life = by_id.get(item["id"])
            if life is not None:
                life["score_history"].append((q, int(scores.get(item["id"], 0))))
        # Death detection: items evicted at this query
        for ev in t["items_evicted"]:
            life = by_id.get(ev["id"])
            if life is not None and life["death_q"] is None:
                life["death_q"] = q
                life["death_reason"] = ev.get("reason", "evicted_or_deleted")
                life["score_at_death"] = ev.get("score", 0)
    # "Used" proxy: how many later trajectories reference this id?
    for life_id, life in by_id.items():
        marker = f"[{life_id}]"
        birth = life["birth_q"]
        for t in traces:
            if t["step_idx"] <= birth:
                continue
            if marker in t.get("trajectory", ""):
                life["used_in_trajectories"] += 1
    return by_id


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def fmt_row(d: dict[str, Any], cols: list[str]) -> str:
    return " ".join(f"{d[c]:>6}" if isinstance(d[c], int | bool) else f"{d[c]:>6.2f}" for c in cols)


def print_context_report(cid: int, traces: list[dict[str, Any]], top_items: int = 10) -> None:
    print(f"\n{'='*100}\nCONTEXT {cid} — {len(traces)} updates")
    print("=" * 100)

    rows = [per_query_row(t) for t in traces]
    # Per-query table
    print("\nPER-QUERY METRICS")
    print(
        f"{'q':>3} {'evol':>5} {'#in':>4} {'#out':>4} {'+':>3} {'-':>3} {'~':>3} "
        f"{'ops':>4} {'hlp':>4} {'hrm':>4} {'stl':>4} {'neu':>4} {'#cand':>5} {'dups':>5}"
    )
    print("-" * 90)
    for r in rows:
        print(
            f"{r['q']:>3} {'T' if r['evolving'] else 'F':>5} "
            f"{r['items_before']:>4} {r['items_after']:>4} "
            f"{r['added']:>3} {r['evicted']:>3} {r['modified']:>3} "
            f"{r['ops_applied']:>4} "
            f"{r['tag_helpful']:>4} {r['tag_harmful']:>4} {r['tag_stale']:>4} {r['tag_neutral']:>4} "
            f"{r['n_cache_candidates']:>5} {r['duplicate_pairs']:>5}"
        )

    # Aggregate
    n = len([r for r in rows if r["evolving"]])
    if n:
        total_emit = sum(r["emit_del"] + r["emit_repl"] for r in rows)
        total_fail = sum(r["fail_del"] + r["fail_repl"] for r in rows)
        agg = {
            "queries_evolving": n,
            "avg_items_after": sum(r["items_after"] for r in rows) / len(rows),
            "avg_added": sum(r["added"] for r in rows) / len(rows),
            "avg_evicted": sum(r["evicted"] for r in rows) / len(rows),
            "avg_ops_applied": sum(r["ops_applied"] for r in rows) / len(rows),
            "cartographer_emit_add": sum(r["emit_add"] for r in rows),
            "cartographer_emit_del": sum(r["emit_del"] for r in rows),
            "cartographer_emit_repl": sum(r["emit_repl"] for r in rows),
            "silent_failures_del_repl": total_fail,
            "silent_failure_rate": (total_fail / total_emit) if total_emit else 0.0,
            "total_helpful_tags": sum(r["tag_helpful"] for r in rows),
            "total_harmful_tags": sum(r["tag_harmful"] for r in rows),
            "total_stale_tags": sum(r["tag_stale"] for r in rows),
            "total_neutral_tags": sum(r["tag_neutral"] for r in rows),
            "neutral_rate_in_tags": (
                sum(r["tag_neutral"] for r in rows)
                / max(
                    1,
                    sum(
                        r["tag_helpful"] + r["tag_harmful"] + r["tag_stale"] + r["tag_neutral"]
                        for r in rows
                    ),
                )
            ),
            "max_duplicate_pairs": max(r["duplicate_pairs"] for r in rows),
            "final_items": rows[-1]["items_after"],
        }
        print("\nAGGREGATE")
        for k, v in agg.items():
            if isinstance(v, float):
                print(f"  {k:>22}: {v:.2f}")
            else:
                print(f"  {k:>22}: {v}")

    # Per-item lifetimes
    lives = item_lifetimes(traces)
    print(f"\nPER-ITEM LIFETIMES ({len(lives)} items born across run)")
    sorted_lives = sorted(
        lives.items(),
        key=lambda kv: (kv[1]["score_history"][-1][1] if kv[1]["score_history"] else 0),
        reverse=True,
    )
    print(
        f"  {'id':<15} {'sec':<22} {'birth':>5} {'death':>5} {'final_score':>11} "
        f"{'used_after':>10}  content"
    )
    print("  " + "-" * 110)
    for life_id, life in sorted_lives[:top_items]:
        final_score = life["score_history"][-1][1] if life["score_history"] else "—"
        print(
            f"  {life_id:<15} {life['section'][:22]:<22} "
            f"{life['birth_q']:>5} "
            f"{(life['death_q'] if life['death_q'] is not None else '—'):>5} "
            f"{final_score!s:>11} "
            f"{life['used_in_trajectories']:>10}  "
            f"{life['content_first'][:80]}"
        )

    # Sticky-zero analysis: items with score=0 that lived past 5 queries
    sticky = [
        (life_id, life)
        for life_id, life in lives.items()
        if all(s == 0 for _, s in life["score_history"])
        and len(life["score_history"]) >= 5
    ]
    if sticky:
        print(
            f"\nSTICKY-ZERO ITEMS (never-tagged, persisted ≥5 queries): {len(sticky)}"
        )
        for life_id, life in sticky[:5]:
            print(
                f"  [{life_id}] section={life['section']} "
                f"survived={len(life['score_history'])} queries  "
                f"used_after={life['used_in_trajectories']}  "
                f"content={life['content_first'][:80]}"
            )


def compare_contexts(
    left: tuple[int, list[dict[str, Any]]],
    right: tuple[int, list[dict[str, Any]]],
) -> None:
    """Side-by-side aggregate comparison of two contexts."""
    (lcid, ltraces), (rcid, rtraces) = left, right
    lrows = [per_query_row(t) for t in ltraces]
    rrows = [per_query_row(t) for t in rtraces]

    def agg(rows: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "avg_items": sum(r["items_after"] for r in rows) / len(rows),
            "avg_added": sum(r["added"] for r in rows) / len(rows),
            "avg_evicted": sum(r["evicted"] for r in rows) / len(rows),
            "avg_ops": sum(r["ops_applied"] for r in rows) / len(rows),
            "helpful_tags": sum(r["tag_helpful"] for r in rows),
            "harmful_tags": sum(r["tag_harmful"] for r in rows),
            "neutral_tags": sum(r["tag_neutral"] for r in rows),
            "max_dup_pairs": max(r["duplicate_pairs"] for r in rows),
            "final_items": rows[-1]["items_after"],
        }

    la, ra = agg(lrows), agg(rrows)
    print(f"\n{'='*100}\nSIDE-BY-SIDE: ctx {lcid} vs ctx {rcid}\n{'='*100}")
    print(f"  {'metric':<22} {'ctx '+str(lcid):>15} {'ctx '+str(rcid):>15} {'Δ (right-left)':>18}")
    print("  " + "-" * 75)
    for k in la:
        d = ra[k] - la[k]
        print(f"  {k:<22} {la[k]:>15.2f} {ra[k]:>15.2f} {d:>+18.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze PEEK trace files")
    p.add_argument("traces_root", type=Path, help="Path to runs/<tag>/traces/")
    p.add_argument(
        "--ctx",
        type=int,
        action="append",
        default=None,
        help="Only analyze this context_window_id (repeatable)",
    )
    p.add_argument(
        "--compare",
        nargs=2,
        type=int,
        metavar=("CTX_A", "CTX_B"),
        help="Side-by-side compare two contexts",
    )
    p.add_argument(
        "--top-items", type=int, default=10, help="Show top-N items per context (default: 10)"
    )
    args = p.parse_args()

    if not args.traces_root.exists():
        raise SystemExit(f"traces dir not found: {args.traces_root}")
    contexts = discover_contexts(args.traces_root)
    if not contexts:
        raise SystemExit(f"no ctx_<id>/ subdirectories found under {args.traces_root}")

    targets = args.ctx if args.ctx else list(contexts.keys())
    loaded: dict[int, list[dict[str, Any]]] = {}
    for cid in targets:
        if cid not in contexts:
            print(f"  warning: ctx {cid} not in traces dir, skipping")
            continue
        loaded[cid] = load_context_traces(contexts[cid])
        print_context_report(cid, loaded[cid], top_items=args.top_items)

    if args.compare:
        a, b = args.compare
        if a in loaded and b in loaded:
            compare_contexts((a, loaded[a]), (b, loaded[b]))
        else:
            print(f"  warning: --compare requires both contexts to be loaded ({a}, {b})")


if __name__ == "__main__":
    main()
