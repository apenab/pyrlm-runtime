#!/usr/bin/env python3
"""OBLIQ-Bench Palanca 1 v2 — Multi-Query Union + Original Query + Listwise Rerank.

Pipeline (v2, default):

    query oblicua
        ↓
    rewriter LLM (1 call, N=5 reformulaciones)
        ↓
    BM25 × 5 reformulaciones + BM25 × query original (6 búsquedas total)
        ↓
    unión deduplicada por doc_id (~125 docs)
        ↓
    ListwiseReranker sobre la unión (query ORIGINAL)
        ↓
    top-10

v2 vs v1: añade la query original al fan-out BM25 (coste cero, BM25
in-memory). Elimina las ~10-15 regresiones de v1 donde la query original
ya tenía gold en BM25 pero los rewrites movieron el haz lejos. Pasar
``--no-include-original`` para reproducir comportamiento v1.

Cache OFF por defecto (NoopCache). Pasar ``--cache-dir`` para opt-in.

Ver ``docs/OBLIQ-PALANCA1-MULTIQUERY.md`` para diseño y motivación.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from pyrlm_runtime import (
    FileCache,
    ListwiseReranker,
    QueryRewriter,
    TournamentReranker,
    ndcg_at_k,
    recall_at_k,
    union_pool,
)
from pyrlm_runtime.adapters import FakeAdapter, ModelAdapter
from pyrlm_runtime.cache import CacheRecord

load_dotenv()


# ---------------------------------------------------------------------------
# Dataset (idéntico a oblique_rerank_bench.py / oblique_agentic_bench.py)
# ---------------------------------------------------------------------------


DATASET_REPO = "dianetc/OBLIQ-Bench"
SUBSET_PATHS = {
    "math": "analogues/math/queries+qrels",
    "writing": "analogues/writing/queries+qrels",
    "twitter": "descriptive/twitter/queries+qrels",
    "wildchat": "descriptive/wildchat/queries+qrels",
    "congress": "tip-of-tongue/congress/queries+qrels",
}


def load_obliq_subset(subset: str) -> tuple[
    dict[str, str], dict[str, str], dict[str, dict[str, float]], dict[str, list[str]]
]:
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with: uv pip install '.[examples]'"
        ) from exc

    ds = load_dataset(DATASET_REPO, subset)
    corpus = {row["_id"]: row["text"] for row in ds["corpus"]}
    queries = {row["_id"]: row["text"] for row in ds["queries"]}

    qrels_relpath = f"{SUBSET_PATHS[subset]}/qrels.tsv"
    qrels_path = hf_hub_download(DATASET_REPO, qrels_relpath, repo_type="dataset")
    qrels: dict[str, dict[str, float]] = {}
    with open(qrels_path) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            q_id, c_id, score = parts[0], parts[1], float(parts[2])
            qrels.setdefault(q_id, {})[c_id] = score

    excluded_ids: dict[str, list[str]] = {}
    excl_relpath = f"{SUBSET_PATHS[subset]}/per_query_excluded_ids.json"
    try:
        excl_path = hf_hub_download(DATASET_REPO, excl_relpath, repo_type="dataset")
        with open(excl_path) as f:
            excluded_ids = json.load(f)
    except Exception:
        pass

    return corpus, queries, qrels, excluded_ids


class InMemoryBM25:
    def __init__(self, corpus: dict[str, str], excluded_ids: dict[str, list[str]]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise SystemExit(
                "Missing rank-bm25. Install with: uv pip install '.[examples]'"
            ) from exc
        self._ids = list(corpus.keys())
        self._texts = [corpus[c] for c in self._ids]
        self._bm25 = BM25Okapi([self._tokenise(t) for t in self._texts])
        self._excluded = {q: set(ids) for q, ids in excluded_ids.items()}

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return [t.lower() for t in text.split() if t]

    def search(
        self, query_id: str, query_text: str, top_n: int
    ) -> list[dict[str, Any]]:
        excluded = self._excluded.get(query_id, set())
        scores = self._bm25.get_scores(self._tokenise(query_text))
        ranked = sorted(range(len(self._ids)), key=lambda i: scores[i], reverse=True)
        out: list[dict[str, Any]] = []
        for idx in ranked:
            c = self._ids[idx]
            if c in excluded:
                continue
            text = self._texts[idx]
            out.append(
                {
                    "doc_id": c,
                    "content": text,
                    "preview": text[:500],
                    "score": float(scores[idx]),
                    "metadata": {},
                }
            )
            if len(out) >= top_n:
                break
        return out


# ---------------------------------------------------------------------------
# Math-oblique rewriter prompt (domain-specific, lives in examples not src/)
# ---------------------------------------------------------------------------


def _make_rewriter_system_prompt(n: int) -> str:
    return f"""\
You are a search-query reformulation expert for an oblique retrieval
task. The user is searching a corpus of math problems for problems
sharing the same proof technique or reasoning structure as their query
— even when the surface topic differs.

Your job: given the ORIGINAL QUERY, produce exactly {n} reformulations
that each attack the same underlying technique from a DIFFERENT angle.
Each reformulation should use vocabulary that a problem author might
plausibly choose when applying the same technique to a different
topic.

Constraints:
- Each reformulation must be a single concise phrase (10-20 words).
- The reformulations together must span as much vocabulary diversity
  as possible. Do NOT generate near-duplicates of each other.
- Avoid vocabulary that is too generic ("math problem", "competition
  problem", "find problems") — that hurts BM25 precision.
- Avoid named entities, specific numbers, or specific theorem names
  from the original query.
- Stay faithful to the underlying technique. Do not drift into
  unrelated math areas.

Return ONE JSON object on a single line, no prose, no markdown fence:
{{"rewrites": ["...", "...", "...", "...", "..."]}}
"""


# ---------------------------------------------------------------------------
# No-op cache (forces fresh LLM responses)
# ---------------------------------------------------------------------------


class _NoopCache:
    def get(self, key: str) -> CacheRecord | None:  # noqa: ARG002
        return None

    def set(self, key: str, record: CacheRecord) -> None:  # noqa: ARG002
        return None


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def build_rewriter_adapter(args: argparse.Namespace) -> ModelAdapter:
    if args.adapter == "fake":
        return FakeAdapter(
            script=[
                '{"rewrites": ["alpha alternative 1", "beta alternative 2", '
                '"gamma alternative 3", "delta alternative 4", '
                '"epsilon alternative 5"]}'
            ]
            * 100
        )
    if args.adapter == "azure":
        from pyrlm_runtime.adapters import AzureOpenAIAdapter

        kw: dict[str, Any] = {"model": args.rewriter_model, "timeout": 60.0}
        if args.api_version:
            kw["api_version"] = args.api_version
        return AzureOpenAIAdapter(**kw)
    raise SystemExit(f"Unknown adapter {args.adapter!r}")


def build_rerank_adapter(args: argparse.Namespace) -> ModelAdapter:
    if args.adapter == "fake":
        # FakeAdapter that returns a plausible-looking permutation
        return FakeAdapter(script=["[1] > [2] > [3] > [4] > [5]"] * 1000)
    if args.adapter == "azure":
        from pyrlm_runtime.adapters import AzureOpenAIAdapter

        kw: dict[str, Any] = {"model": args.rerank_model, "timeout": 120.0}
        if args.api_version:
            kw["api_version"] = args.api_version
        return AzureOpenAIAdapter(**kw)
    raise SystemExit(f"Unknown adapter {args.adapter!r}")


# ---------------------------------------------------------------------------
# Single-query evaluation
# ---------------------------------------------------------------------------


def evaluate_one(
    *,
    query_id: str,
    query_text: str,
    qrels: dict[str, float],
    bm25: InMemoryBM25,
    rewriter: QueryRewriter,
    reranker: ListwiseReranker,
    args: argparse.Namespace,
) -> dict[str, Any]:
    gold = {d for d, s in qrels.items() if s > 0}

    # Baseline: BM25 sobre query original
    t0 = time.perf_counter()
    baseline_pool = bm25.search(query_id, query_text, args.bm25_top_n)
    baseline_elapsed = time.perf_counter() - t0
    baseline_top_k = [d["doc_id"] for d in baseline_pool[: args.top_k]]

    error: str | None = None
    rewrites: list[str] = []
    pools_per_rewrite: list[list[dict[str, Any]]] = []
    union: list[dict[str, Any]] = []
    multiquery_top_k: list[str] = []
    rewriter_elapsed = 0.0
    rerank_elapsed = 0.0

    try:
        # Rewrite
        t1 = time.perf_counter()
        rewrites = rewriter.rewrite(query_text)
        rewriter_elapsed = time.perf_counter() - t1
        if not rewrites:
            # Fallback: degrade gracefully to single-query rerank
            rewrites = [query_text]

        # BM25 × N rewrites (+ original query if v2 behavior enabled)
        searches = rewrites[:]
        if args.include_original and query_text not in searches:
            searches.append(query_text)
        for r in searches:
            pool = bm25.search(query_id, r, args.per_rewrite_top_n)
            pools_per_rewrite.append(pool)

        # Union dedup
        union = union_pool(pools_per_rewrite)
        if args.union_cap and len(union) > args.union_cap:
            union = union[: args.union_cap]

        # Rerank on union (using the ORIGINAL query — the rewriter only
        # served to fan out the retrieval, not to replace the user intent)
        t2 = time.perf_counter()
        reranked = reranker.rerank(query_text, union, top_k=args.top_k)
        rerank_elapsed = time.perf_counter() - t2
        multiquery_top_k = [d["doc_id"] for d in reranked]
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"

    # Pad with baseline tail if rerank produced fewer than top_k
    if len(multiquery_top_k) < args.top_k:
        for doc_id in baseline_top_k:
            if doc_id not in multiquery_top_k:
                multiquery_top_k.append(doc_id)
            if len(multiquery_top_k) >= args.top_k:
                break
    multiquery_top_k = multiquery_top_k[: args.top_k]

    # Diagnostics
    union_ids = {d["doc_id"] for d in union}
    baseline_ids_set = {d["doc_id"] for d in baseline_pool}
    n_hits_per_rewrite = [len(p) for p in pools_per_rewrite]
    n_unique_pool = len(union)
    sum_hits = sum(n_hits_per_rewrite) or 1
    overlap_rate = 1.0 - (n_unique_pool / sum_hits)
    gold_in_union = gold & union_ids
    gold_in_baseline = gold & baseline_ids_set

    # Pool-level recall (Recall@N over the union, before rerank)
    recall_pool = (len(gold_in_union) / len(gold)) if gold else 0.0
    recall_baseline_pool = (len(gold_in_baseline) / len(gold)) if gold else 0.0

    return {
        "query_id": query_id,
        "original_query": query_text,
        "rewrites": rewrites,
        "n_hits_per_rewrite": n_hits_per_rewrite,
        "n_unique_pool": n_unique_pool,
        "overlap_rate": overlap_rate,
        "n_gold_total": len(gold),
        "n_gold_in_pool": len(gold_in_union),
        "n_gold_in_baseline_pool": len(gold_in_baseline),
        "gold_pool_recall": recall_pool,
        "gold_baseline_pool_recall": recall_baseline_pool,
        "baseline_top_k": baseline_top_k,
        "multiquery_top_k": multiquery_top_k,
        "baseline_ndcg10": ndcg_at_k(baseline_top_k, qrels, k=args.top_k),
        "multiquery_ndcg10": ndcg_at_k(multiquery_top_k, qrels, k=args.top_k),
        "baseline_recall10": recall_at_k(baseline_top_k, qrels, k=args.top_k),
        "multiquery_recall10": recall_at_k(multiquery_top_k, qrels, k=args.top_k),
        "baseline_s": baseline_elapsed,
        "rewriter_s": rewriter_elapsed,
        "rerank_s": rerank_elapsed,
        "error": error,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, float]:
    n = len(records)
    if n == 0:
        return {}
    out: dict[str, float] = {}
    for key in (
        "baseline_ndcg10",
        "multiquery_ndcg10",
        "baseline_recall10",
        "multiquery_recall10",
        "n_unique_pool",
        "overlap_rate",
        "n_gold_total",
        "n_gold_in_pool",
        "n_gold_in_baseline_pool",
        "gold_pool_recall",
        "gold_baseline_pool_recall",
    ):
        out[key] = sum(r.get(key, 0.0) for r in records) / n
    out["mean_baseline_s"] = sum(r["baseline_s"] for r in records) / n
    out["mean_rewriter_s"] = sum(r["rewriter_s"] for r in records) / n
    out["mean_rerank_s"] = sum(r["rerank_s"] for r in records) / n
    out["n_errors"] = float(sum(1 for r in records if r.get("error")))
    out["n_examples"] = float(n)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OBLIQ-Bench Palanca 1 — Multi-Query Union + Listwise Rerank.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subset", default="math", choices=sorted(SUBSET_PATHS))
    parser.add_argument("--adapter", default="azure", choices=["fake", "azure"])
    parser.add_argument("--rewriter-model", default="gpt-5.4-mini")
    parser.add_argument("--rerank-model", default="gpt-5.1")
    parser.add_argument("--api-version", default=None)
    parser.add_argument("--n-rewrites", type=int, default=5)
    parser.add_argument(
        "--per-rewrite-top-n",
        type=int,
        default=25,
        help="BM25 hits to take per reformulation before deduplication.",
    )
    parser.add_argument(
        "--bm25-top-n",
        type=int,
        default=50,
        help="BM25 hits for the baseline (single-query) bucket.",
    )
    parser.add_argument(
        "--union-cap",
        type=int,
        default=150,
        help="Hard cap on union pool size after dedup (0 = no cap).",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--reranker-mode",
        choices=["sliding", "tournament"],
        default="sliding",
        help="Which reranker to use. 'sliding' = ListwiseReranker (RankGPT, "
        "window+step). 'tournament' = TournamentReranker (paper App. C, "
        "shuffled batches + recursive elimination). Default sliding (tournament "
        "was tested at pool~108 and lost — it's designed for pools 300-2500).",
    )
    parser.add_argument(
        "--no-include-original",
        dest="include_original",
        action="store_false",
        default=True,
        help="Exclude the original query from the BM25 fan-out (reverts to "
        "Palanca 1 v1 behavior). Default: include original as extra BM25 search.",
    )
    parser.add_argument(
        "--rerank-window-size",
        type=int,
        default=20,
        help="Window size (sliding) / batch_size (tournament). Default 20.",
    )
    parser.add_argument(
        "--rerank-step",
        type=int,
        default=10,
        help="ListwiseReranker step size. Ignored in tournament mode.",
    )
    parser.add_argument(
        "--rerank-top-k-per-batch",
        type=int,
        default=4,
        help="TournamentReranker survivors per batch. Ignored in sliding mode.",
    )
    parser.add_argument(
        "--rerank-max-passage-chars",
        type=int,
        default=300,
        help="Max chars per passage in the rerank prompt.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--per-query-timeout",
        type=float,
        default=300.0,
        help="Hard per-query timeout in seconds.",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="If set, ListwiseReranker uses a FileCache here. Default: no "
        "cache, fresh LLM responses every run.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny smoke run (FakeAdapter, 1 query, no Azure needed).",
    )
    args = parser.parse_args()

    if args.smoke:
        args.adapter = "fake"
        args.max_examples = 1
        args.workers = 1

    print(f"Loading OBLIQ-Bench / {args.subset} from {DATASET_REPO} …")
    corpus, queries, qrels, excluded_ids = load_obliq_subset(args.subset)
    print(f"  corpus={len(corpus)}  queries={len(queries)}  qrels={len(qrels)}")

    eligible = [q for q in queries if q in qrels and qrels[q]]
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    if args.max_examples is not None:
        eligible = eligible[: args.max_examples]
    print(f"  evaluating {len(eligible)} queries")

    print("Building in-memory BM25 index …")
    bm25 = InMemoryBM25(corpus, excluded_ids)

    rewriter = QueryRewriter(
        build_rewriter_adapter(args),
        n=args.n_rewrites,
        system_prompt=_make_rewriter_system_prompt(args.n_rewrites),
    )

    cache: FileCache | None = None
    if args.cache_dir:
        cache_path = Path(args.cache_dir).expanduser()
        cache = FileCache(cache_path)
        print(f"  rerank cache: {cache_path}")
    else:
        print("  rerank cache: DISABLED (fresh LLM responses)")
    if args.reranker_mode == "tournament":
        reranker: Any = TournamentReranker(
            build_rerank_adapter(args),
            batch_size=args.rerank_window_size,
            top_k_per_batch=args.rerank_top_k_per_batch,
            max_passage_chars=args.rerank_max_passage_chars,
            cache=cache,
            cache_namespace=(
                f"{args.adapter}:{args.rerank_model}:multiquery:tournament"
            ),
            shuffle_seed=args.seed,
        )
        print(
            f"  reranker: tournament (batch={args.rerank_window_size}, "
            f"top_k_per_batch={args.rerank_top_k_per_batch})"
        )
    else:
        reranker = ListwiseReranker(
            build_rerank_adapter(args),
            window_size=args.rerank_window_size,
            step=args.rerank_step,
            max_passage_chars=args.rerank_max_passage_chars,
            cache=cache,
            cache_namespace=(
                f"{args.adapter}:{args.rerank_model}:multiquery:sliding"
            ),
        )
        print(
            f"  reranker: sliding-window (window={args.rerank_window_size}, "
            f"step={args.rerank_step})"
        )

    run_label = (
        f"run_{_now_tag()}_{args.subset}_multiquery_{args.reranker_mode}_"
        f"{args.adapter}_{_safe(args.rewriter_model)}_x_{_safe(args.rerank_model)}"
    )
    run_dir = Path(args.output_dir or f"examples/exports/oblique_multiquery_bench/{run_label}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  output: {run_dir}\n")

    def run_one(q: str) -> dict[str, Any]:
        return evaluate_one(
            query_id=q,
            query_text=queries[q],
            qrels=qrels[q],
            bm25=bm25,
            rewriter=rewriter,
            reranker=reranker,
            args=args,
        )

    def _stub(q: str, reason: str) -> dict[str, Any]:
        return {
            "query_id": q,
            "original_query": queries.get(q, ""),
            "rewrites": [],
            "n_hits_per_rewrite": [],
            "n_unique_pool": 0,
            "overlap_rate": 0.0,
            "n_gold_total": 0,
            "n_gold_in_pool": 0,
            "n_gold_in_baseline_pool": 0,
            "gold_pool_recall": 0.0,
            "gold_baseline_pool_recall": 0.0,
            "baseline_top_k": [],
            "multiquery_top_k": [],
            "baseline_ndcg10": 0.0,
            "multiquery_ndcg10": 0.0,
            "baseline_recall10": 0.0,
            "multiquery_recall10": 0.0,
            "baseline_s": 0.0,
            "rewriter_s": 0.0,
            "rerank_s": 0.0,
            "error": reason,
        }

    def _run_with_timeout(q: str) -> dict[str, Any]:
        result_box: dict[str, Any] = {}

        def worker() -> None:
            try:
                result_box["rec"] = run_one(q)
            except Exception as exc:  # noqa: BLE001
                result_box["error"] = exc

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout=args.per_query_timeout)
        if "rec" in result_box:
            return result_box["rec"]
        if "error" in result_box:
            exc = result_box["error"]
            return _stub(q, f"error: {type(exc).__name__}: {str(exc)[:120]}")
        return _stub(q, f"per_query_timeout after {args.per_query_timeout:.0f}s")

    records: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_with_timeout, q): q for q in eligible}
            for fut in as_completed(futures):
                rec = fut.result()
                records.append(rec)
                err = f" ERR={rec['error'][:80]!r}" if rec.get("error") else ""
                print(
                    f"  [{len(records)}/{len(eligible)}] {rec['query_id']} "
                    f"base={rec['baseline_ndcg10']:.3f} "
                    f"mq={rec['multiquery_ndcg10']:.3f} "
                    f"pool={rec['n_unique_pool']} "
                    f"gold={rec['n_gold_in_pool']}/{rec['n_gold_total']} "
                    f"(vs base {rec['n_gold_in_baseline_pool']}) "
                    f"t={rec['baseline_s'] + rec['rewriter_s'] + rec['rerank_s']:.1f}s{err}",
                    flush=True,
                )
    else:
        for i, q in enumerate(eligible, start=1):
            rec = _run_with_timeout(q)
            records.append(rec)
            err = f" ERR={rec['error'][:80]!r}" if rec.get("error") else ""
            print(
                f"  [{i}/{len(eligible)}] {rec['query_id']} "
                f"base={rec['baseline_ndcg10']:.3f} "
                f"mq={rec['multiquery_ndcg10']:.3f} "
                f"pool={rec['n_unique_pool']} "
                f"gold={rec['n_gold_in_pool']}/{rec['n_gold_total']} "
                f"(vs base {rec['n_gold_in_baseline_pool']}) "
                f"t={rec['baseline_s'] + rec['rewriter_s'] + rec['rerank_s']:.1f}s{err}",
                flush=True,
            )

    wall_s = time.perf_counter() - t0
    agg = aggregate(records)
    agg["wall_time_s"] = wall_s
    agg["total_rewriter_calls"] = float(rewriter.calls)
    agg["total_rerank_llm_calls"] = float(reranker.llm_calls)
    agg["total_rerank_cache_hits"] = float(reranker.cache_hits)

    metrics = {
        "subset": args.subset,
        "adapter": args.adapter,
        "rewriter_model": args.rewriter_model,
        "rerank_model": args.rerank_model,
        "reranker_mode": args.reranker_mode,
        "cache_enabled": bool(args.cache_dir),
        "params": {
            "n_rewrites": args.n_rewrites,
            "per_rewrite_top_n": args.per_rewrite_top_n,
            "bm25_top_n": args.bm25_top_n,
            "union_cap": args.union_cap,
            "top_k": args.top_k,
            "rerank_window_size": args.rerank_window_size,
            "rerank_step": args.rerank_step,
            "rerank_top_k_per_batch": args.rerank_top_k_per_batch,
            "workers": args.workers,
            "seed": args.seed,
        },
        "aggregate": agg,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (run_dir / "per_query.jsonl").open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    summary = "\n".join(
        [
            f"OBLIQ-Bench / {args.subset}  (Palanca 1: multi-query + {args.reranker_mode} rerank, "
            f"rewriter={args.rewriter_model}, rerank={args.rerank_model}, "
            f"cache={'ON' if args.cache_dir else 'OFF'})",
            f"  examples:        {int(agg.get('n_examples', 0))}   "
            f"errors: {int(agg.get('n_errors', 0))}",
            f"  baseline BM25:   NDCG@{args.top_k}={agg['baseline_ndcg10']:.4f}   "
            f"Recall@{args.top_k}={agg['baseline_recall10']:.4f}",
            f"  multi-query:     NDCG@{args.top_k}={agg['multiquery_ndcg10']:.4f}   "
            f"Recall@{args.top_k}={agg['multiquery_recall10']:.4f}",
            f"  Δ NDCG vs BM25:  {agg['multiquery_ndcg10'] - agg['baseline_ndcg10']:+.4f}",
            f"  mean pool size:  {agg['n_unique_pool']:.1f}   "
            f"overlap_rate: {agg['overlap_rate']:.2%}",
            f"  gold in pool:    {agg['n_gold_in_pool']:.2f}/{agg['n_gold_total']:.2f} "
            f"(baseline BM25 pool: {agg['n_gold_in_baseline_pool']:.2f})",
            f"  pool-level recall (multi/baseline): "
            f"{agg['gold_pool_recall']:.3f} / {agg['gold_baseline_pool_recall']:.3f}",
            f"  llm calls:       rewriter={int(agg['total_rewriter_calls'])}   "
            f"rerank={int(agg['total_rerank_llm_calls'])}   "
            f"rerank_cache_hits={int(agg['total_rerank_cache_hits'])}",
            f"  timing/q:        baseline={agg['mean_baseline_s']:.2f}s   "
            f"rewriter={agg['mean_rewriter_s']:.2f}s   "
            f"rerank={agg['mean_rerank_s']:.2f}s",
            f"  wall_time:       {wall_s:.1f}s",
        ]
    )
    (run_dir / "summary.txt").write_text(summary + "\n")
    print()
    print(summary)


if __name__ == "__main__":
    main()
