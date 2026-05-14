#!/usr/bin/env python3
"""
OBLIQ-Bench reranker benchmark: first-stage retrieval vs listwise LLM rerank.

Measures NDCG@10 and Recall@10 over the OBLIQ-Bench ``math`` subset
(Tchuindjo et al. 2026, arXiv:2605.06235), comparing the top-k from a
first-stage retriever against the same top-k after ``ListwiseReranker``.

First-stage options (``--retriever``):
  oracle  Mix gold-positive docs with random distractors to reach ``--top-n``.
          Best for isolating rerank quality from first-stage recall.
  bm25    In-memory BM25 (``rank_bm25``) over the subset's corpus.
          No external services. Default.
  es      ElasticsearchRetriever via env vars (ES_HOST, ES_API_KEY, ES_INDEX).
          Use ``--qrels-id-field`` to map ES doc_id → dataset corpus_id when
          they differ.

Adapter options (``--adapter``):
  fake    FakeAdapter — for smoke tests; rerank does nothing meaningful.
  azure   AzureOpenAIAdapter (requires AZURE_OPENAI_API_KEY + OPENAI_ENDPOINT).
  openai  OpenAICompatAdapter (requires OPENAI_API_KEY).

Usage:
  uv run python examples/oblique_rerank_bench.py \\
    --adapter openai --model gpt-4o-mini \\
    --retriever bm25 --max-examples 30

  AZURE_OPENAI_API_KEY=... OPENAI_ENDPOINT=... \\
  uv run python examples/oblique_rerank_bench.py \\
    --adapter azure --model gpt-5.1 \\
    --retriever oracle --top-n 50 --top-k 10 \\
    --window-size 20 --step 10 --workers 4 \\
    --cache-dir .cache/oblique_rerank
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from pyrlm_runtime import FileCache, ListwiseReranker, ndcg_at_k, recall_at_k
from pyrlm_runtime.adapters import FakeAdapter, ModelAdapter

load_dotenv()

DATASET_REPO = "dianetc/OBLIQ-Bench"
SUBSET_PATHS = {
    "math": "analogues/math/queries+qrels",
    "writing": "analogues/writing/queries+qrels",
    "twitter": "descriptive/twitter/queries+qrels",
    "wildchat": "descriptive/wildchat/queries+qrels",
    "congress": "tip-of-tongue/congress/queries+qrels",
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_obliq_subset(
    subset: str,
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, dict[str, float]],
    dict[str, list[str]],
]:
    """Load (corpus, queries, qrels, excluded_ids) for an OBLIQ-Bench subset.

    Returns
    -------
    corpus: dict[corpus_id -> text]
    queries: dict[query_id -> text]
    qrels: dict[query_id -> dict[corpus_id -> score]]
    excluded_ids: dict[query_id -> list[corpus_id]] (may be empty)
    """
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets + huggingface_hub. "
            "Install with: uv pip install '.[examples]'"
        ) from exc

    if subset not in SUBSET_PATHS:
        raise SystemExit(f"Unknown subset {subset!r}. Choose from {sorted(SUBSET_PATHS)}.")

    ds = load_dataset(DATASET_REPO, subset)
    corpus = {row["_id"]: row["text"] for row in ds["corpus"]}
    queries = {row["_id"]: row["text"] for row in ds["queries"]}

    qrels_relpath = f"{SUBSET_PATHS[subset]}/qrels.tsv"
    qrels_path = hf_hub_download(DATASET_REPO, qrels_relpath, repo_type="dataset")
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    with open(qrels_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            q_id, c_id, score = parts[0], parts[1], float(parts[2])
            qrels[q_id][c_id] = score

    excluded_ids: dict[str, list[str]] = {}
    excl_relpath = f"{SUBSET_PATHS[subset]}/per_query_excluded_ids.json"
    try:
        excl_path = hf_hub_download(DATASET_REPO, excl_relpath, repo_type="dataset")
        with open(excl_path) as f:
            excluded_ids = json.load(f)
    except Exception:
        pass  # not all subsets have this file

    return corpus, queries, dict(qrels), excluded_ids


# ---------------------------------------------------------------------------
# First-stage retrievers
# ---------------------------------------------------------------------------


class OracleRetriever:
    """Mix gold positives with random negatives to reach top_n.

    Useful for isolating rerank quality from first-stage recall: every query
    is guaranteed to have all gold docs in the pool.
    """

    def __init__(
        self,
        corpus: dict[str, str],
        qrels: dict[str, dict[str, float]],
        excluded_ids: dict[str, list[str]],
        seed: int = 42,
    ) -> None:
        self._corpus = corpus
        self._qrels = qrels
        self._excluded = {q: set(ids) for q, ids in excluded_ids.items()}
        self._corpus_ids = list(corpus.keys())
        self._rng = random.Random(seed)

    def search(self, query_id: str, top_n: int) -> list[dict[str, Any]]:
        gold = [c for c, s in self._qrels.get(query_id, {}).items() if s > 0]
        excluded = self._excluded.get(query_id, set())
        positives = [c for c in gold if c in self._corpus]
        rng = random.Random(hash(query_id) & 0xFFFFFFFF)
        pool: list[str] = []
        seen: set[str] = set()
        for c in positives:
            if c in seen:
                continue
            seen.add(c)
            pool.append(c)
        # Fill with random distractors
        candidates = [
            c
            for c in self._corpus_ids
            if c not in seen and c not in excluded and c not in self._qrels.get(query_id, {})
        ]
        rng.shuffle(candidates)
        for c in candidates:
            if len(pool) >= top_n:
                break
            pool.append(c)
        # Shuffle the whole pool so positives aren't always at the top
        rng.shuffle(pool)
        return [self._as_candidate(c, rank=i) for i, c in enumerate(pool[:top_n])]

    def _as_candidate(self, corpus_id: str, *, rank: int) -> dict[str, Any]:
        text = self._corpus[corpus_id]
        return {
            "doc_id": corpus_id,
            "content": text,
            "preview": text[:500],
            "score": 1.0 / (rank + 1),
            "metadata": {},
        }


class InMemoryBM25Retriever:
    """In-memory BM25 over the subset's corpus."""

    def __init__(
        self,
        corpus: dict[str, str],
        excluded_ids: dict[str, list[str]],
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: rank-bm25. Install with: uv pip install '.[examples]'"
            ) from exc
        self._ids = list(corpus.keys())
        self._texts = [corpus[c] for c in self._ids]
        tokenised = [self._tokenise(t) for t in self._texts]
        self._bm25 = BM25Okapi(tokenised)
        self._corpus = corpus
        self._excluded = {q: set(ids) for q, ids in excluded_ids.items()}

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return [tok.lower() for tok in text.split() if tok]

    def search(self, query_id: str, query_text: str, top_n: int) -> list[dict[str, Any]]:
        excluded = self._excluded.get(query_id, set())
        tokens = self._tokenise(query_text)
        scores = self._bm25.get_scores(tokens)
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


class ESRetrieverWrapper:
    """Thin wrapper around ElasticsearchRetriever for the bench.

    If ``qrels_id_field`` is set, each candidate's ``doc_id`` is remapped to
    that metadata field so it lines up with the dataset's corpus ids.
    """

    def __init__(self, qrels_id_field: str | None = None) -> None:
        try:
            from pyrlm_runtime import ElasticsearchRetriever
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: elasticsearch. Install with: uv pip install '.[elasticsearch]'"
            ) from exc
        host = os.environ.get("ES_HOST")
        api_key = os.environ.get("ES_API_KEY", "")
        index = os.environ.get("ES_INDEX")
        if not host or not index:
            raise SystemExit("--retriever es requires ES_HOST and ES_INDEX env vars")
        self._retriever = ElasticsearchRetriever(host=host, api_key=api_key, index=index)
        self._qrels_id_field = qrels_id_field

    def search(self, query_text: str, top_n: int) -> list[dict[str, Any]]:
        try:
            results = self._retriever.hybrid_search(query_text, top_k=top_n)
        except Exception:
            # Fall back to keyword search if hybrid unavailable (no embeddings configured).
            results = self._retriever.search(query_text, top_k=top_n)
        if self._qrels_id_field:
            for r in results:
                mapped = (r.get("metadata") or {}).get(self._qrels_id_field)
                if mapped is not None:
                    r["doc_id"] = str(mapped)
        return results


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------


def build_adapter(args: argparse.Namespace) -> ModelAdapter:
    """Build a ModelAdapter from CLI args."""
    kind = args.adapter
    if kind == "fake":
        return FakeAdapter(
            rules=[],
            script=["[1] > [2] > [3] > [4] > [5] > [6] > [7] > [8] > [9] > [10]"] * 10000,
        )
    if kind == "azure":
        from pyrlm_runtime.adapters import AzureOpenAIAdapter

        kw: dict[str, Any] = {"model": args.model, "timeout": 600.0}
        if args.api_version:
            kw["api_version"] = args.api_version
        return AzureOpenAIAdapter(**kw)
    if kind == "openai":
        from pyrlm_runtime.adapters import OpenAICompatAdapter

        return OpenAICompatAdapter(
            model=args.model,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            timeout=600.0,
        )
    raise SystemExit(f"Unknown adapter: {kind}")


# ---------------------------------------------------------------------------
# Bench loop
# ---------------------------------------------------------------------------


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_model_name(model: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in model)


def evaluate_one(
    *,
    query_id: str,
    query_text: str,
    qrels: dict[str, float],
    first_stage_fn: Any,
    reranker: ListwiseReranker,
    top_n: int,
    top_k: int,
) -> dict[str, Any]:
    t_first = time.perf_counter()
    candidates = first_stage_fn(query_id=query_id, query_text=query_text, top_n=top_n)
    first_elapsed = time.perf_counter() - t_first
    candidate_ids = [c["doc_id"] for c in candidates]
    baseline_ids = candidate_ids[:top_k]

    llm_calls_before = reranker.llm_calls
    cache_hits_before = reranker.cache_hits
    t_rer = time.perf_counter()
    reranked = reranker.rerank(query_text, candidates, top_k=top_k)
    rerank_elapsed = time.perf_counter() - t_rer
    reranked_ids = [c["doc_id"] for c in reranked]
    llm_calls = reranker.llm_calls - llm_calls_before
    cache_hits = reranker.cache_hits - cache_hits_before

    return {
        "query_id": query_id,
        "n_candidates": len(candidates),
        "baseline_top_k": baseline_ids,
        "rerank_top_k": reranked_ids,
        "baseline_ndcg10": ndcg_at_k(baseline_ids, qrels, k=top_k),
        "rerank_ndcg10": ndcg_at_k(reranked_ids, qrels, k=top_k),
        "baseline_recall10": recall_at_k(baseline_ids, qrels, k=top_k),
        "rerank_recall10": recall_at_k(reranked_ids, qrels, k=top_k),
        "first_stage_s": first_elapsed,
        "rerank_s": rerank_elapsed,
        "llm_calls": llm_calls,
        "cache_hits": cache_hits,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, float]:
    n = len(records)
    if n == 0:
        return {}
    out: dict[str, float] = {}
    for key in (
        "baseline_ndcg10",
        "rerank_ndcg10",
        "baseline_recall10",
        "rerank_recall10",
    ):
        out[key] = sum(r[key] for r in records) / n
    out["mean_first_stage_s"] = sum(r["first_stage_s"] for r in records) / n
    out["mean_rerank_s"] = sum(r["rerank_s"] for r in records) / n
    # Per-query llm_calls/cache_hits are derived from racy `after - before`
    # diffs of a shared counter, so they over-count when workers run in
    # parallel. Track the true totals at aggregation time via the caller.
    out["total_llm_calls_per_query_sum"] = sum(r["llm_calls"] for r in records)
    out["total_cache_hits_per_query_sum"] = sum(r["cache_hits"] for r in records)
    out["n_examples"] = float(n)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OBLIQ-Bench listwise LLM rerank benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subset", default="math", choices=sorted(SUBSET_PATHS))
    parser.add_argument("--retriever", default="bm25", choices=["oracle", "bm25", "es"])
    parser.add_argument(
        "--qrels-id-field",
        default=None,
        help="ES metadata field that holds the dataset corpus_id (es retriever only).",
    )
    parser.add_argument("--adapter", default="fake", choices=["fake", "azure", "openai"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-version", default=None)
    parser.add_argument("--top-n", type=int, default=50, help="First-stage pool size.")
    parser.add_argument("--top-k", type=int, default=10, help="Final cutoff for metrics.")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--max-passage-chars", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    print(f"Loading OBLIQ-Bench / {args.subset} from {DATASET_REPO} …")
    corpus, queries, qrels, excluded_ids = load_obliq_subset(args.subset)
    print(f"  corpus={len(corpus)}  queries={len(queries)}  qrels={len(qrels)}")

    # Restrict to queries that have at least one relevance judgment.
    eligible_qids = [q for q in queries if q in qrels and qrels[q]]
    rng = random.Random(args.seed)
    rng.shuffle(eligible_qids)
    if args.max_examples is not None:
        eligible_qids = eligible_qids[: args.max_examples]
    print(f"  evaluating {len(eligible_qids)} queries")

    # Build first-stage retriever.
    if args.retriever == "oracle":
        oracle = OracleRetriever(corpus, qrels, excluded_ids, seed=args.seed)

        def first_stage(query_id: str, query_text: str, top_n: int) -> list[dict[str, Any]]:
            return oracle.search(query_id, top_n)

    elif args.retriever == "bm25":
        print("Building in-memory BM25 index …")
        bm25 = InMemoryBM25Retriever(corpus, excluded_ids)

        def first_stage(query_id: str, query_text: str, top_n: int) -> list[dict[str, Any]]:
            return bm25.search(query_id, query_text, top_n)

    else:  # es
        es = ESRetrieverWrapper(qrels_id_field=args.qrels_id_field)

        def first_stage(query_id: str, query_text: str, top_n: int) -> list[dict[str, Any]]:
            return es.search(query_text, top_n)

    # Build reranker.
    adapter = build_adapter(args)
    cache: FileCache | None = None
    if args.cache_dir:
        cache_path = Path(args.cache_dir).expanduser()
        cache = FileCache(cache_path)
        print(f"  cache: {cache_path}")
    reranker = ListwiseReranker(
        adapter,
        window_size=args.window_size,
        step=args.step,
        max_passage_chars=args.max_passage_chars,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        cache=cache,
        cache_namespace=f"{args.adapter}:{args.model}",
    )

    # Output dir.
    run_label = f"run_{_now_tag()}_{args.subset}_{args.adapter}_{_safe_model_name(args.model)}"
    run_dir = Path(args.output_dir or f"examples/exports/oblique_rerank_bench/{run_label}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  output: {run_dir}\n")

    # Run.
    records: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    if args.workers > 1 and args.adapter != "fake":
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    evaluate_one,
                    query_id=q,
                    query_text=queries[q],
                    qrels=qrels[q],
                    first_stage_fn=first_stage,
                    reranker=reranker,
                    top_n=args.top_n,
                    top_k=args.top_k,
                ): q
                for q in eligible_qids
            }
            for fut in as_completed(futures):
                rec = fut.result()
                records.append(rec)
                print(
                    f"  [{len(records)}/{len(eligible_qids)}] {rec['query_id']} "
                    f"base ndcg={rec['baseline_ndcg10']:.3f} "
                    f"rerank ndcg={rec['rerank_ndcg10']:.3f}"
                )
    else:
        # Sequential — needed for FakeAdapter (single script) and for low-concurrency debugging.
        for i, q in enumerate(eligible_qids, start=1):
            rec = evaluate_one(
                query_id=q,
                query_text=queries[q],
                qrels=qrels[q],
                first_stage_fn=first_stage,
                reranker=reranker,
                top_n=args.top_n,
                top_k=args.top_k,
            )
            records.append(rec)
            print(
                f"  [{i}/{len(eligible_qids)}] {rec['query_id']} "
                f"base ndcg={rec['baseline_ndcg10']:.3f} "
                f"rerank ndcg={rec['rerank_ndcg10']:.3f}"
            )
    wall_s = time.perf_counter() - t0

    agg = aggregate(records)
    agg["wall_time_s"] = wall_s
    # Read the true totals from the reranker's thread-safe counters. The
    # per-query sums in `agg` are racy under parallel workers.
    agg["total_llm_calls"] = reranker.llm_calls
    agg["total_cache_hits"] = reranker.cache_hits
    total_lookups = agg["total_llm_calls"] + agg["total_cache_hits"]
    agg["cache_hit_rate"] = (
        agg["total_cache_hits"] / total_lookups if total_lookups > 0 else 0.0
    )

    # Outputs.
    metrics = {
        "subset": args.subset,
        "retriever": args.retriever,
        "adapter": args.adapter,
        "model": args.model,
        "params": {
            "top_n": args.top_n,
            "top_k": args.top_k,
            "window_size": args.window_size,
            "step": args.step,
            "max_passage_chars": args.max_passage_chars,
            "workers": args.workers,
            "seed": args.seed,
        },
        "aggregate": agg,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (run_dir / "per_query.jsonl").open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    summary_lines = [
        f"OBLIQ-Bench / {args.subset}  ({args.retriever} → {args.adapter}:{args.model})",
        f"  examples:     {int(agg.get('n_examples', 0))}",
        f"  baseline:     NDCG@{args.top_k}={agg['baseline_ndcg10']:.4f}   "
        f"Recall@{args.top_k}={agg['baseline_recall10']:.4f}",
        f"  rerank :      NDCG@{args.top_k}={agg['rerank_ndcg10']:.4f}   "
        f"Recall@{args.top_k}={agg['rerank_recall10']:.4f}",
        f"  Δ NDCG:       {agg['rerank_ndcg10'] - agg['baseline_ndcg10']:+.4f}",
        f"  Δ Recall:     {agg['rerank_recall10'] - agg['baseline_recall10']:+.4f}",
        f"  llm_calls:    {int(agg['total_llm_calls'])}   "
        f"cache_hits: {int(agg['total_cache_hits'])}   "
        f"hit_rate: {agg['cache_hit_rate']:.2%}",
        f"  wall_time:    {wall_s:.1f}s",
    ]
    summary = "\n".join(summary_lines)
    (run_dir / "summary.txt").write_text(summary + "\n")
    print()
    print(summary)


if __name__ == "__main__":
    main()
