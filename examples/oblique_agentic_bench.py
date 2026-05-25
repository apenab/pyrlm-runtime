#!/usr/bin/env python3
"""OBLIQ-Bench Condition 4 — RLM **agentic search** (no rerank, no pre-fetched pool).

Mirrors the "GPT-5.2 Multi-Hop Agent" from Tchuindjo et al. 2026 §5:
the RLM gets two tools, ``search(query, top_n)`` and ``read_doc(doc_id)``,
and must iteratively explore the corpus to produce a final ``top10_ids``.

The point of this bench is to answer the question: **does the library need
the new ``ListwiseReranker`` primitive, or could the existing RLM loop +
retrieval tools already attack oblique queries on its own?**

Cache is OFF by default. Use ``--use-cache`` to opt back in.

Usage:

    # Smoke test (FakeAdapter, no Azure credentials needed)
    uv run python examples/oblique_agentic_bench.py --smoke

    # Real run (workers=1 by default to keep things deterministic)
    AZURE_OPENAI_API_KEY=... OPENAI_ENDPOINT=... \\
    uv run python examples/oblique_agentic_bench.py \\
      --adapter azure --root-model gpt-5.1 \\
      --max-examples 5 --workers 1
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from pyrlm_runtime import Context, Policy, RLM, ndcg_at_k, recall_at_k
from pyrlm_runtime.adapters import FakeAdapter, ModelAdapter
from pyrlm_runtime.cache import CacheRecord

load_dotenv()


class _NoopCache:
    """Always-miss cache. Bench numbers must reflect fresh LLM responses."""

    def get(self, key: str) -> CacheRecord | None:  # noqa: ARG002
        return None

    def set(self, key: str, record: CacheRecord) -> None:  # noqa: ARG002
        return None


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
    """Same BM25 used by oblique_rlm_bench.py — re-implemented locally to
    keep this bench standalone."""

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
        self._corpus = corpus

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return [t.lower() for t in text.split() if t]

    def search(
        self, query_text: str, top_n: int, exclude: set[str] | None = None
    ) -> list[dict[str, Any]]:
        exclude = exclude or set()
        scores = self._bm25.get_scores(self._tokenise(query_text))
        ranked = sorted(range(len(self._ids)), key=lambda i: scores[i], reverse=True)
        out: list[dict[str, Any]] = []
        for idx in ranked:
            c = self._ids[idx]
            if c in exclude:
                continue
            text = self._texts[idx]
            out.append(
                {
                    "doc_id": c,
                    "preview": text[:300],
                    "score": float(scores[idx]),
                }
            )
            if len(out) >= top_n:
                break
        return out


# ---------------------------------------------------------------------------
# REPL extensions: only search + read_doc. No pool, no verify, no rerank.
# ---------------------------------------------------------------------------


def build_agentic_extensions(
    *,
    corpus: dict[str, str],
    bm25: InMemoryBM25,
    per_query_excluded: set[str],
    query_text: str,
    max_preview_chars: int = 300,
    max_doc_chars: int = 2500,
    search_call_log: list[dict[str, Any]] | None = None,
) -> Any:
    def extensions(*, rlm: Any, repl: Any, retriever: Any, log_diag: Any) -> dict[str, Any]:
        repl.set("query", query_text)

        def search(q: str, top_n: int = 25) -> list[dict[str, Any]]:
            if not isinstance(q, str) or not q.strip():
                return []
            n = max(1, min(int(top_n), 50))
            hits = bm25.search(q, top_n=n, exclude=per_query_excluded)
            if max_preview_chars != 300:
                for h in hits:
                    h["preview"] = h["preview"][:max_preview_chars]
            if search_call_log is not None:
                search_call_log.append(
                    {
                        "q": q[:200],
                        "top_n": n,
                        "returned": len(hits),
                        "ids": [h["doc_id"] for h in hits[:10]],
                    }
                )
            return hits

        def read_doc(doc_id: str) -> str:
            text = corpus.get(str(doc_id), "")
            if max_doc_chars and len(text) > max_doc_chars:
                return text[:max_doc_chars] + "…"
            return text

        return {"search": search, "read_doc": read_doc}

    return extensions


AGENTIC_SYSTEM_PROMPT = """\
You are an information-retrieval agent working inside a Python REPL. You
will be given a SEARCH QUERY and your job is to return the 10 most
relevant document IDs from a mathematics-problems corpus. Relevance in
this task is *latent*: two problems are relevant if they share the same
proof technique, reasoning structure, or solution approach — even if
their surface topic differs.

The REPL has these variables and helpers in scope (already defined; do
NOT redefine them):

* ``query`` (str): the original search query string.
* ``search(q, top_n=25) -> list[dict]``: BM25 search over the corpus.
  Returns up to ``top_n`` hits, each with keys ``doc_id`` (str),
  ``preview`` (first ~300 chars of the document), ``score`` (BM25 score).
  You can — and should — reformulate ``q`` across calls to probe different
  facets of the query (proof technique, mathematical structure, problem
  type, etc.).
* ``read_doc(doc_id) -> str``: full text of a single corpus document
  (truncated to ~2500 chars).
* Standard pyrlm-runtime REPL helpers (``llm_query``, ``llm_batch``, etc.)
  if you want to reason on the side.

REQUIRED procedure:

1. Issue multiple ``search`` calls with **reformulated queries** that
   target the latent attribute (e.g. "induction on n", "pigeonhole",
   "generating functions", "characteristic polynomial of a recurrence",
   whatever the original query implies). Keep a running set of candidate
   doc_ids.
2. For the most promising candidates, call ``read_doc(doc_id)`` and
   inspect the full text to confirm shared technique / structure.
3. Build ``top10_ids`` — a Python list of exactly 10 doc_id strings —
   ranked by your best estimate of latent relevance. Every position must
   be filled; if fewer than 10 candidates look good, pad with the next
   best from your accumulated search results.
4. Print ``top10_ids`` and stop:

       top10_ids = ["...", "...", ...]   # 10 real doc_id strings
       print(top10_ids)

CRITICAL: the elements of ``top10_ids`` must be actual doc_id values
returned by ``search`` (e.g. ``"american-math-monthly___2015___11716"``).
Placeholders like ``"doc_1"`` are an invalid answer.
"""


# ---------------------------------------------------------------------------
# Output parsing (same logic as oblique_rlm_bench.py)
# ---------------------------------------------------------------------------


_PY_LIST_RE = re.compile(r"\[[^\[\]]+\]", re.DOTALL)


def parse_top10(text: str, valid_doc_ids: set[str]) -> list[str]:
    candidates = _PY_LIST_RE.findall(text)
    for candidate in reversed(candidates):
        try:
            parsed = ast.literal_eval(candidate)
        except Exception:
            continue
        if not isinstance(parsed, list):
            continue
        result = [str(x) for x in parsed if str(x) in valid_doc_ids]
        if result:
            return result
    return []


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def build_root_adapter(args: argparse.Namespace) -> ModelAdapter:
    if args.adapter == "fake":
        return _make_fake_adapter()
    if args.adapter == "azure":
        from pyrlm_runtime.adapters import AzureOpenAIAdapter

        kw: dict[str, Any] = {"model": args.root_model, "timeout": 900.0}
        if args.api_version:
            kw["api_version"] = args.api_version
        return AzureOpenAIAdapter(**kw)
    raise SystemExit(f"Unknown adapter {args.adapter!r}")


def _make_fake_adapter() -> FakeAdapter:
    code = (
        "```python\n"
        "hits = search(query, top_n=10)\n"
        "top10_ids = [h['doc_id'] for h in hits][:10]\n"
        "print(top10_ids)\n"
        "```"
    )
    return FakeAdapter(script=[code] + ["FINAL: done"] * 30)


# ---------------------------------------------------------------------------
# Single-query evaluation
# ---------------------------------------------------------------------------


def evaluate_one(
    *,
    query_id: str,
    query_text: str,
    qrels: dict[str, float],
    corpus: dict[str, str],
    bm25: InMemoryBM25,
    excluded_ids_per_query: dict[str, set[str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    per_query_excluded = excluded_ids_per_query.get(query_id, set())

    # Baseline: BM25 top-k on the original query (no agent).
    t_first = time.perf_counter()
    baseline_pool = bm25.search(query_text, top_n=args.top_k, exclude=per_query_excluded)
    first_elapsed = time.perf_counter() - t_first
    baseline_ids = [h["doc_id"] for h in baseline_pool[: args.top_k]]

    search_log: list[dict[str, Any]] = []
    root_adapter = build_root_adapter(args)
    extensions = build_agentic_extensions(
        corpus=corpus,
        bm25=bm25,
        per_query_excluded=per_query_excluded,
        query_text=query_text,
        max_preview_chars=args.max_preview_chars,
        max_doc_chars=args.max_doc_chars,
        search_call_log=search_log,
    )

    rlm = RLM(
        adapter=root_adapter,
        policy=Policy(
            max_steps=args.max_steps,
            max_subcalls=args.max_subcalls,
            max_total_tokens=args.max_total_tokens,
        ),
        max_tokens=args.root_max_tokens,
        system_prompt=AGENTIC_SYSTEM_PROMPT,
        repl_extensions=extensions,
        auto_finalize_var="top10_ids",
        require_repl_before_final=True,
        require_subcall_before_final=False,
        cache=None if args.use_cache else _NoopCache(),
    )

    rlm_ids: list[str] = []
    raw_output = ""
    error: str | None = None
    rlm_top_k_source = "baseline_fallback"
    valid_ids: set[str] = set()  # will be the union of everything search returned
    t_rlm = time.perf_counter()
    steps = 0
    subcalls = 0
    try:
        output, trace = rlm.run(query_text, Context.from_text(""))
        raw_output = output or ""
        # Valid docs = anything search returned during this run, plus the
        # baseline BM25 top-k (so padding works).
        for call in search_log:
            valid_ids.update(call.get("ids", []))
        valid_ids.update(baseline_ids)
        rlm_ids = parse_top10(raw_output, valid_ids)
        if rlm_ids:
            rlm_top_k_source = "final"
        if not rlm_ids and trace and trace.steps:
            for step in reversed(trace.steps):
                step_text = (
                    getattr(step, "stdout", None) or getattr(step, "output", None) or ""
                )
                if not step_text:
                    continue
                rlm_ids = parse_top10(str(step_text), valid_ids)
                if rlm_ids:
                    rlm_top_k_source = "trace_fallback"
                    break
        if trace and trace.steps:
            steps = len(trace.steps)
            subcalls = sum(
                1 for s in trace.steps if s.kind in {"subcall", "recursive_subcall"}
            )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
    rlm_elapsed = time.perf_counter() - t_rlm

    # Pad with baseline BM25 tail if agent returned <top_k.
    if len(rlm_ids) < args.top_k:
        for doc_id in baseline_ids:
            if doc_id not in rlm_ids:
                rlm_ids.append(doc_id)
            if len(rlm_ids) >= args.top_k:
                break
    rlm_ids = rlm_ids[: args.top_k]

    # Recall diagnostic: of the gold for this query, how many surfaced in
    # the agent's accumulated search results?
    gold = {d for d, s in qrels.items() if s > 0}
    n_search_calls = len(search_log)
    gold_seen_by_agent = len(gold & valid_ids)

    return {
        "query_id": query_id,
        "baseline_top_k": baseline_ids,
        "rlm_top_k": rlm_ids,
        "rlm_top_k_source": rlm_top_k_source,
        "rlm_raw_output_tail": raw_output[-1500:] if raw_output else "",
        "baseline_ndcg10": ndcg_at_k(baseline_ids, qrels, k=args.top_k),
        "rlm_ndcg10": ndcg_at_k(rlm_ids, qrels, k=args.top_k),
        "baseline_recall10": recall_at_k(baseline_ids, qrels, k=args.top_k),
        "rlm_recall10": recall_at_k(rlm_ids, qrels, k=args.top_k),
        "n_search_calls": n_search_calls,
        "n_docs_seen": len(valid_ids),
        "n_gold_total": len(gold),
        "n_gold_seen_by_agent": gold_seen_by_agent,
        "search_queries": [c["q"] for c in search_log],
        "first_stage_s": first_elapsed,
        "rlm_s": rlm_elapsed,
        "rlm_steps": steps,
        "rlm_subcalls": subcalls,
        "error": error,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, float]:
    n = len(records)
    if n == 0:
        return {}
    out: dict[str, float] = {}
    for key in (
        "baseline_ndcg10",
        "rlm_ndcg10",
        "baseline_recall10",
        "rlm_recall10",
        "n_search_calls",
        "n_docs_seen",
        "n_gold_seen_by_agent",
        "n_gold_total",
    ):
        out[key] = sum(r[key] for r in records) / n
    out["mean_first_stage_s"] = sum(r["first_stage_s"] for r in records) / n
    out["mean_rlm_s"] = sum(r["rlm_s"] for r in records) / n
    out["mean_rlm_steps"] = sum(r["rlm_steps"] for r in records) / n
    out["mean_rlm_subcalls"] = sum(r["rlm_subcalls"] for r in records) / n
    out["n_errors"] = float(sum(1 for r in records if r.get("error")))
    out["n_examples"] = float(n)
    for src in ("final", "trace_fallback", "baseline_fallback"):
        out[f"n_source_{src}"] = float(
            sum(1 for r in records if r.get("rlm_top_k_source") == src)
        )
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
        description="OBLIQ-Bench Condition 4 — RLM agentic search (no rerank).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subset", default="math", choices=sorted(SUBSET_PATHS))
    parser.add_argument("--adapter", default="azure", choices=["fake", "azure"])
    parser.add_argument("--root-model", default="gpt-5.1")
    parser.add_argument("--api-version", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-subcalls", type=int, default=20)
    parser.add_argument("--max-total-tokens", type=int, default=500_000)
    parser.add_argument("--root-max-tokens", type=int, default=4096)
    parser.add_argument("--max-preview-chars", type=int, default=300)
    parser.add_argument("--max-doc-chars", type=int, default=2500)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--per-query-timeout",
        type=float,
        default=300.0,
        help="Hard per-query timeout. Stuck queries get marked as errors.",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Enable RLM FileCache (default: OFF — bench numbers must reflect "
        "fresh LLM responses, per project policy).",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny smoke run (FakeAdapter, 1 query).",
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
    excluded_ids_per_query = {q: set(ids) for q, ids in excluded_ids.items()}

    run_label = (
        f"run_{_now_tag()}_{args.subset}_agentic_{args.adapter}_{_safe(args.root_model)}"
    )
    run_dir = Path(args.output_dir or f"examples/exports/oblique_agentic_bench/{run_label}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  output: {run_dir}")
    print(f"  cache: {'ENABLED' if args.use_cache else 'DISABLED (NoopCache)'}\n")

    def run_one(q: str) -> dict[str, Any]:
        return evaluate_one(
            query_id=q,
            query_text=queries[q],
            qrels=qrels[q],
            corpus=corpus,
            bm25=bm25,
            excluded_ids_per_query=excluded_ids_per_query,
            args=args,
        )

    def _stub(q: str, reason: str) -> dict[str, Any]:
        return {
            "query_id": q,
            "baseline_top_k": [],
            "rlm_top_k": [],
            "rlm_top_k_source": "baseline_fallback",
            "rlm_raw_output_tail": "",
            "baseline_ndcg10": 0.0,
            "rlm_ndcg10": 0.0,
            "baseline_recall10": 0.0,
            "rlm_recall10": 0.0,
            "n_search_calls": 0,
            "n_docs_seen": 0,
            "n_gold_total": 0,
            "n_gold_seen_by_agent": 0,
            "search_queries": [],
            "first_stage_s": 0.0,
            "rlm_s": 0.0,
            "rlm_steps": 0,
            "rlm_subcalls": 0,
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
                    f"base={rec['baseline_ndcg10']:.3f} rlm={rec['rlm_ndcg10']:.3f} "
                    f"searches={rec['n_search_calls']} seen={rec['n_docs_seen']} "
                    f"gold_seen={rec['n_gold_seen_by_agent']}/{rec['n_gold_total']} "
                    f"steps={rec['rlm_steps']} t={rec['rlm_s']:.1f}s{err}",
                    flush=True,
                )
    else:
        for i, q in enumerate(eligible, start=1):
            rec = _run_with_timeout(q)
            records.append(rec)
            err = f" ERR={rec['error'][:80]!r}" if rec.get("error") else ""
            print(
                f"  [{i}/{len(eligible)}] {rec['query_id']} "
                f"base={rec['baseline_ndcg10']:.3f} rlm={rec['rlm_ndcg10']:.3f} "
                f"searches={rec['n_search_calls']} seen={rec['n_docs_seen']} "
                f"gold_seen={rec['n_gold_seen_by_agent']}/{rec['n_gold_total']} "
                f"steps={rec['rlm_steps']} t={rec['rlm_s']:.1f}s{err}",
                flush=True,
            )

    wall_s = time.perf_counter() - t0
    agg = aggregate(records)
    agg["wall_time_s"] = wall_s

    metrics = {
        "subset": args.subset,
        "adapter": args.adapter,
        "root_model": args.root_model,
        "cache_enabled": args.use_cache,
        "params": {
            "top_k": args.top_k,
            "max_steps": args.max_steps,
            "max_subcalls": args.max_subcalls,
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
            f"OBLIQ-Bench / {args.subset}  (Condition 4: agentic search, "
            f"root={args.root_model}, cache={'ON' if args.use_cache else 'OFF'})",
            f"  examples:     {int(agg.get('n_examples', 0))}   "
            f"errors: {int(agg.get('n_errors', 0))}",
            f"  baseline:     NDCG@{args.top_k}={agg['baseline_ndcg10']:.4f}   "
            f"Recall@{args.top_k}={agg['baseline_recall10']:.4f}",
            f"  rlm-agentic:  NDCG@{args.top_k}={agg['rlm_ndcg10']:.4f}   "
            f"Recall@{args.top_k}={agg['rlm_recall10']:.4f}",
            f"  Δ NDCG:       {agg['rlm_ndcg10'] - agg['baseline_ndcg10']:+.4f}",
            f"  Δ Recall:     {agg['rlm_recall10'] - agg['baseline_recall10']:+.4f}",
            f"  mean searches/q: {agg['n_search_calls']:.1f}   "
            f"mean docs seen: {agg['n_docs_seen']:.1f}   "
            f"gold seen/total: {agg['n_gold_seen_by_agent']:.1f}/{agg['n_gold_total']:.1f}",
            f"  mean steps: {agg['mean_rlm_steps']:.1f}   "
            f"mean t/query: {agg['mean_rlm_s']:.1f}s   "
            f"wall_time: {wall_s:.1f}s",
        ]
    )
    (run_dir / "summary.txt").write_text(summary + "\n")
    print()
    print(summary)


if __name__ == "__main__":
    main()
