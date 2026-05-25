#!/usr/bin/env python3
"""
OBLIQ-Bench RLM rerank benchmark.

Compares three reranking strategies on the OBLIQ-Bench ``math`` subset
(Tchuindjo et al. 2026, arXiv:2605.06235):

  * baseline   — BM25 top-10 directly
  * (reference) — listwise rerank from ``oblique_rerank_bench.py``
  * **rlm**    — pyrlm-runtime ``RLM`` loop with parallel subcalls verifying
                 each candidate via ``verify_relevance_batch``

For each query the RLM root receives the BM25 top-50 as a ``bm25_pool``
variable in the REPL. The root decides how to use ``verify_relevance_batch``
(parallel sub-LLM verification) and prints the final top-10.

Adapter wiring (Phase B decisions, see docs/OBLIQ-EXPERIMENTS.md):

  * Root: ``gpt-5.1`` via :class:`AzureOpenAIAdapter`
  * Subcall: ``gpt-5.4-mini`` via :class:`AzureOpenAIAdapter` — fast, cheap
    verification per candidate

Usage:

  # Smoke test (1 query, FakeAdapter, no Azure credentials needed)
  uv run python examples/oblique_rlm_bench.py --smoke

  # Real run (Phase B headline number, 151 queries)
  AZURE_OPENAI_API_KEY=... OPENAI_ENDPOINT=... \\
  uv run python examples/oblique_rlm_bench.py \\
    --root-model gpt-5.1 --subcall-model gpt-5.4-mini \\
    --max-examples 151 --workers 2 \\
    --output-dir examples/exports/oblique_rlm_bench/<run>
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

from _rlm_rerank_tools import RLM_RERANK_SYSTEM_PROMPT, build_rlm_rerank_extensions
from pyrlm_runtime import Context, Policy, RLM, ndcg_at_k, recall_at_k
from pyrlm_runtime.adapters import FakeAdapter, ModelAdapter
from pyrlm_runtime.cache import CacheRecord

load_dotenv()


class _NoopCache:
    """No-op cache that always misses. Forces every subcall to hit the
    real LLM, so bench results reflect fresh model behavior rather than
    cached responses from earlier exploratory runs.
    """

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


# ---------------------------------------------------------------------------
# Dataset (mirrors oblique_rerank_bench.py)
# ---------------------------------------------------------------------------


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

    def search(self, query_id: str, query_text: str, top_n: int) -> list[dict[str, Any]]:
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


class OracleRetriever:
    """Mirror of the listwise bench's oracle pool — gold + random distractors.

    Lets the RLM bench be compared directly against
    ``oblique_rerank_bench.py --retriever oracle`` since the candidate
    pool composition is identical (same seed, same ordering).
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
        self._seed = seed

    def search(self, query_id: str, query_text: str, top_n: int) -> list[dict[str, Any]]:
        gold = [c for c, s in self._qrels.get(query_id, {}).items() if s > 0]
        excluded = self._excluded.get(query_id, set())
        positives = [c for c in gold if c in self._corpus]
        rng = random.Random(f"{self._seed}:{query_id}")
        seen: set[str] = set()
        pool: list[str] = []
        for c in positives:
            if c not in seen:
                seen.add(c)
                pool.append(c)
        candidates = [
            c
            for c in self._corpus_ids
            if c not in seen
            and c not in excluded
            and c not in self._qrels.get(query_id, {})
        ]
        rng.shuffle(candidates)
        for c in candidates:
            if len(pool) >= top_n:
                break
            pool.append(c)
        rng.shuffle(pool)
        return [
            {
                "doc_id": c,
                "content": self._corpus[c],
                "preview": self._corpus[c][:500],
                "score": 1.0 / (i + 1),
                "metadata": {},
            }
            for i, c in enumerate(pool[:top_n])
        ]


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


_PY_LIST_RE = re.compile(r"\[[^\[\]]+\]", re.DOTALL)


def parse_top10_from_rlm_output(text: str, valid_doc_ids: set[str]) -> list[str]:
    """Find the last Python-list literal in the RLM output that contains doc_ids."""
    # Try the final answer line first (everything after the last newline often
    # is the print(top10_ids) output)
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
# Adapter construction
# ---------------------------------------------------------------------------


def build_root_adapter(args: argparse.Namespace) -> ModelAdapter:
    if args.adapter == "fake":
        return _make_fake_root_adapter()
    if args.adapter == "azure":
        from pyrlm_runtime.adapters import AzureOpenAIAdapter

        kw: dict[str, Any] = {"model": args.root_model, "timeout": 900.0}
        if args.api_version:
            kw["api_version"] = args.api_version
        return AzureOpenAIAdapter(**kw)
    raise SystemExit(f"Unknown adapter {args.adapter!r}")


def build_subcall_adapter(args: argparse.Namespace) -> ModelAdapter:
    if args.adapter == "fake":
        return _make_fake_subcall_adapter()
    if args.adapter == "azure":
        from pyrlm_runtime.adapters import AzureOpenAIAdapter

        kw: dict[str, Any] = {"model": args.subcall_model, "timeout": args.subcall_timeout}
        if args.api_version:
            kw["api_version"] = args.api_version
        return AzureOpenAIAdapter(**kw)
    raise SystemExit(f"Unknown adapter {args.adapter!r}")


def _make_fake_root_adapter() -> FakeAdapter:
    """Fake root adapter that performs a single verify_relevance_batch then prints top-10.

    Used only for the ``--smoke`` test path to confirm the bench wiring.
    """
    code_block = (
        "```python\n"
        "verdicts = verify_relevance_batch(query, [d['doc_id'] for d in bm25_pool])\n"
        "relevant = [v for v in verdicts if v['relevant']]\n"
        "by_score = {d['doc_id']: d['score'] for d in bm25_pool}\n"
        "ranked = sorted(relevant, key=lambda v: by_score.get(v['doc_id'], 0.0), "
        "reverse=True)\n"
        "remaining = [d['doc_id'] for d in bm25_pool if d['doc_id'] not in "
        "{v['doc_id'] for v in ranked}]\n"
        "top10_ids = [v['doc_id'] for v in ranked[:10]]\n"
        "top10_ids += remaining[: 10 - len(top10_ids)]\n"
        "print(top10_ids)\n"
        "```"
    )
    # Script: first call returns code, subsequent calls return FINAL.
    return FakeAdapter(script=[code_block] + ["FINAL: done"] * 30)


def _make_fake_subcall_adapter() -> FakeAdapter:
    """Fake subcall adapter that always returns score=5 for smoke."""
    adapter = FakeAdapter()
    adapter.add_rule(
        "",
        '{"score": 5, "reason": "smoke-test verifier always says strong match"}',
        regex=False,
    )
    return adapter


# ---------------------------------------------------------------------------
# Single-query evaluation
# ---------------------------------------------------------------------------


def evaluate_one(
    *,
    query_id: str,
    query_text: str,
    qrels: dict[str, float],
    corpus: dict[str, str],
    first_stage: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    t_first = time.perf_counter()
    pool = first_stage.search(query_id, query_text, args.top_n)
    first_elapsed = time.perf_counter() - t_first

    baseline_ids = [d["doc_id"] for d in pool[: args.top_k]]
    valid_ids = {d["doc_id"] for d in pool}

    root_adapter = build_root_adapter(args)
    subcall_adapter = build_subcall_adapter(args)
    verify_log: list[dict[str, Any]] = []
    extensions = build_rlm_rerank_extensions(
        corpus=corpus,
        query_text=query_text,
        bm25_pool=pool,
        max_passage_chars=args.max_passage_chars,
        subcall_max_tokens=args.subcall_max_tokens,
        verify_log=verify_log,
    )

    rlm = RLM(
        adapter=root_adapter,
        subcall_adapter=subcall_adapter,
        policy=Policy(
            max_steps=args.max_steps,
            max_subcalls=args.max_subcalls,
            max_total_tokens=args.max_total_tokens,
        ),
        max_tokens=args.root_max_tokens,
        subcall_max_tokens=args.subcall_max_tokens,
        parallel_subcalls=True,
        max_concurrent_subcalls=args.max_concurrent_subcalls,
        system_prompt=RLM_RERANK_SYSTEM_PROMPT,
        repl_extensions=extensions,
        # Auto-finalize when the REPL has a `top10_ids` variable set to a
        # non-empty Python list. Combined with require_repl_before_final and
        # require_subcall_before_final this forces the loop to actually run
        # verify_relevance_batch before it can terminate.
        auto_finalize_var="top10_ids",
        require_repl_before_final=True,
        require_subcall_before_final=True,
        # By default disable the RLM's FileCache so bench numbers reflect
        # fresh LLM responses. Pass --use-cache to opt back into the
        # default .rlm_cache directory.
        cache=None if args.use_cache else _NoopCache(),
    )

    rlm_ids: list[str] = []
    raw_output: str = ""
    trace_kinds: dict[str, int] = {}
    error: str | None = None
    rlm_top_k_source = "baseline_fallback"
    t_rlm = time.perf_counter()
    try:
        output, trace = rlm.run(query_text, Context.from_text(""))
        raw_output = output or ""
        rlm_ids = parse_top10_from_rlm_output(raw_output, valid_ids)
        if rlm_ids:
            rlm_top_k_source = "final"
        # If FINAL message didn't yield a parseable list, scan the REPL
        # outputs in the trace as a fallback.
        if not rlm_ids and trace and trace.steps:
            for step in reversed(trace.steps):
                step_text = getattr(step, "stdout", None) or getattr(step, "output", None) or ""
                if not step_text:
                    continue
                rlm_ids = parse_top10_from_rlm_output(str(step_text), valid_ids)
                if rlm_ids:
                    rlm_top_k_source = "trace_fallback"
                    break
        steps = len(trace.steps) if trace and trace.steps else 0
        subcalls = sum(
            1
            for step in (trace.steps if trace else [])
            if step.kind in {"subcall", "recursive_subcall"}
        )
        if trace and trace.steps:
            for step in trace.steps:
                trace_kinds[step.kind] = trace_kinds.get(step.kind, 0) + 1
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
        steps = 0
        subcalls = 0
    rlm_elapsed = time.perf_counter() - t_rlm

    # If RLM failed or returned fewer than top_k, pad with baseline tail to
    # ensure metrics get a fair chance to score.
    if len(rlm_ids) < args.top_k:
        for doc_id in baseline_ids:
            if doc_id not in rlm_ids:
                rlm_ids.append(doc_id)
            if len(rlm_ids) >= args.top_k:
                break
    rlm_ids = rlm_ids[: args.top_k]

    # Summarise what the verifier said: how many it marked relevant, and
    # whether any of those overlap with the dataset's gold judgments.
    verify_summary: list[dict[str, Any]] = []
    gold_set = {d for d, s in qrels.items() if s > 0}
    for call in verify_log:
        verdicts = call.get("verdicts", [])
        relevant_ids = {v["doc_id"] for v in verdicts if v["relevant"]}
        pool_gold = gold_set & {v["doc_id"] for v in verdicts}
        gold_verdicts = [v for v in verdicts if v["doc_id"] in gold_set]
        non_gold_verdicts = [v for v in verdicts if v["doc_id"] not in gold_set]
        verify_summary.append(
            {
                "input_count": call.get("input_count", 0),
                "relevant_count": call.get("relevant_count", 0),
                "score_distribution": call.get("score_distribution"),
                "mean_score_gold": (
                    sum(v.get("score", 0) for v in gold_verdicts) / len(gold_verdicts)
                    if gold_verdicts
                    else None
                ),
                "mean_score_non_gold": (
                    sum(v.get("score", 0) for v in non_gold_verdicts)
                    / len(non_gold_verdicts)
                    if non_gold_verdicts
                    else None
                ),
                "pool_gold_count": len(pool_gold),
                "gold_marked_relevant": len(relevant_ids & gold_set),
                "gold_marked_not_relevant": len(pool_gold - relevant_ids),
                "non_gold_marked_relevant": len(relevant_ids - gold_set),
                "first_gold_verdicts": [
                    {
                        "doc_id": v["doc_id"],
                        "score": v.get("score"),
                        "relevant": v.get("relevant"),
                        "reason": v["reason"],
                    }
                    for v in gold_verdicts
                ][:5],
            }
        )

    return {
        "query_id": query_id,
        "n_candidates": len(pool),
        "baseline_top_k": baseline_ids,
        "rlm_top_k": rlm_ids,
        "rlm_top_k_source": rlm_top_k_source,
        "rlm_raw_output_tail": raw_output[-1500:] if raw_output else "",
        "trace_kinds": trace_kinds,
        "verify_summary": verify_summary,
        "baseline_ndcg10": ndcg_at_k(baseline_ids, qrels, k=args.top_k),
        "rlm_ndcg10": ndcg_at_k(rlm_ids, qrels, k=args.top_k),
        "baseline_recall10": recall_at_k(baseline_ids, qrels, k=args.top_k),
        "rlm_recall10": recall_at_k(rlm_ids, qrels, k=args.top_k),
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
        description="OBLIQ-Bench RLM rerank benchmark (Phase B, version B).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subset", default="math", choices=sorted(SUBSET_PATHS))
    parser.add_argument(
        "--retriever",
        default="bm25",
        choices=["bm25", "oracle"],
        help="First-stage pool source. 'oracle' mirrors the listwise bench's "
        "oracle setup (gold injected + random distractors).",
    )
    parser.add_argument("--adapter", default="azure", choices=["fake", "azure"])
    parser.add_argument("--root-model", default="gpt-5.1")
    parser.add_argument("--subcall-model", default="gpt-5.4-mini")
    parser.add_argument("--api-version", default=None)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-subcalls", type=int, default=100)
    parser.add_argument("--max-total-tokens", type=int, default=500_000)
    parser.add_argument("--root-max-tokens", type=int, default=4096)
    parser.add_argument("--subcall-max-tokens", type=int, default=256)
    parser.add_argument("--max-concurrent-subcalls", type=int, default=10)
    parser.add_argument("--max-passage-chars", type=int, default=600)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--subcall-timeout",
        type=float,
        default=60.0,
        help="HTTP timeout for the subcall adapter. Lower = fail fast on "
        "stuck Azure connections.",
    )
    parser.add_argument(
        "--per-query-timeout",
        type=float,
        default=180.0,
        help="Per-query timeout in seconds. Queries that exceed this are "
        "abandoned and recorded as errors so one stuck query cannot block "
        "the rest of the run.",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Enable the RLM's FileCache (default: disabled). Disabling "
        "the cache makes every subcall hit the real LLM, which is the "
        "right default for publication-quality benchmarks.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny smoke test (1 query, FakeAdapter, single worker).",
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

    if args.retriever == "bm25":
        print("Building in-memory BM25 index …")
        first_stage: Any = InMemoryBM25(corpus, excluded_ids)
    else:
        print("Building oracle retriever (gold + random distractors) …")
        first_stage = OracleRetriever(corpus, qrels, excluded_ids, seed=args.seed)

    run_label = (
        f"run_{_now_tag()}_{args.subset}_{args.retriever}_{args.adapter}_"
        f"{_safe(args.root_model)}_x_{_safe(args.subcall_model)}"
    )
    run_dir = Path(args.output_dir or f"examples/exports/oblique_rlm_bench/{run_label}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  output: {run_dir}\n")

    def run_one(q: str) -> dict[str, Any]:
        return evaluate_one(
            query_id=q,
            query_text=queries[q],
            qrels=qrels[q],
            corpus=corpus,
            first_stage=first_stage,
            args=args,
        )

    records: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    def _stub_record(q: str, reason: str) -> dict[str, Any]:
        return {
            "query_id": q,
            "n_candidates": 0,
            "baseline_top_k": [],
            "rlm_top_k": [],
            "rlm_top_k_source": "baseline_fallback",
            "rlm_raw_output_tail": "",
            "trace_kinds": {},
            "verify_summary": [],
            "baseline_ndcg10": 0.0,
            "rlm_ndcg10": 0.0,
            "baseline_recall10": 0.0,
            "rlm_recall10": 0.0,
            "first_stage_s": 0.0,
            "rlm_s": 0.0,
            "rlm_steps": 0,
            "rlm_subcalls": 0,
            "error": reason,
        }

    def _run_with_hard_timeout(q: str) -> dict[str, Any]:
        """Run a query in a daemon thread; abandon it after per_query_timeout.

        ``threading.Thread`` cannot be killed from Python, but daemon threads
        die with the process. We just stop waiting and mark the query as a
        timeout error so the bench can move on.
        """
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
            return _stub_record(
                q, f"error: {type(exc).__name__}: {str(exc)[:120]}"
            )
        return _stub_record(
            q, f"per_query_timeout after {args.per_query_timeout:.0f}s"
        )

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_with_hard_timeout, q): q for q in eligible}
            for fut in as_completed(futures):
                rec = fut.result()
                records.append(rec)
                err = f" ERR={rec['error'][:80]!r}" if rec.get("error") else ""
                print(
                    f"  [{len(records)}/{len(eligible)}] {rec['query_id']} "
                    f"base={rec['baseline_ndcg10']:.3f} "
                    f"rlm={rec['rlm_ndcg10']:.3f} "
                    f"steps={rec['rlm_steps']} subcalls={rec['rlm_subcalls']}"
                    f" t={rec['rlm_s']:.1f}s{err}",
                    flush=True,
                )
    else:
        for i, q in enumerate(eligible, start=1):
            rec = _run_with_hard_timeout(q)
            records.append(rec)
            err = f" ERR={rec['error'][:80]!r}" if rec.get("error") else ""
            print(
                f"  [{i}/{len(eligible)}] {rec['query_id']} "
                f"base={rec['baseline_ndcg10']:.3f} "
                f"rlm={rec['rlm_ndcg10']:.3f} "
                f"steps={rec['rlm_steps']} subcalls={rec['rlm_subcalls']}"
                f" t={rec['rlm_s']:.1f}s{err}",
                flush=True,
            )

    wall_s = time.perf_counter() - t0
    agg = aggregate(records)
    agg["wall_time_s"] = wall_s

    metrics = {
        "subset": args.subset,
        "retriever": args.retriever,
        "adapter": args.adapter,
        "root_model": args.root_model,
        "subcall_model": args.subcall_model,
        "params": {
            "top_n": args.top_n,
            "top_k": args.top_k,
            "max_steps": args.max_steps,
            "max_subcalls": args.max_subcalls,
            "max_concurrent_subcalls": args.max_concurrent_subcalls,
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
        f"OBLIQ-Bench / {args.subset}  ({args.retriever} → RLM-as-reranker, "
        f"root={args.root_model}, subcall={args.subcall_model})",
        f"  examples:     {int(agg.get('n_examples', 0))}   "
        f"errors: {int(agg.get('n_errors', 0))}",
        f"  baseline:     NDCG@{args.top_k}={agg['baseline_ndcg10']:.4f}   "
        f"Recall@{args.top_k}={agg['baseline_recall10']:.4f}",
        f"  rlm-rerank:   NDCG@{args.top_k}={agg['rlm_ndcg10']:.4f}   "
        f"Recall@{args.top_k}={agg['rlm_recall10']:.4f}",
        f"  Δ NDCG:       {agg['rlm_ndcg10'] - agg['baseline_ndcg10']:+.4f}",
        f"  Δ Recall:     {agg['rlm_recall10'] - agg['baseline_recall10']:+.4f}",
        f"  mean steps:   {agg['mean_rlm_steps']:.1f}   "
        f"mean subcalls: {agg['mean_rlm_subcalls']:.1f}   "
        f"mean t/query:  {agg['mean_rlm_s']:.1f}s",
        f"  wall_time:    {wall_s:.1f}s",
    ]
    summary = "\n".join(summary_lines)
    (run_dir / "summary.txt").write_text(summary + "\n")
    print()
    print(summary)


if __name__ == "__main__":
    main()
