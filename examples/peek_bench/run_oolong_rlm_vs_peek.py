#!/usr/bin/env python3
"""PEEK Benchmark: RLM baseline vs RLM+PEEK on oolong-synth.

Evaluates the PEEK orientation-cache on the "same context × N queries" workload
from arXiv:2605.19932.  For each sampled context, ALL its queries are run twice:

  1) baseline  — fresh RLM for every query (no persistent map)
  2) rlm+peek  — RLM with a PeekSession that evolves over the first
                 ``--evolve-steps`` queries and is reused for the rest

Results are written to docs/peek-bench/ and include the per-query scores,
iteration counts, token usage, and the final context map for each context.

**You run this — do NOT invoke it from automation or CI.**

Prerequisites:
  pip install git+https://github.com/zhuohangu/peek.git   # peek-ai
  uv sync --group dev                                       # datasets

Usage (pilot — 5 contexts × their queries, dry-run cost estimate):
  python examples/peek_bench/run_oolong_rlm_vs_peek.py \\
    --model gpt-5.1 --n-contexts 5 --dry-run

Phase 0 baseline (N=10 contexts, cache OFF):
  AZURE_OPENAI_API_KEY=... OPENAI_ENDPOINT=... \\
  python examples/peek_bench/run_oolong_rlm_vs_peek.py \\
    --model gpt-5.1 --n-contexts 10 --mode baseline --seed 42

Phase 2 comparison (N=30 contexts, cache OFF):
  AZURE_OPENAI_API_KEY=... OPENAI_ENDPOINT=... \\
  python examples/peek_bench/run_oolong_rlm_vs_peek.py \\
    --model gpt-5.1 --n-contexts 30 --mode compare --seed 42 --evolve-steps 4
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))  # noqa: E402
sys.path.insert(0, str(_REPO_ROOT / "examples"))  # noqa: E402

from _azure_check import check_azure_connection  # noqa: E402
from pyrlm_runtime import Context, Policy, RLM  # noqa: E402
from pyrlm_runtime.adapters import AzureOpenAIAdapter  # noqa: E402
from pyrlm_runtime.peek_integration import PeekSession  # noqa: E402
from pyrlm_runtime.prompts import BASE_SYSTEM_PROMPT  # noqa: E402


def _load_env() -> None:
    """Load .env walking up from the script. Gracefully skips if python-dotenv is absent."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return  # env vars must be set externally
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
    if os.getenv("AZURE_OPENAI_API_KEY"):
        return
    here = Path(__file__).resolve()
    for candidate in [here.parents[2] / ".env", here.parents[3] / ".env"]:
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            if os.getenv("AZURE_OPENAI_API_KEY"):
                return
    load_dotenv(override=False)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-5.1"
# System-prompt tips from the existing oolong runner (improves RLM on synth tasks)
_OOLONG_ENV_TIPS = """
<env_tips>
Strategy for structured data tasks (dates, labels, user IDs):
1. ALWAYS use Python code for counting — never delegate counting to sub-LLMs.
2. For large contexts (>32K chars), split with ctx.chunk() and process each chunk.
3. Use llm_batch() ONLY for semantic understanding, never for counting.
4. Verify your answer with a second pass before finalising.
</env_tips>
"""



def _parse_gold(datapoint: dict[str, Any]) -> Any:
    try:
        return (
            ast.literal_eval(datapoint["answer"])[0]
            if "datetime" not in str(datapoint["answer"])
            else datetime.strptime(datapoint["answer"], "[datetime.date(%Y, %m, %d)]")
        )
    except Exception:
        return datapoint["answer"]


def _parse_predicted(output: str) -> tuple[str, str]:
    if ":" not in output:
        return (output if len(output) < 20 else output.split()[-1]), "low"
    candidate = output.split(":")[-1].strip().replace("*", "").replace("[", "").replace("]", "")
    confidence = "med"
    if any(kw in output for kw in ("User:", "Answer:", "Date:", "Label")):
        confidence = "high"
    if len(candidate) < 20:
        confidence = "vhigh"
    elif "more common" in candidate:
        candidate = "more common"
    elif "less common" in candidate:
        candidate = "less common"
    elif "same frequency" in candidate:
        candidate = "same frequency"
    return candidate, confidence


def score_example(datapoint: dict[str, Any], output: str) -> float:
    gold = _parse_gold(datapoint)
    trimmed, _ = _parse_predicted(output)
    if str(trimmed) == str(gold):
        return 1.0
    if str(trimmed) in ("more common", "less common", "same frequency"):
        return float(str(trimmed) in str(gold))
    if datapoint.get("answer_type") == "ANSWER_TYPE.NUMERIC":
        try:
            return float(0.75 ** abs(int(trimmed) - int(gold)))
        except Exception:
            return 0.0
    if datapoint.get("answer_type") == "ANSWER_TYPE.DATE":
        try:
            import dateutil.parser

            parsed = dateutil.parser.parse(str(trimmed))
            return float(parsed == gold)
        except Exception:
            return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_model(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def _trace_step_count(trace: Any) -> int:
    if trace is None or not hasattr(trace, "steps"):
        return 0
    return len(trace.steps)


def _trace_total_tokens(trace: Any) -> int:
    if trace is None or not hasattr(trace, "steps"):
        return 0
    return sum((s.usage.total_tokens for s in trace.steps if s.usage), 0)


# ---------------------------------------------------------------------------
# Single-query runners
# ---------------------------------------------------------------------------


def run_baseline_query(
    adapter: AzureOpenAIAdapter,
    context_text: str,
    question: str,
    *,
    max_steps: int,
    max_tokens: int,
    env_tips: bool,
) -> tuple[str, int, int, float, str | None]:
    """Run one query with plain RLM (no PEEK). Returns (answer, steps, tokens, elapsed, error)."""
    system_prompt = BASE_SYSTEM_PROMPT + (_OOLONG_ENV_TIPS if env_tips else "")
    context = Context.from_text(context_text)
    rlm = RLM(
        adapter=adapter,
        policy=Policy(max_steps=max_steps, max_total_tokens=4_000_000),
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        require_repl_before_final=True,
    )
    start = time.time()
    trace = None
    try:
        output, trace = rlm.run(question, context)
        return (
            output or "",
            _trace_step_count(trace),
            _trace_total_tokens(trace),
            time.time() - start,
            None,
        )
    except Exception as exc:
        return (
            "",
            _trace_step_count(trace),
            _trace_total_tokens(trace),
            time.time() - start,
            str(exc),
        )


def run_peek_query(
    adapter: AzureOpenAIAdapter,
    context_text: str,
    question: str,
    session: PeekSession,
    *,
    max_steps: int,
    max_tokens: int,
    env_tips: bool,
) -> tuple[str, int, int, float, str | None]:
    """Run one query with RLM+PEEK. Mutates session in-place. Returns (answer, steps, tokens, elapsed, error)."""
    system_prompt = BASE_SYSTEM_PROMPT + (_OOLONG_ENV_TIPS if env_tips else "")
    context = Context.from_text(context_text)
    rlm = RLM(
        adapter=adapter,
        policy=Policy(max_steps=max_steps, max_total_tokens=4_000_000),
        system_prompt=system_prompt,
        system_prompt_supplement=session.system_prompt_supplement,
        max_tokens=max_tokens,
        require_repl_before_final=True,
    )
    start = time.time()
    trace = None
    try:
        output, trace = rlm.run(question, context)
        session.update_from_run(trace, query=question)
        return (
            output or "",
            _trace_step_count(trace),
            _trace_total_tokens(trace),
            time.time() - start,
            None,
        )
    except Exception as exc:
        if trace is not None:
            session.update_from_run(trace, query=question)
        return (
            "",
            _trace_step_count(trace),
            _trace_total_tokens(trace),
            time.time() - start,
            str(exc),
        )


# ---------------------------------------------------------------------------
# Context-group evaluation
# ---------------------------------------------------------------------------


def evaluate_context_group(
    rows: list[dict[str, Any]],
    context_text: str,
    adapter: AzureOpenAIAdapter,
    peek_adapter: AzureOpenAIAdapter,
    *,
    mode: str,
    max_steps: int,
    max_tokens: int,
    env_tips: bool,
    evolve_steps: int | None,
    token_budget: int,
    peek_map_dir: Path | None,
    peek_trace_dir: Path | None,
    context_window_id: Any,
    dry_run: bool,
) -> dict[str, Any]:
    """Evaluate all queries for one context under baseline, peek, or both modes."""
    n_queries = len(rows)
    ctx_len = len(context_text)
    print(f"  Context {context_window_id}: {n_queries} queries, ctx_len={ctx_len}")

    baseline_results: list[dict[str, Any]] = []
    peek_results: list[dict[str, Any]] = []
    peek_session: PeekSession | None = None

    if mode in ("peek", "compare") and not dry_run:
        peek_session = PeekSession.create(
            peek_adapter,
            token_budget=token_budget,
            evolve_steps=evolve_steps,  # None = fully online (never freezes)
            trace_dir=(peek_trace_dir / f"ctx_{context_window_id}")
            if peek_trace_dir is not None
            else None,
        )

    for qi, row in enumerate(rows):
        question = row["question"]
        q_id = row.get("id", qi)

        if dry_run:
            # Estimate only — don't make real API calls
            est_tokens = ctx_len // 4 + max_tokens
            baseline_results.append(
                {
                    "id": q_id,
                    "score": 0.0,
                    "steps": 0,
                    "tokens": est_tokens,
                    "elapsed": 0.0,
                    "error": "DRY_RUN",
                }
            )
            peek_results.append(
                {
                    "id": q_id,
                    "score": 0.0,
                    "steps": 0,
                    "tokens": est_tokens,
                    "elapsed": 0.0,
                    "error": "DRY_RUN",
                }
            )
            continue

        # --- Baseline ---
        if mode in ("baseline", "compare"):
            ans, steps, tokens, elapsed, err = run_baseline_query(
                adapter,
                context_text,
                question,
                max_steps=max_steps,
                max_tokens=max_tokens,
                env_tips=env_tips,
            )
            score = score_example(row, ans)
            baseline_results.append(
                {
                    "id": q_id,
                    "query_idx": qi,
                    "question": question[:100],
                    "answer": ans,
                    "gold": str(_parse_gold(row)),
                    "score": score,
                    "steps": steps,
                    "tokens": tokens,
                    "elapsed": elapsed,
                    "error": err,
                }
            )
            print(
                f"    [{qi + 1}/{n_queries}] baseline  score={score:.2f} steps={steps} tok={tokens} t={elapsed:.1f}s"
            )

        # --- PEEK ---
        if mode in ("peek", "compare") and peek_session is not None:
            ans, steps, tokens, elapsed, err = run_peek_query(
                adapter,
                context_text,
                question,
                peek_session,
                max_steps=max_steps,
                max_tokens=max_tokens,
                env_tips=env_tips,
            )
            score = score_example(row, ans)
            peek_results.append(
                {
                    "id": q_id,
                    "query_idx": qi,
                    "question": question[:100],
                    "answer": ans,
                    "gold": str(_parse_gold(row)),
                    "score": score,
                    "steps": steps,
                    "tokens": tokens,
                    "elapsed": elapsed,
                    "error": err,
                    "map_evolving": peek_session.evolving,
                    "map_step": peek_session.steps,
                }
            )
            print(
                f"    [{qi + 1}/{n_queries}] rlm+peek  score={score:.2f} steps={steps} tok={tokens} t={elapsed:.1f}s (map_step={peek_session.steps})"
            )

    # Save final map
    if peek_session is not None and peek_map_dir is not None and not dry_run:
        map_path = peek_map_dir / f"ctx_{context_window_id}.peek.json"
        peek_session.save(map_path)

    def _avg(items: list[dict], key: str) -> float:
        vals = [x[key] for x in items if x.get("error") is None or x["error"] in (None, "")]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "context_window_id": context_window_id,
        "context_len": ctx_len,
        "n_queries": n_queries,
        "baseline": {
            "queries": baseline_results,
            "avg_score": _avg(baseline_results, "score"),
            "avg_steps": _avg(baseline_results, "steps"),
            "avg_tokens": _avg(baseline_results, "tokens"),
            "errors": sum(1 for r in baseline_results if r.get("error")),
        }
        if baseline_results
        else None,
        "peek": {
            "queries": peek_results,
            "avg_score": _avg(peek_results, "score"),
            "avg_steps": _avg(peek_results, "steps"),
            "avg_tokens": _avg(peek_results, "tokens"),
            "errors": sum(1 for r in peek_results if r.get("error")),
        }
        if peek_results
        else None,
    }


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------


def aggregate_results(context_results: list[dict[str, Any]]) -> dict[str, Any]:
    def _agg(key: str) -> dict[str, float]:
        scores = [r[key]["avg_score"] for r in context_results if r.get(key)]
        steps = [r[key]["avg_steps"] for r in context_results if r.get(key)]
        tokens = [r[key]["avg_tokens"] for r in context_results if r.get(key)]
        n = len(scores)
        return {
            "n_contexts": n,
            "avg_score": sum(scores) / n if n else 0.0,
            "avg_steps_per_query": sum(steps) / n if n else 0.0,
            "avg_tokens_per_query": sum(tokens) / n if n else 0.0,
        }

    out: dict[str, Any] = {}
    if any(r.get("baseline") for r in context_results):
        out["baseline"] = _agg("baseline")
    if any(r.get("peek") for r in context_results):
        out["peek"] = _agg("peek")
    if "baseline" in out and "peek" in out:
        b, p = out["baseline"], out["peek"]
        out["delta"] = {
            "score": p["avg_score"] - b["avg_score"],
            "steps": p["avg_steps_per_query"] - b["avg_steps_per_query"],
            "tokens": p["avg_tokens_per_query"] - b["avg_tokens_per_query"],
            "score_pct": (p["avg_score"] - b["avg_score"]) / max(b["avg_score"], 1e-6) * 100,
        }
    return out


def print_summary(summary: dict[str, Any], config: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("PEEK BENCHMARK SUMMARY")
    print("=" * 72)
    print(
        f"  model={config['model']}  mode={config['mode']}  n_contexts={config['n_contexts']}  seed={config['seed']}"
    )
    print(f"  evolve_steps={config['evolve_steps']}  token_budget={config['token_budget']}")
    print()

    for engine in ("baseline", "peek"):
        if engine not in summary:
            continue
        s = summary[engine]
        print(
            f"  {engine:10s}  score={s['avg_score']:.4f}  "
            f"steps/q={s['avg_steps_per_query']:.1f}  "
            f"tokens/q={s['avg_tokens_per_query']:.0f}  "
            f"n_contexts={s['n_contexts']}"
        )

    if "delta" in summary:
        d = summary["delta"]
        sign = "+" if d["score"] >= 0 else ""
        print(f"\n  Δ score = {sign}{d['score']:.4f}  ({sign}{d['score_pct']:.1f}%)")
        print(f"  Δ steps/q = {d['steps']:+.1f}")
        print(f"  Δ tokens/q = {d['tokens']:+.0f}")

        # Decision rule evaluation
        print("\n  Decision rule (pre-committed):")
        score_ok = d["score"] >= 0.05
        steps_ok = d["steps"] <= 0.0
        tokens_ok = (
            summary["peek"]["avg_tokens_per_query"]
            <= 1.5 * summary["baseline"]["avg_tokens_per_query"]
        )
        if score_ok and steps_ok and tokens_ok:
            verdict = "PROMOTE"
        elif d["score"] >= 0.02 and steps_ok and tokens_ok:
            verdict = "HOLD (scale to N=150)"
        else:
            verdict = "REJECT — document as ablation"
        print(f"  → {verdict}")
        print(
            f"     score≥+5pp: {'✓' if score_ok else '✗'}  "
            f"steps≤baseline: {'✓' if steps_ok else '✗'}  "
            f"tokens≤1.5×baseline: {'✓' if tokens_ok else '✗'}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(description="PEEK benchmark: RLM vs RLM+PEEK on oolong-synth")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--mode",
        choices=["baseline", "peek", "compare"],
        default="compare",
        help="baseline=RLM only, peek=PEEK only, compare=both (default: compare)",
    )
    parser.add_argument(
        "--n-contexts",
        type=int,
        default=10,
        help="Number of context groups to evaluate (default: 10 for pilot)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--context-ids",
        type=str,
        default=None,
        help="Comma-separated list of explicit context_window_ids to evaluate (overrides --n-contexts)",
    )
    parser.add_argument(
        "--evolve-steps",
        type=int,
        default=4,
        help="PEEK evolve_steps (m in paper, default: 4). Use -1 for fully online (never freezes).",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=1024,
        help="PEEK context map token budget B (default: 1024)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=15, help="Max RLM steps per query (default: 15)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max LLM output tokens per step (default: 2048)",
    )
    parser.add_argument(
        "--env-tips", action="store_true", help="Append oolong strategy tips to system prompt"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan and cost estimate without making API calls",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Write per-query PEEK trace JSON under runs/<tag>/traces/ctx_<id>/q{NN}.json",
    )
    parser.add_argument(
        "--output-dir", default=None, help="Output directory (default: docs/peek-bench/runs/<tag>)"
    )
    args = parser.parse_args()

    if not args.dry_run:
        check_azure_connection(args.model)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing: datasets. Install with: uv add datasets --dev") from exc

    # --- Load and group dataset ---
    print("Loading oolongbench/oolong-synth …")
    data = load_dataset("oolongbench/oolong-synth", split="test")
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in data:
        groups[row["context_window_id"]].append(dict(row))

    # Keep only groups with ≥ 5 queries (filter tiny outliers)
    groups = {k: v for k, v in groups.items() if len(v) >= 5}

    # Pick contexts: explicit IDs override sampling
    if args.context_ids:
        requested = [int(x.strip()) for x in args.context_ids.split(",") if x.strip()]
        missing = [c for c in requested if c not in groups]
        if missing:
            raise SystemExit(f"context_ids not found in dataset or have <5 queries: {missing}")
        selected_ids = requested
        print(f"Selected {len(selected_ids)} explicit contexts: {selected_ids}")
    else:
        rng = random.Random(args.seed)
        ctx_ids = sorted(groups.keys())
        rng.shuffle(ctx_ids)
        selected_ids = ctx_ids[: args.n_contexts]
        print(f"Selected {len(selected_ids)} contexts (seed={args.seed})")

    if args.dry_run:
        total_queries = sum(len(groups[cid]) for cid in selected_ids)
        avg_ctx_len = sum(
            len(groups[cid][0].get("context_window_text_with_labels", "")) for cid in selected_ids
        ) / max(1, len(selected_ids))
        est_tokens_per_q = int(avg_ctx_len / 4 + args.max_tokens)
        runs = 2 if args.mode == "compare" else 1
        est_total_tokens = total_queries * runs * est_tokens_per_q
        print("\nDRY RUN ESTIMATE:")
        print(f"  Total queries: {total_queries} ({runs}× for {args.mode})")
        print(f"  Avg context len: {avg_ctx_len:.0f} chars")
        print(f"  Est. tokens/query: {est_tokens_per_q}")
        print(f"  Est. total tokens: {est_total_tokens:,}")
        print(f"  At $0.40/Mtok input: ~${est_total_tokens * 0.40 / 1_000_000:.2f}")
        print("\n  Run without --dry-run to execute. Authorize spend first.")
        return

    # --- Build adapters (reads AZURE_OPENAI_API_KEY + OPENAI_ENDPOINT from env) ---
    adapter = AzureOpenAIAdapter(model=args.model, timeout=900.0)
    peek_adapter = AzureOpenAIAdapter(model=args.model, timeout=900.0)

    # --- Output directory ---
    run_tag = f"run_{_now_tag()}_{args.mode}_{_safe_model(args.model)}_n{len(selected_ids)}"
    out_dir = Path(args.output_dir or (_REPO_ROOT / "docs" / "peek-bench" / "runs" / run_tag))
    out_dir.mkdir(parents=True, exist_ok=True)
    map_dir = out_dir / "peek_maps"
    trace_dir = (out_dir / "traces") if args.trace else None

    run_config = vars(args)
    run_config["dataset"] = "oolongbench/oolong-synth"
    run_config["selected_context_ids"] = selected_ids
    _write_json(out_dir / "run_config.json", run_config)

    # --- Evaluate ---
    all_results: list[dict[str, Any]] = []
    for i, ctx_id in enumerate(selected_ids):
        rows = groups[ctx_id]
        context_text = rows[0].get("context_window_text_with_labels", "") or rows[0].get(
            "context_window_text", ""
        )
        print(f"\n[{i + 1}/{len(selected_ids)}] context_window_id={ctx_id}")

        result = evaluate_context_group(
            rows,
            context_text,
            adapter,
            peek_adapter,
            mode=args.mode,
            max_steps=args.max_steps,
            max_tokens=args.max_tokens,
            env_tips=args.env_tips,
            evolve_steps=(None if args.evolve_steps < 0 else args.evolve_steps),
            token_budget=args.token_budget,
            peek_map_dir=map_dir,
            peek_trace_dir=trace_dir,
            context_window_id=ctx_id,
            dry_run=False,
        )
        all_results.append(result)
        _write_json(out_dir / f"ctx_{ctx_id}.json", result)

    # --- Aggregate and report ---
    summary = aggregate_results(all_results)
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "all_results.json", all_results)

    print_summary(
        summary,
        {
            "model": args.model,
            "mode": args.mode,
            "n_contexts": len(selected_ids),
            "seed": args.seed,
            "evolve_steps": args.evolve_steps,
            "token_budget": args.token_budget,
        },
    )
    print(f"\nArtifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
