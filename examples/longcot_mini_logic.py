"""LongCoT-mini logic — phase 1 of LOGIC-MATCH-EXPERIMENT.

Runs pyrlm on the 105 logic examples that the original RLM author
published trajectories for at
``../longcot-mini-rlm-results/trajectories/logic/results.jsonl``.

Author's reference: ``reward = 0.9905`` (104 / 105). Goal of this
harness: measure pyrlm's reward on the same examples.

Phase 1 (H1, dataset hypothesis) strips the ``<env_tips>`` suffix the
author appended to each upstream ``longcot/logic`` prompt. With
``--keep-env-tips`` the suffix is preserved (phase 2, H2). The
upstream ``longcot.verify`` is used for scoring after matching each
mini row to its upstream ``Question`` by prompt-body hash.

Single arm (compaction OFF). Compaction is closed as a variable for
this workload — see ``docs/longcot-bench/EXPERIMENTS.md`` closure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Make pyrlm_runtime importable from a clean checkout, and re-use the
# existing compaction-bench harness's runner / summariser to avoid drift.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from longcot_pyrlm_compaction import (  # noqa: E402
    _build_verify_options,
    _run_pyrlm,
    write_json,
)

ENV_TIPS_RE = re.compile(r"\n*<env_tips>.*?</env_tips>\s*$", flags=re.DOTALL)


def strip_env_tips(prompt: str) -> str:
    return ENV_TIPS_RE.sub("", prompt).strip()


def _hash(s: str) -> str:
    return hashlib.md5(s.strip().encode()).hexdigest()


def load_mini_logic(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def index_upstream_logic() -> dict[str, Any]:
    """Hash upstream longcot/logic prompts → Question, for verifier lookup."""
    import longcot

    out: dict[str, Any] = {}
    for q in longcot.load_questions(domain="logic"):
        out[_hash(q.prompt)] = q
    return out


def stratified_by_template(
    pairs: list[tuple[dict[str, Any], Any]],
    *,
    max_examples: int,
    seed: int,
) -> list[tuple[dict[str, Any], Any]]:
    """Sample ``max_examples`` proportionally across templates."""
    if max_examples >= len(pairs):
        return list(pairs)
    rng = random.Random(seed)
    buckets: dict[str, list[tuple[dict[str, Any], Any]]] = defaultdict(list)
    for mini_row, q in pairs:
        tpl = (q.problem or {}).get("template", "?")
        buckets[tpl].append((mini_row, q))

    total = len(pairs)
    target: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    used = 0
    for key, items in buckets.items():
        ideal = len(items) / total * max_examples
        floor = int(ideal)
        target[key] = floor
        used += floor
        remainders.append((ideal - floor, key))
    remainders.sort(reverse=True)
    for _, key in remainders:
        if used >= max_examples:
            break
        target[key] += 1
        used += 1
    chosen: list[tuple[dict[str, Any], Any]] = []
    for key, items in buckets.items():
        items_shuffled = list(items)
        rng.shuffle(items_shuffled)
        chosen.extend(items_shuffled[: target[key]])
    return chosen


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"examples": 0}
    score = sum(r["score"] for r in rows) / n
    tokens = sum(r["tokens"] for r in rows) / n
    elapsed = sum(r["elapsed"] for r in rows) / n
    errors = sum(1 for r in rows if r.get("error"))
    return {
        "examples": n,
        "avg_score": score,
        "avg_tokens": tokens,
        "avg_elapsed": elapsed,
        "errors": errors,
        "score_total": int(sum(r["score"] for r in rows)),
    }


def aggregate_by_template(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[r["template"]].append(r)
    return {tpl: aggregate(rs) for tpl, rs in sorted(by.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mini-jsonl",
        default=str(Path.home() / "projects/longcot-mini-rlm-results/trajectories/logic/results.jsonl"),
    )
    parser.add_argument(
        "--keep-env-tips", action="store_true",
        help="Phase 2: keep the author's <env_tips> suffix in the prompt.",
    )
    parser.add_argument("--provider", choices=["azure", "vertex"], default="azure")
    parser.add_argument("--model", default=None)
    parser.add_argument("--sub-model", default=None)
    parser.add_argument(
        "--api-version",
        default=os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21",
    )
    parser.add_argument("--project-id", default=os.getenv("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument(
        "--location",
        default=os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1",
    )
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument(
        "--sample-strategy", choices=["stratified", "head"], default="stratified",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-subcalls", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--subcall-max-tokens", type=int, default=2048)
    parser.add_argument("--max-concurrent-subcalls", type=int, default=20)
    parser.add_argument(
        "--no-recursive", dest="recursive_subcalls",
        action="store_false", default=True,
    )
    parser.add_argument(
        "--recursion-impl", choices=["child", "fork"], default="child",
        help="Recursive subcall implementation (no-regression guard arm).",
    )
    parser.add_argument("--enable-verifier-fallback", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-traces", dest="save_traces", action="store_true", default=True)
    parser.add_argument("--no-save-traces", dest="save_traces", action="store_false")
    parser.add_argument("--inter-example-sleep", type=float, default=0.0)
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip examples already present in <output-dir>/rows.json (resume mode).",
    )
    args = parser.parse_args()

    if args.provider == "vertex":
        if args.model is None:
            args.model = "gemini-2.5-pro"
        if args.sub_model is None:
            args.sub_model = "gemini-2.5-flash"
        if not args.project_id:
            print("ERROR: --provider=vertex requires --project-id", file=sys.stderr)
            return 2
    else:
        if args.model is None:
            args.model = "gpt-5.2"
        if args.sub_model is None:
            args.sub_model = "gpt-5.4-mini"

    try:
        import longcot
    except ImportError as exc:
        print("ERROR: longcot not importable (uv pip install -e ../longcot)",
              file=sys.stderr)
        print(repr(exc), file=sys.stderr)
        return 2

    mini_path = Path(args.mini_jsonl)
    if not mini_path.exists():
        print(f"ERROR: mini jsonl not found: {mini_path}", file=sys.stderr)
        return 2

    mini_rows = load_mini_logic(mini_path)
    print(f"Loaded {len(mini_rows)} mini rows from {mini_path}")

    upstream = index_upstream_logic()
    pairs: list[tuple[dict[str, Any], Any]] = []
    unmatched: list[int] = []
    for r in mini_rows:
        mini_prompt = r["prompt"][-1]["content"]
        core = strip_env_tips(mini_prompt)
        q = upstream.get(_hash(core))
        if q is None:
            unmatched.append(r["example_id"])
            continue
        pairs.append((r, q))
    if unmatched:
        print(f"WARNING: {len(unmatched)} mini rows did not match upstream "
              f"(first 5: {unmatched[:5]})", file=sys.stderr)
    print(f"Matched {len(pairs)}/{len(mini_rows)} mini rows to upstream questions")

    # Sample
    if args.sample_strategy == "head":
        sampled = sorted(pairs, key=lambda p: p[0]["example_id"])[: args.max_examples]
    else:
        sampled = stratified_by_template(
            pairs, max_examples=args.max_examples, seed=args.seed,
        )
    print(f"Sampled {len(sampled)} examples")
    tpl_dist: dict[str, int] = defaultdict(int)
    for _, q in sampled:
        tpl_dist[(q.problem or {}).get("template", "?")] += 1
    for tpl, n in sorted(tpl_dist.items()):
        print(f"  {tpl}: {n}")

    verify_options = _build_verify_options(args.enable_verifier_fallback)
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", {
        "provider": args.provider,
        "model": args.model,
        "sub_model": args.sub_model,
        "api_version": args.api_version,
        "mini_jsonl": str(mini_path),
        "keep_env_tips": args.keep_env_tips,
        "max_examples": args.max_examples,
        "sample_strategy": args.sample_strategy,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "max_subcalls": args.max_subcalls,
        "max_tokens": args.max_tokens,
        "subcall_max_tokens": args.subcall_max_tokens,
        "recursive_subcalls": args.recursive_subcalls,
        "recursion_impl": args.recursion_impl,
    })

    rows: list[dict[str, Any]] = []
    done_ids: set[str] = set()
    if args.skip_existing:
        existing_path = run_dir / "rows.json"
        if existing_path.exists():
            rows = json.loads(existing_path.read_text())
            done_ids = {r["id"] for r in rows}
            print(f"Resuming: {len(done_ids)} examples already in rows.json")

    for i, (mini_row, q) in enumerate(sampled):
        if q.question_id in done_ids:
            print(f"[{i + 1}/{len(sampled)}] {q.question_id} — already done, skipping")
            continue
        tpl = (q.problem or {}).get("template")
        ex_id = mini_row["example_id"]
        # The prompt we feed pyrlm: with or without env_tips.
        mini_prompt = mini_row["prompt"][-1]["content"]
        prompt_to_send = mini_prompt if args.keep_env_tips else strip_env_tips(mini_prompt)

        print(f"\n[{i + 1}/{len(sampled)}] mini_id={ex_id} qid={q.question_id} "
              f"tpl={tpl} prompt_len={len(prompt_to_send)} "
              f"(env_tips={'on' if args.keep_env_tips else 'off'})")

        trace_path: Path | None = None
        if args.save_traces:
            trace_path = run_dir / "traces" / f"{q.question_id}.json"

        output, tokens, elapsed, err, compaction_fires, trace_summary = _run_pyrlm(
            prompt_to_send,
            provider=args.provider,
            model=args.model,
            sub_model=args.sub_model,
            api_version=args.api_version,
            project_id=args.project_id,
            location=args.location,
            max_steps=args.max_steps,
            max_subcalls=args.max_subcalls,
            max_tokens=args.max_tokens,
            subcall_max_tokens=args.subcall_max_tokens,
            max_concurrent_subcalls=args.max_concurrent_subcalls,
            recursive_subcalls=args.recursive_subcalls,
            recursion_impl=args.recursion_impl,
            compaction=False,
            compaction_threshold_tokens=0,
            trace_path=trace_path,
        )

        score = False
        verify_err: str | None = None
        try:
            score = bool(longcot.verify(q, output, options=verify_options))
        except Exception as ve:
            verify_err = repr(ve)

        row = {
            "mini_example_id": ex_id,
            "id": q.question_id,
            "domain": q.domain,
            "difficulty": q.difficulty,
            "template": tpl,
            "author_reward": mini_row.get("reward"),
            "score": 1.0 if score else 0.0,
            "tokens": tokens,
            "elapsed": elapsed,
            "compaction_fires": compaction_fires,
            "error": err,
            "verify_error": verify_err,
            "output": output,
            "trace_summary": trace_summary,
        }
        rows.append(row)
        print(
            f"  score={int(score)} (author={mini_row.get('reward')}) "
            f"tokens={tokens} elapsed={elapsed:.1f}s "
            f"iters={trace_summary['n_root_calls']} "
            f"hit_max_steps={trace_summary['hit_max_steps']} "
            f"err={err}{' verify_err=' + verify_err if verify_err else ''}"
        )

        write_json(run_dir / "rows.json", rows)

        if args.inter_example_sleep > 0 and i < len(sampled) - 1:
            time.sleep(args.inter_example_sleep)

    summary = aggregate(rows)
    by_template = aggregate_by_template(rows)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "summary_by_template.json", by_template)

    print("\n=== SUMMARY ===")
    print(f"  acc={summary.get('avg_score', 0):.3f} "
          f"({summary.get('score_total', 0)}/{summary.get('examples', 0)})  "
          f"tokens={summary.get('avg_tokens', 0):.0f}  "
          f"elapsed={summary.get('avg_elapsed', 0):.1f}s  "
          f"errors={summary.get('errors', 0)}")
    print("\nBy template:")
    for tpl, s in by_template.items():
        print(f"  {tpl}: {s.get('score_total', 0)}/{s.get('examples', 0)} "
              f"(acc={s.get('avg_score', 0):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
