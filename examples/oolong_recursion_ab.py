"""Old-vs-new recursion A/B on OOLONG (long-context) — pyrlm-runtime.

Measures whether the new child-RLM recursion (``recursion_impl="child"``)
matches or beats the legacy fork (``recursion_impl="fork"``) on a workload that
*actually exercises recursion*: OOLONG-synth at long context (M/L/XL buckets),
where the root delegates chunks to sub-LLM calls. LongCoT-logic barely fires
recursion (~0.2 recursive subcalls/example), so it is used only as a no-regression
guard elsewhere — not here.

Each sampled example is run TWICE back-to-back (arm ``fork`` then arm ``child``)
with IDENTICAL model / policy / seed and a FRESH cache dir per example (cache OFF
across examples — publishable numbers need fresh LLM calls). The only variable is
``recursion_impl``. We record accuracy, tokens, elapsed, AND recursion diagnostics
(``n_recursive_subcalls``, ``max_depth``) so we can confirm recursion actually
fired — if it did not, the A/B is inconclusive by construction.

Decision rule (pre-committed, see docs/pyrlm-vs-rlm-bench / refactor plan):
  ship the child path if  acc(child) >= acc(fork) - 0.01  (non-inferiority),
  the anti-leak test passes, and the LongCoT guard does not regress. The leak fix
  + dead-code removal justify shipping at parity; acc(child) - acc(fork) >= 0.01
  is a headline win.

Usage:

    # Smoke (FakeAdapter, no network, no dataset) — verifies the harness runs
    # both arms and collects recursion diagnostics:
    uv run python examples/oolong_recursion_ab.py --smoke

    # Real A/B (Azure), biased to long context:
    AZURE_OPENAI_API_KEY=... OPENAI_ENDPOINT=... \
    uv run python examples/oolong_recursion_ab.py \
        --model gpt-5.1 --max-examples 30 \
        --min-context-len 8192 --max-context-len 1000000 \
        --output-dir examples/exports/recursion_ab/run01
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any

# Reuse the existing OOLONG harness's sampling + scoring so numbers stay comparable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oolong_rlm_vs_baseline import (  # noqa: E402
    context_bucket,
    now_tag,
    safe_model_name,
    score_output,
    select_rows,
)

from pyrlm_runtime import Context, Policy, RLM  # noqa: E402

ARMS = ("fork", "child")

# OOLONG-synth (extraction/counting) lets the model answer by scanning P in Python,
# so recursion almost never fires naturally — even at 4M-char contexts (measured).
# To get a fork-vs-child signal we must FORCE delegation: this orchestrator prompt
# makes the root answer via sub-LLM calls over chunks (→ recursive subcalls). Both
# arms get the IDENTICAL prompt, so the only variable remains recursion_impl.
ORCHESTRATOR_PROMPT_SUFFIX = """

<orchestrator_mode>
For THIS task you must act as an ORCHESTRATOR, not solve it yourself. The context
in P is too long to read directly. Do NOT scan P with peek/tail/regex/string ops to
find the answer. Instead:

1. Split the context into chunks and delegate to sub-LLMs, e.g.:
   ```python
   answers = ask_chunks(QUESTION_TEXT, ctx, chunk_size=8000)
   ```
   (ask_chunks runs one sub-LLM per chunk in parallel and returns their answers.)
2. Aggregate the sub-LLM answers in Python to produce the final result.
3. Then finalize with FINAL_VAR.

Always gather evidence through sub-LLM delegation before answering.
</orchestrator_mode>
"""


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def recursion_diag(trace: Any) -> dict[str, Any]:
    """Flat counters proving (or disproving) that recursion fired."""
    if trace is None or not getattr(trace, "steps", None):
        return {"total_steps": 0, "n_recursive_subcalls": 0, "max_depth": 0}
    kinds = Counter(s.kind for s in trace.steps)
    return {
        "total_steps": len(trace.steps),
        "n_root_calls": kinds.get("root_call", 0),
        "n_subcalls": kinds.get("subcall", 0),
        "n_recursive_subcalls": kinds.get("recursive_subcall", 0),
        "n_sub_root_calls": kinds.get("sub_root_call", 0),
        "n_sub_repl_exec": kinds.get("sub_repl_exec", 0),
        "n_sub_subcalls": kinds.get("sub_subcall", 0),
        "max_depth": max((s.depth or 0) for s in trace.steps),
    }


def build_rlm(
    root: Any,
    sub: Any,
    *,
    recursion_impl: str,
    cache_dir: Path,
    max_steps: int,
    max_subcalls: int,
    max_tokens: int,
    subcall_max_tokens: int,
    max_recursion_depth: int,
    system_prompt: str | None = None,
) -> RLM:
    kwargs: dict[str, Any] = {}
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    return RLM(
        adapter=root,
        subcall_adapter=sub,
        policy=Policy(
            max_steps=max_steps,
            max_subcalls=max_subcalls,
            max_total_tokens=12_000_000,
        ),
        cache_dir=cache_dir,
        max_tokens=max_tokens,
        subcall_max_tokens=subcall_max_tokens,
        require_repl_before_final=True,
        **kwargs,
        parallel_subcalls=True,
        max_concurrent_subcalls=20,
        conversation_history=True,
        recursive_subcalls=True,
        recursion_impl=recursion_impl,
        max_recursion_depth=max_recursion_depth,
    )


def run_arm(rlm: RLM, question: str, context_text: str) -> dict[str, Any]:
    start = time.time()
    trace = None
    try:
        output, trace = rlm.run(question, Context.from_text(context_text))
        tokens = sum((s.usage.total_tokens if s.usage else 0) for s in trace.steps)
        return {
            "output": output or "",
            "tokens": tokens,
            "elapsed": time.time() - start,
            "error": None,
            "diag": recursion_diag(trace),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "output": "",
            "tokens": 0,
            "elapsed": time.time() - start,
            "error": repr(exc),
            "diag": recursion_diag(trace),
        }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "score": 0.0, "tokens": 0, "elapsed": 0.0, "errors": 0, "rec": 0}
    )
    for r in rows:
        a = by_arm[r["arm"]]
        a["n"] += 1
        a["score"] += r["score"]
        a["tokens"] += r["tokens"]
        a["elapsed"] += r["elapsed"]
        a["errors"] += int(bool(r["error"]))
        a["rec"] += r["diag"].get("n_recursive_subcalls", 0)
    out: dict[str, Any] = {}
    for arm, a in by_arm.items():
        n = max(1, a["n"])
        out[arm] = {
            "examples": a["n"],
            "avg_score": a["score"] / n,
            "avg_tokens": a["tokens"] / n,
            "avg_elapsed": a["elapsed"] / n,
            "errors": a["errors"],
            "total_recursive_subcalls": a["rec"],
        }
    return out


def aggregate_by_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[r["arm"]][r["bucket"]].append(r)
    return {
        arm: {bucket: aggregate(rs)[arm] for bucket, rs in buckets.items()}
        for arm, buckets in grouped.items()
    }


# ---------------------------------------------------------------------------
# Smoke mode (FakeAdapter, no network / no dataset)
# ---------------------------------------------------------------------------

def _smoke() -> int:
    from pyrlm_runtime.adapters.fake import FakeAdapter, FakeRule
    from pyrlm_runtime.prompts import (
        BASE_SYSTEM_PROMPT,
        RECURSIVE_SUBCALL_SYSTEM_PROMPT,
        SUBCALL_SYSTEM_PROMPT,
    )

    ROOT, CHILD, LEAF = BASE_SYSTEM_PROMPT[:50], RECURSIVE_SUBCALL_SYSTEM_PROMPT[:50], SUBCALL_SYSTEM_PROMPT[:50]
    INIT, ITER = "have not interacted", "[REPL Result]"

    def is_root(p: str) -> bool:
        return ROOT in p and CHILD not in p and LEAF not in p

    def rules() -> list[FakeRule]:
        return [
            FakeRule(lambda p: is_root(p) and INIT in p and ITER not in p,
                     "res = ask('subq', P)\nanswer = res"),
            FakeRule(lambda p: is_root(p) and ITER in p, "FINAL_VAR: answer"),
            # llm_query is supported by both arms, so the smoke exercises end-to-end
            # plumbing on each. (Only the child arm also has llm_batch — see the
            # differential tests in tests/test_recursive_subcall.py.)
            FakeRule(lambda p: CHILD in p and INIT in p and ITER not in p,
                     "answer = llm_query('leaf-a')"),
            FakeRule(lambda p: CHILD in p and ITER in p, "FINAL_VAR: answer"),
            FakeRule(lambda p: LEAF in p, "SMOKE_ANSWER"),
        ]

    context_text = "smoke context. " * 50
    question = "What is the smoke answer?"
    print("SMOKE: running both arms with FakeAdapter (no network)\n")
    ok = True
    for arm in ARMS:
        import tempfile

        rlm = build_rlm(
            FakeAdapter(rules=rules()), FakeAdapter(rules=rules()),
            recursion_impl=arm, cache_dir=Path(tempfile.mkdtemp()) / "c",
            max_steps=10, max_subcalls=20, max_tokens=512, subcall_max_tokens=256,
            max_recursion_depth=2,
        )
        res = run_arm(rlm, question, context_text)
        rec = res["diag"]["n_recursive_subcalls"]
        print(f"  arm={arm:5s} output={res['output']!r} err={res['error']} "
              f"rec_subcalls={rec} max_depth={res['diag']['max_depth']} "
              f"steps={res['diag']['total_steps']}")
        if res["error"] is not None:
            ok = False
        if res["output"] != "SMOKE_ANSWER":
            ok = False
        if rec < 1:
            print(f"    WARN: arm {arm} did not fire a recursive subcall")
            ok = False
    print("\nSMOKE", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Run a no-network FakeAdapter smoke test of both arms.")
    parser.add_argument("--model", default="gpt-5.1", help="Azure root deployment.")
    parser.add_argument("--sub-model", default=None,
                        help="Azure subcall deployment (default: same as --model).")
    parser.add_argument("--api-version", default=None)
    parser.add_argument("--max-examples", type=int, default=30)
    parser.add_argument("--sample-strategy", choices=["head", "random", "stratified"],
                        default="stratified")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-context-len", type=int, default=8192,
                        help="Bias toward long context where recursion fires (default 8192).")
    parser.add_argument("--max-context-len", type=int, default=1_000_000)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--max-subcalls", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--subcall-max-tokens", type=int, default=1024)
    parser.add_argument("--max-recursion-depth", type=int, default=2)
    parser.add_argument(
        "--force-delegation", action="store_true",
        help="Prepend an orchestrator prompt that forces sub-LLM delegation so "
             "recursion actually fires (OOLONG never delegates naturally). Both arms "
             "get the identical prompt; the only variable stays recursion_impl.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Resume: skip example ids already present in rows.json.")
    args = parser.parse_args()

    if args.smoke:
        return _smoke()

    from datasets import load_dataset

    from _azure_check import check_azure_connection
    from pyrlm_runtime.adapters import AzureOpenAIAdapter

    sub_model = args.sub_model or args.model
    check_azure_connection(args.model, api_version=args.api_version)

    system_prompt = None
    if args.force_delegation:
        from pyrlm_runtime.prompts import BASE_SYSTEM_PROMPT

        system_prompt = BASE_SYSTEM_PROMPT + ORCHESTRATOR_PROMPT_SUFFIX

    data = load_dataset("oolongbench/oolong-synth")["test"]
    context_col = "context_window_text_with_labels"
    data = data.filter(lambda x: args.min_context_len < x["context_len"] <= args.max_context_len)
    data = select_rows(data, strategy=args.sample_strategy, max_examples=args.max_examples,
                       seed=args.seed)
    print(f"Loaded {len(data)} OOLONG-synth examples "
          f"(ctx {args.min_context_len}-{args.max_context_len})")

    run_dir = Path(
        args.output_dir
        or f"examples/exports/recursion_ab/run_{now_tag()}_{safe_model_name(args.model)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", {
        "model": args.model, "sub_model": sub_model,
        "max_examples": args.max_examples, "sample_strategy": args.sample_strategy,
        "seed": args.seed, "min_context_len": args.min_context_len,
        "max_context_len": args.max_context_len, "max_steps": args.max_steps,
        "max_subcalls": args.max_subcalls, "max_tokens": args.max_tokens,
        "subcall_max_tokens": args.subcall_max_tokens,
        "max_recursion_depth": args.max_recursion_depth,
        "force_delegation": args.force_delegation,
        "arms": list(ARMS), "cache": "off (fresh dir per example)",
    })

    adapter_kwargs: dict[str, Any] = {"model": args.model, "timeout": 900.0}
    if args.api_version:
        adapter_kwargs["api_version"] = args.api_version
    root = AzureOpenAIAdapter(**adapter_kwargs)
    sub = AzureOpenAIAdapter(**{**adapter_kwargs, "model": sub_model})

    rows: list[dict[str, Any]] = []
    done: set[str] = set()
    rows_path = run_dir / "rows.json"
    if args.skip_existing and rows_path.exists():
        rows = json.loads(rows_path.read_text())
        done = {f"{r['id']}:{r['arm']}" for r in rows}
        print(f"Resuming: {len(done)} (id,arm) pairs already done")

    cache_root = run_dir / "_cache"
    for i, dp in enumerate(data):
        context_text = dp[context_col]
        question = dp["question"]
        ex_id = str(dp.get("id", i))
        ctx_len = int(dp["context_len"])
        bucket = context_bucket(ctx_len)
        print(f"\n[{i + 1}/{len(data)}] id={ex_id} ctx_len={ctx_len} bucket={bucket}")

        for arm in ARMS:
            if f"{ex_id}:{arm}" in done:
                print(f"  arm={arm:5s} — already done, skip")
                continue
            rlm = build_rlm(
                root, sub, recursion_impl=arm,
                cache_dir=cache_root / arm / ex_id,  # fresh per (example, arm)
                max_steps=args.max_steps, max_subcalls=args.max_subcalls,
                max_tokens=args.max_tokens, subcall_max_tokens=args.subcall_max_tokens,
                max_recursion_depth=args.max_recursion_depth,
                system_prompt=system_prompt,
            )
            res = run_arm(rlm, question, context_text)
            score = float(score_output("synth", dict(dp), res["output"], args.model)["score"])
            row = {
                "id": ex_id, "arm": arm, "bucket": bucket, "context_len": ctx_len,
                "score": score, "tokens": res["tokens"], "elapsed": res["elapsed"],
                "error": res["error"], "diag": res["diag"], "output": res["output"][:500],
            }
            rows.append(row)
            write_json(rows_path, rows)
            print(f"  arm={arm:5s} score={score:.0f} tokens={res['tokens']} "
                  f"elapsed={res['elapsed']:.1f}s rec={res['diag']['n_recursive_subcalls']} "
                  f"max_depth={res['diag']['max_depth']} err={res['error']}")

    summary = aggregate(rows)
    by_bucket = aggregate_by_bucket(rows)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "summary_by_bucket.json", by_bucket)

    print("\n=== SUMMARY (A/B) ===")
    for arm in ARMS:
        if arm not in summary:
            continue
        s = summary[arm]
        print(f"  {arm:5s}: acc={s['avg_score']:.3f}  tokens={s['avg_tokens']:.0f}  "
              f"elapsed={s['avg_elapsed']:.1f}s  errors={s['errors']}  "
              f"rec_subcalls_total={s['total_recursive_subcalls']}  n={s['examples']}")
    if "fork" in summary and "child" in summary:
        delta = summary["child"]["avg_score"] - summary["fork"]["avg_score"]
        print(f"\n  Δacc(child - fork) = {delta:+.3f}")
        rec_total = summary["child"]["total_recursive_subcalls"] + summary["fork"]["total_recursive_subcalls"]
        if rec_total == 0:
            print("  WARNING: recursion never fired — A/B is INCONCLUSIVE for this sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
