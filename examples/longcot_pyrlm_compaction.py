"""LongCoT bench — pyrlm-runtime, compaction ON vs OFF.

Experiment 1 of ``docs/longcot-bench/EXPERIMENTS.md``: measure whether
history compaction changes accuracy on LongCoT (short input, very long
reasoning), the workload where pyrlm's message stream actually grows
past realistic thresholds — unlike oolong-synth, where ``Context`` keeps
the document outside the message stream and compaction never fired.

The harness loads the upstream LongCoT package (which bundles question
data and the deterministic verifier) and runs **two arms** of
pyrlm-runtime back-to-back on each example so per-example diffs are
clean:

    Arm A: compaction = False
    Arm B: compaction = True with compaction_threshold_tokens = 40000

Both arms share every other knob (model, sub-model, max_steps, parallel
subcalls, recursive subcalls).  The threshold for arm B is **calibrated
to LongCoT** — see the EXPERIMENTS.md "compaction threshold rationale"
section before changing it.

Setup
-----

1. Clone the LongCoT repo and install it into pyrlm's venv:

       git clone https://github.com/LongHorizonReasoning/longcot ../longcot
       uv pip install -e ../longcot

   (rdkit / chess / sympy / pyyaml get pulled in.  We filter to
   math+cs+logic by default so the chemistry/chess verifiers don't have
   to run, but installing them is the cleanest path.)

2. Export Azure credentials, same as the other oolong benches:

       export AZURE_OPENAI_API_KEY=...
       export OPENAI_ENDPOINT=https://<resource>.openai.azure.com

3. Run:

       uv run python examples/longcot_pyrlm_compaction.py \\
           --model gpt-5.5 --sub-model gpt-5.4-mini \\
           --domains math,cs,logic --difficulties medium,hard \\
           --max-examples 60 --sample-strategy stratified --seed 42 \\
           --output-dir examples/exports/longcot/exp1_compaction_on_off

Math / chemistry verifier fallbacks (which call Gemini) are
**disabled by default**: we want deterministic scoring only and no
extra API dependency.  Pass ``--enable-verifier-fallback`` with
``GEMINI_API_KEY`` exported if you want lenient parsing.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Make ``pyrlm_runtime`` importable from a clean checkout (mirrors the
# pattern used by examples/oolong_pyrlm_vs_rlm.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------
def stratified_sample(
    questions: list[Any],
    *,
    max_examples: int,
    seed: int,
) -> list[Any]:
    """Sample ``max_examples`` proportionally to (domain × difficulty) buckets."""
    if max_examples >= len(questions):
        return list(questions)

    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for q in questions:
        buckets[(q.domain, q.difficulty)].append(q)

    # Floor + remainder so we don't drop tiny buckets to zero.
    total = len(questions)
    target: dict[tuple[str, str], int] = {}
    remainders: list[tuple[float, tuple[str, str]]] = []
    used = 0
    for key, qs in buckets.items():
        raw = max_examples * len(qs) / total
        floor = max(1, int(raw)) if qs else 0
        target[key] = floor
        used += floor
        remainders.append((raw - floor, key))
    # Distribute / trim until we hit max_examples exactly.
    remainders.sort(reverse=True)
    i = 0
    while used < max_examples and i < len(remainders):
        key = remainders[i][1]
        if target[key] < len(buckets[key]):
            target[key] += 1
            used += 1
        i += 1
    # If we over-allocated due to the floor=1 guard, trim the lowest
    # remainders first.
    if used > max_examples:
        for _, key in sorted(remainders):
            while used > max_examples and target[key] > 1:
                target[key] -= 1
                used -= 1
            if used <= max_examples:
                break

    out: list[Any] = []
    for key, n in target.items():
        if n <= 0:
            continue
        picked = rng.sample(buckets[key], min(n, len(buckets[key])))
        out.extend(picked)
    # Deterministic order across runs.
    out.sort(key=lambda q: (q.domain, q.difficulty, q.question_id))
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    sub = [r for r in rows if r["arm"] == arm]
    if not sub:
        return {"examples": 0}
    n = len(sub)

    def _avg_ts(field: str) -> float:
        vals = [r.get("trace_summary", {}).get(field, 0) or 0 for r in sub]
        return sum(vals) / n if n else 0.0

    return {
        "examples": n,
        "avg_score": sum(r["score"] for r in sub) / n,
        "avg_tokens": sum(r["tokens"] for r in sub) / n,
        "avg_elapsed": sum(r["elapsed"] for r in sub) / n,
        "errors": sum(1 for r in sub if r.get("error")),
        "compaction_fires_total": sum(r.get("compaction_fires", 0) for r in sub),
        "examples_with_compaction": sum(
            1 for r in sub if r.get("compaction_fires", 0) > 0
        ),
        # Trajectory-shape diagnostics (averaged across examples).
        "avg_root_calls": _avg_ts("n_root_calls"),
        "avg_repl_execs": _avg_ts("n_repl_execs"),
        "avg_subcalls": _avg_ts("n_subcalls"),
        "avg_recursive_subcalls": _avg_ts("n_recursive_subcalls"),
        "avg_max_depth": _avg_ts("max_depth"),
        "avg_prompt_tokens": _avg_ts("prompt_tokens"),
        "avg_completion_tokens": _avg_ts("completion_tokens"),
        "avg_root_tokens": _avg_ts("root_tokens"),
        "avg_subcall_tokens": _avg_ts("subcall_tokens"),
        # Cap-hit rates — the load-bearing signal for the hypothesis:
        # compaction should reduce abandonment on long trajectories.
        "examples_hit_max_steps": sum(
            1 for r in sub
            if r.get("trace_summary", {}).get("hit_max_steps")
        ),
        "examples_hit_max_subcalls": sum(
            1 for r in sub
            if r.get("trace_summary", {}).get("hit_max_subcalls")
        ),
    }


def aggregate_by_cell(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["arm"], r["domain"], r["difficulty"])].append(r)
    for (arm, dom, diff), sub in grouped.items():
        n = len(sub)
        out[arm][f"{dom}/{diff}"] = {
            "examples": n,
            "avg_score": sum(r["score"] for r in sub) / n if n else 0.0,
            "avg_tokens": sum(r["tokens"] for r in sub) / n if n else 0.0,
            "avg_elapsed": sum(r["elapsed"] for r in sub) / n if n else 0.0,
            "compaction_fires_total": sum(r.get("compaction_fires", 0) for r in sub),
        }
    # Convert defaultdict to plain dict for json serialisation.
    return {k: dict(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# pyrlm runner
# ---------------------------------------------------------------------------
def _count_compaction_fires(trace: Any) -> int:
    """Count TraceStep entries of kind=='compaction'."""
    if trace is None:
        return 0
    try:
        return sum(1 for s in trace.steps if getattr(s, "kind", None) == "compaction")
    except Exception:
        return 0


def _summarize_trace(
    trace: Any, *, max_steps_limit: int, max_subcalls_limit: int,
) -> dict[str, Any]:
    """Per-example diagnostic counters extracted from the trace.

    These let us answer the post-hoc questions the hypothesis cares
    about:

    - Did arm B run more iterations than arm A? (compaction → longer
      trajectories allowed)
    - Did arm B hit ``max_steps`` less often? (compaction → fewer
      abandonments)
    - Where do tokens go (prompt vs completion, root vs subcall)?
    - When did compaction fire relative to the trajectory?

    Kept lightweight (a flat dict) — full trace is on disk for deep
    dives.
    """
    out: dict[str, Any] = {
        "n_steps_total": 0,
        "n_root_calls": 0,
        "n_repl_execs": 0,
        "n_subcalls": 0,
        "n_recursive_subcalls": 0,
        "n_compactions": 0,
        "max_depth": 0,
        "first_compaction_step": None,
        "last_step_kind": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "root_tokens": 0,
        "subcall_tokens": 0,
        # Outcome heuristic: best-effort classification — see notes below.
        "hit_max_steps": False,
        "hit_max_subcalls": False,
    }
    if trace is None:
        return out
    try:
        steps = list(trace.steps)
    except Exception:
        return out

    out["n_steps_total"] = len(steps)
    for s in steps:
        kind = getattr(s, "kind", None)
        depth = getattr(s, "depth", 0) or 0
        out["max_depth"] = max(out["max_depth"], depth)
        if kind == "root_call":
            out["n_root_calls"] += 1
        elif kind == "repl_exec":
            out["n_repl_execs"] += 1
        elif kind in ("subcall", "sub_root_call"):
            out["n_subcalls"] += 1
        elif kind == "recursive_subcall":
            out["n_recursive_subcalls"] += 1
        elif kind == "compaction":
            out["n_compactions"] += 1
            if out["first_compaction_step"] is None:
                out["first_compaction_step"] = getattr(s, "step_id", None)
        u = getattr(s, "usage", None)
        if u is not None:
            pt = getattr(u, "prompt_tokens", 0) or 0
            ct = getattr(u, "completion_tokens", 0) or 0
            tt = getattr(u, "total_tokens", 0) or (pt + ct)
            out["prompt_tokens"] += pt
            out["completion_tokens"] += ct
            out["total_tokens"] += tt
            if depth == 0 and kind in ("root_call", "baseline_call"):
                out["root_tokens"] += tt
            elif kind in ("subcall", "sub_root_call", "recursive_subcall",
                           "sub_repl_exec", "sub_subcall"):
                out["subcall_tokens"] += tt
    if steps:
        out["last_step_kind"] = getattr(steps[-1], "kind", None)
    # Heuristic: pyrlm caps root_calls at max_steps and subcalls at
    # max_subcalls.  Hitting the cap doesn't *prove* abandonment (the
    # model may have produced a FINAL on the last allowed iter) but it's
    # a strong correlate the analysis cares about.
    out["hit_max_steps"] = out["n_root_calls"] >= max_steps_limit
    out["hit_max_subcalls"] = (
        out["n_subcalls"] + out["n_recursive_subcalls"]
    ) >= max_subcalls_limit
    return out


def _run_pyrlm(
    prompt: str,
    *,
    provider: str,
    model: str,
    sub_model: str,
    api_version: str | None,
    project_id: str | None,
    location: str | None,
    max_steps: int,
    max_subcalls: int,
    max_tokens: int,
    subcall_max_tokens: int,
    max_concurrent_subcalls: int,
    recursive_subcalls: bool,
    compaction: bool,
    compaction_threshold_tokens: int,
    trace_path: Path | None,
    recursion_impl: str = "child",
) -> tuple[str, int, float, str | None, int, dict[str, Any]]:
    """Run a single pyrlm-runtime invocation.

    Returns ``(output, tokens, elapsed, error_repr, compaction_fires,
    trace_summary)``.

    ``trace_summary`` is a flat dict of diagnostic counters from
    ``_summarize_trace`` — kept in ``rows.json`` for analysis without
    parsing the full trace file.

    Note: LongCoT prompts are short (≤6K chars) so we pass the prompt
    directly as the user query and use an empty Context.  No document
    abstraction needed.
    """
    from pyrlm_runtime import Context, Policy, RLM

    if provider == "vertex":
        from pyrlm_runtime.adapters.vertex_ai import VertexAIAdapter

        if not project_id or not location:
            raise ValueError(
                "provider=vertex requires --project-id and --location "
                "(or GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION env vars)"
            )
        root = VertexAIAdapter(project_id=project_id, location=location, model=model)
        sub = VertexAIAdapter(project_id=project_id, location=location, model=sub_model)
    elif provider == "azure":
        from pyrlm_runtime.adapters import AzureOpenAIAdapter

        adapter_kwargs: dict[str, Any] = {"model": model, "timeout": 900.0}
        if api_version:
            adapter_kwargs["api_version"] = api_version
        root = AzureOpenAIAdapter(**adapter_kwargs)
        sub_kwargs = dict(adapter_kwargs)
        sub_kwargs["model"] = sub_model
        sub = AzureOpenAIAdapter(**sub_kwargs)
    else:
        raise ValueError(f"unknown provider: {provider!r} (expected 'azure' or 'vertex')")

    rlm = RLM(
        adapter=root,
        subcall_adapter=sub,
        policy=Policy(
            max_steps=max_steps,
            max_subcalls=max_subcalls,
            max_total_tokens=12_000_000,
        ),
        max_tokens=max_tokens,
        subcall_max_tokens=subcall_max_tokens,
        parallel_subcalls=True,
        max_concurrent_subcalls=max_concurrent_subcalls,
        conversation_history=True,
        recursive_subcalls=recursive_subcalls,
        recursion_impl=recursion_impl,
        max_recursion_depth=2,
        compaction=compaction,
        # tiktoken-accurate counting via compaction_model_name.
        # _compaction_threshold_effective auto-resolves context limit
        # from the model name, so pct=0.85 now works correctly.
        compaction_model_name=model if compaction else "",
        compaction_threshold_tokens=compaction_threshold_tokens if compaction else 0,
        compaction_threshold_pct=0.0,
        compaction_model_context_limit=0,
    )

    # Per-iter progress: print one line per completed step so long Gemini
    # "thinking" calls don't look like hangs. Cheap, no semantic impact.
    class _ProgressListener:
        def handle(self, event: Any) -> None:
            if event.kind != "step_completed" or event.step is None:
                return
            s = event.step
            if s.kind not in ("root_call", "subcall", "compaction", "recursive_subcall"):
                return
            tok = s.usage.total_tokens if s.usage else 0
            marker = {"compaction": "[COMPACT]"}.get(s.kind, f"[{s.kind}]")
            print(
                f"    {marker} step{s.step_id} depth={s.depth} "
                f"tok={tok} elapsed={s.elapsed:.1f}s",
                flush=True,
            )

    rlm.event_listener = _ProgressListener()

    # LongCoT prompts are short (≤6K chars) and self-contained — the puzzle is
    # IN the prompt text, not a separate document. Originally we passed
    # Context.from_text("") and the prompt as the query; this caused the model
    # to look for puzzle data in the (empty) P/ctx variables that pyrlm's system
    # prompt tells it exist, and to give up on tasks that needed parsing
    # ("solution = <unavailable: grid not accessible in REPL context>" on
    # Dungeon_hard_12, smoke v4 2026-05-27). We now pass the puzzle text as the
    # context so P/ctx are populated, with a meta-instruction as the query.
    context = Context.from_text(prompt)
    start = time.time()
    trace = None
    try:
        output, trace = rlm.run(
            # NOTE: we deliberately avoid telling the model to "ignore"
            # instructions in P — Azure's content filter flags that as
            # prompt-injection and returns HTTP 400 (smoke v5/v6
            # 2026-05-27). pyrlm's system prompt already directs the
            # model to use the REPL, which is enough in practice.
            "Solve the puzzle described in P (the REPL variable holds the "
            "full puzzle text). Use the REPL to compute the solution. "
            "Emit the final answer as a Python statement starting with "
            "'solution = ...' matching the format the puzzle specifies.",
            context,
        )
        tokens = sum(
            (s.usage.total_tokens if s.usage else 0) for s in trace.steps
        )
        if trace_path is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(trace.to_json())
        summary = _summarize_trace(
            trace, max_steps_limit=max_steps, max_subcalls_limit=max_subcalls,
        )
        return (
            output or "",
            tokens,
            time.time() - start,
            None,
            _count_compaction_fires(trace),
            summary,
        )
    except Exception as exc:
        summary = _summarize_trace(
            trace, max_steps_limit=max_steps, max_subcalls_limit=max_subcalls,
        )
        return ("", 0, time.time() - start, repr(exc),
                _count_compaction_fires(trace), summary)


# ---------------------------------------------------------------------------
# Verifier wrapper
# ---------------------------------------------------------------------------
def _build_verify_options(enable_fallback: bool):
    """Construct ``longcot.VerifyOptions`` honouring our defaults."""
    import longcot  # noqa: F401  — ensure available

    return longcot.VerifyOptions(
        math=longcot.MathVerifyOptions(enable_fallback=enable_fallback),
        chemistry=longcot.ChemistryVerifyOptions(enable_fallback=enable_fallback),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Provider
    parser.add_argument(
        "--provider",
        choices=["azure", "vertex"],
        default="azure",
        help="LLM provider. 'azure' uses AzureOpenAIAdapter, 'vertex' uses VertexAIAdapter (ADC).",
    )
    # Models (defaults swap based on --provider; resolved after parse_args)
    parser.add_argument("--model", default=None)
    parser.add_argument("--sub-model", default=None)
    parser.add_argument(
        "--api-version",
        default=os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21",
        help="Azure-only. Ignored for --provider=vertex.",
    )
    # Vertex-only
    parser.add_argument(
        "--project-id",
        default=os.getenv("GOOGLE_CLOUD_PROJECT"),
        help="Vertex-only. GCP project id. Defaults to $GOOGLE_CLOUD_PROJECT.",
    )
    parser.add_argument(
        "--location",
        default=os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1",
        help="Vertex-only. GCP region. Defaults to $GOOGLE_CLOUD_LOCATION or us-central1.",
    )
    # Dataset filtering
    parser.add_argument(
        "--domains",
        default="math,cs,logic",
        help=("Comma-separated subset of {math,cs,logic,chemistry,chess}. "
              "Default excludes chemistry+chess (domain-specific quirks)."),
    )
    parser.add_argument(
        "--difficulties",
        default="medium,hard",
        help=("Comma-separated subset of {easy,medium,hard}. Default "
              "skips easy — trajectories too short to stress compaction."),
    )
    # Sampling
    parser.add_argument("--max-examples", type=int, default=60)
    parser.add_argument(
        "--sample-strategy",
        choices=["stratified", "head"],
        default="stratified",
    )
    parser.add_argument("--seed", type=int, default=42)
    # pyrlm knobs (shared across arms)
    parser.add_argument(
        "--max-steps", type=int, default=50,
        help="LongCoT needs more iterations than oolong; default 50.",
    )
    parser.add_argument("--max-subcalls", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--subcall-max-tokens", type=int, default=2048,
        help="LongCoT subcalls reason more per call than oolong; default 2048.",
    )
    parser.add_argument("--max-concurrent-subcalls", type=int, default=20)
    parser.add_argument(
        "--no-recursive", dest="recursive_subcalls",
        action="store_false", default=True,
        help="Disable recursive subcalls (default ON, paper-tracking).",
    )
    # Compaction threshold for arm B — the variable we actually test
    parser.add_argument(
        "--compaction-threshold-tokens", type=int, default=40000,
        help=("Tiktoken-accurate threshold at which arm B compacts "
              "(default 40000).  Calibrated from smoke-test data: "
              "MFMC_hard_7 (6 iters) built a history of ~46K tiktoken "
              "tokens, so 40K fires at iter 7-8 on hard examples while "
              "leaving shorter trajectories (medium, fast convergence) "
              "untouched.  NOTE: do NOT use pct=0.85×ctx (=231K) here — "
              "that is a safety-net-for-overflow, not a quality threshold. "
              "LongCoT histories rarely exceed 100K even on hard runs."),
    )
    # Verifier behaviour
    parser.add_argument(
        "--enable-verifier-fallback",
        action="store_true",
        help=("Enable LongCoT math/chemistry LLM fallback (uses Gemini, "
              "needs GEMINI_API_KEY).  Default off for deterministic, "
              "no-extra-dependency scoring."),
    )
    # Arms (lets us re-run only one arm if something broke)
    parser.add_argument(
        "--arms", default="a,b",
        help="Comma-separated subset of {a,b}. a=no-compaction, b=compaction.",
    )
    # Output
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--save-traces", dest="save_traces",
        action="store_true", default=True,
    )
    parser.add_argument(
        "--no-save-traces", dest="save_traces", action="store_false",
    )
    parser.add_argument(
        "--inter-example-sleep", type=float, default=0.0,
        help=("Seconds to sleep between examples (after all arms for the "
              "current example complete). Useful to avoid Azure 429 rate "
              "limits on tight TPM quotas. Default 0 (no pause)."),
    )
    args = parser.parse_args()

    # Resolve per-provider model defaults (only when user didn't pass --model)
    if args.provider == "vertex":
        if args.model is None:
            args.model = "gemini-2.5-pro"
        if args.sub_model is None:
            args.sub_model = "gemini-2.5-flash"
        if not args.project_id:
            print(
                "ERROR: --provider=vertex requires --project-id "
                "(or $GOOGLE_CLOUD_PROJECT)",
                file=sys.stderr,
            )
            return 2
    else:  # azure
        if args.model is None:
            args.model = "gpt-5.5"
        if args.sub_model is None:
            args.sub_model = "gpt-5.4-mini"

    # ----- Import longcot now that argparse is done so errors are clear ----
    try:
        import longcot
    except ImportError as exc:
        print("ERROR: longcot package not importable. Install it from a "
              "clone of LongHorizonReasoning/longcot, e.g.:\n"
              "    uv pip install -e ../longcot", file=sys.stderr)
        print(f"({exc!r})", file=sys.stderr)
        return 2

    # ----- Load questions ---------------------------------------------------
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    diffs = [d.strip() for d in args.difficulties.split(",") if d.strip()]
    all_qs: list[Any] = []
    for d in domains:
        for diff in diffs:
            all_qs.extend(longcot.load_questions(domain=d, difficulty=diff))
    if not all_qs:
        print(f"ERROR: no questions matched domains={domains} diffs={diffs}",
              file=sys.stderr)
        return 2
    print(f"Loaded {len(all_qs)} questions from "
          f"domains={domains} difficulties={diffs}")

    # ----- Sample -----------------------------------------------------------
    if args.sample_strategy == "head":
        sampled = sorted(all_qs, key=lambda q: q.question_id)[: args.max_examples]
    else:
        sampled = stratified_sample(
            all_qs, max_examples=args.max_examples, seed=args.seed,
        )
    print(f"Sampled {len(sampled)} questions")
    # Print bucket distribution so the user sees what they're about to run.
    dist: dict[tuple[str, str], int] = defaultdict(int)
    for q in sampled:
        dist[(q.domain, q.difficulty)] += 1
    for (dom, diff), n in sorted(dist.items()):
        print(f"  {dom}/{diff}: {n}")

    # ----- Verifier options -------------------------------------------------
    verify_options = _build_verify_options(args.enable_verifier_fallback)

    # ----- Output dir -------------------------------------------------------
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Persist config for reproducibility
    write_json(run_dir / "config.json", {
        "provider": args.provider,
        "project_id": args.project_id if args.provider == "vertex" else None,
        "location": args.location if args.provider == "vertex" else None,
        "model": args.model,
        "sub_model": args.sub_model,
        "api_version": args.api_version,
        "domains": domains,
        "difficulties": diffs,
        "max_examples": args.max_examples,
        "sample_strategy": args.sample_strategy,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "max_subcalls": args.max_subcalls,
        "max_tokens": args.max_tokens,
        "subcall_max_tokens": args.subcall_max_tokens,
        "max_concurrent_subcalls": args.max_concurrent_subcalls,
        "recursive_subcalls": args.recursive_subcalls,
        "compaction_threshold_tokens": args.compaction_threshold_tokens,
        "enable_verifier_fallback": args.enable_verifier_fallback,
        "arms": args.arms,
    })

    arms = [a.strip() for a in args.arms.split(",") if a.strip() in ("a", "b")]
    if not arms:
        print("ERROR: --arms must contain 'a' and/or 'b'", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    per_example: list[dict[str, Any]] = []

    for i, q in enumerate(sampled):
        print(f"\n[{i + 1}/{len(sampled)}] id={q.question_id} "
              f"{q.domain}/{q.difficulty} template="
              f"{(q.problem or {}).get('template')}")

        example_row: dict[str, Any] = {
            "id": q.question_id,
            "domain": q.domain,
            "difficulty": q.difficulty,
            "template": (q.problem or {}).get("template"),
            "prompt_len_chars": len(q.prompt),
            "results": {},
        }

        for arm in arms:
            arm_dir_name = "arm_a_no_compaction" if arm == "a" else "arm_b_compaction"
            trace_path: Path | None = None
            if args.save_traces:
                trace_path = run_dir / arm_dir_name / "traces" / f"{q.question_id}.json"

            compaction_on = arm == "b"
            print(f"  → arm {arm} (compaction={'ON' if compaction_on else 'OFF'}) ...",
                  flush=True)
            output, tokens, elapsed, err, compaction_fires, trace_summary = _run_pyrlm(
                q.prompt,
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
                compaction=compaction_on,
                compaction_threshold_tokens=args.compaction_threshold_tokens,
                trace_path=trace_path,
            )

            score = False
            verify_err: str | None = None
            try:
                score = bool(longcot.verify(q, output, options=verify_options))
            except Exception as ve:
                verify_err = repr(ve)

            row = {
                "arm": arm,
                "id": q.question_id,
                "domain": q.domain,
                "difficulty": q.difficulty,
                "template": (q.problem or {}).get("template"),
                "score": 1.0 if score else 0.0,
                "tokens": tokens,
                "elapsed": elapsed,
                "compaction_fires": compaction_fires,
                "error": err,
                "verify_error": verify_err,
                "output": output,
                # Per-example diagnostic counters; see _summarize_trace.
                "trace_summary": trace_summary,
            }
            rows.append(row)
            example_row["results"][arm] = row
            print(
                f"     score={int(score)} tokens={tokens} "
                f"elapsed={elapsed:.1f}s "
                f"iters={trace_summary['n_root_calls']} "
                f"subcalls={trace_summary['n_subcalls']}"
                f"+{trace_summary['n_recursive_subcalls']}rec "
                f"depth={trace_summary['max_depth']} "
                f"compaction_fires={compaction_fires} "
                f"hit_max_steps={trace_summary['hit_max_steps']} "
                f"err={err}{' verify_err=' + verify_err if verify_err else ''}"
            )

        per_example.append(example_row)
        # Persist incrementally so a partial run is still useful.
        write_json(run_dir / "per_example.json", per_example)
        write_json(run_dir / "rows.json", rows)

        # Optional pause so Azure TPM quota recovers before the next example.
        if args.inter_example_sleep > 0 and i < len(sampled) - 1:
            print(f"  ... sleeping {args.inter_example_sleep:.0f}s "
                  f"before next example", flush=True)
            time.sleep(args.inter_example_sleep)

    # ----- Final summary ----------------------------------------------------
    summary = {a: aggregate(rows, a) for a in arms}
    summary_by_cell = aggregate_by_cell(rows)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "summary_by_cell.json", summary_by_cell)

    print("\n=== SUMMARY ===")
    for arm, s in summary.items():
        if not s.get("examples"):
            continue
        label = "arm_a_no_compaction" if arm == "a" else "arm_b_compaction"
        print(
            f"  {label}: acc={s['avg_score']:.3f}  tokens={s['avg_tokens']:.0f}  "
            f"elapsed={s['avg_elapsed']:.1f}s  errors={s['errors']}\n"
            f"    iters_avg={s['avg_root_calls']:.1f}  "
            f"subcalls_avg={s['avg_subcalls']:.1f}+{s['avg_recursive_subcalls']:.1f}rec  "
            f"max_depth_avg={s['avg_max_depth']:.1f}\n"
            f"    hit_max_steps={s['examples_hit_max_steps']}/{s['examples']}  "
            f"hit_max_subcalls={s['examples_hit_max_subcalls']}/{s['examples']}\n"
            f"    compaction_fires_total={s['compaction_fires_total']}  "
            f"examples_with_compaction={s['examples_with_compaction']}/{s['examples']}"
        )
    print(f"\nBy cell: {run_dir / 'summary_by_cell.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
