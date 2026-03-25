#!/usr/bin/env python3
"""
Oolong benchmark: baseline (single-shot) vs RLM (pyrlm-runtime).

Evaluates two strategies on the Oolong long-context benchmark:
  1) baseline  – full context stuffed into a single prompt
  2) rlm       – pyrlm-runtime agentic loop with parallel sub-LLM calls

Uses Azure OpenAI by default. Required env vars:
  AZURE_OPENAI_API_KEY  – API key
  OPENAI_ENDPOINT       – e.g. https://<resource>.openai.azure.com
  AZURE_ACCOUNT_NAME    – alternative to OPENAI_ENDPOINT (used if OPENAI_ENDPOINT is unset)

Usage:
  uv run python examples/oolong_rlm_vs_baseline.py --max-examples 30

Full run:
  AZURE_OPENAI_API_KEY=... OPENAI_ENDPOINT=... \\
  uv run python examples/oolong_rlm_vs_baseline.py \\
    --model gpt-5.1 \\
    --dataset synth \\
    --sample-strategy stratified \\
    --seed 42 \\
    --max-examples 100
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from _azure_check import check_azure_connection
from pyrlm_runtime import Context, Policy, RLM
from pyrlm_runtime.adapters import AzureOpenAIAdapter
from pyrlm_runtime.prompts import BASE_SYSTEM_PROMPT, SUBCALL_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-5.1"
ENGINES = ("baseline", "rlm")

# ---------------------------------------------------------------------------
# Environment tips (appended to system prompt when --env-tips is set)
# ---------------------------------------------------------------------------

OOLONG_ENV_TIPS = """
<env_tips>
Strategy for structured data tasks (dates, labels, user IDs):

1. ALWAYS use Python code for counting and filtering — never delegate counting
   to sub-LLMs. Use collections.Counter, list comprehensions, or regex on P.
   Example:
   ```python
   import re
   from collections import Counter
   matches = re.findall(r"label:\\s*(\\w+)", P)
   counts = Counter(matches)
   print(counts)
   ```

2. For large contexts (>32K chars), split with ctx.chunk() and process each
   chunk with Python string operations — not subcalls.
   IMPORTANT: When COUNTING exact occurrences, use overlap=0 to avoid
   double-counting items that fall in overlap zones. Only use overlap>0
   when you need complete sentences at chunk boundaries (e.g., for semantic
   classification of individual items).
   Example (counting):
   ```python
   chunks = ctx.chunk(20000, overlap=0)
   total = 0
   for start, end, text in chunks:
       total += text.count("True")
   print(f"Total True labels: {total}")
   ```

3. Use llm_batch() ONLY for semantic understanding (e.g., classifying free
   text, summarising), never for counting or exact string matching.

4. Verify your answer with a second pass before finalising.
   Print the count AND a sample of matching items to double-check.

5. For date-range filtering, parse dates in Python using a manual month_map
   (datetime.strptime is BLOCKED in this sandbox). Compare as (year, month, day)
   integer tuples — do not ask a sub-LLM to filter by date.
   ```python
   month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
                "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
   def parse_date(s):  # "Oct 24, 2022"
       mo, day, yr = s.split()[0], s.split()[1].rstrip(","), s.split()[2]
       return (int(yr), month_map[mo], int(day))
   cutoff = (2024, 8, 27)  # tuple comparison works directly
   ```

6. NEVER invent or impute a label when a sub-LLM returns NO_ANSWER,
   unclear output, or an empty result. Do not use heuristics like
   "this looks positive" or "probably negative" to break ties.

7. For binary classification tasks (True/False, positive/negative),
   first isolate the exact subset in Python, then classify each item
   independently. If any item returns NO_ANSWER, run a second pass for
   that SAME item with a stricter prompt that:
   - repeats the exact allowed labels
   - requests one label only
   - asks for a short evidence phrase in a separate verification step
   Only finalize after all items are resolved from evidence.

8. If even the second pass returns NO_ANSWER, do NOT finalize with a guessed
   majority label. Keep investigating, print the unresolved items, and use
   another verification strategy.

9. Avoid "planning-only" turns after you have already identified the relevant
   subset. Once the subset is known, the NEXT step should perform useful work:
   classification, counting, or verification.

10. When comparing "more/less common" across time periods, compare RATES
    (count/total_in_period), NOT absolute counts. Example:
    12 neutral in 35 rows (34%) is LESS common than 6 neutral in 10 rows (60%),
    even though 12 > 6.
    ```python
    before = [(label, date) for label, date in records if date < cutoff]
    after  = [(label, date) for label, date in records if date >= cutoff]
    rate_before = sum(1 for l, _ in before if l == target) / max(len(before), 1)
    rate_after  = sum(1 for l, _ in after  if l == target) / max(len(after), 1)
    ```

11. When counting unique dates, ALWAYS normalise to (year, month, day) integer
    tuples before deduplication. The data may mix "May 5, 2023" and "May 05, 2023"
    for the same calendar date — string Counter will wrongly report them as distinct.
    Use this pattern:
    ```python
    import re
    from collections import Counter
    month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
                 "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
    def to_date_tuple(s):  # s = "May 05, 2023" or "May 5, 2023"
        parts = s.strip().split()
        return (int(parts[2]), month_map[parts[0]], int(parts[1].rstrip(",")))
    raw = re.findall(r"Date:\s+([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})", P)
    counts = Counter(to_date_tuple(d) for d in raw)
    exactly_once = sum(1 for v in counts.values() if v == 1)
    print("Total dates:", len(raw), " Unique:", len(counts),
          " Exactly-once:", exactly_once)
    ```
</env_tips>
"""


OOLONG_SUBCALL_TIPS = """

Additional rules for Oolong subcalls:
- If the question asks for one of a closed set of labels, return exactly one
  allowed label or NO_ANSWER.
- Do not add explanations unless explicitly requested.
- Do not guess when the text is genuinely insufficient.
- Do not convert uncertainty into a label.
"""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_model_name(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def context_bucket(context_len: int) -> str:
    if context_len < 8_000:
        return "S(<8k)"
    if context_len < 32_000:
        return "M(8k-32k)"
    if context_len < 128_000:
        return "L(32k-128k)"
    return "XL(>=128k)"


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{} notation used in standard mode."""
    match = re.search(r"\\boxed\{\\text\{([^}]*)\}\}", text) or re.search(
        r"\\boxed[\{]+([^}]*)[\}]+", text
    )
    if match:
        return match.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Scoring (mirrors Oolong paper)
# ---------------------------------------------------------------------------

def synth_attempt_answer_parse(answer: str) -> tuple[str, str]:
    confidence = "low"
    if ":" not in answer:
        return (answer if len(answer) < 20 else answer.split()[-1]), confidence

    candidate = answer.split(":")[-1].strip()
    candidate = candidate.replace("*", "").replace("[", "").replace("]", "")
    confidence = "med"

    if any(kw in answer for kw in ("User:", "Answer:", "Date:", "Label")):
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


def synth_score(datapoint: dict[str, Any], output: str, model: str) -> dict[str, Any]:
    try:
        gold = (
            ast.literal_eval(datapoint["answer"])[0]
            if "datetime" not in datapoint["answer"]
            else datetime.strptime(datapoint["answer"], "[datetime.date(%Y, %m, %d)]")
        )
    except Exception:
        gold = datapoint["answer"]

    trimmed, confidence = synth_attempt_answer_parse(output)
    score = 0.0
    if str(trimmed) == str(gold):
        score = 1.0
    elif str(trimmed) in ("more common", "less common", "same frequency"):
        if str(trimmed) in str(gold):
            score = 1.0
    elif datapoint.get("answer_type") == "ANSWER_TYPE.NUMERIC":
        try:
            score = float(0.75 ** abs(int(trimmed) - int(gold)))
        except Exception:
            confidence = "low"
    elif datapoint.get("answer_type") == "ANSWER_TYPE.DATE":
        try:
            import dateutil.parser
            score = float(dateutil.parser.parse(trimmed) == gold)
        except Exception:
            confidence = "low"

    return {
        "id": datapoint.get("id"),
        "model": model,
        "attempted_parse": str(trimmed),
        "parse_confidence": confidence,
        "full_answer": output,
        "score": score,
        "answer": str(gold),
    }


def dnd_parse_answer(answer: str) -> int | str | list[str]:
    try:
        return int(answer)
    except ValueError:
        pass
    if "," in answer:
        return [item.strip() for item in answer.split(",") if item.strip()]
    return answer


def dnd_parse_response(answer: str) -> tuple[Any, str]:
    match = re.search(r"\\boxed\{\\text\{([^}]*)\}\}", answer) or re.search(
        r"\\boxed[\{]+([^}]*)[\}]+", answer
    )
    if match:
        return dnd_parse_answer(match.group(1)), "high"
    return answer, "low"


def dnd_score(datapoint: dict[str, Any], output: str, model: str) -> dict[str, Any]:
    gold = dnd_parse_answer(datapoint["answer"])
    trimmed, confidence = dnd_parse_response(output)
    score = 0.0
    if isinstance(gold, int) and isinstance(trimmed, int):
        score = float(0.75 ** abs(gold - trimmed))
    elif isinstance(gold, str) and isinstance(trimmed, str):
        score = float(gold.strip().lower() == trimmed.strip().lower())
    elif isinstance(gold, list) and isinstance(trimmed, list):
        overlap = set(gold) & set(trimmed)
        score = float(len(overlap) / len(gold)) if gold else 0.0

    return {
        "id": datapoint.get("id"),
        "model": model,
        "attempted_parse": trimmed,
        "parse_confidence": confidence,
        "full_answer": output,
        "score": score,
        "answer": gold,
    }


def score_output(dataset_kind: str, datapoint: dict[str, Any], output: str, model: str) -> dict[str, Any]:
    if dataset_kind == "synth":
        return synth_score(datapoint, output, model)
    return dnd_score(datapoint, output, model)


# ---------------------------------------------------------------------------
# Engine runners
# ---------------------------------------------------------------------------

def run_baseline(
    adapter: AzureOpenAIAdapter,
    context_text: str,
    question: str,
    *,
    max_tokens: int,
    temperature: float,
) -> tuple[str, int, float, str | None]:
    """Single-shot: full context stuffed into the prompt."""
    messages = [
        {
            "role": "system",
            "content": "You are a precise solver. Return only the final answer. No explanations.",
        },
        {
            "role": "user",
            "content": (
                f"{question}\n\n"
                f"<context>\n{context_text}\n</context>\n\n"
                "Provide your answer inside \\boxed{}."
            ),
        },
    ]
    start = time.time()
    try:
        resp = adapter.complete(messages, max_tokens=max_tokens, temperature=temperature)
        output = extract_boxed_answer(resp.text or "")
        tokens = resp.usage.total_tokens if resp.usage else 0
        return output, tokens, time.time() - start, None
    except Exception as exc:
        return "", 0, time.time() - start, str(exc)


def _trace_total_tokens(trace: Any) -> int:
    if trace is None or not hasattr(trace, "steps"):
        return 0
    return sum((s.usage.total_tokens for s in trace.steps if s.usage), 0)


def _summarize_trace(trace: Any) -> dict[str, int]:
    if trace is None or not hasattr(trace, "steps"):
        return {}
    counts: dict[str, int] = defaultdict(int)
    for step in trace.steps:
        counts[step.kind] += 1
    return dict(counts)


def _serialize_trace(trace: Any, *, text_limit: int = 1200) -> list[dict[str, Any]]:
    if trace is None or not hasattr(trace, "steps"):
        return []

    def clip(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if len(value) <= text_limit:
            return value
        return value[:text_limit] + "...<truncated>"

    serialized: list[dict[str, Any]] = []
    for step in trace.steps:
        usage = getattr(step, "usage", None)
        serialized.append(
            {
                "step_id": getattr(step, "step_id", None),
                "kind": getattr(step, "kind", None),
                "depth": getattr(step, "depth", None),
                "elapsed": getattr(step, "elapsed", None),
                "cache_hit": getattr(step, "cache_hit", False),
                "parallel_group_id": getattr(step, "parallel_group_id", None),
                "parallel_index": getattr(step, "parallel_index", None),
                "parallel_total": getattr(step, "parallel_total", None),
                "prompt_summary": clip(getattr(step, "prompt_summary", None)),
                "code": clip(getattr(step, "code", None)),
                "output": clip(getattr(step, "output", None)),
                "stdout": clip(getattr(step, "stdout", None)),
                "error": clip(getattr(step, "error", None)),
                "usage": (
                    {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    }
                    if usage
                    else None
                ),
            }
        )
    return serialized


def _trace_diagnostics(trace: Any) -> dict[str, Any]:
    if trace is None or not hasattr(trace, "steps"):
        return {}

    steps = list(trace.steps)

    def first_step(kind: str) -> int | None:
        for step in steps:
            if getattr(step, "kind", None) == kind:
                return getattr(step, "step_id", None)
        return None

    def _find_nonempty(attr: str, *, kind: str | None = None, reverse: bool = False) -> dict[str, Any] | None:
        seq = reversed(steps) if reverse else steps
        for step in seq:
            if kind is not None and getattr(step, "kind", None) != kind:
                continue
            value = getattr(step, attr, None)
            if isinstance(value, str) and value.strip():
                return {
                    "step_id": getattr(step, "step_id", None),
                    "kind": getattr(step, "kind", None),
                    attr: value[:400],
                }
        return None

    def first_nonempty(attr: str, *, kind: str | None = None) -> dict[str, Any] | None:
        return _find_nonempty(attr, kind=kind, reverse=False)

    def last_nonempty(attr: str, *, kind: str | None = None) -> dict[str, Any] | None:
        return _find_nonempty(attr, kind=kind, reverse=True)

    return {
        "total_steps": len(steps),
        "first_root_call_step": first_step("root_call"),
        "first_repl_step": first_step("repl_exec"),
        "first_subcall_step": first_step("subcall"),
        "first_recursive_subcall_step": first_step("recursive_subcall"),
        "first_sub_root_call_step": first_step("sub_root_call"),
        "first_error": first_nonempty("error"),
        "first_stdout": first_nonempty("stdout", kind="repl_exec"),
        "first_subcall_output": first_nonempty("output", kind="subcall"),
        "last_output": last_nonempty("output"),
    }


def run_rlm(
    adapter: AzureOpenAIAdapter,
    sub_adapter: AzureOpenAIAdapter,
    context_text: str,
    question: str,
    *,
    max_tokens: int,
    subcall_max_tokens: int,
    max_steps: int,
    max_subcalls: int,
    temperature: float,
    env_tips: bool = False,
    require_repl: bool = True,
 ) -> tuple[str, int, float, str | None, dict[str, int], Any]:
    """pyrlm-runtime agentic loop with parallel sub-LLM calls."""
    system_prompt = BASE_SYSTEM_PROMPT
    if env_tips:
        system_prompt += OOLONG_ENV_TIPS
    subcall_system_prompt = SUBCALL_SYSTEM_PROMPT + OOLONG_SUBCALL_TIPS

    context = Context.from_text(context_text)
    rlm = RLM(
        adapter=adapter,
        subcall_adapter=sub_adapter,
        policy=Policy(
            max_steps=max_steps,
            max_subcalls=max_subcalls,
            max_total_tokens=12_000_000,
        ),
        system_prompt=system_prompt,
        subcall_system_prompt=subcall_system_prompt,
        max_tokens=max_tokens,
        subcall_max_tokens=subcall_max_tokens,
        require_repl_before_final=require_repl,
        parallel_subcalls=True,
        max_concurrent_subcalls=20,
        conversation_history=True,
    )
    start = time.time()
    trace = None
    try:
        output, trace = rlm.run(question, context)
        tokens = _trace_total_tokens(trace)
        return output or "", tokens, time.time() - start, None, _summarize_trace(trace), trace
    except Exception as exc:
        return (
            "",
            _trace_total_tokens(trace),
            time.time() - start,
            str(exc),
            _summarize_trace(trace),
            trace,
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def engine_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"examples": 0, "score_sum": 0.0, "tokens_sum": 0, "elapsed_sum": 0.0, "errors": 0}
    )
    for row in rows:
        e = row["engine"]
        grouped[e]["examples"] += 1
        grouped[e]["score_sum"] += float(row["score"])
        grouped[e]["tokens_sum"] += int(row["tokens"])
        grouped[e]["elapsed_sum"] += float(row["elapsed"])
        grouped[e]["errors"] += int(bool(row["error"]))

    return {
        engine: {
            "examples": acc["examples"],
            "avg_score": acc["score_sum"] / max(1, acc["examples"]),
            "avg_tokens": acc["tokens_sum"] / max(1, acc["examples"]),
            "avg_elapsed": acc["elapsed_sum"] / max(1, acc["examples"]),
            "errors": acc["errors"],
        }
        for engine, acc in grouped.items()
    }


def engine_summary_by_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(
            lambda: {"examples": 0, "score_sum": 0.0, "tokens_sum": 0, "elapsed_sum": 0.0, "errors": 0}
        )
    )
    for row in rows:
        g = grouped[row["engine"]][row["context_bucket"]]
        g["examples"] += 1
        g["score_sum"] += float(row["score"])
        g["tokens_sum"] += int(row["tokens"])
        g["elapsed_sum"] += float(row["elapsed"])
        g["errors"] += int(bool(row["error"]))

    return {
        engine: {
            bucket: {
                "examples": acc["examples"],
                "avg_score": acc["score_sum"] / max(1, acc["examples"]),
                "avg_tokens": acc["tokens_sum"] / max(1, acc["examples"]),
                "avg_elapsed": acc["elapsed_sum"] / max(1, acc["examples"]),
                "errors": acc["errors"],
            }
            for bucket, acc in by_bucket.items()
        }
        for engine, by_bucket in grouped.items()
    }


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def select_rows(data: Any, *, strategy: str, max_examples: int, seed: int) -> Any:
    if max_examples <= 0 or len(data) <= max_examples:
        return data

    if strategy == "head":
        return data.select(range(max_examples))

    if strategy == "random":
        rng = random.Random(seed)
        idxs = list(range(len(data)))
        rng.shuffle(idxs)
        return data.select(sorted(idxs[:max_examples]))

    # Stratified by log2(context_len) bins
    bins: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(data):
        key = int(math.log2(max(2, int(row["context_len"]))))
        bins[key].append(i)

    rng = random.Random(seed)
    keys = sorted(bins.keys())
    selected: list[int] = []
    per_bin = max(1, max_examples // max(1, len(keys)))
    for key in keys:
        choices = bins[key][:]
        rng.shuffle(choices)
        selected.extend(choices[:per_bin])

    if len(selected) < max_examples:
        remaining = [i for i in range(len(data)) if i not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_examples - len(selected)])

    return data.select(sorted(selected[:max_examples]))


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

BUCKET_ORDER = ["S(<8k)", "M(8k-32k)", "L(32k-128k)", "XL(>=128k)"]


def write_markdown_summary(path: Path, summary: dict[str, Any], by_bucket: dict[str, Any]) -> None:
    lines = [
        "# Oolong: Baseline vs RLM Summary",
        "",
        "## Overall",
        "",
        "| Engine | Examples | Avg Score | Avg Tokens | Avg Time (s) | Errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for engine in ENGINES:
        if engine not in summary:
            continue
        s = summary[engine]
        lines.append(
            f"| {engine} | {s['examples']} | {s['avg_score']:.4f}"
            f" | {s['avg_tokens']:.1f} | {s['avg_elapsed']:.2f} | {s['errors']} |"
        )

    lines += [
        "",
        "## By Context Bucket",
        "",
        "| Engine | Bucket | Examples | Avg Score | Avg Tokens | Avg Time (s) | Errors |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for engine in ENGINES:
        if engine not in by_bucket:
            continue
        for bucket in BUCKET_ORDER:
            if bucket not in by_bucket[engine]:
                continue
            s = by_bucket[engine][bucket]
            lines.append(
                f"| {engine} | {bucket} | {s['examples']} | {s['avg_score']:.4f}"
                f" | {s['avg_tokens']:.1f} | {s['avg_elapsed']:.2f} | {s['errors']} |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Oolong benchmark: baseline vs pyrlm-runtime RLM (Azure OpenAI)"
    )
    parser.add_argument("--dataset", choices=["synth", "real"], default="synth",
                        help="Oolong dataset subset (default: synth)")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", DEFAULT_MODEL),
                        help=f"Azure deployment name (default: {DEFAULT_MODEL})")
    parser.add_argument("--api-version", default=None,
                        help="Azure API version (default: from AZURE_OPENAI_API_VERSION or 2024-10-21)")
    parser.add_argument("--max-examples", type=int, default=50,
                        help="Max examples to evaluate (default: 50)")
    parser.add_argument("--sample-strategy", choices=["head", "random", "stratified"],
                        default="stratified", help="How to sample examples (default: stratified)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-context-len", type=int, default=1024)
    parser.add_argument("--max-context-len", type=int, default=131072)
    parser.add_argument("--no-labels", action="store_true",
                        help="Use context_window_text instead of context_window_text_with_labels (synth only)")
    parser.add_argument("--temperature", type=float, default=0.0)
    # Baseline
    parser.add_argument("--baseline-max-tokens", type=int, default=512)
    # RLM
    parser.add_argument("--rlm-max-tokens", type=int, default=2048,
                        help="Max tokens for root RLM responses")
    parser.add_argument("--rlm-subcall-max-tokens", type=int, default=1024,
                        help="Max tokens for sub-LLM responses")
    parser.add_argument("--rlm-max-steps", type=int, default=15)
    parser.add_argument("--rlm-max-subcalls", type=int, default=30)
    parser.add_argument("--env-tips", action="store_true",
                        help="Append strategy tips to RLM system prompt (recommended)")
    parser.add_argument("--no-require-repl", action="store_true",
                        help="Allow RLM to answer without executing code first")
    # Output
    parser.add_argument("--output-dir", default=None,
                        help="Directory to write results (default: examples/exports/oolong_rlm_vs_baseline/<run>)")
    parser.add_argument("--save-rlm-traces", action="store_true",
                        help="Persist per-example RLM traces for failure analysis")
    args = parser.parse_args()

    check_azure_connection(args.model, api_version=args.api_version)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with: uv pip install datasets"
        ) from exc

    # --- Load dataset ---
    if args.dataset == "synth":
        data = load_dataset("oolongbench/oolong-synth")["test"]
        context_col = "context_window_text" if args.no_labels else "context_window_text_with_labels"
    else:
        data = load_dataset("oolongbench/oolong-real", "dnd")["test"]
        context_col = "context_window_text"

    data = data.filter(lambda x: args.min_context_len < x["context_len"] <= args.max_context_len)
    data = select_rows(data, strategy=args.sample_strategy, max_examples=args.max_examples, seed=args.seed)
    print(f"Loaded {len(data)} examples from oolong-{args.dataset}")

    # --- Build adapters ---
    adapter_kwargs: dict[str, Any] = {"model": args.model, "timeout": 900.0}
    if args.api_version:
        adapter_kwargs["api_version"] = args.api_version

    baseline_adapter = AzureOpenAIAdapter(**adapter_kwargs)
    rlm_root_adapter = AzureOpenAIAdapter(**adapter_kwargs)
    rlm_sub_adapter = AzureOpenAIAdapter(**adapter_kwargs)

    # --- Output dir ---
    run_label = f"run_{now_tag()}_{args.dataset}_{safe_model_name(args.model)}"
    run_dir = Path(args.output_dir or f"examples/exports/oolong_rlm_vs_baseline/{run_label}")
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = run_dir / "rlm_traces"

    rows: list[dict[str, Any]] = []
    per_example: list[dict[str, Any]] = []
    trace_index: list[dict[str, Any]] = []

    print(
        f"Evaluating baseline vs RLM | model={args.model} | "
        f"sampling={args.sample_strategy} | seed={args.seed}"
    )
    print(f"Output: {run_dir}\n")

    for i, dp in enumerate(data):
        context_text = dp[context_col]
        question = dp["question"]
        example_id = dp.get("id", i)
        ctx_len = int(dp["context_len"])
        bucket = context_bucket(ctx_len)
        print(f"[{i + 1}/{len(data)}] id={example_id} ctx_len={ctx_len} bucket={bucket}")

        # --- Baseline ---
        b_out, b_tok, b_elapsed, b_err = run_baseline(
            baseline_adapter,
            context_text,
            question,
            max_tokens=args.baseline_max_tokens,
            temperature=args.temperature,
        )
        status_b = f"err={b_err[:60]!r}" if b_err else f"tok={b_tok}"
        print(f"  baseline  {b_elapsed:.1f}s {status_b}")

        # --- RLM ---
        r_out, r_tok, r_elapsed, r_err, r_steps, r_trace = run_rlm(
            rlm_root_adapter,
            rlm_sub_adapter,
            context_text,
            question,
            max_tokens=args.rlm_max_tokens,
            subcall_max_tokens=args.rlm_subcall_max_tokens,
            max_steps=args.rlm_max_steps,
            max_subcalls=args.rlm_max_subcalls,
            temperature=args.temperature,
            env_tips=args.env_tips,
            require_repl=not args.no_require_repl,
        )
        status_r = f"err={r_err[:60]!r}" if r_err else f"tok={r_tok} steps={r_steps}"
        print(f"  rlm       {r_elapsed:.1f}s {status_r}")
        r_trace_diag = _trace_diagnostics(r_trace)
        trace_path: Path | None = None
        if args.save_rlm_traces:
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / f"{example_id}.json"
            write_json(
                trace_path,
                {
                    "id": example_id,
                    "question": question,
                    "context_len": ctx_len,
                    "context_bucket": bucket,
                    "diagnostics": r_trace_diag,
                    "steps": _serialize_trace(r_trace),
                },
            )

        example_payload: dict[str, Any] = {
            "id": example_id,
            "question": question,
            "context_len": ctx_len,
            "context_bucket": bucket,
            "results": {},
        }

        for engine, out, tok, elapsed, err, extra in [
            ("baseline", b_out, b_tok, b_elapsed, b_err, {}),
            ("rlm", r_out, r_tok, r_elapsed, r_err, {"trace_steps": r_steps}),
        ]:
            eval_out = score_output(args.dataset, dict(dp), out, args.model)
            row = {
                "id": example_id,
                "engine": engine,
                "score": float(eval_out["score"]),
                "tokens": int(tok),
                "elapsed": float(elapsed),
                "error": err,
                "context_len": ctx_len,
                "context_bucket": bucket,
                "attempted_parse": str(eval_out["attempted_parse"]),
                "gold_answer": str(eval_out["answer"]),
                "parse_confidence": eval_out["parse_confidence"],
            }
            rows.append(row)
            example_payload["results"][engine] = {
                "output": out,
                "tokens": tok,
                "elapsed": elapsed,
                "error": err,
                "score": eval_out["score"],
                "attempted_parse": str(eval_out["attempted_parse"]),
                "gold_answer": str(eval_out["answer"]),
                "parse_confidence": eval_out["parse_confidence"],
                "extra": extra,
            }

        example_payload["results"]["rlm"]["extra"]["trace_diagnostics"] = r_trace_diag
        if trace_path is not None:
            example_payload["results"]["rlm"]["extra"]["trace_file"] = str(trace_path)

        per_example.append(example_payload)
        trace_index.append(
            {
                "id": example_id,
                "context_len": ctx_len,
                "context_bucket": bucket,
                "baseline_score": example_payload["results"]["baseline"]["score"],
                "rlm_score": example_payload["results"]["rlm"]["score"],
                "score_delta": (
                    example_payload["results"]["rlm"]["score"]
                    - example_payload["results"]["baseline"]["score"]
                ),
                "rlm_tokens": r_tok,
                "rlm_elapsed": r_elapsed,
                "trace_diagnostics": r_trace_diag,
                "trace_file": str(trace_path) if trace_path is not None else None,
            }
        )

    # --- Aggregate ---
    summary = engine_summary(rows)
    summary_by_bucket = engine_summary_by_bucket(rows)

    run_config = {
        "dataset": args.dataset,
        "model": args.model,
        "api_version": args.api_version,
        "max_examples": args.max_examples,
        "examples_evaluated": len(per_example),
        "sample_strategy": args.sample_strategy,
        "seed": args.seed,
        "min_context_len": args.min_context_len,
        "max_context_len": args.max_context_len,
        "labels": not args.no_labels,
        "temperature": args.temperature,
        "baseline_max_tokens": args.baseline_max_tokens,
        "rlm_max_tokens": args.rlm_max_tokens,
        "rlm_subcall_max_tokens": args.rlm_subcall_max_tokens,
        "rlm_max_steps": args.rlm_max_steps,
        "rlm_max_subcalls": args.rlm_max_subcalls,
        "env_tips": args.env_tips,
        "save_rlm_traces": args.save_rlm_traces,
        "require_repl_before_final": not args.no_require_repl,
        "pyrlm_runtime_path": str(
            Path(sys.modules["pyrlm_runtime"].__file__).resolve()
        ) if "pyrlm_runtime" in sys.modules else "?",
    }

    write_json(run_dir / "run_config.json", run_config)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "summary_by_context_bucket.json", summary_by_bucket)
    write_json(run_dir / "per_example.json", per_example)
    write_json(run_dir / "rlm_trace_index.json", trace_index)
    write_jsonl(run_dir / "flat_results.jsonl", rows)
    write_markdown_summary(run_dir / "summary.md", summary, summary_by_bucket)

    # --- Print summary ---
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for engine in ENGINES:
        if engine not in summary:
            continue
        s = summary[engine]
        print(
            f"{engine:10s}  score={s['avg_score']:.4f}  "
            f"avg_tokens={s['avg_tokens']:.1f}  avg_time={s['avg_elapsed']:.2f}s  "
            f"errors={s['errors']}/{s['examples']}"
        )

    print("\nBY CONTEXT BUCKET")
    print("=" * 72)
    for engine in ENGINES:
        if engine not in summary_by_bucket:
            continue
        for bucket in BUCKET_ORDER:
            if bucket not in summary_by_bucket[engine]:
                continue
            s = summary_by_bucket[engine][bucket]
            print(
                f"{engine:10s}  {bucket:12s}  score={s['avg_score']:.4f}  "
                f"avg_tokens={s['avg_tokens']:.1f}  avg_time={s['avg_elapsed']:.2f}s  "
                f"n={s['examples']} err={s['errors']}"
            )

    print(f"\nArtifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
