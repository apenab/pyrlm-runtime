"""
Benchmark: RLM vs Baseline (Conventional Model)
===============================================

PURPOSE:
    Compare RLM against a "baseline" (direct prompt) to show
    when and why RLM beats conventional models.

    This benchmark tests the MIT CSAIL paper's core hypothesis:
    "RLM scales better with large contexts than traditional models."

WHAT IT SHOWS:
    1. Crossover point: the context size where RLM starts to win
    2. Truncation effect: how baseline fails when context is too large
    3. Implemented optimizations: deterministic phase 0, parallel subcalls
    4. Fuzzy metrics: typo tolerance in evaluation (Levenshtein)

PAPER HYPOTHESIS:
    ┌────────────────────────────────────────────────────────────────────┐
    │                                                                    │
    │  Accuracy ▲                                                        │
    │          │      ╭──────── RLM (scales better)                     │
    │          │     ╱                                                   │
    │          │    ╱                                                    │
    │          │   ╱  ╲                                                  │
    │          │  ╱    ╲──────── Baseline (truncated)                   │
    │          │ ╱                                                       │
    │          │╱       CROSSOVER POINT                                 │
    │          └────────────────────────────────► Context size          │
    │                                                                    │
    │  Baseline: sends all context (truncates if too large)             │
    │  RLM: inspects context programmatically (no hard limit)           │
    └────────────────────────────────────────────────────────────────────┘

IMPLEMENTED OPTIMIZATIONS (from prior analysis):
    B) Deterministic phase 0:
       - ALWAYS try extract_after() BEFORE subcalls
       - Reduces ~30 subcalls down to potentially 0-1
       - If extract_after finds the answer, no LLM needed

    F) Fuzzy metrics:
       - Allows typos using Levenshtein distance
       - "ooloong" is accepted for "oolong" (distance <= 2)

    Parallel:
       - ask_chunks with parallel=True runs concurrent subcalls
       - max_concurrent_subcalls controls worker count

CONTEXT STRUCTURE:
    - Generates N documents with M lines each
    - The "needle" (key term) is placed at 80% of the total docs
    - Baseline truncates to 8000 chars by default -> misses the needle in large contexts

ENVIRONMENT VARIABLES:
    LLM_BASE_URL              Server URL
    LLM_MODELS                Models to test (comma-separated)
    LLM_SUBCALL_MODEL         Subcall model (optional, smaller/faster)
    RLM_CONTEXT_SIZES         Context sizes to test (default: 5,30,120 docs)
    RLM_LINES_PER_DOC         Lines per document (default: 8)
    RLM_KEY_DOC_RATIO         Needle position (default: 0.8 = 80%)
    BASELINE_MAX_CHARS        Baseline truncation limit (default: 8000)
    RLM_PARALLEL_SUBCALLS     Enable parallel subcalls (default: 1)
    RLM_FALLBACK              Enable fallback code (default: 1)
    SHOW_TRAJECTORY           Show paper-style RLM trajectory (0/1, default: 0)
    RLM_EXPORT                Export results to a file in examples/ (0/1, default: 0)
    RLM_EXPORT_FORMAT         Export format: md or jsonl (default: md)
    RLM_EXPORT_PATH           Export path override (supports {timestamp})
                             (default: examples/exports/rlm_vs_baseline_results_<timestamp>.md)

HOW TO RUN:
    # Basic
    uv run python examples/rlm_vs_baseline.py

    # Multiple context sizes
    RLM_CONTEXT_SIZES=10,50,100,200 uv run python examples/rlm_vs_baseline.py

    # Separate subcall model (more efficient)
    LLM_SUBCALL_MODEL=qwen2.5:3b uv run python examples/rlm_vs_baseline.py

    # Trajectory visualization (MIT paper Appendix B style)
    SHOW_TRAJECTORY=1 uv run python examples/rlm_vs_baseline.py

    # Export results (default: Markdown)
    RLM_EXPORT=1 uv run python examples/rlm_vs_baseline.py

    # Export results (Markdown)
    RLM_EXPORT=1 RLM_EXPORT_FORMAT=md uv run python examples/rlm_vs_baseline.py

    # Export results (JSONL)
    RLM_EXPORT=1 RLM_EXPORT_FORMAT=jsonl uv run python examples/rlm_vs_baseline.py

EXPECTED OUTPUT:
    ============================================================
    Model: qwen2.5-coder:7b
    Subcall model: same as root
    Parallel subcalls: True max_workers=4
    Baseline max chars: 8000

    Context: docs=5 lines/doc=8 chars=2850
      baseline: oolong  elapsed=0.45s tokens=890 truncated=False contains_key=True
      rlm: oolong  elapsed=1.23s tokens=1250 steps={'root_call': 2}
      winner: baseline (fewer tokens)

    Context: docs=120 lines/doc=8 chars=68400
      baseline: I cannot find...  elapsed=0.52s tokens=920 truncated=True contains_key=False
      rlm: oolong  elapsed=2.15s tokens=1890 steps={'root_call': 2}
      winner: rlm (baseline missed key term)

    Summary:
    docs chars base_tok base_s trunc base_ok rlm_tok rlm_s rlm_ok winner
       5  2850     890   0.45 False    True    1250   1.23   True baseline (fewer tokens)
     120 68400     920   0.52  True   False    1890   2.15   True rlm (baseline missed key term)

INTERPRETATION:
    - Small contexts (~3K chars): baseline wins (less overhead)
    - Large contexts (>8K chars): RLM wins (baseline truncates and misses needle)
    - The crossover point depends on BASELINE_MAX_CHARS and needle position
"""

import io
import json
import logging
import os
import sys
import time
from collections import Counter

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from pyrlm_runtime import Context, Policy, RLM
from pyrlm_runtime.adapters import GenericChatAdapter
# We define our own system prompt tailored to the needle-in-haystack task

# Paper-aligned system prompt for needle-in-haystack tasks (Appendix D.1 style)
RLM_SYSTEM_PROMPT = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A 'P' variable (string) containing the full context - it may be too large for your context window.
2. A 'ctx' variable (Context object) with helpers for safe inspection.
3. Helper functions: peek(n), tail(n), lenP(), ctx.slice, ctx.find, ctx.chunk, ctx.chunk_documents.
4. Sub-LLM functions: llm_query(text), llm_query_batch(chunks), ask(question, text), ask_chunks(question, chunks), ask_chunks_first(question, chunks).
5. Utilities: pick_first_answer(answers), extract_after(marker).
6. The ability to use print() statements to view the output of your REPL code and continue reasoning.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context.

IMPORTANT OPTIMIZATION: For needle-in-haystack tasks, ALWAYS try deterministic extraction FIRST before using expensive LLM subcalls:
1. Try extract_after(marker) to find the answer using string search (0 tokens, instant)
2. Only if that fails, use ask_chunks or llm_query on portions of the context

RECOMMENDED STRATEGY for needle-in-haystack tasks:
```
# Phase 0: Try deterministic extraction first (0 tokens)
key = extract_after('The key term is:')
if key:
    # Found it! No need for LLM subcalls
    print(f"Found key: {key}")
else:
    # Phase 1: Need to search semantically with LLM
    chunks = [c[2] for c in ctx.chunk(2000)]
    key = ask_chunks_first("What is the key term?", chunks)
    print(f"Found via subcalls: {key}")
```

Remember that your sub-LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10-20 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls.

IMPORTANT: Be careful about using llm_query as it incurs high runtime costs. Always batch as much information as reasonably possible into each call (aim for ~200k characters per call). For example, if you have 100 documents to process, split into chunks of 10-20 documents and call llm_query on each chunk (5-10 calls total) rather than 100 individual calls.

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL: <your final answer here> to provide the answer directly
2. Use FINAL_VAR: <variable_name> to return a variable you have created in the REPL environment as your final output

Execute your plan immediately in your response. Output to the REPL environment and recursive LLMs as much as possible.
"""

KEY_MARKER = "The key term is:"
KEY_VALUE = "oolong"


def build_documents(doc_count: int, lines_per_doc: int, *, key_doc_ratio: float) -> list[str]:
    if doc_count <= 0:
        return []
    key_doc_index = max(0, min(doc_count - 1, int(doc_count * key_doc_ratio)))
    filler = "alpha beta gamma delta epsilon zeta eta theta iota kappa.\n"
    docs: list[str] = []
    for doc_idx in range(doc_count):
        lines = [filler for _ in range(lines_per_doc)]
        if doc_idx == key_doc_index:
            lines.insert(lines_per_doc // 2, f"{KEY_MARKER} {KEY_VALUE}.\n")
        if doc_idx == 0:
            lines.insert(
                0,
                "RLMs treat long prompts as environment state and inspect them via code.\n",
            )
        docs.append("".join(lines))
    return docs


def build_context(
    doc_count: int,
    lines_per_doc: int,
    *,
    key_doc_ratio: float,
    separator: str,
) -> Context:
    documents = build_documents(doc_count, lines_per_doc, key_doc_ratio=key_doc_ratio)
    return Context.from_documents(documents, separator=separator)


def run_rlm(
    adapter: GenericChatAdapter,
    subcall_adapter: GenericChatAdapter | None,
    model: str,
    context: Context,
    query: str,
    *,
    require_subcall: bool,
    max_steps: int,
    max_tokens: int,
    max_subcall_tokens: int | None,
    max_subcalls: int,
    fallback_enabled: bool,
    fallback_code: str | None,
    invalid_limit: int | None,
    repl_error_limit: int | None,
    subcall_guard_steps: int | None,
    parallel_subcalls: bool,
    max_concurrent_subcalls: int,
) -> dict:
    # Use our custom needle-in-haystack optimized prompt for all models
    system_prompt = RLM_SYSTEM_PROMPT

    if not fallback_enabled:
        fallback_code = None
        invalid_limit = None
        repl_error_limit = None

    policy = Policy(
        max_steps=max_steps,
        max_subcalls=max_subcalls,
        max_total_tokens=max_tokens,
        max_subcall_tokens=max_subcall_tokens,
    )
    rlm = RLM(
        adapter=adapter,
        subcall_adapter=subcall_adapter,
        policy=policy,
        system_prompt=system_prompt,
        require_repl_before_final=True,
        require_subcall_before_final=require_subcall,
        auto_finalize_var="key",
        invalid_response_limit=invalid_limit,
        repl_error_limit=repl_error_limit,
        fallback_code=fallback_code,
        subcall_guard_steps=subcall_guard_steps,
        parallel_subcalls=parallel_subcalls,
        max_concurrent_subcalls=max_concurrent_subcalls,
    )

    started = time.perf_counter()
    trace = None
    try:
        output, trace = rlm.run(query, context)
    except Exception as exc:  # noqa: BLE001
        return {
            "mode": "rlm",
            "output": f"ERROR: {type(exc).__name__}",
            "elapsed": time.perf_counter() - started,
            "tokens": policy.total_tokens,
            "calls": policy.subcalls + policy.steps,
            "steps": {"error": type(exc).__name__},
            "trace": trace,  # Include trace even on error
        }
    elapsed = time.perf_counter() - started

    counts = Counter(step.kind for step in trace.steps)
    tokens = sum(step.usage.total_tokens for step in trace.steps if step.usage is not None)
    calls = counts.get("root_call", 0) + counts.get("subcall", 0)

    # Collect detailed metrics (token breakdown, cache hits, subcall stats)
    prompt_tokens = sum(step.usage.prompt_tokens for step in trace.steps if step.usage)
    completion_tokens = sum(step.usage.completion_tokens for step in trace.steps if step.usage)
    cache_read_tokens = sum(
        getattr(step.usage, "cache_read_input_tokens", 0) for step in trace.steps if step.usage
    )
    cache_creation_tokens = sum(
        getattr(step.usage, "cache_creation_input_tokens", 0) for step in trace.steps if step.usage
    )
    cache_hits = sum(1 for step in trace.steps if step.cache_hit)

    # Subcall-specific metrics
    subcall_steps = [s for s in trace.steps if s.kind == "subcall"]
    subcall_count = len(subcall_steps)
    subcall_tokens = sum(s.usage.total_tokens for s in subcall_steps if s.usage)
    subcall_avg_tokens = subcall_tokens / subcall_count if subcall_count > 0 else 0

    # REPL execution metrics
    repl_execs = counts.get("repl_exec", 0)
    repl_errors = sum(1 for s in trace.steps if s.kind == "repl_exec" and s.error)

    return {
        "mode": "rlm",
        "output": output,
        "elapsed": elapsed,
        "tokens": tokens,
        "calls": calls,
        "steps": dict(counts),
        "trace": trace,  # Include trace for trajectory visualization
        # Detailed metrics for demo presentation
        "metrics": {
            "token_breakdown": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": tokens,
                "cache_read_tokens": cache_read_tokens,
                "cache_creation_tokens": cache_creation_tokens,
            },
            "cache": {
                "hits": cache_hits,
                "total_steps": len(trace.steps),
                "hit_rate": cache_hits / len(trace.steps) if trace.steps else 0,
            },
            "subcalls": {
                "count": subcall_count,
                "total_tokens": subcall_tokens,
                "avg_tokens_per_subcall": round(subcall_avg_tokens, 1),
            },
            "repl": {
                "executions": repl_execs,
                "errors": repl_errors,
            },
        },
    }


def run_baseline(
    adapter: GenericChatAdapter,
    context: Context,
    query: str,
    *,
    max_tokens: int,
    max_context_chars: int | None,
) -> dict:
    context_text = context.text
    truncated = False
    if max_context_chars is not None and max_context_chars > 0:
        if len(context_text) > max_context_chars:
            context_text = context_text[:max_context_chars]
            truncated = True
    contains_key = KEY_MARKER in context_text
    prompt = (
        "Answer the question using only the provided context.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question:\n{query}\n\n"
        "Answer with only the key term value."
    )
    started = time.perf_counter()
    response = adapter.complete(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    elapsed = time.perf_counter() - started
    return {
        "mode": "baseline",
        "output": response.text.strip(),
        "elapsed": elapsed,
        "tokens": response.usage.total_tokens,
        "calls": 1,
        "steps": {"root_call": 1},
        "truncated": truncated,
        "used_chars": len(context_text),
        "context_chars": context.len_chars(),
        "contains_key": contains_key,
    }


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def is_success(output: str, *, max_distance: int = 2) -> bool:
    """Check if output contains the key value, tolerating typos.

    Handles cases like 'ooloong' (typo for 'oolong') by using Levenshtein distance.
    """
    output_lower = output.lower()
    # Exact match first
    if KEY_VALUE in output_lower:
        return True
    # Fuzzy match: check each word in output
    words = output_lower.replace(",", " ").replace(".", " ").split()
    for word in words:
        word = word.strip("\"'()[]{}:;")
        if not word:
            continue
        distance = _levenshtein(word, KEY_VALUE)
        if distance <= max_distance:
            return True
    return False


def pick_winner(baseline: dict, rlm: dict) -> str:
    baseline_ok = is_success(baseline["output"])
    rlm_ok = is_success(rlm["output"])
    if rlm_ok and not baseline_ok:
        return "rlm (baseline missed key term)"
    if baseline_ok and not rlm_ok:
        return "baseline (rlm missed key term)"
    if rlm_ok and baseline_ok:
        if rlm["tokens"] < baseline["tokens"]:
            return "rlm (fewer tokens)"
        if rlm["tokens"] > baseline["tokens"]:
            return "baseline (fewer tokens)"
        return "tie"
    return "tie"


def export_results(path: str, payload: dict) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def export_results_md(path: str, transcript: str) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as handle:
        if not file_exists:
            handle.write("# RLM vs Baseline Results\n\n")
        handle.write(f"## Run {timestamp}\n\n")
        handle.write("```text\n")
        handle.write(transcript.rstrip() + "\n")
        handle.write("```\n\n")


class _Tee(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for stream in self._streams:
            stream.write(s)
            stream.flush()
        return len(s)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def format_trajectory(trace, title: str = "RLM Trajectory", max_code_lines: int = 50) -> str:
    """
    Format an RLM trace trajectory for display, similar to MIT RLM paper Appendix B.

    Args:
        trace: The Trace object containing execution steps
        title: Title for the trajectory visualization
        max_code_lines: Maximum lines of code to show per step (truncate if longer)

    Returns:
        Formatted string representation of the trajectory
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"{title:^80}")
    lines.append("=" * 80)
    lines.append("")

    # Group steps by execution sequence
    for i, step in enumerate(trace.steps):
        step_header = []

        # Step number and type
        if step.kind == "root_call":
            step_header.append(f"[Step {i + 1}] Root LLM Call")
        elif step.kind == "repl_exec":
            step_header.append(f"[Step {i + 1}] REPL Execution")
        elif step.kind == "subcall":
            step_header.append(f"[Step {i + 1}] Subcall (LLM)")
        elif step.kind == "recursive_subcall":
            step_header.append(f"[Step {i + 1}] Recursive Subcall (depth={step.depth})")
        else:
            step_header.append(f"[Step {i + 1}] {step.kind}")

        # Add token usage info
        if step.usage:
            step_header.append(f" | Tokens: {step.usage.total_tokens}")

        # Add cache hit indicator
        if step.cache_hit:
            step_header.append(" | [CACHED]")

        lines.append("".join(step_header))
        lines.append("-" * 80)

        # Show prompt summary for LLM calls
        if step.prompt_summary and step.kind in ("root_call", "subcall", "recursive_subcall"):
            lines.append(
                f"Prompt: {step.prompt_summary[:200]}{'...' if len(step.prompt_summary) > 200 else ''}"
            )
            lines.append("")

        # Show code if present
        if step.code:
            code_lines = step.code.split("\n")
            if len(code_lines) > max_code_lines:
                truncated_code = "\n".join(code_lines[:max_code_lines])
                lines.append("```python")
                lines.append(truncated_code)
                lines.append(f"... [{len(code_lines) - max_code_lines} more lines truncated]")
                lines.append("```")
            else:
                lines.append("```python")
                lines.append(step.code)
                lines.append("```")
            lines.append("")

        # Show stdout if present
        if step.stdout:
            stdout_display = step.stdout[:1000]  # Limit stdout display
            if len(step.stdout) > 1000:
                stdout_display += f"\n... [{len(step.stdout) - 1000} more chars truncated]"
            lines.append("Output:")
            lines.append(stdout_display)
            lines.append("")

        # Show error if present
        if step.error:
            lines.append(f"ERROR: {step.error}")
            lines.append("")

        lines.append("")

    # Summary section
    lines.append("=" * 80)
    lines.append("TRAJECTORY SUMMARY")
    lines.append("=" * 80)

    total_steps = len(trace.steps)
    total_tokens = sum(s.usage.total_tokens for s in trace.steps if s.usage)
    step_counts = {}
    for step in trace.steps:
        step_counts[step.kind] = step_counts.get(step.kind, 0) + 1
    cache_hits = sum(1 for s in trace.steps if s.cache_hit)
    errors = sum(1 for s in trace.steps if s.error)

    lines.append(f"Total Steps: {total_steps}")
    lines.append(f"Total Tokens: {total_tokens}")
    lines.append(f"Cache Hits: {cache_hits}")
    lines.append(f"Errors: {errors}")
    lines.append("")
    lines.append("Step Breakdown:")
    for kind, count in sorted(step_counts.items()):
        lines.append(f"  - {kind}: {count}")
    lines.append("=" * 80)

    return "\n".join(lines)


def plot_crossover_ascii(summary_rows: list[dict]) -> str:
    """
    Generate an ASCII art plot showing the crossover point between RLM and baseline.

    Visualizes how accuracy and token usage change with context size, highlighting
    where RLM starts outperforming the baseline approach.
    """
    if not summary_rows:
        return ""

    lines = []
    lines.append("")
    lines.append("=" * 100)
    lines.append("CROSSOVER POINT VISUALIZATION".center(100))
    lines.append("=" * 100)
    lines.append("")

    # Extract data
    docs = [r["docs"] for r in summary_rows]
    baseline_ok = [1 if r["baseline_ok"] else 0 for r in summary_rows]
    rlm_ok = [1 if r["rlm_ok"] else 0 for r in summary_rows]
    baseline_tokens = [r["baseline_tokens"] for r in summary_rows]
    rlm_tokens = [r["rlm_tokens"] for r in summary_rows]

    # Plot 1: Accuracy (Success Rate)
    lines.append("SUCCESS RATE vs CONTEXT SIZE:")
    lines.append("-" * 100)
    lines.append("")

    height = 10
    width = min(80, len(docs) * 12)

    # Create grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Draw axes
    for y in range(height):
        grid[y][0] = "│"
    for x in range(width):
        grid[height - 1][x] = "─"
    grid[height - 1][0] = "└"

    # Plot baseline (✗ = fail, ✓ = success)
    for i, (doc_count, ok) in enumerate(zip(docs, baseline_ok)):
        x = int((i / (len(docs) - 1)) * (width - 5)) + 2 if len(docs) > 1 else width // 2
        if x < width:
            if ok:
                y = 1  # Success at top
                grid[y][x] = "B"
            else:
                y = height - 3  # Failure at bottom
                grid[y][x] = "b"

    # Plot RLM (✗ = fail, ✓ = success)
    for i, (doc_count, ok) in enumerate(zip(docs, rlm_ok)):
        x = int((i / (len(docs) - 1)) * (width - 5)) + 2 if len(docs) > 1 else width // 2
        if x < width:
            if ok:
                y = 2  # Success at top (slightly below baseline)
                if grid[y][x] == " ":
                    grid[y][x] = "R"
                else:
                    grid[y][x] = "*"  # Both succeed
            else:
                y = height - 2  # Failure at bottom
                grid[y][x] = "r"

    # Add labels
    lines.append("  100% ┤ Success")
    for row in grid[: height - 1]:
        lines.append("       " + "".join(row))
    lines.append("    0% " + "".join(grid[height - 1]))

    # X-axis labels
    x_labels = "         "
    for i, doc_count in enumerate(docs):
        if len(docs) > 1:
            x_pos = int((i / (len(docs) - 1)) * (width - 5)) + 2
        else:
            x_pos = width // 2
        label = f"{doc_count}"
        # Pad to position
        while len(x_labels) < x_pos + 7:
            x_labels += " "
        x_labels += label

    lines.append(x_labels)
    lines.append("         " + "Context Size (documents)".center(width))
    lines.append("")
    lines.append("  Legend: B=Baseline OK | b=Baseline FAIL | R=RLM OK | r=RLM FAIL | *=Both OK")
    lines.append("")

    # Plot 2: Token Usage Comparison
    lines.append("")
    lines.append("TOKEN USAGE vs CONTEXT SIZE:")
    lines.append("-" * 100)
    lines.append("")

    if baseline_tokens and rlm_tokens:
        max_tokens = max(max(baseline_tokens), max(rlm_tokens))

        # Normalize to 0-100 range
        def normalize(val):
            return int((val / max_tokens) * 40) if max_tokens > 0 else 0

        for i, (doc_count, b_tok, r_tok) in enumerate(zip(docs, baseline_tokens, rlm_tokens)):
            b_bar = normalize(b_tok)
            r_bar = normalize(r_tok)

            label = f"{doc_count:>4} docs │"
            b_visual = "B" * b_bar
            r_visual = "R" * r_bar

            lines.append(f"{label} {b_visual:<42} ({b_tok:>6} tokens)")
            lines.append(f"{'':>10} {r_visual:<42} ({r_tok:>6} tokens)")
            lines.append("")

        lines.append("  Legend: B=Baseline tokens | R=RLM tokens")
        lines.append(f"  Scale: {max_tokens:,} tokens = 40 chars")

    lines.append("")
    lines.append("=" * 100)
    lines.append("")

    return "\n".join(lines)


def format_results_table(summary_rows: list[dict]) -> str:
    """
    Format benchmark results as a presentable table for demo purposes.

    Shows key metrics including tokens, time, correctness, and winner for each context size.
    Highlights the crossover point where RLM starts outperforming baseline.
    """
    lines = []
    lines.append("")
    lines.append("=" * 120)
    lines.append("BENCHMARK RESULTS - RLM vs Baseline".center(120))
    lines.append("=" * 120)
    lines.append("")

    # Table header
    lines.append(f"{'Docs':<6} {'Chars':<8} │ {'Baseline':<35} │ {'RLM':<45} │ {'Winner':<20}")
    lines.append(
        f"{'':6} {'':8} │ "
        f"{'Tokens':>8} {'Time':>6} {'Trunc':>6} {'OK':>4} {'Subcalls':>8} │ "
        f"{'Tokens':>8} {'Time':>6} {'OK':>4} {'Steps':>6} {'Subcalls':>8} {'Cache':>6} │ "
        f"{'':<20}"
    )
    lines.append("─" * 120)

    # Table rows
    for row in summary_rows:
        docs = row["docs"]
        chars = row["chars"]

        # Baseline metrics
        b_tokens = row["baseline_tokens"]
        b_time = row["baseline_elapsed"]
        b_trunc = "✓" if row["baseline_truncated"] else "✗"
        b_ok = "✓" if row["baseline_ok"] else "✗"
        b_subcalls = "-"  # Baseline doesn't use subcalls

        # RLM metrics
        r_tokens = row["rlm_tokens"]
        r_time = row["rlm_elapsed"]
        r_ok = "✓" if row["rlm_ok"] else "✗"

        # Handle error case where steps contains error string instead of counts
        rlm_steps = row.get("rlm_steps", {})
        if "error" in rlm_steps:
            r_steps_total = 0  # Error case
            r_subcalls = "-"
            r_cache = "-"
        else:
            r_steps_total = sum(rlm_steps.values()) if rlm_steps else 0

            # Extract subcalls and cache info from metrics if available
            if row.get("rlm_metrics"):
                r_subcalls = row["rlm_metrics"]["subcalls"]["count"]
                r_cache = f"{row['rlm_metrics']['cache']['hit_rate']:.0%}"
            else:
                r_subcalls = rlm_steps.get("subcall", 0)
                r_cache = "-"

        # Winner indication
        winner = row["winner"]
        if "rlm" in winner.lower():
            winner_symbol = "🏆 RLM"
        elif "baseline" in winner.lower():
            winner_symbol = "🏆 Baseline"
        else:
            winner_symbol = "tie"

        lines.append(
            f"{docs:<6} {chars:<8} │ "
            f"{b_tokens:>8} {b_time:>6.2f}s {b_trunc:>6} {b_ok:>4} {b_subcalls:>8} │ "
            f"{r_tokens:>8} {r_time:>6.2f}s {r_ok:>4} {r_steps_total:>6} {r_subcalls:>8} {r_cache:>6} │ "
            f"{winner_symbol:<20}"
        )

    lines.append("=" * 120)
    lines.append("")

    # Summary statistics
    lines.append("SUMMARY STATISTICS:")
    lines.append("-" * 120)

    total_tests = len(summary_rows)
    rlm_wins = sum(
        1
        for r in summary_rows
        if r["winner"].lower().startswith("rlm") or "🏆 rlm" in r["winner"].lower()
    )
    baseline_wins = sum(
        1
        for r in summary_rows
        if r["winner"].lower().startswith("baseline") or "🏆 baseline" in r["winner"].lower()
    )
    ties = total_tests - rlm_wins - baseline_wins

    lines.append(f"Total tests: {total_tests}")
    lines.append(f"RLM wins: {rlm_wins} ({rlm_wins / total_tests:.1%})")
    lines.append(f"Baseline wins: {baseline_wins} ({baseline_wins / total_tests:.1%})")
    lines.append(f"Ties: {ties}")
    lines.append("")

    # Token efficiency
    total_baseline_tokens = sum(r["baseline_tokens"] for r in summary_rows)
    total_rlm_tokens = sum(r["rlm_tokens"] for r in summary_rows)
    lines.append(f"Total tokens - Baseline: {total_baseline_tokens:,} | RLM: {total_rlm_tokens:,}")

    if total_baseline_tokens > 0:
        token_ratio = total_rlm_tokens / total_baseline_tokens
        lines.append(f"Token efficiency: RLM uses {token_ratio:.2f}x tokens vs baseline")

    # Time efficiency
    total_baseline_time = sum(r["baseline_elapsed"] for r in summary_rows)
    total_rlm_time = sum(r["rlm_elapsed"] for r in summary_rows)
    lines.append(f"Total time - Baseline: {total_baseline_time:.2f}s | RLM: {total_rlm_time:.2f}s")

    lines.append("")

    # Crossover point analysis
    lines.append("CROSSOVER POINT ANALYSIS:")
    lines.append("-" * 120)

    truncation_starts = None
    rlm_starts_winning = None

    for i, row in enumerate(summary_rows):
        if row["baseline_truncated"] and truncation_starts is None:
            truncation_starts = i
        if "rlm" in row["winner"].lower() and rlm_starts_winning is None:
            rlm_starts_winning = i

    if truncation_starts is not None:
        row = summary_rows[truncation_starts]
        lines.append(f"• Baseline starts truncating at {row['docs']} docs ({row['chars']:,} chars)")

    if rlm_starts_winning is not None:
        row = summary_rows[rlm_starts_winning]
        lines.append(f"• RLM starts winning at {row['docs']} docs ({row['chars']:,} chars)")
        lines.append(f"  Reason: {row['winner']}")

    # RLM efficiency metrics
    if any(r.get("rlm_metrics") for r in summary_rows):
        lines.append("")
        lines.append("RLM EFFICIENCY METRICS:")
        lines.append("-" * 120)

        avg_subcalls = sum(
            r["rlm_metrics"]["subcalls"]["count"] for r in summary_rows if r.get("rlm_metrics")
        ) / len([r for r in summary_rows if r.get("rlm_metrics")])

        avg_cache_rate = sum(
            r["rlm_metrics"]["cache"]["hit_rate"] for r in summary_rows if r.get("rlm_metrics")
        ) / len([r for r in summary_rows if r.get("rlm_metrics")])

        lines.append(f"• Average subcalls per test: {avg_subcalls:.1f}")
        lines.append(f"• Average cache hit rate: {avg_cache_rate:.1%}")

    lines.append("=" * 120)
    lines.append("")

    return "\n".join(lines)


def smart_router(
    adapter: "GenericChatAdapter",
    subcall_adapter: "GenericChatAdapter | None",
    model: str,
    context: "Context",
    query: str,
    baseline_query: str,
    *,
    baseline_max_chars: int | None,
    run_rlm_fn: callable,
    run_baseline_fn: callable,
    rlm_kwargs: dict,
) -> tuple[dict, str]:
    """A) Router baseline-first: use baseline if context fits, RLM otherwise.

    This optimization avoids the overhead of RLM when the context is small enough
    for a direct baseline call.
    """
    context_len = context.len_chars()

    # If baseline can see the full context (no truncation), try baseline first
    if baseline_max_chars is None or context_len <= baseline_max_chars:
        baseline_result = run_baseline_fn(
            adapter,
            context,
            baseline_query,
            max_tokens=256,
            max_context_chars=baseline_max_chars,
        )
        if is_success(baseline_result["output"]):
            return baseline_result, "baseline (router: context fits)"

    # Fall back to RLM for large contexts or when baseline fails
    rlm_result = run_rlm_fn(
        adapter,
        subcall_adapter,
        model,
        context,
        query,
        **rlm_kwargs,
    )
    return rlm_result, "rlm (router: context too large or baseline failed)"


def main() -> None:
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    models = os.getenv("LLM_MODELS", "qwen2.5-coder:7b").split(",")
    models = [model.strip() for model in models if model.strip()]
    log_level = os.getenv("LLM_LOG_LEVEL", "WARNING").upper()

    # Configure logging: suppress httpx noise
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("pyrlm_runtime").setLevel(getattr(logging, log_level, logging.WARNING))
    timeout = float(os.getenv("LLM_TIMEOUT", "180"))
    max_steps = int(os.getenv("LLM_MAX_STEPS", "20"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "60000"))
    max_subcall_raw = os.getenv("LLM_MAX_SUBCALL_TOKENS", "").strip()
    max_subcall_tokens = int(max_subcall_raw) if max_subcall_raw else None
    # Disable require_subcall to allow Phase 0 optimization (0 subcalls when extract_after succeeds)
    require_subcall = os.getenv("LLM_REQUIRE_SUBCALL", "0") == "1"
    fallback_enabled = os.getenv("RLM_FALLBACK", "1") == "1"
    invalid_limit = int(os.getenv("RLM_INVALID_LIMIT", "1"))
    repl_error_limit = int(os.getenv("RLM_REPL_ERROR_LIMIT", "2"))
    subcall_guard_raw = os.getenv("RLM_SUBCALL_GUARD_STEPS", "").strip()
    subcall_guard_steps = int(subcall_guard_raw) if subcall_guard_raw else None

    sizes_raw = os.getenv("RLM_CONTEXT_SIZES", "5,30,120")
    doc_counts = [int(item) for item in sizes_raw.split(",") if item.strip()]
    lines_per_doc = int(os.getenv("RLM_LINES_PER_DOC", "8"))
    key_doc_ratio = float(os.getenv("RLM_KEY_DOC_RATIO", "0.8"))
    separator = os.getenv("RLM_DOC_SEPARATOR", "\n\n---\n\n")
    chunk_size = int(os.getenv("RLM_CHUNK_SIZE", "2000"))

    baseline_max_raw = os.getenv("BASELINE_MAX_CHARS", "8000").strip()
    baseline_max_chars = int(baseline_max_raw) if baseline_max_raw else None
    if baseline_max_chars is not None and baseline_max_chars <= 0:
        baseline_max_chars = None

    parallel_subcalls = os.getenv("RLM_PARALLEL_SUBCALLS", "1") == "1"
    max_concurrent_subcalls = int(os.getenv("RLM_MAX_CONCURRENT_SUBCALLS", "4"))
    max_subcalls_raw = os.getenv("RLM_MAX_SUBCALLS", "").strip()
    max_subcalls_override = int(max_subcalls_raw) if max_subcalls_raw else None

    use_subcall_adapter = os.getenv("RLM_USE_SUBCALL_ADAPTER", "1") == "1"
    subcall_model = os.getenv("LLM_SUBCALL_MODEL", "").strip()
    if subcall_model.lower() in {"", "same", "auto"}:
        subcall_model = ""
    subcall_base_url = os.getenv("LLM_SUBCALL_BASE_URL", base_url)
    subcall_timeout = float(os.getenv("LLM_SUBCALL_TIMEOUT", str(timeout)))
    export_enabled = os.getenv("RLM_EXPORT", "0") == "1"
    export_format = os.getenv("RLM_EXPORT_FORMAT", "md").strip().lower()
    export_path = os.getenv("RLM_EXPORT_PATH", "").strip()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if export_enabled and not export_path:
        if export_format == "md":
            export_path = os.path.join(
                "examples",
                "exports",
                f"rlm_vs_baseline_results_{timestamp}.md",
            )
        else:
            export_path = os.path.join(
                "examples",
                "exports",
                f"rlm_vs_baseline_results_{timestamp}.jsonl",
            )
    elif export_enabled:
        if "{timestamp}" in export_path:
            export_path = export_path.replace("{timestamp}", timestamp)
        else:
            base, ext = os.path.splitext(export_path)
            if not ext:
                ext = ".md" if export_format == "md" else ".jsonl"
            export_path = f"{base}_{timestamp}{ext}"

    transcript_buffer = None
    tee_stream = None
    if export_enabled and export_format == "md":
        transcript_buffer = io.StringIO()
        tee_stream = _Tee(sys.stdout, transcript_buffer)
        sys.stdout = tee_stream

    context_variants: list[tuple[int, Context]] = []
    for doc_count in doc_counts:
        context = build_context(
            doc_count,
            lines_per_doc,
            key_doc_ratio=key_doc_ratio,
            separator=separator,
        )
        context_variants.append((doc_count, context))

    sub_question = (
        "Extract the value after the literal text 'The key term is:'. "
        "Return only the value (e.g., oolong). If not present, return NO_ANSWER."
    )
    baseline_query = "What is the key term defined by 'The key term is:'?"

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

    for model in models:
        adapter = GenericChatAdapter(
            base_url=base_url, model=model, timeout=timeout, api_key=api_key
        )
        subcall_adapter = None
        subcall_label = "disabled"
        if use_subcall_adapter:
            if subcall_model:
                subcall_adapter = GenericChatAdapter(
                    base_url=subcall_base_url,
                    model=subcall_model,
                    timeout=subcall_timeout,
                    api_key=api_key,
                )
                subcall_label = subcall_model
            else:
                subcall_adapter = adapter
                subcall_label = "same as root"

        print("=" * 60)
        print(f"Model: {model}")
        print(f"Subcall model: {subcall_label}")
        print(f"Parallel subcalls: {parallel_subcalls} max_workers={max_concurrent_subcalls}")
        if baseline_max_chars is not None:
            print(f"Baseline max chars: {baseline_max_chars}")
        summary_rows: list[dict] = []

        for doc_count, context in context_variants:
            # B) Deterministic phase 0: ALWAYS try extract_after FIRST
            # This reduces ~30 subcalls down to potentially 0-1
            query = (
                "Find the key term defined by 'The key term is:'. "
                "IMPORTANT: First try key = extract_after('The key term is:'). "
                "Only if key is None, then use "
            )
            if parallel_subcalls:
                query += (
                    f"ask_chunks with chunk_size={chunk_size} and parallel=True, "
                    "then pick_first_answer. "
                )
            else:
                query += f"ask_chunks_first with chunk_size={chunk_size}. "
            query += "Set key and reply as FINAL_VAR: key."
            query += f"\n\nSub-question: {sub_question}"

            fallback_code = None
            if fallback_enabled:
                # B) Deterministic phase 0: extract_after FIRST, subcalls only if needed
                # If extract_after finds it: 0 subcalls (Phase 0 optimization!)
                # If extract_after fails: Full chunking approach with subcalls
                fallback_code = "key = extract_after('The key term is:')\nif key is None:\n"
                if parallel_subcalls:
                    fallback_code += (
                        f"    chunks = ctx.chunk({chunk_size})\n"
                        f"    answers = ask_chunks({sub_question!r}, chunks, parallel=True)\n"
                        "    key = pick_first_answer(answers)\n"
                    )
                else:
                    fallback_code += (
                        f"    key = ask_chunks_first({sub_question!r}, ctx.chunk({chunk_size}))\n"
                    )

            estimated_chunks = max(1, (context.len_chars() + chunk_size - 1) // chunk_size)
            max_subcalls = max_subcalls_override or max(12, estimated_chunks + 2)

            rlm_result = run_rlm(
                adapter,
                subcall_adapter,
                model,
                context,
                query,
                require_subcall=require_subcall,
                max_steps=max_steps,
                max_tokens=max_tokens,
                max_subcall_tokens=max_subcall_tokens,
                max_subcalls=max_subcalls,
                fallback_enabled=fallback_enabled,
                fallback_code=fallback_code,
                invalid_limit=invalid_limit if fallback_enabled else None,
                repl_error_limit=repl_error_limit if fallback_enabled else None,
                subcall_guard_steps=subcall_guard_steps,
                parallel_subcalls=parallel_subcalls,
                max_concurrent_subcalls=max_concurrent_subcalls,
            )
            baseline_result = run_baseline(
                adapter,
                context,
                baseline_query,
                max_tokens=256,
                max_context_chars=baseline_max_chars,
            )

            print(
                f"Context: docs={doc_count} lines/doc={lines_per_doc} chars={context.len_chars()}"
            )
            print(
                f"  baseline: {baseline_result['output']}"
                f"  elapsed={baseline_result['elapsed']:.2f}s"
                f" tokens={baseline_result['tokens']} calls={baseline_result['calls']}"
                f" used_chars={baseline_result['used_chars']}"
                f" truncated={baseline_result['truncated']}"
                f" contains_key={baseline_result['contains_key']}"
            )
            print(
                f"  rlm: {rlm_result['output']}"
                f"  elapsed={rlm_result['elapsed']:.2f}s"
                f" tokens={rlm_result['tokens']} calls={rlm_result['calls']}"
                f" steps={rlm_result['steps']}"
            )

            # Display detailed metrics if available
            if rlm_result.get("metrics"):
                m = rlm_result["metrics"]
                tb = m["token_breakdown"]
                print(
                    f"    └─ tokens: prompt={tb['prompt_tokens']} "
                    f"completion={tb['completion_tokens']} "
                    f"cache_read={tb['cache_read_tokens']}"
                )
                print(
                    f"    └─ cache: {m['cache']['hits']}/{m['cache']['total_steps']} hits "
                    f"({m['cache']['hit_rate']:.1%})"
                )
                print(
                    f"    └─ subcalls: {m['subcalls']['count']} calls, "
                    f"{m['subcalls']['total_tokens']} tokens "
                    f"(avg {m['subcalls']['avg_tokens_per_subcall']}/call)"
                )
                print(f"    └─ repl: {m['repl']['executions']} execs, {m['repl']['errors']} errors")

            winner = pick_winner(baseline_result, rlm_result)
            print(f"  winner: {winner}")

            # Display trajectory visualization if enabled (similar to MIT RLM paper Appendix B)
            show_trajectory = os.getenv("SHOW_TRAJECTORY", "0") == "1"
            if show_trajectory and rlm_result.get("trace"):
                print("\n")
                trajectory_output = format_trajectory(
                    rlm_result["trace"], title=f"RLM Trajectory (docs={doc_count})"
                )
                print(trajectory_output)
                print("\n")

            summary_rows.append(
                {
                    "docs": doc_count,
                    "chars": context.len_chars(),
                    "baseline_tokens": baseline_result["tokens"],
                    "baseline_elapsed": baseline_result["elapsed"],
                    "baseline_truncated": baseline_result["truncated"],
                    "baseline_ok": is_success(baseline_result["output"]),
                    "rlm_tokens": rlm_result["tokens"],
                    "rlm_elapsed": rlm_result["elapsed"],
                    "rlm_ok": is_success(rlm_result["output"]),
                    "rlm_steps": rlm_result.get("steps", {}),
                    "rlm_metrics": rlm_result.get("metrics"),
                    "winner": winner,
                }
            )

        if summary_rows:
            # Display ASCII plot showing crossover point
            crossover_plot = plot_crossover_ascii(summary_rows)
            print(crossover_plot)

            # Display enhanced results table with crossover analysis
            results_table = format_results_table(summary_rows)
            print(results_table)
            if export_path and export_format == "jsonl":
                payload = {
                    "model": model,
                    "subcall_model": subcall_label,
                    "timestamp": time.time(),
                    "baseline_max_chars": baseline_max_chars,
                    "parallel_subcalls": parallel_subcalls,
                    "max_concurrent_subcalls": max_concurrent_subcalls,
                    "context_sizes": doc_counts,
                    "lines_per_doc": lines_per_doc,
                    "key_doc_ratio": key_doc_ratio,
                    "summary_rows": summary_rows,
                }
                export_results(export_path, payload)
                print(f"Exported results to {export_path}")

    if tee_stream is not None:
        sys.stdout = sys.__stdout__
    if export_enabled and export_format == "md" and export_path and transcript_buffer:
        export_results_md(export_path, transcript_buffer.getvalue())
        print(f"Exported results to {export_path}")


if __name__ == "__main__":
    main()
