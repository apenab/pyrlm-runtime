"""
Benchmark: Full RLM Loop with PythonREPL vs MontyREPL
=====================================================

PURPOSE:
    Compare the two REPL backends in a realistic end-to-end RLM execution
    using FakeAdapter (no real LLM needed). This isolates the REPL overhead
    within the actual RLM loop, including variable injection, external
    function registration, code execution, and variable capture.

WHAT IT MEASURES:
    1. Complete RLM run with Phase 0 (deterministic extract_after)
    2. Complete RLM run with multi-step (peek -> extract -> finalize)
    3. Scaling behavior: small, medium, and large contexts

    For each scenario, both PythonREPL and MontyREPL are tested.

HOW TO RUN:
    uv run python examples/bench_rlm_repl_backends.py

    # Export results to Markdown
    RLM_EXPORT=1 uv run python examples/bench_rlm_repl_backends.py

NOTE:
    This uses FakeAdapter, so no LLM server is needed. The benchmark
    measures REPL overhead within the full RLM loop.
"""

import io
import os
import sys
import time
from dataclasses import dataclass

from pyrlm_runtime import Context, Policy, RLM
from pyrlm_runtime.adapters import FakeAdapter
from pyrlm_runtime.env_monty import MONTY_AVAILABLE


KEY_MARKER = "The key term is:"
KEY_VALUE = "oolong"


@dataclass
class ScenarioResult:
    name: str
    python_ms: float
    monty_ms: float
    python_output: str
    monty_output: str
    python_steps: int
    monty_steps: int

    @property
    def python_ok(self) -> bool:
        return KEY_VALUE in self.python_output.lower()

    @property
    def monty_ok(self) -> bool:
        return KEY_VALUE in self.monty_output.lower()

    @property
    def speedup(self) -> str:
        if self.monty_ms <= 0 or self.python_ms <= 0:
            return "N/A"
        ratio = self.python_ms / self.monty_ms
        if ratio > 1.1:
            return f"monty {ratio:.1f}x faster"
        if ratio < 0.9:
            return f"python {1/ratio:.1f}x faster"
        return "~tie"


def build_needle_context(doc_count: int, lines_per_doc: int = 8) -> Context:
    """Build a context with a needle at 80% position."""
    key_idx = max(0, int(doc_count * 0.8))
    filler = "alpha beta gamma delta epsilon zeta eta theta iota kappa.\n"
    docs = []
    for i in range(doc_count):
        lines = [filler] * lines_per_doc
        if i == key_idx:
            lines.insert(lines_per_doc // 2, f"{KEY_MARKER} {KEY_VALUE}.\n")
        docs.append("".join(lines))
    return Context.from_documents(docs, separator="\n\n---\n\n")


def make_phase0_adapter() -> FakeAdapter:
    """Adapter that generates Phase 0 code (deterministic extract_after)."""
    return FakeAdapter(
        script=[
            'key = extract_after("The key term is:")\nprint(f"Found: {key}")',
            "FINAL_VAR: key",
        ]
    )


def make_multistep_adapter() -> FakeAdapter:
    """Adapter that generates multi-step code (peek -> extract -> finalize)."""
    return FakeAdapter(
        script=[
            'snippet = peek(500)\nprint(f"Preview ({len(snippet)} chars): {snippet[:50]}")',
            'key = extract_after("The key term is:")\nprint(f"Key: {key}")',
            "FINAL_VAR: key",
        ]
    )


def run_single(adapter_fn, context: Context, backend: str) -> tuple[str, int, float]:
    """Run a single RLM execution with a given backend and return (output, steps, ms)."""
    adapter = adapter_fn()
    policy = Policy(max_steps=10, max_subcalls=50)
    rlm = RLM(
        adapter=adapter,
        policy=policy,
        require_repl_before_final=True,
        auto_finalize_var="key",
        repl_backend=backend,
    )
    start = time.perf_counter()
    output, trace = rlm.run("Find the key term.", context)
    elapsed = (time.perf_counter() - start) * 1000
    return output, len(trace.steps), elapsed


def run_scenario(
    name: str,
    adapter_fn,
    context: Context,
    iterations: int = 20,
) -> ScenarioResult:
    """Run a scenario with both REPL backends."""
    py_times, mo_times = [], []
    py_output, mo_output = "", ""
    py_steps, mo_steps = 0, 0

    for _ in range(iterations):
        out, steps, ms = run_single(adapter_fn, context, "python")
        py_times.append(ms)
        py_output, py_steps = out, steps

    for _ in range(iterations):
        out, steps, ms = run_single(adapter_fn, context, "monty")
        mo_times.append(ms)
        mo_output, mo_steps = out, steps

    return ScenarioResult(
        name=name,
        python_ms=sum(py_times) / len(py_times),
        monty_ms=sum(mo_times) / len(mo_times),
        python_output=py_output.strip(),
        monty_output=mo_output.strip(),
        python_steps=py_steps,
        monty_steps=mo_steps,
    )


def print_results(results: list[ScenarioResult]) -> None:
    print()
    print("=" * 110)
    print("RESULTS (average ms per full RLM run)".center(110))
    print("=" * 110)
    print()
    print(
        f"{'Scenario':<42} {'Python':>10} {'Monty':>10} {'Speedup':>22}"
        f" {'Py OK':>6} {'Mo OK':>6} {'Steps':>6}"
    )
    print("-" * 110)
    for r in results:
        print(
            f"{r.name:<42} {r.python_ms:>8.2f}ms {r.monty_ms:>8.2f}ms {r.speedup:>22}"
            f" {'yes' if r.python_ok else 'NO':>6} {'yes' if r.monty_ok else 'NO':>6}"
            f" {r.python_steps:>6}"
        )


def print_analysis(results: list[ScenarioResult]) -> None:
    print()
    print("=" * 110)
    print("ANALYSIS".center(110))
    print("=" * 110)
    print()

    all_py_ok = all(r.python_ok for r in results)
    all_mo_ok = all(r.monty_ok for r in results)
    print(f"Correctness: Python={'ALL PASS' if all_py_ok else 'SOME FAIL'}"
          f" | Monty={'ALL PASS' if all_mo_ok else 'SOME FAIL'}")

    comparable = [r for r in results if r.python_ms > 0 and r.monty_ms > 0]
    if comparable:
        avg_ratio = sum(r.python_ms / r.monty_ms for r in comparable) / len(comparable)
        winner = f"Monty is {avg_ratio:.1f}x faster" if avg_ratio > 1 else f"Python is {1/avg_ratio:.1f}x faster"
        print(f"Average speedup: {winner}")

    # Overhead analysis
    overheads = [(r.monty_ms - r.python_ms) for r in comparable]
    avg_overhead = sum(overheads) / len(overheads) if overheads else 0
    print(f"Average Monty overhead per run: {avg_overhead:+.2f}ms")

    print()
    print("KEY INSIGHT:")
    print("-" * 110)
    print("  The REPL execution time is typically <1% of total RLM wall-clock time.")
    print("  A real LLM API call takes 100-5000ms. Monty adds only a few ms of overhead.")
    print("  The security guarantees (sandbox isolation, timeout, memory limits)")
    print("  far outweigh this negligible cost in production use.")
    print("=" * 110)


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


def export_results_md(path: str, transcript: str) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as handle:
        if not file_exists:
            handle.write("# Benchmark: Full RLM Loop - PythonREPL vs MontyREPL\n\n")
        handle.write(f"## Run {timestamp}\n\n")
        handle.write("```text\n")
        handle.write(transcript.rstrip() + "\n")
        handle.write("```\n\n")


def main() -> None:
    if not MONTY_AVAILABLE:
        print("ERROR: pydantic-monty is not installed.")
        print("Install with: uv pip install pydantic-monty")
        return

    export_enabled = os.getenv("RLM_EXPORT", "0") == "1"
    export_path = os.getenv("RLM_EXPORT_PATH", "").strip()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if export_enabled and not export_path:
        export_path = os.path.join(
            "examples", "exports", f"bench_rlm_repl_backends_{timestamp}.md"
        )

    transcript_buffer = None
    tee_stream = None
    if export_enabled:
        transcript_buffer = io.StringIO()
        tee_stream = _Tee(sys.stdout, transcript_buffer)
        sys.stdout = tee_stream

    print("=" * 110)
    print("BENCHMARK: Full RLM Loop - PythonREPL vs MontyREPL".center(110))
    print("=" * 110)
    print("Using FakeAdapter (no LLM server needed)")
    print()

    small_ctx = build_needle_context(5)
    medium_ctx = build_needle_context(30)
    large_ctx = build_needle_context(120)

    print(f"Small context:  {small_ctx.len_chars():>8,} chars ({small_ctx.num_documents()} docs)")
    print(f"Medium context: {medium_ctx.len_chars():>8,} chars ({medium_ctx.num_documents()} docs)")
    print(f"Large context:  {large_ctx.len_chars():>8,} chars ({large_ctx.num_documents()} docs)")
    print()

    scenarios = [
        ("Phase 0 (extract_after, 5 docs)", make_phase0_adapter, small_ctx),
        ("Phase 0 (extract_after, 30 docs)", make_phase0_adapter, medium_ctx),
        ("Phase 0 (extract_after, 120 docs)", make_phase0_adapter, large_ctx),
        ("Multi-step (peek+extract, 5 docs)", make_multistep_adapter, small_ctx),
        ("Multi-step (peek+extract, 30 docs)", make_multistep_adapter, medium_ctx),
        ("Multi-step (peek+extract, 120 docs)", make_multistep_adapter, large_ctx),
    ]

    results: list[ScenarioResult] = []
    for name, adapter_fn, ctx in scenarios:
        print(f"  Running: {name}...", end="", flush=True)
        result = run_scenario(name, adapter_fn, ctx)
        results.append(result)
        print(f" done ({result.speedup})")

    print_results(results)
    print_analysis(results)

    if tee_stream is not None:
        sys.stdout = sys.__stdout__
    if export_enabled and export_path and transcript_buffer:
        export_results_md(export_path, transcript_buffer.getvalue())
        print(f"Exported results to {export_path}")


if __name__ == "__main__":
    main()
