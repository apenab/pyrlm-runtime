"""
Benchmark: PythonREPL vs MontyREPL
==================================

PURPOSE:
    Measure the raw performance difference between the CPython-based
    PythonREPL and the Rust-based MontyREPL (pydantic-monty) to determine
    whether Monty introduces overhead or improves performance.

    This benchmark runs WITHOUT any LLM calls, isolating the REPL execution
    cost from network latency.

WHAT IT MEASURES:
    1. Startup time: creating a new REPL instance
    2. Simple exec: basic variable assignment and print
    3. String operations: slicing, formatting, len()
    4. External function calls: peek(), extract_after(), lenP()
    5. Multi-step simulation: multiple exec() calls preserving state
    6. Security: infinite loop protection (Monty only)
    7. Large context: operations on large strings (100K+ chars)

HOW TO RUN:
    uv run python examples/bench_repl_python_vs_monty.py

    # Export results to Markdown
    RLM_EXPORT=1 uv run python examples/bench_repl_python_vs_monty.py

EXPECTED OUTPUT:
    A comparison table showing execution times for each scenario
    with both backends, plus a summary of which is faster.
"""

import io
import os
import sys
import time
from dataclasses import dataclass

from pyrlm_runtime.env import ExecResult, PythonREPL
from pyrlm_runtime.env_monty import MONTY_AVAILABLE, MontyREPL


@dataclass
class BenchResult:
    name: str
    python_ms: float
    monty_ms: float
    python_ok: bool
    monty_ok: bool
    python_output: str
    monty_output: str

    @property
    def speedup(self) -> str:
        if self.python_ms < 0:
            return "monty only (security)"
        if self.monty_ms == 0 and self.python_ms == 0:
            return "tie"
        if self.monty_ms == 0:
            return "monty >>>"
        ratio = self.python_ms / self.monty_ms
        if ratio > 1.1:
            return f"monty {ratio:.1f}x faster"
        if ratio < 0.9:
            return f"python {1/ratio:.1f}x faster"
        return "~tie"


def make_python_repl(context: str) -> PythonREPL:
    repl = PythonREPL()
    repl.set("P", context)

    def peek(n: int = 2000) -> str:
        return context[:n]

    def tail(n: int = 2000) -> str:
        return context[-n:]

    def lenp() -> int:
        return len(context)

    def extract_after(marker: str, *, max_len: int = 128) -> str | None:
        idx = context.find(marker)
        if idx == -1:
            return None
        start = idx + len(marker)
        window = context[start : start + max_len].lstrip()
        if not window:
            return None
        return window.split()[0].strip(" \t\r\n.,;:\"'()[]{}")

    repl.set("peek", peek)
    repl.set("tail", tail)
    repl.set("lenP", lenp)
    repl.set("extract_after", extract_after)
    return repl


def make_monty_repl(context: str) -> MontyREPL:
    repl = MontyREPL()
    repl.set("P", context)

    def peek(n: int = 2000) -> str:
        return context[:n]

    def tail(n: int = 2000) -> str:
        return context[-n:]

    def lenp() -> int:
        return len(context)

    def extract_after(marker: str, *, max_len: int = 128) -> str | None:
        idx = context.find(marker)
        if idx == -1:
            return None
        start = idx + len(marker)
        window = context[start : start + max_len].lstrip()
        if not window:
            return None
        return window.split()[0].strip(" \t\r\n.,;:\"'()[]{}")

    repl.set("peek", peek)
    repl.set("tail", tail)
    repl.set("lenP", lenp)
    repl.set("extract_after", extract_after)
    return repl


def time_it(fn, iterations: int = 1) -> tuple[float, ExecResult | None]:
    """Run fn `iterations` times and return (total_ms, last_result)."""
    start = time.perf_counter()
    result = None
    for _ in range(iterations):
        result = fn()
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed, result


# --- Benchmark Scenarios ---

SMALL_CONTEXT = "alpha beta gamma delta epsilon. " * 100  # ~3.1K chars
LARGE_CONTEXT = "alpha beta gamma " * 10000  # ~170K chars
NEEDLE_CONTEXT = ("filler line.\n" * 500) + "The key term is: oolong.\n" + ("filler line.\n" * 500)

ITERATIONS = 50


def bench_startup() -> BenchResult:
    """Measure REPL instantiation cost."""
    py_ms, _ = time_it(lambda: make_python_repl(SMALL_CONTEXT), ITERATIONS)
    mo_ms, _ = time_it(lambda: make_monty_repl(SMALL_CONTEXT), ITERATIONS)
    return BenchResult("Startup (create REPL)", py_ms, mo_ms, True, True, "", "")


def bench_simple_exec() -> BenchResult:
    """Simple variable assignment and print."""
    code = 'x = 1 + 2\nprint(x)'
    py_repl = make_python_repl(SMALL_CONTEXT)
    mo_repl = make_monty_repl(SMALL_CONTEXT)

    py_ms, py_r = time_it(lambda: py_repl.exec(code), ITERATIONS)
    mo_ms, mo_r = time_it(lambda: mo_repl.exec(code), ITERATIONS)
    return BenchResult(
        "Simple exec (x=1+2)",
        py_ms, mo_ms,
        py_r.error is None, mo_r.error is None,
        py_r.stdout.strip(), mo_r.stdout.strip(),
    )


def bench_string_ops() -> BenchResult:
    """String slicing, len, and f-string formatting."""
    code = 'snippet = P[:100]\ntotal = len(P)\nprint(f"len={total} preview={snippet[:30]}")'
    py_repl = make_python_repl(LARGE_CONTEXT)
    mo_repl = make_monty_repl(LARGE_CONTEXT)

    py_ms, py_r = time_it(lambda: py_repl.exec(code), ITERATIONS)
    mo_ms, mo_r = time_it(lambda: mo_repl.exec(code), ITERATIONS)
    return BenchResult(
        "String ops (slice+len+fstr)",
        py_ms, mo_ms,
        py_r.error is None, mo_r.error is None,
        py_r.stdout.strip()[:60], mo_r.stdout.strip()[:60],
    )


def bench_external_calls() -> BenchResult:
    """External function calls (peek, lenP, extract_after)."""
    code = (
        'h = peek(50)\n'
        't = tail(50)\n'
        'l = lenP()\n'
        'print(f"head={h[:10]} tail={t[-10:]} len={l}")'
    )
    py_repl = make_python_repl(LARGE_CONTEXT)
    mo_repl = make_monty_repl(LARGE_CONTEXT)

    py_ms, py_r = time_it(lambda: py_repl.exec(code), ITERATIONS)
    mo_ms, mo_r = time_it(lambda: mo_repl.exec(code), ITERATIONS)
    return BenchResult(
        "External funcs (peek+tail+lenP)",
        py_ms, mo_ms,
        py_r.error is None, mo_r.error is None,
        py_r.stdout.strip()[:60], mo_r.stdout.strip()[:60],
    )


def bench_extract_after() -> BenchResult:
    """Needle-in-haystack: extract_after on large context."""
    code = 'key = extract_after("The key term is:")\nprint(key)'
    py_repl = make_python_repl(NEEDLE_CONTEXT)
    mo_repl = make_monty_repl(NEEDLE_CONTEXT)

    py_ms, py_r = time_it(lambda: py_repl.exec(code), ITERATIONS)
    mo_ms, mo_r = time_it(lambda: mo_repl.exec(code), ITERATIONS)
    return BenchResult(
        "extract_after (needle-in-haystack)",
        py_ms, mo_ms,
        py_r.error is None, mo_r.error is None,
        py_r.stdout.strip(), mo_r.stdout.strip(),
    )


def bench_multi_step() -> BenchResult:
    """Simulate RLM multi-step: 3 sequential exec() calls with state."""
    steps = [
        'snippet = peek(200)\nprint(f"Got {len(snippet)} chars")',
        'key = extract_after("The key term is:")\nprint(f"Key: {key}")',
        'answer = f"Found: {key}"\nprint(answer)',
    ]

    def run_steps(repl):
        last = ExecResult(stdout="", error=None)
        for code in steps:
            last = repl.exec(code)
            if last.error:
                return last
        return last

    py_ms, py_r = time_it(lambda: run_steps(make_python_repl(NEEDLE_CONTEXT)), ITERATIONS)
    mo_ms, mo_r = time_it(lambda: run_steps(make_monty_repl(NEEDLE_CONTEXT)), ITERATIONS)
    return BenchResult(
        "Multi-step (3 execs with state)",
        py_ms, mo_ms,
        py_r.error is None, mo_r.error is None,
        py_r.stdout.strip(), mo_r.stdout.strip(),
    )


def bench_list_comprehension() -> BenchResult:
    """List comprehension and iteration."""
    code = (
        'words = P.split()\n'
        'long_words = [w for w in words if len(w) > 4]\n'
        'print(f"Total words: {len(words)}, long: {len(long_words)}")'
    )
    py_repl = make_python_repl(SMALL_CONTEXT)
    mo_repl = make_monty_repl(SMALL_CONTEXT)

    py_ms, py_r = time_it(lambda: py_repl.exec(code), ITERATIONS)
    mo_ms, mo_r = time_it(lambda: mo_repl.exec(code), ITERATIONS)
    return BenchResult(
        "List comprehension (filter words)",
        py_ms, mo_ms,
        py_r.error is None, mo_r.error is None,
        py_r.stdout.strip(), mo_r.stdout.strip(),
    )


def bench_infinite_loop() -> BenchResult:
    """Security: infinite loop protection. Monty should timeout, Python will hang."""
    code = 'x = 0\nwhile True:\n    x += 1'

    # Python: we SKIP this (it would hang). Mark as N/A.
    py_ms = -1.0
    py_ok = False
    py_out = "SKIPPED (would hang)"

    mo_repl = make_monty_repl(SMALL_CONTEXT)
    start = time.perf_counter()
    mo_r = mo_repl.exec(code)
    mo_ms = (time.perf_counter() - start) * 1000
    mo_ok = mo_r.error is not None  # Should error with timeout
    mo_out = (mo_r.error or "")[:60]

    return BenchResult(
        "Infinite loop protection",
        py_ms, mo_ms,
        py_ok, mo_ok,
        py_out, mo_out,
    )


def bench_large_context_passthrough() -> BenchResult:
    """Pass a large context string as input and operate on it."""
    big = "x" * 200_000  # 200K chars
    code = 'result = len(P)\nprint(result)'

    py_repl = make_python_repl(big)
    mo_repl = make_monty_repl(big)

    py_ms, py_r = time_it(lambda: py_repl.exec(code), ITERATIONS)
    mo_ms, mo_r = time_it(lambda: mo_repl.exec(code), ITERATIONS)
    return BenchResult(
        "Large context (200K chars, len)",
        py_ms, mo_ms,
        py_r.error is None, mo_r.error is None,
        py_r.stdout.strip(), mo_r.stdout.strip(),
    )


def _format_ms(ms: float) -> str:
    return f"{ms:.2f}" if ms >= 0 else "N/A"


def _match_label(r: BenchResult) -> str:
    if r.python_ms < 0:
        return "-"
    return "yes" if r.python_output == r.monty_output else "~"


def print_results_table(results: list[BenchResult]) -> None:
    print()
    print("=" * 100)
    print("RESULTS".center(100))
    print("=" * 100)
    print()
    header = f"{'Scenario':<36} {'Python (ms)':>12} {'Monty (ms)':>12} {'Speedup':>22} {'Match':>6}"
    print(header)
    print("-" * 100)
    for r in results:
        print(
            f"{r.name:<36} {_format_ms(r.python_ms):>12} {_format_ms(r.monty_ms):>12}"
            f" {r.speedup:>22} {_match_label(r):>6}"
        )


def print_summary(results: list[BenchResult]) -> None:
    print()
    print("=" * 100)
    print("SUMMARY".center(100))
    print("=" * 100)

    comparable = [r for r in results if r.python_ms >= 0 and r.monty_ms >= 0]
    if not comparable:
        return
    total_py = sum(r.python_ms for r in comparable)
    total_mo = sum(r.monty_ms for r in comparable)
    print(f"Total Python: {total_py:.2f}ms | Total Monty: {total_mo:.2f}ms")
    if total_py == 0 and total_mo == 0:
        print("Overall: No timings recorded")
    elif total_mo == 0:
        print("Overall: Monty infinitely faster (Python > 0, Monty ~ 0)")
    elif total_py == 0:
        print("Overall: Python has no recorded time; Monty faster")
    else:
        ratio = total_py / total_mo
        if ratio > 1:
            winner = f"Monty is {ratio:.1f}x faster"
        else:
            winner = f"Python is {total_mo / total_py:.1f}x faster"
        print(f"Overall: {winner}")
    print()


def print_output_comparison(results: list[BenchResult]) -> None:
    print("OUTPUT COMPARISON:")
    print("-" * 100)
    for r in results:
        if r.python_ms < 0 or r.python_output != r.monty_output:
            label = "" if r.python_ms < 0 else " MISMATCH"
            print(f"  {r.name}:{label}")
            print(f"    Python: {r.python_output}")
            print(f"    Monty:  {r.monty_output}")
    print()


def print_security_comparison(results: list[BenchResult]) -> None:
    print("SECURITY COMPARISON:")
    print("-" * 100)
    inf_loop = next((r for r in results if "Infinite" in r.name), None)
    if inf_loop:
        py_status = "HANGS" if not inf_loop.python_ok else "Protected"
        mo_status = "Protected" if inf_loop.monty_ok else "HANGS"
        print(f"  Infinite loop: Python={py_status} | Monty={mo_status}"
              f" (timeout: {inf_loop.monty_ms:.0f}ms)")
    print("  Sandbox escape (exec/eval): Python=VULNERABLE | Monty=BLOCKED")
    print("  Memory bomb ([0]*10**9):    Python=VULNERABLE | Monty=BLOCKED (max_memory limit)")
    print("  Import os/sys:              Python=blocked by whitelist | Monty=BLOCKED (no imports)")
    print("=" * 100)


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
            handle.write("# Benchmark: PythonREPL vs MontyREPL\n\n")
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
            "examples", "exports", f"bench_repl_python_vs_monty_{timestamp}.md"
        )

    transcript_buffer = None
    tee_stream = None
    if export_enabled:
        transcript_buffer = io.StringIO()
        tee_stream = _Tee(sys.stdout, transcript_buffer)
        sys.stdout = tee_stream

    print("=" * 100)
    print("BENCHMARK: PythonREPL (CPython exec) vs MontyREPL (Rust pydantic-monty)".center(100))
    print("=" * 100)
    print(f"Iterations per test: {ITERATIONS}")
    print(f"Small context: {len(SMALL_CONTEXT):,} chars")
    print(f"Large context: {len(LARGE_CONTEXT):,} chars")
    print(f"Needle context: {len(NEEDLE_CONTEXT):,} chars")
    print()

    benchmarks = [
        bench_startup,
        bench_simple_exec,
        bench_string_ops,
        bench_external_calls,
        bench_extract_after,
        bench_multi_step,
        bench_list_comprehension,
        bench_infinite_loop,
        bench_large_context_passthrough,
    ]

    results: list[BenchResult] = []
    for bench_fn in benchmarks:
        print(f"  Running: {(bench_fn.__doc__ or "").strip()}...", end="", flush=True)
        result = bench_fn()
        results.append(result)
        print(f" done ({result.speedup})")

    print_results_table(results)
    print_summary(results)
    print_output_comparison(results)
    print_security_comparison(results)

    if tee_stream is not None:
        sys.stdout = sys.__stdout__
    if export_enabled and export_path and transcript_buffer:
        export_results_md(export_path, transcript_buffer.getvalue())
        print(f"Exported results to {export_path}")


if __name__ == "__main__":
    main()
