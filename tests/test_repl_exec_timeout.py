"""Tests for the per-exec CPU-time timeout in PythonREPL.

The guard interrupts runaway pure-Python compute (e.g. O(n^3) loops) via an
ITIMER_PROF / SIGPROF timer on the Unix main thread. It measures CPU time, not
wall-clock, so time blocked in I/O (e.g. an LLM subcall made from within the
cell) does not count against it. It no-ops where SIGPROF cannot be delivered.
"""

from __future__ import annotations

import signal
import threading
import time

import pytest

from pyrlm_runtime.env import PythonREPL

HAS_SIGPROF = (
    hasattr(signal, "SIGPROF")
    and hasattr(signal, "setitimer")
    and threading.current_thread() is threading.main_thread()
)

needs_sigprof = pytest.mark.skipif(
    not HAS_SIGPROF, reason="SIGPROF/setitimer not available on this platform/thread"
)


@needs_sigprof
def test_runaway_loop_is_aborted() -> None:
    """An infinite pure-Python loop burns CPU and is interrupted as an error."""
    repl = PythonREPL(exec_timeout=1.0)
    result = repl.exec("while True:\n    pass")
    assert result.error is not None
    assert "exceeded" in result.error
    assert "TimeoutError" in result.error


def test_fast_code_is_not_affected() -> None:
    """Normal fast code completes without tripping the timeout."""
    repl = PythonREPL(exec_timeout=5.0)
    result = repl.exec("x = 21 * 2\nprint(x)")
    assert result.error is None
    assert "42" in result.stdout


@needs_sigprof
def test_io_wait_inside_exec_does_not_trip_cpu_guard() -> None:
    """Regression: a subcall that blocks on I/O (LLM latency) inside the cell
    must NOT be killed by the exec guard.

    The guard is a *CPU-time* timer, so wall-clock time spent blocked (here a
    sleep standing in for ``ask``/``llm_query`` latency) does not advance it.
    With the old wall-clock (ITIMER_REAL) guard this exec would trip after
    ``exec_timeout`` mid-subcall; with the CPU-time guard it completes.
    """
    repl = PythonREPL(exec_timeout=0.2)

    def slow_subcall() -> str:
        time.sleep(0.6)  # LLM latency >> exec_timeout, but ~zero CPU
        return "answer"

    repl.set("ask", slow_subcall)
    result = repl.exec("print(ask())")
    assert result.error is None, result.error
    assert "answer" in result.stdout


@needs_sigprof
def test_timer_is_disarmed_after_exec() -> None:
    """The guard leaves no armed ITIMER_PROF behind after a normal exec."""
    repl = PythonREPL(exec_timeout=5.0)
    repl.exec("y = 1")
    assert signal.getitimer(signal.ITIMER_PROF)[0] == 0.0


@needs_sigprof
def test_nested_guard_does_not_cancel_outer_timer() -> None:
    """A nested exec must not disarm an already-armed outer ITIMER_PROF timer."""
    received: list[str] = []

    def _on_prof(signum: int, frame: object) -> None:
        received.append("outer fired")

    old = signal.signal(signal.SIGPROF, _on_prof)
    signal.setitimer(signal.ITIMER_PROF, 5.0)
    try:
        repl = PythonREPL(exec_timeout=1.0)
        repl.exec("z = 2 + 2")
        # Outer timer must still be armed (nested guard refrained from re-arming).
        remaining = signal.getitimer(signal.ITIMER_PROF)[0]
        assert remaining > 0.0
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
        signal.signal(signal.SIGPROF, old)


@needs_sigprof
def test_repl_guard_does_not_touch_wall_clock_timer() -> None:
    """The REPL guard uses ITIMER_PROF, so it composes with a Vertex-style
    ITIMER_REAL wall-clock guard armed higher up the stack without cancelling
    it."""
    old = signal.signal(signal.SIGALRM, lambda *a: None)
    signal.setitimer(signal.ITIMER_REAL, 30.0)
    try:
        repl = PythonREPL(exec_timeout=1.0)
        repl.exec("w = 5")
        # The outer wall-clock timer is untouched by the REPL's CPU-time guard.
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0.0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
