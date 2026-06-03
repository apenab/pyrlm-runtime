"""Tests for the loop-collapse guard.

When the model emits consecutive non-productive responses (no code, no FINAL,
no subcall), the loop aborts with the sentinel ``LOOP_COLLAPSED`` instead of
burning all of max_steps on prose.
"""

from __future__ import annotations

from pyrlm_runtime.adapters import FakeAdapter
from pyrlm_runtime.context import Context
from pyrlm_runtime.policy import Policy
from pyrlm_runtime.rlm import RLM


def test_loop_collapse_aborts_on_unproductive_streak() -> None:
    adapter = FakeAdapter()
    # Empty pattern matches every prompt → always returns prose (never code/FINAL).
    adapter.add_rule("", "I am still thinking about how to approach this.")

    rlm = RLM(adapter=adapter, loop_collapse_limit=3)
    out, trace = rlm.run("Solve it.", Context.from_text("ctx"))

    assert out == "LOOP_COLLAPSED"
    # Aborted well before max_steps (3 unproductive iters, not 40).
    root_calls = [s for s in trace.steps if s.kind == "root_call"]
    assert len(root_calls) <= 4


def test_loop_collapse_disabled_does_not_emit_sentinel() -> None:
    adapter = FakeAdapter()
    adapter.add_rule("", "Still thinking.")
    rlm = RLM(adapter=adapter, loop_collapse_limit=None, policy=Policy(max_steps=2))
    out, _ = rlm.run("Solve it.", Context.from_text("ctx"))
    assert out != "LOOP_COLLAPSED"


def test_loop_collapse_aborts_on_empty_response_streak() -> None:
    """Regression: a streak of *empty* responses (no code, no FINAL, no subcall)
    must also trip the collapse guard. Before the fix the abort was only checked
    in the non-code-prose branch, so an empty-response loop burned all of
    max_steps without ever emitting the sentinel."""
    adapter = FakeAdapter()
    adapter.add_rule("", "")  # every turn returns an empty completion

    rlm = RLM(adapter=adapter, loop_collapse_limit=3, policy=Policy(max_steps=40))
    out, trace = rlm.run("Solve it.", Context.from_text("ctx"))

    assert out == "LOOP_COLLAPSED"
    root_calls = [s for s in trace.steps if s.kind == "root_call"]
    assert len(root_calls) <= 4  # aborted at the streak limit, not at max_steps
