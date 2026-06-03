"""Tests for conversation-history compaction in the RLM loop."""

from __future__ import annotations

import pytest

from pyrlm_runtime.adapters import FakeAdapter
from pyrlm_runtime.context import Context
from pyrlm_runtime.policy import Policy
from pyrlm_runtime.rlm import RLM


def _is_summary_call(messages: list[dict]) -> bool:
    return any("Summarize your progress" in (m.get("content") or "") for m in messages)


def test_compaction_triggers_and_records_step() -> None:
    """A low threshold forces a compaction, which appears as a trace step."""
    adapter = FakeAdapter(script=["FINAL: done"])
    # The compaction summary request is answered by a rule (order-independent).
    adapter.add_rule("Summarize your progress", "Compacted summary of progress.")

    rlm = RLM(
        adapter=adapter,
        compaction=True,
        compaction_threshold_tokens=10,  # tiny → triggers immediately
        conversation_history=True,
    )
    out, trace = rlm.run("What is the answer?", Context.from_text("some context here"))

    assert out == "done"
    compaction_steps = [s for s in trace.steps if s.kind == "compaction"]
    assert len(compaction_steps) >= 1
    # The trace step logs the effective threshold, not a stale config value.
    assert "threshold=10" in (compaction_steps[0].prompt_summary or "")


def test_no_compaction_when_disabled() -> None:
    """With compaction off, no compaction step is ever recorded."""
    adapter = FakeAdapter(script=["FINAL: done"])
    rlm = RLM(adapter=adapter, compaction=False, conversation_history=True)
    _, trace = rlm.run("Q?", Context.from_text("ctx"))
    assert not any(s.kind == "compaction" for s in trace.steps)


def test_compaction_requires_conversation_history() -> None:
    """compaction=True with conversation_history=False is rejected at construction."""
    with pytest.raises(ValueError, match="conversation_history"):
        RLM(adapter=FakeAdapter(), compaction=True, conversation_history=False)


class _RaisingSummaryAdapter(FakeAdapter):
    """FakeAdapter that fails only on the compaction summary call."""

    def complete(self, messages, **kwargs):  # type: ignore[override]
        if _is_summary_call(messages):
            raise RuntimeError("summary backend down")
        return super().complete(messages, **kwargs)


def test_compaction_summary_failure_does_not_destroy_history() -> None:
    """Regression: a failed summary call must skip compaction (keep the
    trajectory), not commit a destructive replace. Before the fix the exception
    was swallowed into a '[compaction failed]' summary that REPLACED the whole
    history and recorded a compaction step; now the round is skipped and the run
    proceeds normally."""
    adapter = _RaisingSummaryAdapter(script=["FINAL: done"])
    rlm = RLM(
        adapter=adapter,
        compaction=True,
        compaction_threshold_tokens=10,  # tiny → would trigger every iteration
        conversation_history=True,
    )
    out, trace = rlm.run("What is the answer?", Context.from_text("some context"))

    assert out == "done"  # run completes despite the summary failure
    # No compaction step is recorded because the destructive replace was skipped.
    assert not any(s.kind == "compaction" for s in trace.steps)


def test_compaction_summary_tokens_counted_in_policy() -> None:
    """Regression: the summary call's tokens are charged to the policy budget,
    like any other root-level adapter call (previously they leaked uncounted)."""
    adapter = FakeAdapter(script=["FINAL: done"])
    adapter.add_rule("Summarize your progress", "Compacted summary of progress so far.")
    policy = Policy(max_total_tokens=10_000_000)

    rlm = RLM(
        adapter=adapter,
        compaction=True,
        compaction_threshold_tokens=10,
        conversation_history=True,
        policy=policy,
    )
    _, trace = rlm.run("What is the answer?", Context.from_text("some context"))

    compaction_steps = [s for s in trace.steps if s.kind == "compaction"]
    assert compaction_steps and compaction_steps[0].usage is not None
    summary_tokens = compaction_steps[0].usage.total_tokens
    assert summary_tokens > 0
    # The policy total includes both the root call(s) and the summary call.
    root_tokens = sum(
        s.usage.total_tokens for s in trace.steps if s.kind == "root_call" and s.usage
    )
    assert policy.total_tokens >= root_tokens + summary_tokens
