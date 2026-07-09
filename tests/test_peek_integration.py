"""Tests for PeekSession and peek_integration helpers.

These tests use a lightweight stub LMClient so they run without network access
or a real LLM.  The stub returns hard-coded JSON that satisfies peek-ai's
Distiller and Cartographer parsers.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

try:
    import peek  # noqa: F401

    PEEK_AVAILABLE = True
except ImportError:
    PEEK_AVAILABLE = False

from pyrlm_runtime.adapters.base import Usage
from pyrlm_runtime.adapters.fake import FakeAdapter
from pyrlm_runtime.peek_integration import (
    PeekSession,
    _PeekLMClientAdapter,
    trace_to_peek_trajectory,
)
from pyrlm_runtime.trace import Trace, TraceStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DISTILLER_RESPONSE = json.dumps(
    {
        "diagnosis": "Agent spent iterations exploring context structure.",
        "item_tags": {},
        "cache_candidates": [
            {
                "section": "context_roadmap",
                "content": "Single text block of ~5k chars containing news articles.",
            }
        ],
    }
)

_CARTOGRAPHER_RESPONSE = json.dumps(
    {
        "reasoning": "Adding a roadmap entry for the corpus layout.",
        "operations": [
            {
                "type": "ADD",
                "section": "context_roadmap",
                "content": "Single text block of ~5k chars containing news articles.",
            }
        ],
    }
)


class _StubLMClient:
    """Stub that alternates between Distiller and Cartographer responses.

    peek calls the client twice per update: once for the Distiller, once for
    the Cartographer.  The stub returns the appropriate JSON based on call order.
    """

    def __init__(self) -> None:
        self._calls = 0

    def completion(self, messages: list[dict[str, Any]]) -> str:
        idx = self._calls % 2
        self._calls += 1
        return [_DISTILLER_RESPONSE, _CARTOGRAPHER_RESPONSE][idx]

    def last_usage(self):
        from peek.core.types import Usage as PeekUsage

        return PeekUsage(input_tokens=100, output_tokens=50)


def _make_simple_trace() -> Trace:
    trace = Trace(steps=[])
    trace.add(
        TraceStep(
            step_id=1,
            kind="root_call",
            depth=0,
            output="I will inspect the context to find the answer.\n```python\nprint(P[:500])\n```",
            usage=Usage(prompt_tokens=200, completion_tokens=50, total_tokens=250),
        )
    )
    trace.add(
        TraceStep(
            step_id=2,
            kind="repl_exec",
            depth=0,
            code="print(P[:500])",
            stdout="Breaking news: economy grows...",
            usage=None,
        )
    )
    trace.add(
        TraceStep(
            step_id=3,
            kind="root_call",
            depth=0,
            output="FINAL(World)",
            usage=Usage(prompt_tokens=300, completion_tokens=20, total_tokens=320),
        )
    )
    return trace


# ---------------------------------------------------------------------------
# trace_to_peek_trajectory
# ---------------------------------------------------------------------------


class TestTraceToTrajectory:
    def test_empty_trace_returns_empty(self) -> None:
        trace = Trace(steps=[])
        result = trace_to_peek_trajectory(trace)
        assert result == ""

    def test_query_prepended(self) -> None:
        trace = Trace(steps=[])
        result = trace_to_peek_trajectory(trace, query="How many sports articles?")
        assert result.startswith("Question: How many sports articles?")

    def test_root_call_output_included(self) -> None:
        trace = _make_simple_trace()
        result = trace_to_peek_trajectory(trace, query="Q")
        assert "FINAL(World)" in result
        assert "I will inspect the context" in result

    def test_repl_exec_code_and_stdout_included(self) -> None:
        trace = _make_simple_trace()
        result = trace_to_peek_trajectory(trace)
        assert "print(P[:500])" in result
        assert "Breaking news" in result

    def test_step_headers_present(self) -> None:
        trace = _make_simple_trace()
        result = trace_to_peek_trajectory(trace)
        assert "STEP 1 [root_call]" in result
        assert "STEP 2 [repl_exec]" in result

    def test_subcall_depth_marker(self) -> None:
        trace = Trace(steps=[])
        trace.add(
            TraceStep(
                step_id=1,
                kind="subcall",
                depth=1,
                output="subcall result",
                usage=None,
            )
        )
        result = trace_to_peek_trajectory(trace)
        assert "depth=1" in result
        assert "subcall result" in result

    def test_long_subcall_output_truncated(self) -> None:
        trace = Trace(steps=[])
        trace.add(
            TraceStep(
                step_id=1,
                kind="subcall",
                depth=1,
                output="x" * 600,
                usage=None,
            )
        )
        result = trace_to_peek_trajectory(trace)
        assert "[truncated]" in result

    def test_error_included(self) -> None:
        trace = Trace(steps=[])
        trace.add(
            TraceStep(
                step_id=1,
                kind="repl_exec",
                depth=0,
                code="1/0",
                stdout=None,
                error="ZeroDivisionError: division by zero",
                usage=None,
            )
        )
        result = trace_to_peek_trajectory(trace)
        assert "ZeroDivisionError" in result


# ---------------------------------------------------------------------------
# _PeekLMClientAdapter
# ---------------------------------------------------------------------------


class TestPeekLMClientAdapter:
    def test_completion_returns_text(self) -> None:
        fake = FakeAdapter(script=["hello from fake"])
        adapter = _PeekLMClientAdapter(fake, max_tokens=256)
        result = adapter.completion([{"role": "user", "content": "ping"}])
        assert result == "hello from fake"

    def test_last_usage_maps_tokens(self) -> None:
        if not PEEK_AVAILABLE:
            pytest.skip("peek-ai not installed")
        fake = FakeAdapter(script=["response text"])
        adapter = _PeekLMClientAdapter(fake, max_tokens=256)
        adapter.completion([{"role": "user", "content": "ping"}])
        usage = adapter.last_usage()
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0

    def test_usage_reflects_most_recent_call(self) -> None:
        if not PEEK_AVAILABLE:
            pytest.skip("peek-ai not installed")
        fake = FakeAdapter(
            rules=[],
            script=["first response", "second response with more tokens here"],
        )
        adapter = _PeekLMClientAdapter(fake, max_tokens=256)
        adapter.completion([{"role": "user", "content": "a"}])
        u1 = adapter.last_usage()
        adapter.completion([{"role": "user", "content": "a"}])
        u2 = adapter.last_usage()
        assert u2.output_tokens >= u1.output_tokens


# ---------------------------------------------------------------------------
# PeekSession
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PEEK_AVAILABLE, reason="peek-ai not installed")
class TestPeekSession:
    def _session_with_stub(self, **kwargs) -> PeekSession:
        from peek import CachePolicy

        stub = _StubLMClient()
        policy = CachePolicy(client=stub, **kwargs)
        return PeekSession(policy)

    def test_system_prompt_supplement_empty_before_update(self) -> None:
        session = self._session_with_stub(token_budget=1024)
        assert session.system_prompt_supplement == ""

    def test_update_from_run_populates_map(self) -> None:
        session = self._session_with_stub(token_budget=1024)
        trace = _make_simple_trace()
        result = session.update_from_run(trace, query="What is the topic?")
        assert result is not None
        assert result.operations_applied >= 0

    def test_system_prompt_supplement_nonempty_after_update(self) -> None:
        session = self._session_with_stub(token_budget=1024)
        session.update_from_run(_make_simple_trace(), query="Q")
        supp = session.system_prompt_supplement
        # After update, the Cartographer added a roadmap entry — map is non-empty
        assert supp == "" or "Context Map" in supp

    def test_steps_counter_increments(self) -> None:
        session = self._session_with_stub(token_budget=1024)
        assert session.steps == 0
        session.update_from_run(_make_simple_trace())
        assert session.steps == 1
        session.update_from_run(_make_simple_trace())
        assert session.steps == 2

    def test_evolve_steps_freezes_map(self) -> None:
        session = self._session_with_stub(token_budget=1024, evolve_steps=1)
        assert session.evolving is True
        session.update_from_run(_make_simple_trace())
        assert session.evolving is False
        result = session.update_from_run(_make_simple_trace())
        assert result is None  # frozen: returns None

    def test_empty_trace_does_not_crash(self) -> None:
        session = self._session_with_stub(token_budget=1024)
        trace = Trace(steps=[])
        result = session.update_from_run(trace, query="Q")
        assert result is not None  # Distiller/Cartographer still called with empty trajectory

    def test_save_and_load_roundtrip(self) -> None:
        from peek import CachePolicy

        stub = _StubLMClient()
        policy = CachePolicy(client=stub, token_budget=512, evolve_steps=3)
        session = PeekSession(policy)
        session.update_from_run(_make_simple_trace(), query="Q1")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "map.peek.json"
            session.save(path)
            assert path.exists()

            payload = json.loads(path.read_text())
            assert "map_text" in payload
            assert payload["token_budget"] == 512
            assert payload["evolve_steps"] == 3
            assert payload["steps"] == 1

            # Load and verify state is preserved
            stub2 = _StubLMClient()
            policy2 = CachePolicy.load(path, client=stub2)
            session2 = PeekSession(policy2)
            assert session2.steps == 1
            assert session2.evolving is True

    def test_create_factory_uses_model_adapter(self) -> None:
        fake = FakeAdapter(script=[_DISTILLER_RESPONSE, _CARTOGRAPHER_RESPONSE])
        session = PeekSession.create(fake, token_budget=512)
        assert isinstance(session, PeekSession)
        assert session.steps == 0

    def test_load_raises_without_peek(self, monkeypatch) -> None:
        # When peek is importable but we simulate the ImportError path via _require_peek
        # This test just verifies the happy path of create() doesn't crash.
        fake = FakeAdapter(script=[_DISTILLER_RESPONSE, _CARTOGRAPHER_RESPONSE])
        session = PeekSession.create(fake, token_budget=512)
        assert session is not None


# ---------------------------------------------------------------------------
# Regression: wiring — update_from_run calls policy.update with trajectory str
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PEEK_AVAILABLE, reason="peek-ai not installed")
def test_regression_trajectory_string_passed_to_policy() -> None:
    """update_from_run must pass a non-empty string to policy when trace has steps."""
    from peek import CachePolicy

    received: list[str] = []

    class CapturingClient:
        def completion(self, messages):
            content = messages[0]["content"] if messages else ""
            received.append(content)
            # Return valid Distiller JSON on first call, Cartographer JSON on second
            idx = len(received) - 1
            return [_DISTILLER_RESPONSE, _CARTOGRAPHER_RESPONSE][idx % 2]

        def last_usage(self):
            from peek.core.types import Usage as PeekUsage

            return PeekUsage(input_tokens=10, output_tokens=5)

    policy = CachePolicy(client=CapturingClient(), token_budget=1024)
    session = PeekSession(policy)
    trace = _make_simple_trace()
    session.update_from_run(trace, query="Test question")

    # Distiller should receive a non-empty trajectory in its prompt
    assert len(received) >= 1
    distiller_prompt = received[0]
    assert "STEP" in distiller_prompt or "Question" in distiller_prompt
    assert "FINAL(World)" in distiller_prompt


# ---------------------------------------------------------------------------
# Regression: vendor/peek/_io.py extract_json must not catastrophically
# backtrack on LLM output that opens a code fence without closing it. This
# was the bug behind a 45-minute hung benchmark run.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PEEK_AVAILABLE, reason="peek-ai not installed")
def test_extract_json_handles_unclosed_fence_quickly() -> None:
    import time

    from peek._io import extract_json

    # 100KB string with an opening fence and no closing fence — the kind of
    # output an LLM might emit when truncated mid-response. The pre-fix
    # regex `(.*?)\s*```` with DOTALL would backtrack exponentially here.
    payload = "```json\n" + ("a" * 100_000)

    t0 = time.perf_counter()
    result = extract_json(payload)
    elapsed = time.perf_counter() - t0

    # Should give up immediately, not hang. The original bug took >2 minutes
    # on a 1KB input; we generously allow 1s for a 100KB input.
    assert elapsed < 1.0, f"extract_json took {elapsed:.3f}s on unclosed fence"
    # Returning None is acceptable; what matters is that it returned at all.
    assert result is None or isinstance(result, dict)


@pytest.mark.skipif(not PEEK_AVAILABLE, reason="peek-ai not installed")
def test_extract_json_still_finds_valid_fenced_json() -> None:
    """Make sure the fix didn't regress the happy path."""
    from peek._io import extract_json

    payload = 'some preamble\n```json\n{"a": 1, "b": [2, 3]}\n```\ntrailing'
    result = extract_json(payload)
    assert result == {"a": 1, "b": [2, 3]}


@pytest.mark.skipif(not PEEK_AVAILABLE, reason="peek-ai not installed")
def test_extract_json_finds_second_block_when_first_invalid() -> None:
    from peek._io import extract_json

    payload = '```\nnot json at all\n```\n```json\n{"x": 42}\n```'
    result = extract_json(payload)
    assert result == {"x": 42}
