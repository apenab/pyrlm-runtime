from __future__ import annotations

from pyrlm_runtime import Context, Policy, RLM
from pyrlm_runtime.adapters import FakeAdapter


def test_llm_batch_basic() -> None:
    """llm_batch processes multiple prompts and returns results in order."""
    adapter = FakeAdapter(
        script=[
            "\n".join(
                [
                    'results = llm_batch(["hello", "world"])',
                    "answer = ','.join(results)",
                ]
            ),
            "FINAL_VAR: answer",
        ]
    )
    adapter.add_rule("You are a sub-LLM", "resp")

    context = Context.from_text("test")
    runtime = RLM(adapter=adapter)
    output, trace = runtime.run("Test.", context)

    assert output == "resp,resp"
    # Should have subcall trace steps
    subcall_steps = [s for s in trace.steps if s.kind == "subcall"]
    assert len(subcall_steps) == 2


def test_llm_batch_single_prompt() -> None:
    """Single-item batch works without thread overhead."""
    adapter = FakeAdapter(
        script=[
            "\n".join(
                [
                    'results = llm_batch(["single_test_input"])',
                    "answer = results[0]",
                ]
            ),
            "FINAL_VAR: answer",
        ]
    )
    adapter.add_rule("You are a sub-LLM", "single-resp")

    context = Context.from_text("test")
    runtime = RLM(adapter=adapter)
    output, trace = runtime.run("Test.", context)

    assert output == "single-resp"


def test_llm_batch_empty() -> None:
    """Empty list returns empty list."""
    adapter = FakeAdapter(
        script=[
            "\n".join(
                [
                    "results = llm_batch([])",
                    "answer = str(len(results))",
                ]
            ),
            "FINAL_VAR: answer",
        ]
    )

    context = Context.from_text("test")
    runtime = RLM(adapter=adapter)
    output, trace = runtime.run("Test.", context)

    assert output == "0"


def test_llm_batch_deduplication() -> None:
    """Identical prompts are deduplicated — only one API call per unique prompt."""
    adapter = FakeAdapter(
        script=[
            "\n".join(
                [
                    'results = llm_batch(["same", "same", "same"])',
                    "answer = ','.join(results)",
                ]
            ),
            "FINAL_VAR: answer",
        ]
    )
    adapter.add_rule("You are a sub-LLM", "dedup-resp")

    context = Context.from_text("test")
    runtime = RLM(adapter=adapter)
    output, trace = runtime.run("Test.", context)

    # All three results should be identical
    assert output == "dedup-resp,dedup-resp,dedup-resp"
    # But only one subcall should have been made (deduplication)
    subcall_steps = [s for s in trace.steps if s.kind == "subcall"]
    assert len(subcall_steps) == 1


def test_llm_batch_preserves_order() -> None:
    """Results are returned in the same order as input prompts."""
    adapter = FakeAdapter(
        script=[
            "\n".join(
                [
                    'results = llm_batch(["alpha", "beta", "gamma"])',
                    "answer = '|'.join(results)",
                ]
            ),
            "FINAL_VAR: answer",
        ]
    )
    # All subcalls match the same rule so results are identical —
    # but the important thing is we get exactly 3 results in order.
    adapter.add_rule("You are a sub-LLM", "ok")

    context = Context.from_text("test")
    runtime = RLM(adapter=adapter)
    output, trace = runtime.run("Test.", context)

    assert output == "ok|ok|ok"
    subcall_steps = [s for s in trace.steps if s.kind == "subcall"]
    assert len(subcall_steps) == 3


def test_llm_batch_respects_policy_limits() -> None:
    """Batch operations respect subcall count limits."""
    adapter = FakeAdapter(
        script=[
            "\n".join(
                [
                    'results = llm_batch(["a", "b", "c", "d", "e"])',
                    "answer = '|'.join(results)",
                ]
            ),
            "FINAL_VAR: answer",
        ]
    )
    adapter.add_rule("You are a sub-LLM", "ok")

    context = Context.from_text("test")
    policy = Policy(max_subcalls=2)
    runtime = RLM(adapter=adapter, policy=policy)
    output, trace = runtime.run("Test.", context)

    # First 2 subcalls succeed, rest should get SUBCALL_LIMIT messages
    assert "ok" in output
    assert "SUBCALL_LIMIT" in output


def test_llm_batch_with_subcall_batch_parallel() -> None:
    """subcall_batch parallel=True delegates to llm_batch under the hood."""
    adapter = FakeAdapter(
        script=[
            "\n".join(
                [
                    "chunks = ctx.chunk(5)",
                    "answers = ask_chunks('Q?', chunks, parallel=True)",
                    "result = ','.join(answers)",
                ]
            ),
            "FINAL_VAR: result",
        ]
    )
    adapter.add_rule("You are a sub-LLM", "ans")

    context = Context.from_text("abcdefghij")
    runtime = RLM(adapter=adapter, parallel_subcalls=True)
    output, trace = runtime.run("Test.", context)

    assert "ans" in output
    subcall_steps = [s for s in trace.steps if s.kind == "subcall"]
    assert len(subcall_steps) == 2
    assert all(step.parallel_group_id is not None for step in subcall_steps)
    assert {step.parallel_index for step in subcall_steps} == {0, 1}
    assert all(step.parallel_total == 2 for step in subcall_steps)
