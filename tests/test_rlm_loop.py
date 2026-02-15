from pyrlm_runtime import Context, RLM
from pyrlm_runtime.adapters import FakeAdapter


def test_rlm_loop_runs_to_final() -> None:
    adapter = FakeAdapter(
        script=[
            "\n".join(
                [
                    "snippet = peek(20)",
                    "summary = llm_query(f'Summarize: {snippet}')",
                    "answer = f'Result: {summary}'",
                ]
            ),
            "FINAL_VAR: answer",
        ]
    )
    adapter.add_rule("You are a sub-LLM", "ok")

    context = Context.from_text("RLMs inspect prompts via code.")
    runtime = RLM(adapter=adapter)
    output, trace = runtime.run("Return a result.", context)

    assert output == "Result: ok"
    kinds = [step.kind for step in trace.steps]
    assert "root_call" in kinds
    assert "repl_exec" in kinds
    assert "subcall" in kinds


def test_rlm_with_subcall_adapter() -> None:
    """Test using a different adapter for subcalls."""
    root_adapter = FakeAdapter(
        script=[
            "\n".join(
                [
                    "snippet = peek(20)",
                    "summary = llm_query(f'Summarize: {snippet}')",
                    "answer = f'Got: {summary}'",
                ]
            ),
            "FINAL_VAR: answer",
        ]
    )

    # Subcall adapter returns different response
    subcall_adapter = FakeAdapter(script=[])
    subcall_adapter.add_rule("You are a sub-LLM", "subcall-response")

    context = Context.from_text("Test context.")
    runtime = RLM(adapter=root_adapter, subcall_adapter=subcall_adapter)
    output, trace = runtime.run("Test query.", context)

    assert output == "Got: subcall-response"


def test_rlm_with_parallel_subcalls() -> None:
    """Test parallel subcall execution."""
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

    # Should have multiple answers joined
    assert "ans" in output


def test_recursive_subcall_uses_configured_repl_backend() -> None:
    """Regression: recursive subcalls must use the configured REPL backend,
    not a hardcoded PythonREPL."""
    from unittest.mock import patch

    from pyrlm_runtime.trace import Trace

    adapter = FakeAdapter(
        script=[
            "result = llm_query('summarize')\nanswer = result",
            "FINAL_VAR: answer",
        ]
    )

    context = Context.from_text("test context")
    runtime = RLM(
        adapter=adapter,
        recursive_subcalls=True,
        max_recursion_depth=2,
    )

    with patch("pyrlm_runtime.rlm._run_recursive_subcall") as mock_rrs:
        mock_rrs.return_value = ("mocked-answer", Trace(steps=[]))
        output, trace = runtime.run("Test", context)

        # The key assertion: create_repl must be passed (not None)
        assert mock_rrs.called, "_run_recursive_subcall was never called"
        call_kwargs = mock_rrs.call_args.kwargs
        assert "create_repl" in call_kwargs, (
            "create_repl was not passed to _run_recursive_subcall"
        )
        assert call_kwargs["create_repl"] is not None, (
            "create_repl should not be None"
        )
        assert call_kwargs["create_repl"] == runtime._create_repl, (
            "create_repl should be RLM._create_repl"
        )

    assert output == "mocked-answer"


def test_rlm_with_document_context() -> None:
    """Test RLM with document list context."""
    adapter = FakeAdapter(
        script=[
            "\n".join(
                [
                    "n = ctx.num_documents()",
                    "doc0 = ctx.get_document(0)",
                    "answer = f'Docs: {n}, First: {doc0[:10]}'",
                ]
            ),
            "FINAL_VAR: answer",
        ]
    )

    docs = ["First document content", "Second document content"]
    context = Context.from_documents(docs)
    runtime = RLM(adapter=adapter)
    output, trace = runtime.run("Count docs.", context)

    assert "Docs: 2" in output
    assert "First: First docu" in output
