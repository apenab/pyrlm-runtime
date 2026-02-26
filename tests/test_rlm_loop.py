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


def test_recursive_subcall_uses_configured_repl_backend(tmp_path) -> None:
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
        cache_dir=tmp_path / "cache",
    )

    with patch("pyrlm_runtime.rlm._run_recursive_subcall") as mock_rrs:
        mock_rrs.return_value = ("mocked-answer", Trace(steps=[]))
        output, trace = runtime.run("Test", context)

        # The key assertion: create_repl must be passed (not None)
        assert mock_rrs.called, "_run_recursive_subcall was never called"
        call_kwargs = mock_rrs.call_args.kwargs
        assert "create_repl" in call_kwargs, "create_repl was not passed to _run_recursive_subcall"
        assert call_kwargs["create_repl"] is not None, "create_repl should not be None"
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


# ---------------------------------------------------------------------------
# Conversation history tests
# ---------------------------------------------------------------------------


def test_conversation_history_messages_grow() -> None:
    """Verify the messages list grows across iterations in history mode."""
    adapter = FakeAdapter(
        script=[
            "x = 1\nprint(x)",
            "y = x + 1\nprint(y)",
            "FINAL_VAR: y",
        ]
    )
    context = Context.from_text("test data")
    runtime = RLM(adapter=adapter, conversation_history=True)
    output, trace = runtime.run("Compute something.", context)

    assert output == "2"
    # Call 1: [system, user_initial] = 2 messages
    # Call 2: + [assistant, user_repl] = 4 messages
    # Call 3: + [assistant, user_repl] = 6 messages
    assert len(adapter.call_log) == 3
    assert len(adapter.call_log[0]) == 2
    assert len(adapter.call_log[1]) == 4
    assert len(adapter.call_log[2]) == 6


def test_conversation_history_contains_previous_output() -> None:
    """Verify the LLM sees its own previous code and stdout in the history."""
    adapter = FakeAdapter(
        script=[
            'print("hello_marker")',
            "FINAL: done",
        ]
    )
    context = Context.from_text("test")
    runtime = RLM(adapter=adapter, conversation_history=True)
    output, trace = runtime.run("Test.", context)

    assert output == "done"
    # The second call should contain the first assistant response and REPL output
    second_call_msgs = adapter.call_log[1]
    # Message at index 2: assistant with the code
    assert second_call_msgs[2]["role"] == "assistant"
    assert 'print("hello_marker")' in second_call_msgs[2]["content"]
    # Message at index 3: user with REPL result containing stdout
    assert second_call_msgs[3]["role"] == "user"
    assert "hello_marker" in second_call_msgs[3]["content"]


def test_conversation_history_trimming() -> None:
    """Verify that history trimming keeps system + initial user + recent turns."""
    adapter = FakeAdapter(
        script=[
            "a = 1\nprint(a)",
            "b = 2\nprint(b)",
            "c = 3\nprint(c)",
            "FINAL: done",
        ]
    )
    context = Context.from_text("test")
    # Very small budget to force trimming
    runtime = RLM(
        adapter=adapter,
        conversation_history=True,
        max_history_tokens=200,
    )
    output, trace = runtime.run("Test.", context)

    assert output == "done"
    # System message should always be present in every call
    for call_msgs in adapter.call_log:
        assert call_msgs[0]["role"] == "system"
        if len(call_msgs) > 1:
            assert call_msgs[1]["role"] == "user"
    # Later calls should have fewer messages than unbounded growth would give
    # Without trimming call 4 would have 8 messages; with trimming it should be less
    assert len(adapter.call_log[-1]) < 8


def test_conversation_history_disabled_backward_compat() -> None:
    """When conversation_history=False, every call has exactly 2 messages."""
    adapter = FakeAdapter(
        script=[
            "x = 42\nprint(x)",
            "FINAL_VAR: x",
        ]
    )
    context = Context.from_text("test context")
    runtime = RLM(adapter=adapter, conversation_history=False)
    output, trace = runtime.run("Get x.", context)

    assert output == "42"
    # In stateless mode, every call must have exactly [system, user]
    for call_msgs in adapter.call_log:
        assert len(call_msgs) == 2
        assert call_msgs[0]["role"] == "system"
        assert call_msgs[1]["role"] == "user"
        # The user message should contain the query each time
        assert "Get x." in call_msgs[1]["content"]


def test_recursive_subcall_accepts_conversation_history_params() -> None:
    """Verify _run_recursive_subcall accepts conversation_history kwargs."""
    import inspect

    from pyrlm_runtime.rlm import _run_recursive_subcall

    sig = inspect.signature(_run_recursive_subcall)
    assert "conversation_history" in sig.parameters, (
        "_run_recursive_subcall must accept conversation_history"
    )
    assert "max_history_tokens" in sig.parameters, (
        "_run_recursive_subcall must accept max_history_tokens"
    )
    # Verify defaults
    assert sig.parameters["conversation_history"].default is True
    assert sig.parameters["max_history_tokens"].default == 0


def test_trim_history_preserves_role_alternation() -> None:
    """Regression: _trim_history must keep (assistant, user) pairs so role
    alternation is never broken by partial trimming."""
    from pyrlm_runtime.rlm import _trim_history

    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "initial user message"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "u3"},
    ]
    # Budget that fits head + only the last pair, not all of tail
    # estimate_tokens counts chars/4; head ≈ (1+20)/4 ≈ 5 tokens
    # Each tail message ≈ 1 token → pair ≈ 2 tokens
    # Set budget so only 1 pair from tail fits after head
    trimmed = _trim_history(msgs, max_tokens=8)

    # Must always start with system, user
    assert trimmed[0]["role"] == "system"
    assert trimmed[1]["role"] == "user"

    # Verify strict role alternation after the head
    for i in range(2, len(trimmed)):
        expected = "assistant" if i % 2 == 0 else "user"
        assert trimmed[i]["role"] == expected, (
            f"Role alternation broken at index {i}: expected {expected}, "
            f"got {trimmed[i]['role']}"
        )
