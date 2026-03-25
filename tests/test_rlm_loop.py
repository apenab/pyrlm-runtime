import logging


from pyrlm_runtime import Context, Policy, RLM
from pyrlm_runtime.adapters import FakeAdapter, ModelResponse, Usage
from pyrlm_runtime.events import RLMEvent
from pyrlm_runtime.prompts import build_iteration_message


class RecordingListener:
    def __init__(self) -> None:
        self.events: list[RLMEvent] = []

    def handle(self, event: RLMEvent) -> None:
        self.events.append(event)


class MetaSequenceAdapter:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.call_log: list[list[dict[str, str]]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ModelResponse:
        del max_tokens, temperature
        self.call_log.append(list(messages))
        if not self._responses:
            raise RuntimeError("MetaSequenceAdapter has no remaining responses")
        return self._responses.pop(0)


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


def test_llm_query_json_parses_fenced_json_response() -> None:
    adapter = FakeAdapter(
        script=[
            'data = llm_query_json("extract")\nanswer = str(data["value"])\nprint(answer)',
            "FINAL_VAR: answer",
        ]
    )
    adapter.add_rule("You are a sub-LLM", '```json\n{"value": 7}\n```')

    context = Context.from_text("Test context.")
    runtime = RLM(adapter=adapter)
    output, trace = runtime.run("Test query.", context)

    assert output == "7"


def test_llm_batch_records_coerces_common_extraction_shapes() -> None:
    adapter = FakeAdapter(
        script=[
            (
                'rows = llm_batch_records(["entity-one", "no-answer", "wrapped"])\n'
                'answer = f"{rows[0][0][\'name\']}|{len(rows[1])}|{rows[2][0][\'name\']}"\n'
                "print(answer)"
            ),
            "FINAL_VAR: answer",
        ]
    )
    adapter.add_rule(r"You are a sub-LLM[\s\S]*entity-one", '[{"name": "A"}]', regex=True)
    adapter.add_rule(r"You are a sub-LLM[\s\S]*no-answer", '"NO_ANSWER"', regex=True)
    adapter.add_rule(
        r"You are a sub-LLM[\s\S]*wrapped",
        '{"records": [{"name": "B"}]}',
        regex=True,
    )

    context = Context.from_text("Test context.")
    runtime = RLM(adapter=adapter)
    output, trace = runtime.run("Test query.", context)

    assert output == "A|0|B"


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


def test_conversation_history_contains_state_summary_after_silent_repl_step() -> None:
    """Silent REPL steps should still surface created state to the next turn."""
    adapter = FakeAdapter(
        script=[
            "values = [1, 2, 3]\nsummary = {'count': len(values)}",
            'answer = str(summary["count"])\nprint(answer)',
            "FINAL_VAR: answer",
        ]
    )
    context = Context.from_text("test")
    runtime = RLM(adapter=adapter, conversation_history=True)
    output, trace = runtime.run("Count values.", context)

    assert output == "3"
    second_call_msgs = adapter.call_log[1]
    assert second_call_msgs[3]["role"] == "user"
    assert "stdout:\n<none>" in second_call_msgs[3]["content"]
    assert "error:\n<none>" in second_call_msgs[3]["content"]
    assert "state:" in second_call_msgs[3]["content"]
    assert "values = list[len=3]" in second_call_msgs[3]["content"]
    assert "summary = dict[len=1]" in second_call_msgs[3]["content"]


def test_empty_length_responses_abort_with_explicit_no_answer(caplog) -> None:
    adapter = MetaSequenceAdapter([
        ModelResponse(
            text="",
            usage=Usage(100, 40, 140),
            meta={
                "provider": "openai_compatible",
                "finish_reason": "length",
                "content_kind": "null",
                "reasoning_present": True,
            },
        ),
        ModelResponse(
            text="",
            usage=Usage(120, 45, 165),
            meta={
                "provider": "openai_compatible",
                "finish_reason": "length",
                "content_kind": "null",
                "reasoning_present": True,
            },
        ),
    ])
    runtime = RLM(
        adapter=adapter,
        invalid_response_limit=2,
        rlm_diagnostics=True,
    )

    with caplog.at_level(logging.DEBUG, logger="pyrlm_runtime"):
        output, trace = runtime.run("Q?", Context.from_text("context"))

    assert output == "NO_ANSWER"
    assert len(adapter.call_log) == 2
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "invalid_response_detail=" in log_text
    assert '"detail": "empty_content_length"' in log_text
    assert "empty_length limit reached (2), aborting with NO_ANSWER" in log_text


def test_invalid_response_limit_resets_after_valid_code_step() -> None:
    adapter = FakeAdapter(
        script=[
            "",
            'answer = "ok"\nprint(answer)',
            "",
            "FINAL_VAR: answer",
        ]
    )
    runtime = RLM(
        adapter=adapter,
        invalid_response_limit=2,
        fallback_code='print("fallback-ran")',
    )

    output, trace = runtime.run("Q?", Context.from_text("context"))

    assert output == "ok"
    assert not any(step.stdout == "fallback-ran\n" for step in trace.steps)


def test_rlm_diagnostics_logging_is_opt_in(caplog) -> None:
    runtime_on = RLM(
        adapter=FakeAdapter(script=["FINAL: done"]),
        rlm_diagnostics=True,
    )
    with caplog.at_level(logging.DEBUG, logger="pyrlm_runtime"):
        output_on, trace_on = runtime_on.run("Q?", Context.from_text("context"))

    assert output_on == "done"
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "runtime_fingerprint=" in log_text
    assert "root_call request_meta=" in log_text
    assert "root_call response_meta=" in log_text

    caplog.clear()

    runtime_off = RLM(
        adapter=FakeAdapter(script=["FINAL: done"]),
        rlm_diagnostics=False,
    )
    with caplog.at_level(logging.DEBUG, logger="pyrlm_runtime"):
        output_off, trace_off = runtime_off.run("Q?", Context.from_text("context"))

    assert output_off == "done"
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "runtime_fingerprint=" not in log_text
    assert "root_call request_meta=" not in log_text
    assert "root_call response_meta=" not in log_text


def test_build_iteration_message_truncates_large_stdout() -> None:
    huge_stdout = "X" * 5000

    msg = build_iteration_message(
        last_stdout=huge_stdout,
        last_error=None,
        last_state_summary=None,
        step=2,
        max_steps=40,
    )

    assert "stdout:\n" in msg
    assert "...[truncated " in msg
    assert len(msg) < 2600


def test_max_tokens_exceeded_executes_last_code_and_returns_materialized_answer() -> None:
    adapter = MetaSequenceAdapter([
        ModelResponse(
            text='final_answer = "done"\nprint(final_answer)',
            usage=Usage(10, 10, 20),
        ),
    ])
    runtime = RLM(
        adapter=adapter,
        policy=Policy(max_total_tokens=5),
    )

    output, trace = runtime.run("Q?", Context.from_text("context"))

    assert output == "done"
    assert any(step.kind == "repl_exec" and step.stdout == "done\n" for step in trace.steps)


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
            f"Role alternation broken at index {i}: expected {expected}, got {trimmed[i]['role']}"
        )


# ---------------------------------------------------------------------------
# min_steps regression tests
# ---------------------------------------------------------------------------


def test_min_steps_blocks_early_auto_finalize() -> None:
    """Auto-finalize must NOT trigger before min_steps is reached."""
    # The adapter produces code that sets final_answer on every step,
    # but min_steps=3 should keep the RLM iterating until step 3.
    adapter = FakeAdapter(
        script=[
            'final_answer = "step1_answer"',
            'final_answer = "step2_answer"',
            'final_answer = "step3_answer"',
            'final_answer = "step4_answer"',
        ]
    )

    context = Context.from_text("test context")
    runtime = RLM(
        adapter=adapter,
        auto_finalize_var="final_answer",
        min_steps=3,
    )
    output, trace = runtime.run("Q?", context)

    # Should have run at least 3 steps (the first two are blocked by min_steps)
    repl_steps = [s for s in trace.steps if s.kind == "repl_exec"]
    assert len(repl_steps) >= 3, f"Expected ≥3 REPL steps, got {len(repl_steps)}"
    assert output == "step3_answer"


def test_min_steps_blocks_explicit_final() -> None:
    """FINAL: must be rejected before min_steps is reached."""
    adapter = FakeAdapter(
        script=[
            # Step 1: try to finalize immediately — should be blocked
            "FINAL: early_answer",
            # Step 2: produce code after the guard blocks us
            'x = "working"',
            # Step 3: now we can finalize
            "FINAL: late_answer",
        ]
    )

    context = Context.from_text("test context")
    runtime = RLM(
        adapter=adapter,
        min_steps=2,
        require_repl_before_final=False,
    )
    output, trace = runtime.run("Q?", context)

    assert output == "late_answer"
    # Verify we went through more than one root call
    root_calls = [s for s in trace.steps if s.kind == "root_call"]
    assert len(root_calls) >= 2


def test_min_steps_zero_allows_immediate_finalize() -> None:
    """With min_steps=0 (default), FINAL on step 1 works normally."""
    adapter = FakeAdapter(
        script=[
            "FINAL: immediate",
        ]
    )

    context = Context.from_text("test context")
    runtime = RLM(
        adapter=adapter,
        min_steps=0,
        require_repl_before_final=False,
    )
    output, trace = runtime.run("Q?", context)

    assert output == "immediate"


# ---------------------------------------------------------------------------
# auto_finalize_reject_patterns regression tests
# ---------------------------------------------------------------------------


def test_auto_finalize_rejects_meta_reference() -> None:
    """Regression: GPT-5.2 writes '[La respuesta anterior...]' as final_answer.

    When auto_finalize_reject_patterns is configured, answers matching any
    pattern must be rejected so the RLM continues iterating.
    """
    adapter = FakeAdapter(
        script=[
            'final_answer = "[La respuesta anterior completa constituye la respuesta]"',
            (
                'final_answer = "Respuesta real con contenido completo que tiene '
                "más de cien caracteres de texto para superar cualquier limite mínimo "
                'de longitud configurado."'
            ),
        ]
    )

    context = Context.from_text("test context")
    runtime = RLM(
        adapter=adapter,
        auto_finalize_var="final_answer",
        auto_finalize_reject_patterns=[
            r"respuesta anterior",
            r"see above",
            r"the previous response",
        ],
    )
    output, trace = runtime.run("test query", context)

    assert "Respuesta real" in output
    # First answer was rejected, so we must have at least 2 REPL steps
    repl_steps = [s for s in trace.steps if s.kind == "repl_exec"]
    assert len(repl_steps) >= 2


def test_auto_finalize_reject_patterns_none_allows_anything() -> None:
    """Default: no reject patterns → any string accepted."""
    adapter = FakeAdapter(
        script=[
            'final_answer = "[La respuesta anterior completa]"',
        ]
    )

    context = Context.from_text("test context")
    runtime = RLM(
        adapter=adapter,
        auto_finalize_var="final_answer",
    )
    output, trace = runtime.run("test", context)

    assert "respuesta anterior" in output.lower()


def test_min_steps_does_not_block_at_max_steps_exhaustion() -> None:
    """When max_steps is exhausted, min_steps must NOT prevent returning
    whatever value is available in auto_finalize_var."""
    adapter = FakeAdapter(
        script=[
            'final_answer = "partial_result"',
            'final_answer = "still_working"',
            'final_answer = "almost_done"',
        ]
    )

    context = Context.from_text("test context")
    runtime = RLM(
        adapter=adapter,
        policy=Policy(max_steps=3),
        auto_finalize_var="final_answer",
        min_steps=10,  # Much higher than max_steps
    )
    output, trace = runtime.run("Q?", context)

    # MaxStepsExceeded handler should bypass min_steps and return the value
    assert "partial_result" in output or "still_working" in output or "almost_done" in output


def test_event_listener_emits_run_and_step_events_with_enriched_trace() -> None:
    adapter = FakeAdapter(
        script=[
            "\n".join(
                [
                    "snippet = peek(20)",
                    "summary = llm_query(f'Summarize: {snippet}')",
                    "print(summary)",
                    "answer = f'Result: {summary}'",
                ]
            ),
            "FINAL_VAR: answer",
        ]
    )
    adapter.add_rule("You are a sub-LLM", "ok")
    listener = RecordingListener()

    context = Context.from_text("RLMs inspect prompts via code.")
    runtime = RLM(adapter=adapter, event_listener=listener)
    output, trace = runtime.run("Return a result.", context)

    assert output == "Result: ok"
    assert listener.events[0].kind == "run_started"
    assert listener.events[-1].kind == "run_finished"

    step_events = [event for event in listener.events if event.kind == "step_completed"]
    assert len(step_events) == len(trace.steps)
    assert [event.step.kind for event in step_events if event.step is not None][:3] == [
        "root_call",
        "subcall",
        "repl_exec",
    ]

    root_step = next(step for step in trace.steps if step.kind == "root_call")
    repl_step = next(step for step in trace.steps if step.kind == "repl_exec")
    subcall_step = next(step for step in trace.steps if step.kind == "subcall")

    assert root_step.output is not None
    assert "summary = llm_query" in root_step.output
    assert repl_step.elapsed is not None
    assert subcall_step.output == "ok"
    assert listener.events[-1].tokens_used == sum(
        step.usage.total_tokens for step in trace.steps if step.usage is not None
    )


def test_run_logs_full_final_answer(caplog) -> None:
    answer = "Resultado final " + ("muy largo " * 30).strip()
    adapter = FakeAdapter(script=[f"FINAL: {answer}"])

    runtime = RLM(adapter=adapter)
    with caplog.at_level(logging.INFO, logger="pyrlm_runtime"):
        output, _trace = runtime.run("Return a result.", Context.from_text("ctx"))

    assert output == answer
    assert any(
        record.name == "pyrlm_runtime" and record.getMessage() == f"final answer={answer}"
        for record in caplog.records
    )


def test_multiline_final_answer_is_preserved() -> None:
    adapter = FakeAdapter(
        script=[
            "FINAL: Resumen de ingresos:\n\n1) Empresa A | 2024 | 100\n2) Empresa B | 2023 | 80"
        ]
    )

    runtime = RLM(adapter=adapter)
    output, _trace = runtime.run("Return a result.", Context.from_text("ctx"))

    assert output == (
        "Resumen de ingresos:\n\n"
        "1) Empresa A | 2024 | 100\n"
        "2) Empresa B | 2023 | 80"
    )
