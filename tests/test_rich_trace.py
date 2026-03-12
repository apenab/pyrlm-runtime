import builtins
import importlib
import importlib.util
import sys
from pathlib import Path

import pyrlm_runtime
import pytest
from pyrlm_runtime import Context, RLM, TraceStep
from pyrlm_runtime.adapters import FakeAdapter
from pyrlm_runtime.adapters.base import Usage
from pyrlm_runtime.events import RLMEvent
from pyrlm_runtime.rich_trace import RichTraceListener
from rich.console import Console


def test_root_package_does_not_import_rich_trace_by_default() -> None:
    sys.modules.pop("pyrlm_runtime.rich_trace", None)
    importlib.reload(pyrlm_runtime)

    assert "pyrlm_runtime.rich_trace" not in sys.modules


def test_rich_trace_listener_renders_repl_cycle() -> None:
    console = Console(record=True, width=120)
    listener = RichTraceListener(console=console, max_output_length=120)
    adapter = FakeAdapter(
        script=[
            "\n".join(
                [
                    "snippet = peek(10)",
                    "summary = llm_query(f'Summarize: {snippet}')",
                    "print(summary)",
                    "answer = summary",
                ]
            ),
            "FINAL_VAR: answer",
        ]
    )
    adapter.add_rule("You are a sub-LLM", "subcall-answer")

    runtime = RLM(
        adapter=adapter,
        event_listener=listener,
    )
    output, trace = runtime.run(
        "Summarize the first chunk.", Context.from_text("abcdefghijabcdefghij")
    )

    rendered = console.export_text()

    assert output == "subcall-answer"
    assert len(trace.steps) >= 3
    assert "RLM Run" in rendered
    assert "In [1]" in rendered
    assert "Out [1]" in rendered
    assert "Subcall" in rendered
    assert "Run Finished" in rendered
    assert "subcall-answer" in rendered


def test_rich_trace_listener_renders_errors_truncation_and_cache_hits() -> None:
    console = Console(record=True, width=120)
    listener = RichTraceListener(console=console, max_output_length=40)

    listener.handle(
        RLMEvent(
            kind="run_started",
            query="Q?",
            context_metadata={"total_length": 123, "context_type": "text", "num_documents": 1},
            repl_backend="python",
        )
    )
    listener.handle(
        RLMEvent(
            kind="step_completed",
            step=TraceStep(
                step_id=1,
                kind="subcall",
                depth=1,
                prompt_summary="prompt",
                output="x" * 120,
                usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
                elapsed=0.01,
                cache_hit=True,
            ),
        )
    )
    listener.handle(
        RLMEvent(
            kind="step_completed",
            step=TraceStep(
                step_id=2,
                kind="repl_exec",
                depth=0,
                code="1/0",
                error="ZeroDivisionError: division by zero",
                elapsed=0.02,
            ),
        )
    )
    listener.handle(
        RLMEvent(
            kind="run_finished",
            output="done",
            total_steps=2,
            tokens_used=3,
            elapsed=0.03,
        )
    )

    rendered = console.export_text()

    assert "TRUNCATED" in rendered
    assert "cached" in rendered
    assert "Error [1]" in rendered
    assert "ZeroDivisionError" in rendered


def test_rich_trace_listener_renders_parallel_subcalls() -> None:
    console = Console(record=True, width=120)
    listener = RichTraceListener(console=console, max_output_length=5000)
    adapter = FakeAdapter(
        script=[
            "\n".join(
                [
                    "chunks = ctx.chunk(5)",
                    "answers = ask_chunks('Summarize.', chunks, parallel=True)",
                    "answer = ','.join(answers)",
                ]
            ),
            "FINAL_VAR: answer",
        ]
    )
    adapter.add_rule("You are a sub-LLM", "subcall-answer")

    runtime = RLM(adapter=adapter, event_listener=listener, parallel_subcalls=True)
    output, trace = runtime.run("Summarize.", Context.from_text("abcdefghij"))

    rendered = console.export_text()
    subcall_steps = [step for step in trace.steps if step.kind == "subcall"]

    assert output == "subcall-answer,subcall-answer"
    assert len(subcall_steps) == 2
    assert "Parallel Subcalls" in rendered
    assert "Parallel Batch Progress" in rendered
    assert "2/2 done" in rendered
    assert "cached" in rendered
    assert "errors 0" in rendered
    assert "Parallel Subcall [1/2]" in rendered or "Parallel Subcall [2/2]" in rendered
    assert "Parallel Batch Finished" in rendered


def test_rich_trace_listener_clears_parallel_batch_on_completion() -> None:
    console = Console(record=True, width=120)
    listener = RichTraceListener(console=console, max_output_length=5000)

    listener.handle(
        RLMEvent(
            kind="run_started",
            query="Q?",
            context_metadata={"total_length": 10, "context_type": "text", "num_documents": 1},
            repl_backend="python",
        )
    )
    listener.handle(
        RLMEvent(
            kind="step_completed",
            step=TraceStep(
                step_id=1,
                kind="subcall",
                depth=1,
                prompt_summary="a",
                output="A",
                parallel_group_id="parallel-1",
                parallel_index=0,
                parallel_total=2,
            ),
        )
    )
    assert "parallel-1" in listener._parallel_batches

    listener.handle(
        RLMEvent(
            kind="step_completed",
            step=TraceStep(
                step_id=2,
                kind="subcall",
                depth=1,
                prompt_summary="b",
                output="B",
                parallel_group_id="parallel-1",
                parallel_index=1,
                parallel_total=2,
            ),
        )
    )

    assert "parallel-1" not in listener._parallel_batches


def test_rich_trace_listener_resets_stale_parallel_state_on_run_start() -> None:
    console = Console(record=True, width=120)
    listener = RichTraceListener(console=console, max_output_length=5000)
    listener._repl_count = 7
    listener._parallel_batches["stale"] = object()  # type: ignore[assignment]

    listener.handle(
        RLMEvent(
            kind="run_started",
            query="Fresh run",
            context_metadata={"total_length": 10, "context_type": "text", "num_documents": 1},
            repl_backend="python",
        )
    )

    assert listener._repl_count == 0
    assert listener._parallel_batches == {}


def test_rich_trace_import_error_message_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "pyrlm_runtime._rich_trace_import_error_test"
    path = Path(pyrlm_runtime.__file__).resolve().parent / "rich_trace.py"
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "rich" or name.startswith("rich."):
            raise ImportError("simulated missing rich")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    with pytest.raises(ImportError, match='pip install "pyrlm-runtime\\[rich\\]"'):
        spec.loader.exec_module(module)

    sys.modules.pop(module_name, None)
