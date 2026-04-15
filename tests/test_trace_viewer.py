from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from pyrlm_runtime.adapters.base import Usage
from pyrlm_runtime.trace import Trace, TraceStep


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_step(
    step_id: int,
    kind: str = "root_call",
    depth: int = 0,
    **kwargs,
) -> TraceStep:
    return TraceStep(step_id=step_id, kind=kind, depth=depth, **kwargs)


def _flat_trace() -> Trace:
    return Trace(
        steps=[
            _make_step(0, "root_call", depth=0, output="LLM response"),
            _make_step(1, "repl_exec", depth=0, code="print('hi')", stdout="hi\n"),
            _make_step(2, "root_call", depth=0, output="Final answer"),
        ]
    )


def _nested_trace() -> Trace:
    """Depth 0 → 1 → 2 → back to 1 → back to 0."""
    return Trace(
        steps=[
            _make_step(0, "root_call", depth=0),
            _make_step(1, "repl_exec", depth=0, code="x = llm_query('q')"),
            _make_step(2, "subcall", depth=1, prompt_summary="What is x?"),
            _make_step(3, "sub_root_call", depth=2, output="x is 42"),
            _make_step(4, "sub_repl_exec", depth=1, code="print(x)", stdout="42\n"),
            _make_step(5, "root_call", depth=0, output="Answer: 42"),
        ]
    )


def _trace_with_error_and_cache() -> Trace:
    return Trace(
        steps=[
            _make_step(0, "root_call", depth=0, usage=Usage(100, 50, 150)),
            _make_step(1, "repl_exec", depth=0, code="1/0", error="ZeroDivisionError"),
            _make_step(
                2,
                "subcall",
                depth=0,
                cache_hit=True,
                usage=Usage(200, 80, 280),
                parallel_group_id="grp1",
                parallel_index=0,
                parallel_total=2,
            ),
            _make_step(
                3,
                "subcall",
                depth=0,
                cache_hit=False,
                usage=Usage(200, 80, 280),
                parallel_group_id="grp1",
                parallel_index=1,
                parallel_total=2,
            ),
        ]
    )


# ── import guard ─────────────────────────────────────────────────────────────


def test_root_package_does_not_import_trace_viewer_by_default() -> None:
    import pyrlm_runtime

    sys.modules.pop("pyrlm_runtime.trace_viewer", None)
    importlib.reload(pyrlm_runtime)

    assert "pyrlm_runtime.trace_viewer" not in sys.modules


# ── tree reconstruction ───────────────────────────────────────────────────────


def _collect_tree_structure(trace: Trace) -> list[tuple[int, int]]:
    """
    Return (step_id, parent_step_id_or_-1) pairs in the order they are added,
    by replaying the same stack algorithm used in TraceTree.load_trace().
    """
    from pyrlm_runtime.trace_viewer import TraceTree

    # We exercise the algorithm directly by inspecting the built tree.
    tree = TraceTree("Steps")
    tree.load_trace(trace)

    result: list[tuple[int, int]] = []

    def walk(node, parent_data):
        for child in node.children:
            step: TraceStep | None = child.data
            if step is not None:
                result.append((step.step_id, parent_data))
                walk(child, step.step_id)

    walk(tree.root, -1)
    return result


def test_flat_trace_all_under_root() -> None:
    trace = _flat_trace()
    structure = _collect_tree_structure(trace)
    # All depth-0 steps should have root as parent (-1)
    assert all(parent == -1 for _, parent in structure)
    assert [sid for sid, _ in structure] == [0, 1, 2]


def test_nested_trace_structure() -> None:
    trace = _nested_trace()
    structure = _collect_tree_structure(trace)
    by_id = {sid: parent for sid, parent in structure}

    # depth-0 steps have root parent
    assert by_id[0] == -1
    assert by_id[1] == -1
    assert by_id[5] == -1

    # depth-1 steps are under last depth-0 step before them
    assert by_id[2] == 1  # subcall under repl_exec[1]
    assert by_id[4] == 1  # sub_repl_exec under repl_exec[1]

    # depth-2 step is under last depth-1 step before it
    assert by_id[3] == 2  # sub_root_call under subcall[2]


def test_single_step_trace() -> None:
    trace = Trace(steps=[_make_step(0, "root_call", depth=0)])
    structure = _collect_tree_structure(trace)
    assert structure == [(0, -1)]


def test_empty_trace_no_crash() -> None:
    from pyrlm_runtime.trace_viewer import TraceTree

    tree = TraceTree("Steps")
    tree.load_trace(Trace(steps=[]))
    # Should have added a placeholder child (empty message)
    assert len(list(tree.root.children)) == 1


# ── detail rendering ──────────────────────────────────────────────────────────


def _static_content(w) -> object | None:
    """Extract the Rich renderable from a Textual Static widget (Textual 8.x)."""
    # Textual 8.x stores content as _Static__content (name-mangled __content)
    return getattr(w, "_Static__content", None)


def _widgets_to_text(widgets) -> str:
    """Render all Static widgets in a widget list to plain text for assertions."""
    from rich.console import Console
    from textual.widgets import Collapsible, Static

    console = Console(record=True, width=120)

    def _render_static(w) -> None:
        if isinstance(w, Static):
            content = _static_content(w)
            if content is not None:
                console.print(content)

    for w in widgets:
        _render_static(w)
        # Also render Static children inside Collapsible
        if isinstance(w, Collapsible):
            for child in w._nodes:
                _render_static(child)

    return console.export_text()


def test_build_step_widgets_repl_with_error() -> None:
    from pyrlm_runtime.trace_viewer import _build_step_widgets

    step = _make_step(1, "repl_exec", depth=0, code="1/0", error="ZeroDivisionError: division by zero")
    widgets = _build_step_widgets(step)
    assert widgets  # not empty
    rendered = _widgets_to_text(widgets)
    assert "REPL" in rendered
    assert "ZeroDivisionError" in rendered


def test_build_step_widgets_subcall_with_tokens() -> None:
    from pyrlm_runtime.trace_viewer import _build_step_widgets

    step = _make_step(
        2,
        "subcall",
        depth=1,
        prompt_summary="Summarise this paragraph",
        output="It talks about X",
        usage=Usage(500, 100, 600),
        cache_hit=True,
        parallel_group_id="g1",
        parallel_index=0,
        parallel_total=3,
    )
    rendered = _widgets_to_text(_build_step_widgets(step))
    assert "600" in rendered  # total tokens
    assert "parallel" in rendered.lower()
    assert "hit" in rendered.lower()
    assert "Summarise" in rendered


def test_build_step_widgets_code_strips_fences() -> None:
    from pyrlm_runtime.trace_viewer import _build_step_widgets

    step = _make_step(0, "repl_exec", depth=0, code="```python\nprint('hi')\n```")
    rendered = _widgets_to_text(_build_step_widgets(step))
    assert "print" in rendered
    assert "```" not in rendered


def test_text_widgets_collapsible_for_long_text() -> None:
    """Long text (> _PREVIEW_LINES lines) produces a Collapsible widget."""
    from textual.widgets import Collapsible

    from pyrlm_runtime.trace_viewer import _PREVIEW_LINES, _text_widgets

    long_text = "\n".join(f"line {i}" for i in range(_PREVIEW_LINES + 5))
    widgets = _text_widgets("stdout", "green", long_text)
    assert any(isinstance(w, Collapsible) for w in widgets)
    # The collapsible title should mention the overflow count
    coll = next(w for w in widgets if isinstance(w, Collapsible))
    assert "5 more lines" in coll.title


def test_text_widgets_no_collapsible_for_short_text() -> None:
    """Short text (≤ _PREVIEW_LINES lines) has no Collapsible."""
    from textual.widgets import Collapsible

    from pyrlm_runtime.trace_viewer import _PREVIEW_LINES, _text_widgets

    short_text = "\n".join(f"line {i}" for i in range(_PREVIEW_LINES))
    widgets = _text_widgets("stdout", "green", short_text)
    assert not any(isinstance(w, Collapsible) for w in widgets)


# ── from_json_file ────────────────────────────────────────────────────────────


def test_from_json_file_standard_format(tmp_path: Path) -> None:
    from pyrlm_runtime.trace_viewer import TraceViewerApp

    trace = _flat_trace()
    json_file = tmp_path / "trace.json"
    json_file.write_text(trace.to_json(), encoding="utf-8")

    app = TraceViewerApp.from_json_file(json_file)
    assert len(app._trace.steps) == 3


def test_from_json_file_export_format(tmp_path: Path) -> None:
    import json

    from pyrlm_runtime.trace_viewer import TraceViewerApp

    trace = _flat_trace()
    import json as _json

    export_data = {
        "id": 123,
        "question": "test question",
        "rlm_output": "answer",
        "steps": _json.loads(trace.to_json()),
    }
    json_file = tmp_path / "export.json"
    json_file.write_text(json.dumps(export_data), encoding="utf-8")

    app = TraceViewerApp.from_json_file(json_file)
    assert len(app._trace.steps) == 3


def test_from_json_file_real_export() -> None:
    """Smoke-test against an actual export file in examples/exports/."""
    exports_dir = Path(__file__).parent.parent / "examples" / "exports"
    export_files = list(exports_dir.glob("debug_*.json"))
    if not export_files:
        pytest.skip("no debug_*.json export files found")

    from pyrlm_runtime.trace_viewer import TraceViewerApp

    app = TraceViewerApp.from_json_file(export_files[0])
    assert len(app._trace.steps) > 0


def test_from_json_file_invalid_raises(tmp_path: Path) -> None:
    import json

    from pyrlm_runtime.trace_viewer import TraceViewerApp

    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps({"no_steps_key": True}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unrecognised trace JSON format"):
        TraceViewerApp.from_json_file(bad_file)


# ── summary bar ───────────────────────────────────────────────────────────────


def test_summary_bar_stats() -> None:
    from pyrlm_runtime.trace_viewer import TraceSummaryBar

    trace = _trace_with_error_and_cache()
    bar = TraceSummaryBar()
    bar.load_trace(trace)
    rendered = bar.summary_text
    assert "4" in rendered  # 4 steps
    assert "710" in rendered  # total tokens: 150 + 280 + 280
    assert "1" in rendered  # 1 error, 1 cached
