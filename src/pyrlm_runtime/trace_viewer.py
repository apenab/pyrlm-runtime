from __future__ import annotations

import json
import re
from pathlib import Path

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, VerticalScroll
    from textual.widget import Widget
    from textual.widgets import Collapsible, Footer, Header, Static, Tree
    from textual.widgets._tree import TreeNode
except ImportError as exc:  # pragma: no cover - covered via import-failure test
    raise ImportError(
        "textual is required to use pyrlm_runtime.trace_viewer. "
        'Install it with: pip install "pyrlm-runtime[tui]"'
    ) from exc

try:
    from rich.rule import Rule
    from rich.syntax import Syntax
    from rich.text import Text
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "rich is required to use pyrlm_runtime.trace_viewer. "
        'Install it with: pip install "pyrlm-runtime[tui]"'
    ) from exc

from .trace import Trace, TraceStep

# ── colour/icon helpers ───────────────────────────────────────────────────────

_KIND_COLOR: dict[str, str] = {
    "root_call": "cyan",
    "sub_root_call": "cyan",
    "baseline_call": "cyan",
    "repl_exec": "blue",
    "sub_repl_exec": "blue",
    "subcall": "magenta",
    "sub_subcall": "magenta",
    "recursive_subcall": "yellow",
}

_KIND_LABEL: dict[str, str] = {
    "root_call": "Root Call",
    "sub_root_call": "Sub Root Call",
    "baseline_call": "Baseline Call",
    "repl_exec": "REPL",
    "sub_repl_exec": "Sub REPL",
    "subcall": "Subcall",
    "sub_subcall": "Sub Subcall",
    "recursive_subcall": "Recursive Subcall",
}

_MAX_LABEL_EXTRA = 28   # max chars for code/prompt snippet in tree label
_PREVIEW_LINES = 8      # lines shown before the "N more lines ▶" collapsible


def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing ```python / ``` fences if present."""
    text = text.strip()
    text = re.sub(r"^```\s*\w*\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _step_node_label(step: TraceStep) -> Text:
    color = _KIND_COLOR.get(step.kind, "white")
    kind_str = _KIND_LABEL.get(step.kind, step.kind)
    label = Text(f"[{step.step_id}] ", style="dim")
    label.append(kind_str, style=f"bold {color}")

    # Append a brief snippet: first line of code (REPL) or prompt summary (subcall)
    snippet: str | None = None
    if step.code:
        first_line = _strip_code_fences(step.code).splitlines()[0] if step.code else ""
        snippet = first_line[:_MAX_LABEL_EXTRA]
    elif step.prompt_summary:
        snippet = step.prompt_summary[:_MAX_LABEL_EXTRA]

    if snippet:
        label.append(f": {snippet}", style="dim")

    if step.model:
        label.append(f" [{step.model}]", style="dim cyan")

    if step.error:
        label.append(" [E]", style="bold red")
    if step.cache_hit:
        label.append(" ✓", style="bold green")
    if step.parallel_total and step.parallel_total > 1 and step.parallel_index is not None:
        label.append(f" [{step.parallel_index + 1}/{step.parallel_total}]", style="dim magenta")

    return label


def _is_final_var_step(step: TraceStep) -> bool:
    """True when the step's output is a FINAL_VAR signal (e.g. 'FINAL_VAR: result')."""
    return bool(step.output and step.output.strip().startswith("FINAL_VAR:"))


def _text_widgets(
    rule_label: str, rule_style: str, text: str, text_style: str = ""
) -> list[Widget]:
    """Return [Rule header, content] widgets, with a Collapsible for overflow lines."""
    widgets: list[Widget] = [Static(Rule(rule_label, style=rule_style))]
    lines = text.splitlines()

    if len(lines) <= _PREVIEW_LINES:
        content = Text(text, style=text_style) if text_style else Text(text)
        widgets.append(Static(content))
    else:
        preview = "\n".join(lines[:_PREVIEW_LINES])
        rest = "\n".join(lines[_PREVIEW_LINES:])
        remaining = len(lines) - _PREVIEW_LINES
        preview_widget = Text(preview, style=text_style) if text_style else Text(preview)
        rest_widget = Text(rest, style=text_style) if text_style else Text(rest)
        widgets.append(Static(preview_widget))
        widgets.append(
            Collapsible(
                Static(rest_widget),
                title=f"{remaining} more line{'s' if remaining != 1 else ''}",
                collapsed=True,
            )
        )

    return widgets


def _build_step_widgets(
    step: TraceStep, final_answer: str | None = None
) -> list[Widget]:
    """Build Textual widgets for the step detail panel."""
    widgets: list[Widget] = []
    color = _KIND_COLOR.get(step.kind, "white")

    # — Header —
    widgets.append(
        Static(Rule(f"[bold {color}]{_KIND_LABEL.get(step.kind, step.kind)}[/bold {color}]", style=color))
    )

    # — Metadata —
    meta = Text()
    meta.append(f"step {step.step_id}", style="bold")
    meta.append(f"  depth={step.depth}", style="dim")
    if step.elapsed is not None:
        meta.append(f"  elapsed={step.elapsed:.3f}s", style="dim")
    if step.model:
        meta.append(f"  model={step.model}", style="cyan")
    if step.cache_hit:
        meta.append("  cache=hit", style="bold green")
    else:
        meta.append("  cache=miss", style="dim")
    if step.cache_key:
        meta.append(f"  key={step.cache_key}", style="dim")
    widgets.append(Static(meta))

    # — Tokens —
    if step.usage is not None:
        u = step.usage
        tok = Text()
        tok.append("tokens  ", style="dim")
        tok.append(f"prompt={u.prompt_tokens:,}", style="cyan")
        tok.append("  ")
        tok.append(f"completion={u.completion_tokens:,}", style="cyan")
        tok.append("  ")
        tok.append(f"total={u.total_tokens:,}", style="bold cyan")
        widgets.append(Static(tok))

    # — Parallel —
    if step.parallel_group_id and step.parallel_total and step.parallel_index is not None:
        par = Text()
        par.append("parallel  ", style="dim")
        par.append(f"{step.parallel_index + 1}/{step.parallel_total}", style="magenta")
        par.append(f"  group={step.parallel_group_id}", style="dim")
        widgets.append(Static(par))

    # — Prompt (input context, before code) —
    if step.prompt_summary:
        widgets.extend(_text_widgets("prompt", "dim", step.prompt_summary))

    # — Code —
    if step.code:
        clean_code = _strip_code_fences(step.code)
        widgets.append(Static(Rule("code", style="blue")))
        widgets.append(Static(Syntax(clean_code, "python", theme="monokai", line_numbers=True)))

    # — Output —
    if step.output:
        if _is_final_var_step(step) and final_answer is not None:
            var_name = step.output.strip().split(":", 1)[-1].strip()
            hdr = Text()
            hdr.append("FINAL_VAR: ", style="dim")
            hdr.append(var_name, style="bold cyan")
            widgets.append(Static(Rule("output", style="cyan")))
            widgets.append(Static(hdr))
            widgets.extend(_text_widgets("", "dim", final_answer)[1:])  # skip Rule, add content
        else:
            widgets.extend(_text_widgets("output", "cyan", step.output))

    # — Stdout —
    if step.stdout:
        widgets.extend(_text_widgets("stdout", "green", step.stdout))

    # — Error —
    if step.error:
        widgets.extend(_text_widgets("error", "red", step.error, text_style="red"))

    return widgets


# ── Textual widgets ───────────────────────────────────────────────────────────


class TraceTree(Tree[TraceStep]):
    """Left-panel tree showing all trace steps with depth-based nesting."""

    def load_trace(self, trace: Trace) -> None:
        self.clear()
        if not trace.steps:
            self.root.add_leaf("[dim](empty trace)[/dim]")
            return

        # Pass 1: determine which step_ids will have children in the tree.
        # A step gets a child when a later step has a strictly greater depth
        # and no intervening step has already "claimed" that slot.
        steps_with_children: set[int] = set()
        depth_stack: list[int] = [-1]
        id_stack: list[int] = [-1]  # -1 = virtual root
        for step in trace.steps:
            while depth_stack[-1] >= step.depth:
                depth_stack.pop()
                id_stack.pop()
            parent_id = id_stack[-1]
            if parent_id != -1:
                steps_with_children.add(parent_id)
            depth_stack.append(step.depth)
            id_stack.append(step.step_id)

        # Pass 2: build the Textual tree, using add_leaf for nodes without children.
        node_stack: list[tuple[TreeNode[TraceStep], int]] = [(self.root, -1)]

        for step in trace.steps:
            while node_stack[-1][1] >= step.depth:
                node_stack.pop()
            parent = node_stack[-1][0]
            label = _step_node_label(step)
            if step.step_id in steps_with_children:
                node: TreeNode[TraceStep] = parent.add(label, data=step)
            else:
                node = parent.add_leaf(label, data=step)
            node_stack.append((node, step.depth))

        self.root.expand_all()


class StepDetail(Widget):
    """Right-panel detail view for the currently selected step."""

    DEFAULT_CSS = """
    StepDetail {
        padding: 0 1;
        height: auto;
    }
    StepDetail Static {
        height: auto;
    }
    StepDetail Collapsible {
        height: auto;
        border: none;
        padding: 0;
    }
    StepDetail CollapsibleTitle {
        padding: 0;
    }
    """

    final_answer: str | None = None

    def on_mount(self) -> None:
        self.mount(Static(Text("Select a step from the tree on the left.", style="dim")))

    def update_step(self, step: TraceStep) -> None:
        self.remove_children()
        self.mount(*_build_step_widgets(step, final_answer=self.final_answer))


class TraceSummaryBar(Static):
    """Bottom bar showing aggregate trace statistics."""

    DEFAULT_CSS = """
    TraceSummaryBar {
        height: 1;
        background: $panel;
        color: $text-muted;
        padding: 0 1;
    }
    """

    summary_text: str = ""

    def load_trace(self, trace: Trace) -> None:
        total_tokens = sum(
            s.usage.total_tokens for s in trace.steps if s.usage is not None
        )
        errors = sum(1 for s in trace.steps if s.error)
        cached = sum(1 for s in trace.steps if s.cache_hit)
        self.summary_text = (
            f"Steps: {len(trace.steps)}  |  "
            f"Tokens: {total_tokens:,}  |  "
            f"Errors: {errors}  |  "
            f"Cached: {cached}"
        )
        self.update(self.summary_text)


# ── App ───────────────────────────────────────────────────────────────────────


class TraceViewerApp(App[None]):
    """Interactive TUI viewer for pyrlm-runtime execution traces."""

    TITLE = "pyrlm-runtime Trace Viewer"

    CSS = """
    TraceTree {
        width: 35%;
        border-right: solid $accent;
        scrollbar-size: 1 1;
    }
    VerticalScroll {
        width: 65%;
    }
    Horizontal {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]

    def __init__(self, trace: Trace, source_name: str = "", answer: str | None = None) -> None:
        super().__init__()
        self._trace = trace
        self._source_name = source_name
        self._answer = answer

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield TraceTree("Steps", id="trace-tree")
            with VerticalScroll():
                yield StepDetail(id="step-detail")
        yield TraceSummaryBar(id="summary-bar")
        yield Footer()

    def on_mount(self) -> None:
        tree = self.query_one(TraceTree)
        tree.load_trace(self._trace)

        bar = self.query_one(TraceSummaryBar)
        bar.load_trace(self._trace)

        if self._answer is not None:
            self.query_one(StepDetail).final_answer = self._answer

        if self._source_name:
            self.sub_title = self._source_name

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[TraceStep]) -> None:
        step: TraceStep | None = event.node.data
        if step is not None:
            self.query_one(StepDetail).update_step(step)

    def action_cursor_down(self) -> None:
        self.query_one(TraceTree).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one(TraceTree).action_cursor_up()

    @classmethod
    def from_trace(
        cls, trace: Trace, source_name: str = "", answer: str | None = None
    ) -> "TraceViewerApp":
        """Create the app from an in-memory Trace object.

        Pass ``answer`` (the string returned by ``RLM.run()``) to display the
        resolved value on the final FINAL_VAR step instead of just the variable name.
        """
        return cls(trace=trace, source_name=source_name, answer=answer)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "TraceViewerApp":
        """Load a trace from a JSON file.

        Handles two formats:
        - Standard ``Trace.to_json()`` format: a JSON array of step objects.
        - Export format: a JSON object with a ``"steps"`` key containing the array.
        """
        path = Path(path)
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)

        if isinstance(data, list):
            # Standard Trace.to_json() format
            trace = Trace.from_json(raw)
        elif isinstance(data, dict) and "steps" in data:
            # Export format (e.g. examples/exports/debug_*.json)
            steps_raw = json.dumps(data["steps"])
            trace = Trace.from_json(steps_raw)
        else:
            raise ValueError(
                f"Unrecognised trace JSON format in {path}. "
                "Expected a JSON array of steps or a dict with a 'steps' key."
            )

        return cls(trace=trace, source_name=path.name)


def main() -> None:
    """CLI entry point: ``pyrlm-trace-viewer <trace.json>``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive TUI viewer for pyrlm-runtime trace JSON files."
    )
    parser.add_argument("trace_file", help="Path to the trace JSON file")
    args = parser.parse_args()

    app = TraceViewerApp.from_json_file(args.trace_file)
    app.run()
