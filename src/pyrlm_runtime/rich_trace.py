from __future__ import annotations

import threading
from dataclasses import dataclass

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.syntax import Syntax
    from rich.text import Text
except ImportError as exc:  # pragma: no cover - covered via import-failure test
    raise ImportError(
        "rich is required to use pyrlm_runtime.rich_trace. "
        'Install it with: pip install "pyrlm-runtime[rich]"'
    ) from exc

from .events import RLMEvent, RLMEventListener
from .trace import TraceStep


@dataclass
class _ParallelBatchState:
    total: int
    seen: set[int]
    cached: int = 0
    errors: int = 0


class RichTraceListener(RLMEventListener):
    """Render RLM execution events to the terminal with Rich."""

    def __init__(self, console: Console | None = None, max_output_length: int = 2000) -> None:
        self.console = console or Console()
        self.max_output_length = max_output_length
        self._repl_count = 0
        self._parallel_batches: dict[str, _ParallelBatchState] = {}
        self._lock = threading.Lock()

    def handle(self, event: RLMEvent) -> None:
        with self._lock:
            if event.kind == "run_started":
                self._render_run_started(event)
                return
            if event.kind == "step_completed" and event.step is not None:
                self._render_step(event.step)
                return
            if event.kind == "run_finished":
                self._render_run_finished(event)

    def _render_run_started(self, event: RLMEvent) -> None:
        query = event.query or "(no query)"
        meta = event.context_metadata or {}
        summary = [
            f"Query: {query}",
            f"REPL backend: {event.repl_backend or 'unknown'}",
        ]
        if meta:
            summary.append(
                "Context: "
                f"{meta.get('total_length', 0):,} chars | "
                f"{meta.get('context_type', 'unknown')} | "
                f"{meta.get('num_documents', 1)} document(s)"
            )
        self.console.print(Rule("RLM Run", style="cyan"))
        self.console.print(Panel("\n".join(summary), border_style="cyan", box=box.ROUNDED))

    def _render_run_finished(self, event: RLMEvent) -> None:
        lines = [
            f"Answer: {self._truncate(event.output or 'NO_ANSWER')}",
            f"Steps: {event.total_steps or 0}",
            f"Tokens: {event.tokens_used or 0}",
        ]
        if event.elapsed is not None:
            lines.append(f"Elapsed: {event.elapsed:.4f}s")
        self.console.print(
            Panel(
                "\n".join(lines),
                title="[bold green]Run Finished[/bold green]",
                border_style="green",
                box=box.ROUNDED,
            )
        )

    def _render_step(self, step: TraceStep) -> None:
        if step.kind in {"repl_exec", "sub_repl_exec"}:
            self._render_repl_step(step)
            return
        if step.kind in {"subcall", "sub_subcall", "recursive_subcall"}:
            self._render_subcall_step(step)
            return
        self._render_model_step(step)

    def _render_repl_step(self, step: TraceStep) -> None:
        self._repl_count += 1
        code = self._truncate(step.code or "")
        self.console.print(
            Panel(
                Syntax(code, "python", theme="monokai", line_numbers=True),
                title=f"[bold blue]In [{self._repl_count}][/bold blue]",
                subtitle=self._step_meta(step),
                border_style="blue",
                box=box.ROUNDED,
            )
        )

        if step.error:
            self.console.print(
                Panel(
                    Text(self._truncate(step.error), style="bold red"),
                    title=f"[bold red]Error [{self._repl_count}][/bold red]",
                    subtitle=self._step_meta(step),
                    border_style="red",
                    box=box.ROUNDED,
                )
            )
            return

        output = step.stdout or "No output"
        style = "white" if step.stdout else "dim"
        self.console.print(
            Panel(
                Text(self._truncate(output), style=style),
                title=f"[bold green]Out [{self._repl_count}][/bold green]",
                subtitle=self._step_meta(step),
                border_style="green" if step.stdout else "dim",
                box=box.ROUNDED,
            )
        )

    def _render_subcall_step(self, step: TraceStep) -> None:
        title = {
            "subcall": "[bold magenta]Subcall[/bold magenta]",
            "sub_subcall": "[bold magenta]Nested Subcall[/bold magenta]",
            "recursive_subcall": "[bold yellow]Recursive Subcall[/bold yellow]",
        }.get(step.kind, "[bold magenta]Subcall[/bold magenta]")
        group_header = None
        group_progress = None
        group_footer = None
        if step.parallel_group_id and step.parallel_total and step.parallel_index is not None:
            batch = self._parallel_batches.setdefault(
                step.parallel_group_id,
                _ParallelBatchState(total=step.parallel_total, seen=set()),
            )
            if not batch.seen:
                group_header = (
                    f"[bold magenta]Parallel Subcalls[/bold magenta] "
                    f"({step.parallel_total} workers)"
                )
            batch.seen.add(step.parallel_index)
            if step.cache_hit:
                batch.cached += 1
            if step.error:
                batch.errors += 1
            done = len(batch.seen)
            group_progress = (
                f"Progress: {done}/{batch.total} done"
                f" | cached {batch.cached}"
                f" | errors {batch.errors}"
            )
            title = (
                f"[bold magenta]Parallel Subcall "
                f"[{step.parallel_index + 1}/{step.parallel_total}][/bold magenta]"
            )
            if done == step.parallel_total:
                group_footer = (
                    f"Completed parallel batch with {step.parallel_total} subcalls"
                    f" | cached {batch.cached}"
                    f" | errors {batch.errors}"
                )
        body = []
        if step.prompt_summary:
            body.append(f"Prompt: {self._truncate(step.prompt_summary)}")
        if step.output:
            body.append(f"Output: {self._truncate(step.output)}")
        if step.cache_hit:
            body.append("Cache: hit")
        if group_header:
            self.console.print(Rule(group_header, style="magenta"))
        if group_progress:
            self.console.print(
                Panel(
                    group_progress,
                    title="[bold magenta]Parallel Batch Progress[/bold magenta]",
                    border_style="magenta",
                    box=box.ROUNDED,
                )
            )
        self.console.print(
            Panel(
                "\n".join(body) if body else "No details",
                title=title,
                subtitle=self._step_meta(step),
                border_style="magenta" if step.kind != "recursive_subcall" else "yellow",
                box=box.ROUNDED,
            )
        )
        if group_footer:
            self.console.print(
                Panel(
                    group_footer,
                    title="[bold magenta]Parallel Batch Finished[/bold magenta]",
                    border_style="magenta",
                    box=box.ROUNDED,
                )
            )

    def _render_model_step(self, step: TraceStep) -> None:
        title = {
            "root_call": "[bold cyan]Root Call[/bold cyan]",
            "sub_root_call": "[bold cyan]Nested Root Call[/bold cyan]",
            "baseline_call": "[bold cyan]Baseline Call[/bold cyan]",
        }.get(step.kind, "[bold cyan]Model Step[/bold cyan]")
        content = step.output or step.code or step.prompt_summary or ""
        self.console.print(
            Panel(
                Text(self._truncate(content), style="white"),
                title=title,
                subtitle=self._step_meta(step),
                border_style="cyan",
                box=box.ROUNDED,
            )
        )

    def _step_meta(self, step: TraceStep) -> str:
        parts = [f"step={step.step_id}", f"depth={step.depth}"]
        if step.usage is not None:
            parts.append(f"tokens={step.usage.total_tokens}")
        if step.elapsed is not None:
            parts.append(f"time={step.elapsed:.4f}s")
        if step.cache_hit:
            parts.append("cached")
        if step.parallel_total and step.parallel_index is not None:
            parts.append(f"parallel={step.parallel_index + 1}/{step.parallel_total}")
        return " | ".join(parts)

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_length:
            return text
        half = self.max_output_length // 2
        omitted = len(text) - self.max_output_length
        return f"{text[:half]}\n\n... [TRUNCATED {omitted} characters] ...\n\n{text[-half:]}"
