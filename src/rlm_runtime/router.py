"""Smart Router for automatic RLM vs Baseline selection.

This module implements the routing logic described in the RLM paper:
- For small contexts (< threshold), use baseline (direct LLM call)
- For large contexts (>= threshold), use RLM with programmatic inspection

Additionally, it provides execution profiles for different task types:
- deterministic_first: Try regex/extract_after before subcalls (default)
- semantic_batches: Use controlled subcalls for classification/aggregation
- hybrid: Try deterministic, fall back to semantic if no answer
- verify: Double-check with a second approach for high confidence
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .adapters.base import ModelAdapter, ModelResponse, Usage
from .context import Context
from .policy import Policy
from .rlm import RLM
from .trace import Trace, TraceStep


class ExecutionProfile(Enum):
    """Execution profiles for different task types."""

    DETERMINISTIC_FIRST = "deterministic_first"
    SEMANTIC_BATCHES = "semantic_batches"
    HYBRID = "hybrid"
    VERIFY = "verify"


@dataclass
class RouterConfig:
    """Configuration for the SmartRouter."""

    # Threshold in characters - below this, use baseline
    baseline_threshold: int = 8000

    # Default execution profile
    default_profile: ExecutionProfile = ExecutionProfile.DETERMINISTIC_FIRST

    # Auto-calibration: learn optimal threshold from runs
    auto_calibrate: bool = False

    # Collected calibration data: (context_size, baseline_ok, rlm_ok, baseline_time, rlm_time)
    _calibration_data: list[tuple[int, bool, bool, float, float]] = field(
        default_factory=list
    )


@dataclass
class RouterResult:
    """Result from a SmartRouter run."""

    output: str
    trace: Trace
    method: str  # "baseline" or "rlm"
    profile: ExecutionProfile
    context_chars: int
    tokens_used: int
    elapsed: float | None = None


class SmartRouter:
    """Automatically routes between baseline and RLM based on context size.

    Usage:
        router = SmartRouter(adapter, config=RouterConfig(baseline_threshold=8000))
        result = router.run(query, context)
        print(f"Used {result.method}, answer: {result.output}")

    The router automatically selects:
    - Baseline (direct LLM call) for small contexts
    - RLM (programmatic inspection) for large contexts

    This eliminates RLM overhead for simple cases while maintaining
    accuracy for complex, large-context tasks.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        subcall_adapter: ModelAdapter | None = None,
        config: RouterConfig | None = None,
        policy: Policy | None = None,
        system_prompt: str | None = None,
        fallback_code: str | None = None,
        auto_finalize_var: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.adapter = adapter
        self.subcall_adapter = subcall_adapter
        self.config = config or RouterConfig()
        self.policy = policy
        self.system_prompt = system_prompt
        self.fallback_code = fallback_code
        self.auto_finalize_var = auto_finalize_var
        self.logger = logger or logging.getLogger("rlm_runtime.router")

    def run(
        self,
        query: str,
        context: Context,
        *,
        profile: ExecutionProfile | None = None,
        force_method: str | None = None,
    ) -> RouterResult:
        """Run query with automatic method selection.

        Args:
            query: The question to answer
            context: The context to search
            profile: Override the default execution profile
            force_method: Force "baseline" or "rlm" (for testing/debugging)

        Returns:
            RouterResult with output, trace, and metadata
        """
        import time

        effective_profile = profile or self.config.default_profile
        context_chars = context.len_chars()

        # Determine which method to use
        if force_method:
            method = force_method
        elif context_chars < self.config.baseline_threshold:
            method = "baseline"
        else:
            method = "rlm"

        self.logger.debug(
            "router selecting method=%s context_chars=%d threshold=%d profile=%s",
            method,
            context_chars,
            self.config.baseline_threshold,
            effective_profile.value,
        )

        started = time.perf_counter()

        if method == "baseline":
            output, trace, tokens = self._run_baseline(query, context)
        else:
            output, trace, tokens = self._run_rlm(query, context, effective_profile)

        elapsed = time.perf_counter() - started

        return RouterResult(
            output=output,
            trace=trace,
            method=method,
            profile=effective_profile,
            context_chars=context_chars,
            tokens_used=tokens,
            elapsed=elapsed,
        )

    def _run_baseline(
        self, query: str, context: Context
    ) -> tuple[str, Trace, int]:
        """Run baseline: direct LLM call with full context."""
        prompt = (
            "Answer the question using only the provided context. "
            "If the answer is not present, reply with NO_ANSWER.\n\n"
            f"Context:\n{context.text}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        response = self.adapter.complete(messages, max_tokens=512, temperature=0.0)

        # Create a minimal trace for baseline
        trace = Trace(steps=[])
        trace.add(
            TraceStep(
                step_id=1,
                kind="baseline_call",
                depth=0,
                prompt_summary=f"Baseline query: {query[:100]}...",
                code=None,
                usage=response.usage,
            )
        )

        return response.text.strip(), trace, response.usage.total_tokens

    def _run_rlm(
        self, query: str, context: Context, profile: ExecutionProfile
    ) -> tuple[str, Trace, int]:
        """Run RLM with the specified execution profile."""
        from .prompts import LLAMA_SYSTEM_PROMPT

        # Build RLM configuration based on profile
        rlm_kwargs = self._build_rlm_kwargs(profile)

        rlm = RLM(
            adapter=self.adapter,
            subcall_adapter=self.subcall_adapter,
            policy=self.policy or Policy(),
            system_prompt=self.system_prompt or LLAMA_SYSTEM_PROMPT,
            fallback_code=self.fallback_code,
            auto_finalize_var=self.auto_finalize_var,
            logger=self.logger,
            **rlm_kwargs,
        )

        output, trace = rlm.run(query, context)

        # Calculate total tokens from trace
        tokens = sum(step.usage.total_tokens for step in trace.steps if step.usage)

        return output, trace, tokens

    def _build_rlm_kwargs(self, profile: ExecutionProfile) -> dict:
        """Build RLM kwargs based on execution profile."""
        base_kwargs = {
            "require_repl_before_final": True,
            "invalid_response_limit": 2,
        }

        if profile == ExecutionProfile.DETERMINISTIC_FIRST:
            # Prioritize regex/extract_after, minimal subcalls
            return {
                **base_kwargs,
                "require_subcall_before_final": False,
                "fallback_guard_steps": 1,
                "parallel_subcalls": False,
            }

        elif profile == ExecutionProfile.SEMANTIC_BATCHES:
            # Use controlled subcalls for classification/aggregation
            return {
                **base_kwargs,
                "require_subcall_before_final": True,
                "subcall_guard_steps": 2,
                "parallel_subcalls": True,
                "max_concurrent_subcalls": 8,
            }

        elif profile == ExecutionProfile.HYBRID:
            # Try deterministic first, then semantic
            return {
                **base_kwargs,
                "require_subcall_before_final": False,
                "fallback_guard_steps": 2,
                "repl_error_limit": 2,
                "parallel_subcalls": True,
            }

        elif profile == ExecutionProfile.VERIFY:
            # Double-check for high confidence
            return {
                **base_kwargs,
                "require_subcall_before_final": False,
                "fallback_guard_steps": 1,
                "recursive_subcalls": True,
                "max_recursion_depth": 2,
            }

        return base_kwargs


@dataclass
class TraceFormatter:
    """Format traces for documentation and debugging.

    Provides human-readable output showing:
    - What strategy was used (regex, subcall, etc.)
    - What operations were performed
    - Results at each step

    Usage:
        formatter = TraceFormatter()
        print(formatter.format(trace))
        print(formatter.format_table([result1, result2, result3]))
    """

    # Show full code snippets (vs truncated)
    verbose: bool = False

    # Max width for code snippets
    max_code_width: int = 60

    # Include timing info
    show_timing: bool = True

    def format(self, trace: Trace, title: str | None = None) -> str:
        """Format a single trace for display."""
        lines: list[str] = []

        if title:
            lines.append(f"=== {title} ===")
            lines.append("")

        for step in trace.steps:
            step_line = self._format_step(step)
            lines.append(step_line)

        # Summary
        lines.append("")
        lines.append(self._format_summary(trace))

        return "\n".join(lines)

    def _format_step(self, step: TraceStep) -> str:
        """Format a single trace step."""
        parts = [f"[{step.step_id}]"]

        # Step kind with icon
        kind_icons = {
            "root_call": "🔷",
            "repl_exec": "⚙️",
            "subcall": "📤",
            "recursive_subcall": "🔄",
            "baseline_call": "📝",
        }
        icon = kind_icons.get(step.kind, "•")
        parts.append(f"{icon} {step.kind}")

        if step.depth and step.depth > 0:
            parts.append(f"depth={step.depth}")

        if step.cache_hit:
            parts.append("(cached)")

        if step.error:
            parts.append("❌")

        # Content summary
        if step.code:
            snippet = self._extract_strategy(step.code)
            parts.append(f"→ {snippet}")
        elif step.prompt_summary:
            summary = step.prompt_summary[:50] + "..." if len(step.prompt_summary) > 50 else step.prompt_summary
            parts.append(f"→ {summary}")

        # Tokens
        if step.usage:
            parts.append(f"[{step.usage.total_tokens} tok]")

        return " ".join(parts)

    def _extract_strategy(self, code: str) -> str:
        """Extract the key strategy from code."""
        code_lower = code.lower()

        # Detect regex usage
        if "re.search" in code_lower or "re.findall" in code_lower:
            # Try to extract the pattern
            match = re.search(r're\.(search|findall)\s*\(\s*r?["\']([^"\']+)["\']', code)
            if match:
                pattern = match.group(2)
                if len(pattern) > 30:
                    pattern = pattern[:30] + "..."
                return f"regex: {pattern}"
            return "regex search"

        # Detect extract_after
        if "extract_after" in code_lower:
            match = re.search(r'extract_after\s*\(\s*["\']([^"\']+)["\']', code)
            if match:
                marker = match.group(1)
                if len(marker) > 30:
                    marker = marker[:30] + "..."
                return f"extract_after('{marker}')"
            return "extract_after()"

        # Detect subcalls
        if "llm_query" in code_lower or "ask_chunks" in code_lower:
            return "subcall"

        # Detect FINAL
        if "final_var" in code_lower or "final:" in code_lower:
            match = re.search(r'FINAL(?:_VAR)?:\s*(\w+)', code)
            if match:
                return f"FINAL → {match.group(1)}"
            return "FINAL"

        # Detect peek/tail
        if "peek(" in code_lower:
            return "peek()"
        if "tail(" in code_lower:
            return "tail()"

        # Truncate for display
        first_line = code.split("\n")[0].strip()
        if len(first_line) > self.max_code_width:
            first_line = first_line[:self.max_code_width] + "..."
        return first_line

    def _format_summary(self, trace: Trace) -> str:
        """Format trace summary."""
        total_tokens = sum(s.usage.total_tokens for s in trace.steps if s.usage)
        step_counts: dict[str, int] = {}
        for step in trace.steps:
            step_counts[step.kind] = step_counts.get(step.kind, 0) + 1

        cache_hits = sum(1 for s in trace.steps if s.cache_hit)
        errors = sum(1 for s in trace.steps if s.error)

        parts = [
            f"Total: {len(trace.steps)} steps",
            f"{total_tokens} tokens",
        ]
        if cache_hits:
            parts.append(f"{cache_hits} cached")
        if errors:
            parts.append(f"{errors} errors")

        return "Summary: " + ", ".join(parts)

    def format_table(
        self,
        results: list[RouterResult],
        headers: list[str] | None = None,
    ) -> str:
        """Format multiple results as a comparison table.

        Default columns: context_chars, method, profile, tokens, time, output_preview
        """
        if not results:
            return "No results to display."

        # Build table data
        rows: list[list[str]] = []
        default_headers = ["chars", "method", "profile", "tokens", "time", "output"]

        for r in results:
            output_preview = r.output[:30] + "..." if len(r.output) > 30 else r.output
            output_preview = output_preview.replace("\n", " ")
            time_str = f"{r.elapsed:.2f}s" if r.elapsed else "-"

            rows.append([
                f"{r.context_chars:,}",
                r.method,
                r.profile.value[:12],
                str(r.tokens_used),
                time_str,
                output_preview,
            ])

        # Calculate column widths
        headers = headers or default_headers
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        # Build table
        lines: list[str] = []

        # Header
        header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        lines.append(header_line)
        lines.append("-" * len(header_line))

        # Rows
        for row in rows:
            row_line = " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
            lines.append(row_line)

        return "\n".join(lines)

    def format_comparison(
        self,
        baseline_result: RouterResult,
        rlm_result: RouterResult,
    ) -> str:
        """Format side-by-side comparison of baseline vs RLM."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("BASELINE vs RLM Comparison")
        lines.append("=" * 60)
        lines.append("")

        # Context info
        lines.append(f"Context: {baseline_result.context_chars:,} chars")
        lines.append("")

        # Baseline
        lines.append("BASELINE:")
        lines.append(f"  Output: {baseline_result.output[:80]}...")
        lines.append(f"  Tokens: {baseline_result.tokens_used}")
        if baseline_result.elapsed:
            lines.append(f"  Time: {baseline_result.elapsed:.2f}s")
        lines.append("")

        # RLM
        lines.append("RLM:")
        lines.append(f"  Output: {rlm_result.output[:80]}...")
        lines.append(f"  Tokens: {rlm_result.tokens_used}")
        if rlm_result.elapsed:
            lines.append(f"  Time: {rlm_result.elapsed:.2f}s")
        lines.append(f"  Profile: {rlm_result.profile.value}")
        lines.append("")

        # Trace summary
        lines.append("RLM Trace:")
        for step in rlm_result.trace.steps[:5]:  # Show first 5 steps
            lines.append(f"  {self._format_step(step)}")
        if len(rlm_result.trace.steps) > 5:
            lines.append(f"  ... ({len(rlm_result.trace.steps) - 5} more steps)")

        return "\n".join(lines)
