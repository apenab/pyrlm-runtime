from __future__ import annotations

import ast
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any
from .adapters.base import ModelAdapter
from .cache import CacheRecord, FileCache
from .context import Context
from .env import PythonREPL, REPLProtocol
from .events import RLMEvent, RLMEventListener
from .policy import (
    MaxStepsExceeded,
    MaxSubcallsExceeded,
    MaxTokensExceeded,
    Policy,
    estimate_tokens,
)
from .prompts import (
    BASE_SYSTEM_PROMPT,
    SUBCALL_SYSTEM_PROMPT,
    RECURSIVE_SUBCALL_SYSTEM_PROMPT,
    build_root_user_message,
    build_iteration_message,
    build_system_prompt,
)
from .trace import Trace, TraceStep


def _package_origin(package_name: str) -> str | None:
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return None
    return spec.origin


def _adapter_diagnostic_metadata(adapter: Any) -> dict[str, Any]:
    def _read_attr(obj: Any, name: str) -> Any:
        if hasattr(obj, name):
            return getattr(obj, name)
        nested = getattr(obj, "_adapter", None)
        if nested is not None and hasattr(nested, name):
            return getattr(nested, name)
        return None

    return {
        "adapter": type(adapter).__name__,
        "model": _read_attr(adapter, "model"),
        "endpoint": _read_attr(adapter, "endpoint"),
        "base_url": _read_attr(adapter, "base_url"),
    }


_PAGE_MARKER_RE = re.compile(r"<!-- Page (\d+) -->")


def _split_paged_text(text: str) -> list[tuple[int | None, str]]:
    matches = list(_PAGE_MARKER_RE.finditer(text))
    if not matches:
        return [(None, text)]
    pages: list[tuple[int | None, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        page_num: int | None
        try:
            page_num = int(match.group(1))
        except Exception:
            page_num = None
        pages.append((page_num, text[start:end].strip()))
    return pages


def _normalize_patterns(patterns: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(patterns, str):
        return [patterns]
    if isinstance(patterns, tuple):
        return [pattern for pattern in patterns if isinstance(pattern, str) and pattern]
    if isinstance(patterns, list):
        return [pattern for pattern in patterns if isinstance(pattern, str) and pattern]
    return []



def _loose_text(text: str, *, case_sensitive: bool) -> str:
    normalized = re.sub(r"[\W_]+", " ", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized if case_sensitive else normalized.lower()


def _line_snippet(text: str, start: int, end: int, *, limit: int = 220) -> str:
    line_start = text.rfind("\n", 0, start)
    line_end = text.find("\n", end)
    if line_start == -1:
        line_start = 0
    else:
        line_start += 1
    if line_end == -1:
        line_end = len(text)
    snippet = text[line_start:line_end].strip()
    if len(snippet) <= limit:
        return snippet
    return snippet[:limit] + "..."


def _find_pages_in_paged_text(
    text: str,
    patterns: str | list[str] | tuple[str, ...],
    *,
    regex: bool = False,
    case_sensitive: bool = False,
    max_matches: int = 20,
) -> list[dict[str, Any]]:
    normalized_patterns = _normalize_patterns(patterns)
    if not text or not normalized_patterns or max_matches <= 0:
        return []

    compiled_patterns: list[tuple[str, Any]] = []
    if regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        for pattern in normalized_patterns:
            compiled_patterns.append((pattern, re.compile(pattern, flags)))

    results: list[dict[str, Any]] = []
    for page_num, page_text in _split_paged_text(text):
        if not page_text:
            continue
        matched_patterns: list[str] = []
        snippet: str | None = None
        for pattern in normalized_patterns:
            if regex:
                compiled = next(
                    compiled for raw_pattern, compiled in compiled_patterns if raw_pattern == pattern
                )
                match = compiled.search(page_text)
                if match is None:
                    continue
                matched_patterns.append(pattern)
                if snippet is None:
                    snippet = _line_snippet(page_text, match.start(), match.end())
                continue

            haystack = page_text if case_sensitive else page_text.lower()
            needle = pattern if case_sensitive else pattern.lower()
            index = haystack.find(needle)
            if index != -1:
                matched_patterns.append(pattern)
                if snippet is None:
                    snippet = _line_snippet(page_text, index, index + len(needle))
                continue

            loose_haystack = _loose_text(page_text, case_sensitive=case_sensitive)
            loose_needle = _loose_text(pattern, case_sensitive=case_sensitive)
            if not loose_needle or loose_needle not in loose_haystack:
                continue
            matched_patterns.append(pattern)
            if snippet is None:
                snippet = _line_snippet(page_text, 0, min(len(page_text), 80))

        if not matched_patterns:
            continue
        results.append(
            {
                "page_num": page_num,
                "matched_patterns": matched_patterns,
                "snippet": snippet or "",
            }
        )
        if len(results) >= max_matches:
            break
    return results



@dataclass
class RLM:
    """Recursive Language Model runtime.

    Key parameters for subcall configuration (paper-aligned):
    - subcall_adapter: Use a different (often smaller/cheaper) model for subcalls.
      If None, uses the same adapter as the root.
    - recursive_subcalls: If True, subcalls themselves run a mini-RLM loop instead
      of a single LLM call. This enables true recursive processing as in the paper.
    - max_recursion_depth: Maximum depth for recursive subcalls (default 2).
    """

    adapter: ModelAdapter
    policy: Policy | None = None
    cache: FileCache | None = None
    max_tokens: int = 512
    system_prompt: str = BASE_SYSTEM_PROMPT
    subcall_system_prompt: str = SUBCALL_SYSTEM_PROMPT
    cache_dir: Path | str = ".rlm_cache"
    require_repl_before_final: bool = False
    require_subcall_before_final: bool = False
    auto_finalize_var: str | None = None
    # Minimum character length for auto_finalize_var to trigger (prevents premature finalization)
    auto_finalize_min_length: int = 0
    # Regex patterns to reject from auto-finalize (e.g. meta-references like
    # "the previous response" instead of actual content).  If the value of
    # auto_finalize_var matches any pattern the answer is rejected and the
    # loop continues.
    auto_finalize_reject_patterns: list[str] | None = None
    logger: logging.Logger | None = None
    invalid_response_limit: int | None = None
    fallback_code: str | None = None
    repl_error_limit: int | None = None
    subcall_guard_steps: int | None = None
    fallback_guard_steps: int | None = None
    # Paper-aligned: support different adapter for subcalls
    subcall_adapter: ModelAdapter | None = None
    # Paper-aligned: enable recursive subcalls (subcall runs a mini-RLM)
    recursive_subcalls: bool = False
    # Maximum recursion depth for nested RLM calls
    max_recursion_depth: int = 2
    # System prompt for recursive subcalls
    recursive_subcall_system_prompt: str = RECURSIVE_SUBCALL_SYSTEM_PROMPT
    # Max steps for recursive subcall RLMs (should be small)
    recursive_subcall_max_steps: int = 3
    # Paper-aligned: enable parallel subcalls for batch operations
    parallel_subcalls: bool = False
    # Max concurrent subcalls when parallel_subcalls=True
    max_concurrent_subcalls: int = 10
    # Default max_tokens for subcall responses (increase for reasoning models)
    subcall_max_tokens: int = 256
    # REPL backend: "python" (default) or "monty" (pydantic-monty sandbox)
    repl_backend: str = "python"
    # Multi-turn conversation history (default: enabled).
    # When True the LLM sees all previous assistant responses and REPL
    # results, enabling self-correction across iterations.
    conversation_history: bool = True
    # Maximum estimated tokens for conversation history (0 = unlimited).
    max_history_tokens: int = 0
    # Minimum number of steps before finalization is allowed (0 = no minimum).
    # When set, both auto-finalize and explicit FINAL are blocked until this
    # many policy steps have been taken.  The MaxStepsExceeded handler is *not*
    # affected – if the model runs out of steps it can still return whatever
    # value is available.
    min_steps: int = 0
    event_listener: RLMEventListener | None = None
    # Optional retriever for external document search (e.g. Elasticsearch).
    # When set, es_search/es_vector_search/es_hybrid_search/es_get functions
    # are registered in the REPL environment.
    retriever: Any | None = None
    rlm_diagnostics: bool = False
    # Optional REPL extensions: a callable that receives (rlm, repl, retriever,
    # log_diag) and returns a dict of {name: callable} to register.  This lets
    # downstream projects inject domain-specific REPL functions without
    # modifying pyrlm-runtime itself.
    repl_extensions: Any | None = None
    # Optional extra system prompt text appended after retrieval docs.
    system_prompt_supplement: str = ""
    # Truncation limits for debug log messages (chars).
    log_truncate_code: int = 2000
    log_truncate_output: int = 1000

    def _create_repl(self) -> REPLProtocol:
        if self.repl_backend == "python":
            return PythonREPL()
        if self.repl_backend == "monty":
            from .env_monty import MontyREPL

            return MontyREPL()
        raise ValueError(
            f"Invalid repl_backend={self.repl_backend!r}. Expected 'python' or 'monty'."
        )

    def run(self, query: str, context: Context | None = None) -> tuple[str, Trace]:
        if self.retriever is not None:
            import inspect

            search_method = getattr(self.retriever, "search", None)
            if search_method is not None and inspect.iscoroutinefunction(search_method):
                raise TypeError(
                    "RLM.run() requires a synchronous retriever, but received an "
                    f"async retriever ({type(self.retriever).__name__}). "
                    "Use a synchronous retriever like ElasticsearchRetriever instead."
                )
        if context is None:
            if self.retriever is not None:
                context = Context.from_documents([])
            else:
                raise ValueError(
                    "context is required when no retriever is configured. "
                    "Either pass a Context or set retriever on the RLM instance."
                )
        logger = self.logger or logging.getLogger("pyrlm_runtime")
        policy = self.policy or Policy()
        cache = self.cache or FileCache(self.cache_dir)
        trace = Trace(steps=[])
        repl = self._create_repl()
        run_started = time.perf_counter()
        context_meta = context.metadata()

        def emit(event: RLMEvent) -> None:
            if self.event_listener is not None:
                self.event_listener.handle(event)

        def add_step(step: TraceStep) -> None:
            trace.add(step)
            emit(RLMEvent(kind="step_completed", query=query, step=step))

        def log_diag(label: str, payload: dict[str, Any]) -> None:
            if not self.rlm_diagnostics:
                return
            logger.debug(
                "%s=%s",
                label,
                json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str),
            )

        def finish(output: str) -> tuple[str, Trace]:
            logger.info("final answer=%s", output)
            emit(
                RLMEvent(
                    kind="run_finished",
                    query=query,
                    output=output,
                    total_steps=len(trace.steps),
                    tokens_used=_trace_total_tokens(trace),
                    elapsed=time.perf_counter() - run_started,
                )
            )
            return output, trace

        emit(
            RLMEvent(
                kind="run_started",
                query=query,
                context_metadata=context_meta,
                repl_backend=self.repl_backend,
            )
        )
        log_diag(
            "runtime_fingerprint",
            {
                "pyrlm_runtime_origin": _package_origin("pyrlm_runtime"),
                "texto_origin": _package_origin("texto"),
                "adapter": _adapter_diagnostic_metadata(self.adapter),
                "retriever": type(self.retriever).__name__ if self.retriever is not None else None,
                "retriever_index": getattr(self.retriever, "index", None),
            },
        )

        repl.set("P", context.text)
        repl.set("ctx", context)

        def peek(n: int = 2000) -> str:
            return context.text[:n]

        def tail(n: int = 2000) -> str:
            return context.text[-n:]

        def lenp() -> int:
            return context.len_chars()

        repl.set("peek", peek)
        repl.set("tail", tail)
        repl.set("lenP", lenp)

        step_id = 0
        _step_lock = threading.Lock()
        parallel_group_id = 0

        def next_step_id() -> int:
            nonlocal step_id
            with _step_lock:
                step_id += 1
                return step_id

        def next_parallel_group_id() -> str:
            nonlocal parallel_group_id
            with _step_lock:
                parallel_group_id += 1
                return f"parallel-{parallel_group_id}"

        # Select adapter for subcalls (paper-aligned: can use different/cheaper model)
        effective_subcall_adapter = self.subcall_adapter or self.adapter

        def subcall(
            text: str,
            *,
            model: str | None = None,
            max_tokens: int | None = None,
            depth: int = 1,
            parallel_group: str | None = None,
            parallel_index: int | None = None,
            parallel_total: int | None = None,
            reserved_tokens: int = 0,
        ) -> str:
            if max_tokens is None:
                max_tokens = self.subcall_max_tokens
            nonlocal subcall_made
            subcall_started = time.perf_counter()
            try:
                policy.check_subcall(depth)
            except (MaxSubcallsExceeded, MaxTokensExceeded) as exc:
                if reserved_tokens > 0:
                    policy.release_subcall_tokens(reserved_tokens)
                return (
                    f"[SUBCALL_LIMIT] {exc}. "
                    "You have used all available sub-LLM calls. "
                    "Build your final answer now using the information you already have."
                )

            # Include recursive flag in cache key for correct cache separation
            recursive_flag = self.recursive_subcalls and depth < self.max_recursion_depth
            cache_key = _cache_key(
                text=text, model=model, max_tokens=max_tokens, recursive=recursive_flag
            )
            input_hash = _hash_text(text)
            cached = cache.get(cache_key)
            if cached:
                subcall_made = True
                logger.debug(
                    "subcall cache hit depth=%s tokens=%s", depth, cached.usage.total_tokens
                )
                if reserved_tokens > 0:
                    policy.finalize_subcall_tokens(reserved_tokens, cached.usage.total_tokens)
                else:
                    policy.add_subcall_tokens(cached.usage.total_tokens)
                add_step(
                    TraceStep(
                        step_id=next_step_id(),
                        kind="subcall",
                        depth=depth,
                        prompt_summary=_truncate(text, 240),
                        output=_truncate(cached.text, self.log_truncate_output),
                        usage=cached.usage,
                        elapsed=time.perf_counter() - subcall_started,
                        cache_hit=True,
                        input_hash=input_hash,
                        output_hash=_hash_text(cached.text),
                        cache_key=cache_key,
                        parallel_group_id=parallel_group,
                        parallel_index=parallel_index,
                        parallel_total=parallel_total,
                    )
                )
                return cached.text

            # Paper-aligned: recursive subcalls run a mini-RLM instead of single LLM call
            if self.recursive_subcalls and depth < self.max_recursion_depth:
                try:
                    result_text, sub_trace = _run_recursive_subcall(
                        text=text,
                        adapter=effective_subcall_adapter,
                        system_prompt=self.recursive_subcall_system_prompt,
                        max_steps=self.recursive_subcall_max_steps,
                        max_tokens=max_tokens,
                        depth=depth,
                        logger=logger,
                        create_repl=self._create_repl,
                        conversation_history=self.conversation_history,
                        max_history_tokens=self.max_history_tokens,
                        log_truncate_code=self.log_truncate_code,
                    )
                except Exception:
                    if reserved_tokens > 0:
                        policy.release_subcall_tokens(reserved_tokens)
                    raise
                subcall_made = True
                # Aggregate usage from sub-trace
                total_tokens = sum(s.usage.total_tokens for s in sub_trace.steps if s.usage)
                from .adapters.base import Usage

                aggregated_usage = Usage(
                    prompt_tokens=0, completion_tokens=0, total_tokens=total_tokens
                )
                try:
                    if reserved_tokens > 0:
                        policy.finalize_subcall_tokens(reserved_tokens, total_tokens)
                    else:
                        policy.add_subcall_tokens(total_tokens)
                except MaxTokensExceeded:
                    logger.warning(
                        "Token budget exceeded after recursive subcall; returning partial result"
                    )
                cache.set(cache_key, CacheRecord(text=result_text, usage=aggregated_usage))
                add_step(
                    TraceStep(
                        step_id=next_step_id(),
                        kind="recursive_subcall",
                        depth=depth,
                        prompt_summary=_truncate(text, 240),
                        output=_truncate(result_text, self.log_truncate_output),
                        usage=aggregated_usage,
                        elapsed=time.perf_counter() - subcall_started,
                        cache_hit=False,
                        input_hash=input_hash,
                        output_hash=_hash_text(result_text),
                        cache_key=cache_key,
                        parallel_group_id=parallel_group,
                        parallel_index=parallel_index,
                        parallel_total=parallel_total,
                    )
                )
                # Merge sub-trace steps into main trace for visibility
                kind_map = {
                    "root_call": "sub_root_call",
                    "repl_exec": "sub_repl_exec",
                    "subcall": "sub_subcall",
                }
                for sub_step in sub_trace.steps:
                    # Map the sub-step kind to the appropriate sub_ variant
                    sub_kind = kind_map.get(sub_step.kind, sub_step.kind)
                    sub_step_copy = TraceStep(
                        step_id=next_step_id(),
                        kind=sub_kind,  # type: ignore[arg-type]
                        depth=depth + (sub_step.depth or 0),
                        prompt_summary=sub_step.prompt_summary,
                        code=sub_step.code,
                        output=sub_step.output,
                        stdout=sub_step.stdout,
                        error=sub_step.error,
                        usage=sub_step.usage,
                        elapsed=sub_step.elapsed,
                        cache_hit=sub_step.cache_hit,
                        input_hash=sub_step.input_hash,
                        output_hash=sub_step.output_hash,
                        cache_key=sub_step.cache_key,
                    )
                    add_step(sub_step_copy)
                return result_text

            # Standard subcall: single LLM call
            messages = [
                {"role": "system", "content": self.subcall_system_prompt},
                {"role": "user", "content": text},
            ]
            try:
                response = effective_subcall_adapter.complete(
                    messages, max_tokens=max_tokens, temperature=0.0
                )
            except Exception:
                if reserved_tokens > 0:
                    policy.release_subcall_tokens(reserved_tokens)
                raise
            subcall_made = True
            logger.debug("subcall complete depth=%s tokens=%s", depth, response.usage.total_tokens)
            if reserved_tokens > 0:
                policy.finalize_subcall_tokens(reserved_tokens, response.usage.total_tokens)
            else:
                try:
                    policy.add_subcall_tokens(response.usage.total_tokens)
                except MaxTokensExceeded:
                    logger.warning("Token budget exceeded after subcall; returning partial result")
            cache.set(cache_key, CacheRecord(text=response.text, usage=response.usage))
            add_step(
                TraceStep(
                    step_id=next_step_id(),
                    kind="subcall",
                    depth=depth,
                    prompt_summary=_truncate(text, 240),
                    output=_truncate(response.text, self.log_truncate_code),
                    usage=response.usage,
                    elapsed=time.perf_counter() - subcall_started,
                    cache_hit=False,
                    input_hash=input_hash,
                    output_hash=_hash_text(response.text),
                    cache_key=cache_key,
                    parallel_group_id=parallel_group,
                    parallel_index=parallel_index,
                    parallel_total=parallel_total,
                )
            )
            return response.text

        def _normalize_chunks(
            chunks: object,
            *,
            chunk_size: int | None,
            overlap: int,
        ) -> list[str]:
            if isinstance(chunks, Context):
                size = chunk_size or 2000
                return [chunk for _, _, chunk in chunks.chunk(size, overlap=overlap)]
            if isinstance(chunks, str):
                if chunk_size:
                    ctx_chunks = Context.from_text(chunks).chunk(chunk_size, overlap=overlap)
                    return [chunk for _, _, chunk in ctx_chunks]
                return [chunks]
            if isinstance(chunks, list):
                if not chunks:
                    return []
                first = chunks[0]
                if isinstance(first, tuple) and len(first) >= 3:
                    return [
                        str(item[2])
                        for item in chunks
                        if isinstance(item, tuple) and len(item) >= 3
                    ]
                if isinstance(first, str):
                    return [item for item in chunks if isinstance(item, str)]
            return []

        def subcall_batch(
            chunks: object,
            *args: object,
            model: str | None = None,
            max_tokens: int | None = None,
            chunk_size: int | None = None,
            overlap: int = 0,
            question: str | None = None,
            parallel: bool | None = None,
        ) -> list[str]:
            if max_tokens is None:
                max_tokens = self.subcall_max_tokens
            remaining_args = list(args)
            if isinstance(chunks, str) and remaining_args:
                first = remaining_args[0]
                if isinstance(first, list):
                    question = chunks
                    chunks = remaining_args.pop(0)

            prepared = _normalize_chunks(chunks, chunk_size=chunk_size, overlap=overlap)
            if question is None:
                for arg in remaining_args:
                    if isinstance(arg, str):
                        question = arg
                        break
                    if isinstance(arg, list) and len(arg) == 1 and isinstance(arg[0], str):
                        question = arg[0]
                        break
            if question:
                prepared = [f"Question: {question}\nSnippet:\n{chunk}" for chunk in prepared]

            # Deduplicate chunks while preserving order
            unique_chunks: list[str] = []
            seen_set: set[str] = set()
            chunk_indices: dict[str, int] = {}
            for i, chunk in enumerate(prepared):
                if chunk not in seen_set:
                    seen_set.add(chunk)
                    chunk_indices[chunk] = len(unique_chunks)
                    unique_chunks.append(chunk)

            # Determine if we should run in parallel
            use_parallel = parallel if parallel is not None else self.parallel_subcalls

            if use_parallel and len(unique_chunks) > 1:
                # Delegate to llm_batch for parallel execution
                unique_results_list = llm_batch(unique_chunks, model=model, max_tokens=max_tokens)
                chunk_to_result = {c: unique_results_list[i] for i, c in enumerate(unique_chunks)}
                return [chunk_to_result[c] for c in prepared]
            else:
                # Sequential processing (original behavior)
                results: list[str] = []
                seen: dict[str, str] = {}
                for chunk in prepared:
                    if chunk in seen:
                        results.append(seen[chunk])
                        continue
                    result = subcall(chunk, model=model, max_tokens=max_tokens)
                    seen[chunk] = result
                    results.append(result)
                return results

        def llm_batch(
            prompts: list[str],
            *,
            model: str | None = None,
            max_tokens: int | None = None,
        ) -> list[str]:
            """Process a batch of prompts in parallel using sub-LLM calls.

            Args:
                prompts: List of prompt strings to process.
                model: Optional model override for subcalls.
                max_tokens: Optional max tokens per response.

            Returns:
                List of response strings in the same order as input prompts.
            """
            if max_tokens is None:
                max_tokens = self.subcall_max_tokens
            if not prompts:
                return []
            if not isinstance(prompts, list):
                raise ValueError("llm_batch expects a list of prompt strings.")

            error_message = "[ERROR] llm_batch expects a list of prompt strings."
            results: list[str | None] = [None] * len(prompts)

            # Validate and deduplicate prompts while preserving order.
            unique_prompts: list[str] = []
            seen_set: set[str] = set()
            prompt_positions: dict[str, list[int]] = {}
            for idx, p in enumerate(prompts):
                if not isinstance(p, str):
                    results[idx] = error_message
                    continue
                if p not in seen_set:
                    seen_set.add(p)
                    unique_prompts.append(p)
                prompt_positions.setdefault(p, []).append(idx)

            if not unique_prompts:
                return [error_message if result is None else result for result in results]

            # Single valid prompt: call subcall directly (no thread overhead)
            if len(unique_prompts) == 1:
                prompt = unique_prompts[0]
                result = subcall(prompt, model=model, max_tokens=max_tokens)
                for idx in prompt_positions[prompt]:
                    results[idx] = result
                return [error_message if result is None else result for result in results]

            unique_results: list[str | None] = [None] * len(unique_prompts)
            group_id = next_parallel_group_id()

            def _estimate_subcall_token_budget(prompt: str) -> int:
                prompt_budget = estimate_tokens(self.subcall_system_prompt) + estimate_tokens(prompt)
                return prompt_budget + max_tokens

            def _process_one(idx: int, prompt: str) -> tuple[int, str]:
                reserved_tokens = _estimate_subcall_token_budget(prompt)
                policy.reserve_subcall_tokens(reserved_tokens)
                return (
                    idx,
                    subcall(
                        prompt,
                        model=model,
                        max_tokens=max_tokens,
                        parallel_group=group_id,
                        parallel_index=idx,
                        parallel_total=len(unique_prompts),
                        reserved_tokens=reserved_tokens,
                    )
                )

            max_workers = min(self.max_concurrent_subcalls, len(unique_prompts))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_process_one, i, p): i for i, p in enumerate(unique_prompts)
                }
                for future in as_completed(futures):
                    idx, result = future.result()
                    unique_results[idx] = result

            # Map back to original order (handles duplicates and invalid prompts)
            for i, prompt in enumerate(unique_prompts):
                result = unique_results[i] if unique_results[i] is not None else ""
                for idx in prompt_positions[prompt]:
                    results[idx] = result
            return [error_message if result is None else result for result in results]

        def _strip_markdown_fences(text: str) -> str:
            stripped = text.strip()
            if not stripped.startswith("```"):
                return stripped
            lines = stripped.splitlines()
            if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
                return "\n".join(lines[1:-1]).strip()
            return stripped

        def _parse_jsonish_response(text: str) -> Any:
            stripped = _strip_markdown_fences(text)
            candidates: list[str] = []
            for candidate in (stripped, stripped.split(":", 1)[1].strip() if ":" in stripped else ""):
                if candidate and candidate not in candidates:
                    candidates.append(candidate)
            for opener, closer in (("{", "}"), ("[", "]")):
                start = stripped.find(opener)
                end = stripped.rfind(closer)
                if start != -1 and end != -1 and end > start:
                    candidate = stripped[start : end + 1].strip()
                    if candidate and candidate not in candidates:
                        candidates.append(candidate)
            for candidate in candidates:
                try:
                    return json.loads(candidate)
                except Exception:
                    continue
            return {
                "_parse_error": "Could not parse JSON from subcall response.",
                "_raw": stripped,
            }

        def llm_query_json(
            prompt: str,
            *,
            model: str | None = None,
            max_tokens: int | None = None,
        ) -> Any:
            raw = subcall(prompt, model=model, max_tokens=max_tokens)
            return _parse_jsonish_response(raw)

        def _coerce_json_records(parsed: Any) -> list[dict[str, Any]]:
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
            if isinstance(parsed, dict):
                for key in ("records", "items", "entities", "results", "data"):
                    nested = parsed.get(key)
                    if isinstance(nested, list):
                        return [item for item in nested if isinstance(item, dict)]
                if "_parse_error" in parsed:
                    return []
                return [parsed]
            return []

        def llm_batch_json(
            prompts: list[str],
            *,
            model: str | None = None,
            max_tokens: int | None = None,
        ) -> list[Any]:
            return [
                _parse_jsonish_response(raw)
                for raw in llm_batch(prompts, model=model, max_tokens=max_tokens)
            ]

        def llm_query_records(
            prompt: str,
            *,
            model: str | None = None,
            max_tokens: int | None = None,
        ) -> list[dict[str, Any]]:
            raw = subcall(prompt, model=model, max_tokens=max_tokens)
            return _coerce_json_records(_parse_jsonish_response(raw))

        def llm_batch_records(
            prompts: list[str],
            *,
            model: str | None = None,
            max_tokens: int | None = None,
        ) -> list[list[dict[str, Any]]]:
            return [
                _coerce_json_records(_parse_jsonish_response(raw))
                for raw in llm_batch(prompts, model=model, max_tokens=max_tokens)
            ]

        def ask(question: str, text: str, *, max_tokens: int | None = None) -> str:
            return subcall(f"Question: {question}\nSnippet:\n{text}", max_tokens=max_tokens)

        def ask_chunk(question: str, text: str, *, max_tokens: int | None = None) -> str:
            return ask(question, text, max_tokens=max_tokens)

        def ask_chunked(
            question: str,
            chunks: object,
            *,
            max_tokens: int | None = None,
            chunk_size: int | None = None,
            overlap: int = 0,
            parallel: bool | None = None,
        ) -> list[str]:
            return ask_chunks(
                question,
                chunks,
                max_tokens=max_tokens,
                chunk_size=chunk_size,
                overlap=overlap,
                parallel=parallel,
            )

        def ask_chunks(
            question: str,
            chunks: object,
            *,
            max_tokens: int | None = None,
            chunk_size: int | None = None,
            overlap: int = 0,
            parallel: bool | None = None,
        ) -> list[str]:
            return subcall_batch(
                chunks,
                question=question,
                max_tokens=max_tokens,
                chunk_size=chunk_size,
                overlap=overlap,
                parallel=parallel,
            )

        def _sanitize_answer(text: str) -> str | None:
            cleaned = text.strip()
            if not cleaned:
                return None
            lowered = cleaned.lower()
            if "```" in cleaned or "<think>" in lowered:
                return None
            if lowered.startswith(("answer:", "final:", "result:")):
                cleaned = cleaned.split(":", 1)[1].strip()
                lowered = cleaned.lower()
            if cleaned == "NO_ANSWER":
                return None
            marker = "the key term is:"
            if marker in lowered:
                idx = lowered.find(marker) + len(marker)
                tail = cleaned[idx:].strip()
                if not tail:
                    return None
                token = tail.split()[0]
                token = token.strip(" \t\r\n.,;:\"'()[]{}")
                return token or None
            if "\n" in cleaned:
                return None
            if len(cleaned) > 80:
                return None
            return cleaned

        def ask_chunks_first(
            question: str,
            chunks: object,
            *,
            max_tokens: int | None = None,
            chunk_size: int | None = None,
            overlap: int = 0,
        ) -> str | None:
            prepared = _normalize_chunks(chunks, chunk_size=chunk_size, overlap=overlap)
            if question:
                prepared = [f"Question: {question}\nSnippet:\n{chunk}" for chunk in prepared]
            seen: set[str] = set()
            for chunk in prepared:
                if chunk in seen:
                    continue
                seen.add(chunk)
                result = subcall(chunk, max_tokens=max_tokens)
                cleaned = _sanitize_answer(result)
                if cleaned is not None:
                    return cleaned
            return None

        def pick_first_answer(answers: object) -> str | None:
            if not isinstance(answers, list):
                return None
            for item in answers:
                if not isinstance(item, str):
                    continue
                cleaned = _sanitize_answer(item)
                if cleaned is not None:
                    return cleaned
            return None

        def extract_after(marker: str, *, max_len: int = 128) -> str | None:
            idx = context.text.find(marker)
            if idx == -1:
                return None
            start = idx + len(marker)
            window = context.text[start : start + max_len]
            window = window.lstrip()
            if not window:
                return None
            token = window.split()[0]
            return token.strip(" \t\r\n.,;:\"'()[]{}") or None

        repl.set("llm_query", subcall)
        repl.set("llm_query_batch", subcall_batch)
        repl.set("llm_batch", llm_batch)
        repl.set("llm_query_json", llm_query_json)
        repl.set("llm_batch_json", llm_batch_json)
        repl.set("llm_query_records", llm_query_records)
        repl.set("llm_batch_records", llm_batch_records)
        # Expose _parse_jsonish_response for REPL extensions that need JSON parsing
        repl.set("_parse_jsonish_response", _parse_jsonish_response)
        repl.set("ask", ask)
        repl.set("ask_chunk", ask_chunk)
        repl.set("ask_chunked", ask_chunked)
        repl.set("ask_chunks", ask_chunks)
        repl.set("ask_chunks_first", ask_chunks_first)
        repl.set("pick_first_answer", pick_first_answer)
        repl.set("extract_after", extract_after)

        # SHOW_VARS — lets the model inspect user-created variables before
        def show_vars_fn() -> str:
            if hasattr(repl, "show_vars"):
                return repl.show_vars()
            # Fallback for MontyREPL or other backends
            return "(SHOW_VARS not supported by this REPL backend)"

        repl.set("SHOW_VARS", show_vars_fn)

        # -- Retrieval functions (when a retriever is configured) -----------
        _retrieval_fns: dict[str, Any] = {}
        if self.retriever is not None:
            _retriever = self.retriever

            def _normalize_search_results(results: Any) -> Any:
                if not isinstance(results, list):
                    return results
                normalized: list[Any] = []
                for item in results:
                    if not isinstance(item, dict):
                        normalized.append(item)
                        continue
                    clone = dict(item)
                    metadata = clone.get("metadata")
                    page_doc_id = clone.get("page_doc_id") or clone.get("doc_id")
                    if page_doc_id is not None:
                        clone["doc_id"] = page_doc_id
                        clone["page_doc_id"] = page_doc_id
                    if isinstance(metadata, dict):
                        logical_doc_id = metadata.get("doc_id")
                        if logical_doc_id is not None:
                            clone.setdefault("logical_doc_id", logical_doc_id)
                        page_num = metadata.get("page_num")
                        if page_num is not None:
                            clone.setdefault("page_num", page_num)
                    normalized.append(clone)
                return normalized

            def _log_search_result_schema(method: str, results: Any) -> None:
                if not self.rlm_diagnostics or not isinstance(results, list):
                    return
                sample: list[dict[str, Any]] = []
                for item in results[:3]:
                    if not isinstance(item, dict):
                        sample.append({"item_type": type(item).__name__})
                        continue
                    metadata = item.get("metadata")
                    logical_doc_id = None
                    if isinstance(metadata, dict):
                        logical_doc_id = metadata.get("doc_id")
                    sample.append({
                        "page_doc_id": item.get("page_doc_id") or item.get("doc_id"),
                        "logical_doc_id": item.get("logical_doc_id") or logical_doc_id,
                    })
                log_diag("search_result_schema", {"method": method, "sample": sample})

            def _collapse_doc_results(results: Any, *, top_k: int) -> list[dict[str, Any]]:
                if not isinstance(results, list):
                    return []
                grouped: dict[str, dict[str, Any]] = {}
                order: list[str] = []
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    logical_doc_id = item.get("logical_doc_id")
                    page_doc_id = item.get("page_doc_id") or item.get("doc_id")
                    doc_key = logical_doc_id or page_doc_id
                    if not isinstance(doc_key, str) or not doc_key:
                        continue
                    page_num = item.get("page_num")
                    if doc_key not in grouped:
                        clone = dict(item)
                        clone.setdefault("logical_doc_id", logical_doc_id or doc_key)
                        clone.setdefault("page_doc_id", page_doc_id)
                        clone["hit_count"] = 0
                        clone["matched_page_doc_ids"] = []
                        clone["matched_page_numbers"] = []
                        grouped[doc_key] = clone
                        order.append(doc_key)
                    entry = grouped[doc_key]
                    entry["hit_count"] = int(entry.get("hit_count", 0)) + 1
                    if isinstance(page_doc_id, str) and page_doc_id:
                        matched_ids = entry.setdefault("matched_page_doc_ids", [])
                        if isinstance(matched_ids, list) and page_doc_id not in matched_ids:
                            matched_ids.append(page_doc_id)
                    if isinstance(page_num, int):
                        matched_pages = entry.setdefault("matched_page_numbers", [])
                        if isinstance(matched_pages, list) and page_num not in matched_pages:
                            matched_pages.append(page_num)
                collapsed = [grouped[key] for key in order]
                return collapsed[:top_k]

            def es_search(
                query: str, top_k: int = 10, filters: dict | None = None
            ) -> list[dict]:
                results = _normalize_search_results(
                    _retriever.search(query, top_k=top_k, filters=filters)
                )
                _log_search_result_schema("es_search", results)
                return results

            def es_vector_search(
                query: str, top_k: int = 10, filters: dict | None = None
            ) -> list[dict]:
                results = _normalize_search_results(
                    _retriever.vector_search(query, top_k=top_k, filters=filters)
                )
                _log_search_result_schema("es_vector_search", results)
                return results

            def es_hybrid_search(
                query: str, top_k: int = 10, filters: dict | None = None
            ) -> list[dict]:
                results = _normalize_search_results(
                    _retriever.hybrid_search(query, top_k=top_k, filters=filters)
                )
                _log_search_result_schema("es_hybrid_search", results)
                return results

            def es_hybrid_doc_search(
                query: str,
                top_k: int = 10,
                candidate_pages: int | None = None,
                filters: dict | None = None,
            ) -> list[dict]:
                page_limit = candidate_pages or max(top_k * 8, 40)
                results = es_hybrid_search(query, top_k=page_limit, filters=filters)
                collapsed = _collapse_doc_results(results, top_k=top_k)
                _log_search_result_schema("es_hybrid_doc_search", collapsed)
                return collapsed

            def es_hybrid_search_in_doc(
                logical_doc_id: str,
                query: str,
                top_k: int = 5,
                filters: dict | None = None,
            ) -> list[dict]:
                doc_filters = dict(filters or {})
                doc_filters["doc_id"] = logical_doc_id
                results = es_hybrid_search(query, top_k=top_k, filters=doc_filters)
                _log_search_result_schema("es_hybrid_search_in_doc", results)
                return results

            def _get_document_with_fallback(doc_id: str) -> dict[str, Any]:
                try:
                    return _retriever.get(doc_id)
                except Exception as exc:
                    get_logical_document = getattr(_retriever, "get_logical_document", None)
                    if callable(get_logical_document):
                        try:
                            return get_logical_document(doc_id)
                        except Exception:
                            pass
                    raise exc

            def es_get(doc_id: str) -> dict:
                return _get_document_with_fallback(doc_id)

            def es_get_text(doc_id: str) -> str:
                doc = _get_document_with_fallback(doc_id)
                if isinstance(doc, dict):
                    content = doc.get("content", "")
                    return content if isinstance(content, str) else str(content)
                return str(doc)

            def es_get_pages(
                logical_doc_id: str,
                pages: list[int] | None = None,
                radius: int = 0,
                max_pages: int = 20,
            ) -> list[dict]:
                get_pages = getattr(_retriever, "get_pages", None)
                if not callable(get_pages):
                    raise NotImplementedError("Retriever does not support get_pages().")
                return get_pages(
                    logical_doc_id,
                    pages=pages,
                    radius=radius,
                    max_pages=max_pages,
                )

            def es_get_pages_text(
                logical_doc_id: str,
                pages: list[int] | None = None,
                radius: int = 0,
                max_pages: int = 20,
            ) -> str:
                get_pages_text = getattr(_retriever, "get_pages_text", None)
                if callable(get_pages_text):
                    return get_pages_text(
                        logical_doc_id,
                        pages=pages,
                        radius=radius,
                        max_pages=max_pages,
                    )
                docs = es_get_pages(
                    logical_doc_id,
                    pages=pages,
                    radius=radius,
                    max_pages=max_pages,
                )
                parts: list[str] = []
                for doc in docs:
                    page_num = doc.get("page_num")
                    content = doc.get("content", "")
                    if isinstance(page_num, int):
                        parts.append(f"<!-- Page {page_num} -->\n{content}")
                    else:
                        parts.append(str(content))
                return "\n\n".join(parts)

            def _logical_doc_index_status(doc: Any) -> dict[str, Any]:
                metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
                if not isinstance(metadata, dict):
                    metadata = {}
                expected = metadata.get("expected_page_count")
                if not isinstance(expected, int) or expected <= 0:
                    fallback_expected = metadata.get("page_count")
                    expected = fallback_expected if isinstance(fallback_expected, int) and fallback_expected > 0 else None
                indexed = metadata.get("indexed_source_page_count")
                if not isinstance(indexed, int) or indexed < 0:
                    indexed = metadata.get("indexed_page_count")
                if not isinstance(indexed, int) or indexed < 0:
                    source_pages = metadata.get("source_pages")
                    if isinstance(source_pages, list):
                        indexed = len([page for page in source_pages if isinstance(page, int)])
                if not isinstance(indexed, int) or indexed < 0:
                    page_doc_ids = metadata.get("page_doc_ids")
                    if isinstance(page_doc_ids, list):
                        indexed = len(page_doc_ids)
                    elif isinstance(expected, int):
                        indexed = expected
                    else:
                        indexed = None
                chunk_count = metadata.get("chunk_count")
                if not isinstance(chunk_count, int) or chunk_count < 0:
                    page_doc_ids = metadata.get("page_doc_ids")
                    if isinstance(page_doc_ids, list):
                        chunk_count = len(page_doc_ids)
                    else:
                        chunk_count = None
                explicit_incomplete = metadata.get("index_incomplete")
                if isinstance(explicit_incomplete, bool):
                    incomplete = explicit_incomplete
                else:
                    incomplete = bool(
                        isinstance(expected, int)
                        and expected > 0
                        and isinstance(indexed, int)
                        and indexed < expected
                    )
                return {
                    "expected_page_count": expected,
                    "indexed_page_count": indexed,
                    "chunk_count": chunk_count,
                    "index_incomplete": incomplete,
                }

            def es_find_pages(
                logical_doc_id: str,
                patterns: str | list[str],
                *,
                regex: bool = False,
                case_sensitive: bool = False,
                max_matches: int = 20,
                logical_max_pages: int = 2000,
            ) -> list[dict[str, Any]]:
                get_logical_document = getattr(_retriever, "get_logical_document", None)
                if callable(get_logical_document):
                    doc = get_logical_document(logical_doc_id, max_pages=logical_max_pages)
                else:
                    doc = _get_document_with_fallback(logical_doc_id)
                content = ""
                metadata: dict[str, Any] = {}
                if isinstance(doc, dict):
                    raw_content = doc.get("content", "")
                    content = raw_content if isinstance(raw_content, str) else str(raw_content)
                    raw_metadata = doc.get("metadata", {})
                    if isinstance(raw_metadata, dict):
                        metadata = raw_metadata
                else:
                    content = str(doc)
                matches = _find_pages_in_paged_text(
                    content,
                    patterns,
                    regex=regex,
                    case_sensitive=case_sensitive,
                    max_matches=max_matches,
                )
                page_doc_ids = metadata.get("page_doc_ids")
                pages = _split_paged_text(content)
                page_doc_id_by_page: dict[int, str] = {}
                if (
                    isinstance(page_doc_ids, list)
                    and len(page_doc_ids) == len(pages)
                ):
                    for (page_num, _page_text), page_doc_id in zip(pages, page_doc_ids, strict=False):
                        if isinstance(page_num, int) and isinstance(page_doc_id, str) and page_doc_id:
                            page_doc_id_by_page[page_num] = page_doc_id
                for item in matches:
                    page_num = item.get("page_num")
                    if isinstance(page_num, int) and page_num in page_doc_id_by_page:
                        item["page_doc_id"] = page_doc_id_by_page[page_num]
                    item["logical_doc_id"] = logical_doc_id
                index_status = _logical_doc_index_status(doc)
                log_diag(
                    "page_scan_result",
                    {
                        "logical_doc_id": logical_doc_id,
                        "patterns": _normalize_patterns(patterns),
                        "pages": [item.get("page_num") for item in matches],
                        "expected_page_count": index_status["expected_page_count"],
                        "indexed_page_count": index_status["indexed_page_count"],
                        "chunk_count": index_status["chunk_count"],
                        "index_incomplete": index_status["index_incomplete"],
                    },
                )
                return matches

            def es_find_pages_text(
                logical_doc_id: str,
                patterns: str | list[str],
                *,
                regex: bool = False,
                case_sensitive: bool = False,
                radius: int = 0,
                max_pages: int = 20,
                match_limit: int = 20,
                logical_max_pages: int = 2000,
            ) -> str:
                matches = es_find_pages(
                    logical_doc_id,
                    patterns,
                    regex=regex,
                    case_sensitive=case_sensitive,
                    max_matches=match_limit,
                    logical_max_pages=logical_max_pages,
                )
                pages = [
                    page_num
                    for item in matches
                    if isinstance((page_num := item.get("page_num")), int)
                ]
                if not pages:
                    return ""
                return es_get_pages_text(
                    logical_doc_id,
                    pages=pages,
                    radius=radius,
                    max_pages=max_pages,
                )

            _retrieval_fns = {
                "es_search": es_search,
                "es_vector_search": es_vector_search,
                "es_hybrid_search": es_hybrid_search,
                "es_hybrid_doc_search": es_hybrid_doc_search,
                "es_hybrid_search_in_doc": es_hybrid_search_in_doc,
                "es_find_pages": es_find_pages,
                "es_find_pages_text": es_find_pages_text,
                "es_get": es_get,
                "es_get_text": es_get_text,
                "es_get_pages": es_get_pages,
                "es_get_pages_text": es_get_pages_text,
            }
            for name, fn in _retrieval_fns.items():
                repl.set(name, fn)

        # -- REPL extensions (domain-specific functions from downstream) ----
        _extension_fns: dict[str, Any] = {}
        if callable(self.repl_extensions):
            _extension_fns = self.repl_extensions(
                rlm=self,
                repl=repl,
                retriever=self.retriever,
                log_diag=log_diag,
            ) or {}
            for name, fn in _extension_fns.items():
                repl.set(name, fn)

        # Scaffold: mapping of every injected name → its current value.
        # Used by restore_scaffold() to undo accidental overwrites
        # (e.g. model writes `llm_query = None` or `P = "x"`).
        _scaffold: dict[str, Any] = {
            "P": context.text,
            "ctx": context,
            "peek": peek,
            "tail": tail,
            "lenP": lenp,
            "llm_query": subcall,
            "llm_query_batch": subcall_batch,
            "llm_batch": llm_batch,
            "llm_query_json": llm_query_json,
            "llm_batch_json": llm_batch_json,
            "llm_query_records": llm_query_records,
            "llm_batch_records": llm_batch_records,
            "ask": ask,
            "ask_chunk": ask_chunk,
            "ask_chunked": ask_chunked,
            "ask_chunks": ask_chunks,
            "ask_chunks_first": ask_chunks_first,
            "pick_first_answer": pick_first_answer,
            "extract_after": extract_after,
            "SHOW_VARS": show_vars_fn,
            **_retrieval_fns,
            **_extension_fns,
        }
        # Inform REPL backends with SHOW_VARS support which names belong to
        # the scaffold so user-facing variable dumps can hide framework internals.
        if hasattr(repl, "show_vars"):
            try:
                repl._scaffold_names = set(_scaffold.keys())  # type: ignore[attr-defined]
            except Exception:
                # Some custom/frozen REPL backends may not allow dynamic attrs.
                pass

        def restore_scaffold() -> None:
            """Restore scaffold names after each REPL exec (mirrors original's _restore_scaffold)."""
            if hasattr(repl, "restore_names"):
                repl.restore_names(_scaffold)

        def snapshot_repl_state() -> dict[str, str]:
            if hasattr(repl, "snapshot_state"):
                try:
                    snapshot = repl.snapshot_state()
                    if isinstance(snapshot, dict):
                        return snapshot
                except Exception:
                    return {}
            return {}

        def summarize_repl_state_change(
            before: dict[str, str],
            after: dict[str, str],
            *,
            max_items: int = 12,
        ) -> str | None:
            new_items = [(name, after[name]) for name in after if name not in before]
            changed_items = [
                (name, before[name], after[name])
                for name in after
                if name in before and before[name] != after[name]
            ]
            removed_items = [name for name in before if name not in after]

            if not new_items and not changed_items and not removed_items:
                return None

            lines = ["REPL state changes:"]
            shown = 0
            for name, summary in new_items:
                if shown >= max_items:
                    break
                lines.append(f"- new {name} = {summary}")
                shown += 1
            for name, _old, new in changed_items:
                if shown >= max_items:
                    break
                lines.append(f"- updated {name} = {new}")
                shown += 1
            for name in removed_items:
                if shown >= max_items:
                    break
                lines.append(f"- removed {name}")
                shown += 1
            omitted = len(new_items) + len(changed_items) + len(removed_items) - shown
            if omitted > 0:
                lines.append(f"- ... and {omitted} more change(s)")
            return "\n".join(lines)

        last_stdout: str | None = None
        last_error: str | None = None
        last_state_summary: str | None = None
        repl_executed = False
        subcall_made = False
        invalid_responses = 0
        empty_length_streak = 0
        fallback_executed = False
        repl_errors = 0

        def maybe_auto_finalize() -> str | None:
            nonlocal last_error
            if not self.auto_finalize_var:
                return None
            value = repl.get(self.auto_finalize_var)
            if value is None:
                return None
            if isinstance(value, str):
                cleaned = value.strip()
                if not cleaned:
                    last_error = "Auto-finalize blocked: empty value."
                    return None
                if cleaned.upper() == "NO_ANSWER" and self.fallback_code and not fallback_executed:
                    last_error = "Auto-finalize blocked: NO_ANSWER."
                    return None
                if (
                    self.auto_finalize_min_length > 0
                    and len(cleaned) < self.auto_finalize_min_length
                ):
                    last_error = (
                        f"Auto-finalize blocked: answer too short ({len(cleaned)} chars, "
                        f"minimum {self.auto_finalize_min_length}). Keep processing."
                    )
                    return None
                if self.auto_finalize_reject_patterns:
                    import re

                    for pattern in self.auto_finalize_reject_patterns:
                        if re.search(pattern, cleaned, re.IGNORECASE):
                            last_error = (
                                f"Auto-finalize blocked: answer matches reject pattern "
                                f"'{pattern}'. Rewrite {self.auto_finalize_var} with the "
                                f"FULL content — do not use references."
                            )
                            return None
                value = cleaned
            if self.min_steps > 0 and policy.steps < self.min_steps:
                last_error = (
                    f"Auto-finalize blocked: step {policy.steps}/{self.min_steps} "
                    f"(min_steps={self.min_steps}). Keep processing."
                )
                return None
            if _can_finalize(
                require_repl=self.require_repl_before_final,
                repl_executed=repl_executed,
                require_subcall=self.require_subcall_before_final,
                subcall_made=subcall_made,
                min_steps=self.min_steps,
                current_step=policy.steps,
            ):
                return str(value)
            return None

        def maybe_finish_common_result_var() -> str | None:
            for var_name in ("final_answer", "answer", "result", "summary"):
                value = repl.get(var_name)
                if not isinstance(value, str):
                    continue
                cleaned = value.strip()
                if cleaned and cleaned.upper() != "NO_ANSWER":
                    return cleaned
            return None

        def run_fallback(reason: str) -> bool:
            nonlocal last_stdout, last_error, last_state_summary, repl_executed, fallback_executed
            if not self.fallback_code or fallback_executed:
                return False
            logger.debug("executing fallback code reason=%s", reason)
            fallback_started = time.perf_counter()
            state_before = snapshot_repl_state()
            result = repl.exec(self.fallback_code)
            restore_scaffold()
            state_after = snapshot_repl_state()
            last_stdout = result.stdout
            last_error = result.error
            last_state_summary = None
            if not result.stdout and not result.error:
                last_state_summary = summarize_repl_state_change(state_before, state_after)
            repl_executed = True
            fallback_executed = True
            if result.error:
                logger.debug("fallback error=%s", result.error)
            if result.stdout:
                logger.debug("fallback stdout=%s", _truncate(result.stdout, self.log_truncate_output))
            if last_state_summary:
                logger.debug("fallback state=%s", _truncate(last_state_summary, self.log_truncate_output))
            add_step(
                TraceStep(
                    step_id=next_step_id(),
                    kind="repl_exec",
                    depth=0,
                    code=_truncate(self.fallback_code, self.log_truncate_code),
                    stdout=result.stdout,
                    error=result.error,
                    elapsed=time.perf_counter() - fallback_started,
                )
            )
            return True

        def maybe_run_subcall_guard() -> bool:
            if (
                self.require_subcall_before_final
                and not subcall_made
                and self.subcall_guard_steps is not None
                and policy.steps >= self.subcall_guard_steps
            ):
                return run_fallback("subcall_guard")
            return False

        def maybe_run_fallback_guard() -> bool:
            if self.fallback_guard_steps is None:
                return False
            if policy.steps < self.fallback_guard_steps:
                return False
            if not self.auto_finalize_var:
                return False
            value = repl.get(self.auto_finalize_var)
            if value is not None:
                if isinstance(value, str):
                    cleaned = value.strip()
                    lowered = cleaned.lower()
                    if cleaned and lowered not in {"no_answer", "none", "0"}:
                        return False
                else:
                    return False
            return run_fallback("fallback_guard")

        def log_invalid_response_detail(
            detail: str,
            *,
            finish_reason: Any = None,
            text_len: int | None = None,
            usage_total: int | None = None,
        ) -> None:
            payload: dict[str, Any] = {"detail": detail, "step": policy.steps}
            if finish_reason is not None:
                payload["finish_reason"] = finish_reason
            if text_len is not None:
                payload["text_len"] = text_len
            if usage_total is not None:
                payload["usage_total_tokens"] = usage_total
            log_diag("invalid_response_detail", payload)

        # Build effective system prompt (append retrieval docs when retriever is set)
        effective_system_prompt = build_system_prompt(
            self.system_prompt,
            retriever_available=self.retriever is not None,
        )
        if self.system_prompt_supplement:
            effective_system_prompt += self.system_prompt_supplement

        # Initialize conversation history for multi-turn mode
        if self.conversation_history:
            ctx_meta = context.metadata()
            initial_user_message = build_root_user_message(
                query=query,
                context_len=ctx_meta["total_length"],
                context_type=ctx_meta["context_type"],
                num_documents=ctx_meta["num_documents"],
                document_lengths=ctx_meta.get("document_lengths"),
                retriever_available=self.retriever is not None,
                repl_executed=False,
                last_stdout=None,
                last_error=None,
                last_state_summary=None,
                step=1,
                max_steps=policy.max_steps,
            )
            history: list[dict[str, str]] = [
                {"role": "system", "content": effective_system_prompt},
                {"role": "user", "content": initial_user_message},
            ]

        while True:
            try:
                policy.check_step()
            except MaxStepsExceeded:
                # Check auto_finalize_var first (even if below min length, accept at exhaustion)
                if self.auto_finalize_var:
                    value = repl.get(self.auto_finalize_var)
                    if value is not None:
                        text = str(value).strip()
                        if text and text.upper() != "NO_ANSWER":
                            return finish(text)
                # Graceful fallback: ask model for a summary of progress
                if self.conversation_history and history:
                    try:
                        summary_started = time.perf_counter()
                        summary_msgs = list(history) + [
                            {
                                "role": "user",
                                "content": (
                                    "You have used all available steps. Based on all the "
                                    "information you have gathered so far, provide your best "
                                    "final answer NOW. Do NOT write code. Just write the answer "
                                    "directly as plain text."
                                ),
                            }
                        ]
                        if self.max_history_tokens > 0:
                            summary_msgs = _trim_history(summary_msgs, self.max_history_tokens)
                        summary_resp = self.adapter.complete(
                            summary_msgs, max_tokens=self.max_tokens, temperature=0.0
                        )
                        if summary_resp.text and summary_resp.text.strip():
                            add_step(
                                TraceStep(
                                    step_id=next_step_id(),
                                    kind="root_call",
                                    depth=0,
                                    prompt_summary="[max_steps_summary]",
                                    code=_truncate(summary_resp.text, self.log_truncate_code),
                                    output=summary_resp.text,
                                    usage=summary_resp.usage,
                                    elapsed=time.perf_counter() - summary_started,
                                )
                            )
                            return finish(summary_resp.text.strip())
                    except Exception:
                        pass
                if last_stdout and last_stdout.strip():
                    return finish(last_stdout.strip())
                return finish("NO_ANSWER")

            if self.conversation_history:
                # From step 2+, append REPL result from the previous iteration
                if policy.steps > 1:
                    iter_msg = build_iteration_message(
                        last_stdout=last_stdout,
                        last_error=last_error,
                        last_state_summary=last_state_summary,
                        step=policy.steps,
                        max_steps=policy.max_steps,
                    )
                    history.append({"role": "user", "content": iter_msg})
                # Trim history if a token budget is configured
                if self.max_history_tokens > 0:
                    history = _trim_history(history, self.max_history_tokens)
                messages = list(history)  # snapshot for this call
            else:
                # Legacy stateless mode: rebuild from scratch each iteration
                ctx_meta = context.metadata()
                user_message = build_root_user_message(
                    query=query,
                    context_len=ctx_meta["total_length"],
                    context_type=ctx_meta["context_type"],
                    num_documents=ctx_meta["num_documents"],
                    document_lengths=ctx_meta.get("document_lengths"),
                    retriever_available=self.retriever is not None,
                    repl_executed=repl_executed,
                    last_stdout=last_stdout,
                    last_error=last_error,
                    last_state_summary=last_state_summary,
                    step=policy.steps,
                    max_steps=policy.max_steps,
                )
                messages = [
                    {"role": "system", "content": effective_system_prompt},
                    {"role": "user", "content": user_message},
                ]

            log_diag(
                "root_call request_meta",
                {
                    "history_tokens_est": sum(
                        estimate_tokens(msg.get("content", "")) for msg in messages
                    ),
                    "last_user_tokens_est": (
                        estimate_tokens(messages[-1].get("content", "")) if messages else 0
                    ),
                    "max_tokens": self.max_tokens,
                    "messages_count": len(messages),
                    "step": policy.steps,
                },
            )
            logger.debug("root_call step=%s/%s", policy.steps, policy.max_steps)
            root_started = time.perf_counter()
            response = self.adapter.complete(messages, max_tokens=self.max_tokens, temperature=0.0)
            root_elapsed = time.perf_counter() - root_started
            response_meta = response.meta or {}
            finish_reason = response_meta.get("finish_reason")
            log_diag(
                "root_call response_meta",
                {
                    "completion_tokens": response.usage.completion_tokens,
                    "content_kind": response_meta.get("content_kind", "unknown"),
                    "empty_text": not response.text.strip(),
                    "finish_reason": finish_reason,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "reasoning_present": response_meta.get("reasoning_present"),
                    "text_len": len(response.text),
                    "total_tokens": response.usage.total_tokens,
                },
            )
            try:
                policy.add_tokens(response.usage.total_tokens)
            except MaxTokensExceeded:
                logger.warning(
                    "Token budget exceeded after root call; triggering graceful finalization"
                )
                # Still record the response, then finalize with what we have
                if self.conversation_history:
                    history.append({"role": "assistant", "content": response.text})
                prompt_summary = messages[-1]["content"] if messages else ""
                add_step(
                    TraceStep(
                        step_id=next_step_id(),
                        kind="root_call",
                        depth=0,
                        prompt_summary=_truncate(prompt_summary, 240),
                        code=_truncate(response.text, self.log_truncate_code),
                        output=response.text,
                        usage=response.usage,
                        elapsed=root_elapsed,
                    )
                )
                cleaned = response.text.strip()
                final = _parse_final(cleaned)
                # Try auto-finalize first
                if self.auto_finalize_var:
                    value = repl.get(self.auto_finalize_var)
                    if value is not None:
                        text = str(value).strip()
                        if text and text.upper() != "NO_ANSWER":
                            return finish(text)
                resolved = maybe_finish_common_result_var()
                if resolved is not None:
                    return finish(resolved)
                code = _extract_code(cleaned)
                if _looks_like_code(code):
                    logger.debug("repl exec code=%s", _truncate(code, self.log_truncate_code))
                    repl_started = time.perf_counter()
                    state_before = snapshot_repl_state()
                    result = repl.exec(code)
                    restore_scaffold()
                    state_after = snapshot_repl_state()
                    last_stdout = result.stdout
                    last_error = result.error
                    last_state_summary = None
                    if not result.stdout and not result.error:
                        last_state_summary = summarize_repl_state_change(state_before, state_after)
                    repl_executed = True
                    if result.error:
                        logger.debug("repl error=%s", result.error)
                    if result.stdout:
                        logger.debug("repl stdout=%s", _truncate(result.stdout, self.log_truncate_output))
                    if last_state_summary:
                        logger.debug("repl state=%s", _truncate(last_state_summary, self.log_truncate_output))
                    add_step(
                        TraceStep(
                            step_id=next_step_id(),
                            kind="repl_exec",
                            depth=0,
                            code=_truncate(code, self.log_truncate_code),
                            stdout=result.stdout,
                            error=result.error,
                            elapsed=time.perf_counter() - repl_started,
                        )
                    )
                    resolved = maybe_auto_finalize()
                    if resolved is not None:
                        return finish(resolved)
                    resolved = maybe_finish_common_result_var()
                    if resolved is not None:
                        return finish(resolved)
                    if final and _can_finalize(
                        require_repl=self.require_repl_before_final,
                        repl_executed=repl_executed,
                        require_subcall=self.require_subcall_before_final,
                        subcall_made=subcall_made,
                        min_steps=self.min_steps,
                        current_step=policy.steps,
                    ):
                        resolved = _try_resolve_final(final, repl)
                        if resolved is not None:
                            return finish(resolved)
                    if last_stdout and last_stdout.strip():
                        return finish(last_stdout.strip())
                # Run fallback code
                if run_fallback("max_tokens_exceeded"):
                    resolved = maybe_auto_finalize()
                    if resolved is not None:
                        return finish(resolved)
                    resolved = maybe_finish_common_result_var()
                    if resolved is not None:
                        return finish(resolved)
                    if last_stdout and last_stdout.strip():
                        return finish(last_stdout.strip())
                if final and _can_finalize(
                    require_repl=self.require_repl_before_final,
                    repl_executed=repl_executed,
                    require_subcall=self.require_subcall_before_final,
                    subcall_made=subcall_made,
                    min_steps=self.min_steps,
                    current_step=policy.steps,
                ):
                    resolved = _try_resolve_final(final, repl)
                    if resolved is not None:
                        return finish(resolved)
                return finish(cleaned or "NO_ANSWER")

            # Append assistant response to conversation history
            if self.conversation_history:
                history.append({"role": "assistant", "content": response.text})

            # For trace, use the last user message as prompt summary
            prompt_summary = messages[-1]["content"] if messages else ""
            add_step(
                TraceStep(
                    step_id=next_step_id(),
                    kind="root_call",
                    depth=0,
                    prompt_summary=_truncate(prompt_summary, 240),
                    code=_truncate(response.text, self.log_truncate_code),
                    output=response.text,
                    usage=response.usage,
                    elapsed=root_elapsed,
                )
            )

            cleaned = response.text.strip()
            logger.debug("root_call output=%s", _truncate(cleaned, 200))
            if not cleaned:
                invalid_responses += 1
                last_stdout = ""
                last_state_summary = None
                if finish_reason == "length":
                    empty_length_streak += 1
                    last_error = (
                        "Invalid response: empty visible content with finish_reason=length; "
                        f"completion_tokens={response.usage.completion_tokens}; "
                        f"total_tokens={response.usage.total_tokens}."
                    )
                    log_invalid_response_detail(
                        "empty_content_length",
                        finish_reason=finish_reason,
                        text_len=0,
                        usage_total=response.usage.total_tokens,
                    )
                    if (
                        self.invalid_response_limit is not None
                        and empty_length_streak >= self.invalid_response_limit
                    ):
                        logger.warning(
                            "empty_length limit reached (%s), aborting with NO_ANSWER",
                            empty_length_streak,
                        )
                        return finish("NO_ANSWER")
                else:
                    empty_length_streak = 0
                    last_error = "Invalid response: empty visible content."
                    log_invalid_response_detail(
                        "empty_content",
                        finish_reason=finish_reason,
                        text_len=0,
                        usage_total=response.usage.total_tokens,
                    )
                logger.debug("invalid response, skipping repl exec")
                if maybe_run_subcall_guard():
                    resolved = maybe_auto_finalize()
                    if resolved is not None:
                        return finish(resolved)
                if maybe_run_fallback_guard():
                    resolved = maybe_auto_finalize()
                    if resolved is not None:
                        return finish(resolved)
                if (
                    self.invalid_response_limit is not None
                    and invalid_responses >= self.invalid_response_limit
                ):
                    if run_fallback("invalid_response"):
                        resolved = maybe_auto_finalize()
                        if resolved is not None:
                            return finish(resolved)
                continue
            empty_length_streak = 0
            final = _parse_final(cleaned)
            has_fence = "```" in cleaned
            final_unfenced = None if has_fence else final
            logger.debug("root_call classify final=%s fenced=%s", bool(final_unfenced), has_fence)

            if final_unfenced:
                if not _can_finalize(
                    require_repl=self.require_repl_before_final,
                    repl_executed=repl_executed,
                    require_subcall=self.require_subcall_before_final,
                    subcall_made=subcall_made,
                    min_steps=self.min_steps,
                    current_step=policy.steps,
                ):
                    invalid_responses += 1
                    last_stdout = ""
                    last_error = _guard_error(
                        require_repl=self.require_repl_before_final,
                        repl_executed=repl_executed,
                        require_subcall=self.require_subcall_before_final,
                        subcall_made=subcall_made,
                        min_steps=self.min_steps,
                        current_step=policy.steps,
                    )
                    last_state_summary = None
                    log_invalid_response_detail(
                        "blocked_final",
                        finish_reason=finish_reason,
                        text_len=len(cleaned),
                        usage_total=response.usage.total_tokens,
                    )
                    logger.debug("final blocked: %s", last_error)
                    if maybe_run_subcall_guard():
                        resolved = maybe_auto_finalize()
                        if resolved is not None:
                            return finish(resolved)
                    if maybe_run_fallback_guard():
                        resolved = maybe_auto_finalize()
                        if resolved is not None:
                            return finish(resolved)
                    if (
                        self.invalid_response_limit is not None
                        and invalid_responses >= self.invalid_response_limit
                    ):
                        if run_fallback("guard"):
                            resolved = maybe_auto_finalize()
                            if resolved is not None:
                                return finish(resolved)
                    continue
                resolved = _try_resolve_final(final_unfenced, repl)
                if resolved is None:
                    last_stdout = ""
                    last_error = "FINAL_VAR missing in REPL; set the variable before finalizing."
                    last_state_summary = None
                    log_invalid_response_detail(
                        "missing_final_var",
                        finish_reason=finish_reason,
                        text_len=len(cleaned),
                        usage_total=response.usage.total_tokens,
                    )
                    if run_fallback("final_var_missing"):
                        resolved = maybe_auto_finalize()
                        if resolved is not None:
                            return finish(resolved)
                    continue
                return finish(resolved)

            code = _extract_code(cleaned)
            logger.debug("root_call extracted code=%s", _truncate(code, self.log_truncate_code))
            final_in_code = _parse_final(code)
            looks_like_code = _looks_like_code(code) or (has_fence and _can_compile_python(code))
            if final_in_code and not looks_like_code:
                if not _can_finalize(
                    require_repl=self.require_repl_before_final,
                    repl_executed=repl_executed,
                    require_subcall=self.require_subcall_before_final,
                    subcall_made=subcall_made,
                    min_steps=self.min_steps,
                    current_step=policy.steps,
                ):
                    invalid_responses += 1
                    last_stdout = ""
                    last_error = _guard_error(
                        require_repl=self.require_repl_before_final,
                        repl_executed=repl_executed,
                        require_subcall=self.require_subcall_before_final,
                        subcall_made=subcall_made,
                        min_steps=self.min_steps,
                        current_step=policy.steps,
                    )
                    last_state_summary = None
                    log_invalid_response_detail(
                        "blocked_final",
                        finish_reason=finish_reason,
                        text_len=len(code),
                        usage_total=response.usage.total_tokens,
                    )
                    logger.debug("final in code blocked: %s", last_error)
                    if maybe_run_subcall_guard():
                        resolved = maybe_auto_finalize()
                        if resolved is not None:
                            return finish(resolved)
                    if maybe_run_fallback_guard():
                        resolved = maybe_auto_finalize()
                        if resolved is not None:
                            return finish(resolved)
                    if (
                        self.invalid_response_limit is not None
                        and invalid_responses >= self.invalid_response_limit
                    ):
                        if run_fallback("guard"):
                            resolved = maybe_auto_finalize()
                            if resolved is not None:
                                return finish(resolved)
                    continue
                resolved = _try_resolve_final(final_in_code, repl)
                if resolved is None:
                    last_stdout = ""
                    last_error = "FINAL_VAR missing in REPL; set the variable before finalizing."
                    last_state_summary = None
                    log_invalid_response_detail(
                        "missing_final_var",
                        finish_reason=finish_reason,
                        text_len=len(code),
                        usage_total=response.usage.total_tokens,
                    )
                    if run_fallback("final_var_missing"):
                        resolved = maybe_auto_finalize()
                        if resolved is not None:
                            return finish(resolved)
                    continue
                return finish(resolved)

            if not looks_like_code:
                invalid_responses += 1
                last_stdout = ""
                last_error = (
                    "Invalid response: expected Python code or FINAL. "
                    f"Received non-code text (len={len(cleaned)})."
                )
                last_state_summary = None
                log_invalid_response_detail(
                    "non_code_text",
                    finish_reason=finish_reason,
                    text_len=len(cleaned),
                    usage_total=response.usage.total_tokens,
                )
                logger.debug("invalid response, skipping repl exec")
                if maybe_run_subcall_guard():
                    resolved = maybe_auto_finalize()
                    if resolved is not None:
                        return finish(resolved)
                if maybe_run_fallback_guard():
                    resolved = maybe_auto_finalize()
                    if resolved is not None:
                        return finish(resolved)
                if (
                    self.invalid_response_limit is not None
                    and invalid_responses >= self.invalid_response_limit
                ):
                    if run_fallback("invalid_response"):
                        resolved = maybe_auto_finalize()
                        if resolved is not None:
                            return finish(resolved)
                continue

            invalid_responses = 0
            logger.debug("repl exec code=%s", _truncate(code, self.log_truncate_code))
            repl_started = time.perf_counter()
            state_before = snapshot_repl_state()
            result = repl.exec(code)
            # Restore scaffold names immediately after execution so accidental
            # overwrites (e.g. `llm_query = None`, `P = "x"`) don't persist.
            restore_scaffold()
            state_after = snapshot_repl_state()
            last_stdout = result.stdout
            last_error = result.error
            last_state_summary = None
            if not result.stdout and not result.error:
                last_state_summary = summarize_repl_state_change(state_before, state_after)
            repl_executed = True
            if result.error:
                repl_errors += 1
                logger.debug("repl error=%s", result.error)
                if self.repl_error_limit is not None and repl_errors >= self.repl_error_limit:
                    if run_fallback("repl_error_limit"):
                        resolved = maybe_auto_finalize()
                        if resolved is not None:
                            return finish(resolved)
            if result.stdout:
                logger.debug("repl stdout=%s", _truncate(result.stdout, self.log_truncate_output))
            if last_state_summary:
                logger.debug("repl state=%s", _truncate(last_state_summary, self.log_truncate_output))
            add_step(
                TraceStep(
                    step_id=next_step_id(),
                    kind="repl_exec",
                    depth=0,
                    code=_truncate(code, self.log_truncate_code),
                    stdout=result.stdout,
                    error=result.error,
                    elapsed=time.perf_counter() - repl_started,
                )
            )
            if self.auto_finalize_var:
                value = repl.get(self.auto_finalize_var)
                if (
                    isinstance(value, str)
                    and value.strip().upper() == "NO_ANSWER"
                    and self.fallback_code
                    and not fallback_executed
                ):
                    if run_fallback("no_answer"):
                        resolved = maybe_auto_finalize()
                        if resolved is not None:
                            return finish(resolved)
            if maybe_run_fallback_guard():
                resolved = maybe_auto_finalize()
                if resolved is not None:
                    return finish(resolved)
            resolved = maybe_auto_finalize()
            if resolved is not None:
                return finish(resolved)
            if maybe_run_subcall_guard():
                resolved = maybe_auto_finalize()
                if resolved is not None:
                    return finish(resolved)
            if final_unfenced and _can_finalize(
                require_repl=self.require_repl_before_final,
                repl_executed=repl_executed,
                require_subcall=self.require_subcall_before_final,
                subcall_made=subcall_made,
                min_steps=self.min_steps,
                current_step=policy.steps,
            ):
                resolved = _try_resolve_final(final_unfenced, repl)
                if resolved is None:
                    last_stdout = ""
                    last_error = "FINAL_VAR missing in REPL; set the variable before finalizing."
                    last_state_summary = None
                    if run_fallback("final_var_missing"):
                        resolved = maybe_auto_finalize()
                        if resolved is not None:
                            return finish(resolved)
                    continue
                return finish(resolved)


def _can_finalize(
    *,
    require_repl: bool,
    repl_executed: bool,
    require_subcall: bool,
    subcall_made: bool,
    min_steps: int = 0,
    current_step: int = 0,
) -> bool:
    if min_steps > 0 and current_step < min_steps:
        return False
    if require_repl and not repl_executed:
        return False
    if require_subcall and not subcall_made:
        return False
    return True


def _guard_error(
    *,
    require_repl: bool,
    repl_executed: bool,
    require_subcall: bool,
    subcall_made: bool,
    min_steps: int = 0,
    current_step: int = 0,
) -> str:
    if min_steps > 0 and current_step < min_steps:
        return f"Guard: step {current_step}/{min_steps}, keep exploring before FINAL."
    if require_repl and not repl_executed:
        return "Guard: execute REPL code before FINAL."
    if require_subcall and not subcall_made:
        return "Guard: execute at least one subcall before FINAL."
    return "Guard: conditions not met."


def _extract_code(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        if lines and lines[0].strip().lower() in {"python", "repl"}:
            lines = lines[1:]
        return "\n".join(lines).strip()

    if "```" in stripped:
        parts = stripped.split("```")
        if len(parts) >= 3:
            code = parts[1].strip()
            lines = code.splitlines()
            if lines and lines[0].strip().lower() in {"python", "repl"}:
                code = "\n".join(lines[1:]).strip()
            return code
    lines = stripped.splitlines()
    if lines and lines[0].strip().lower() in {"python", "repl"}:
        return "\n".join(lines[1:]).strip()
    return stripped


def _looks_like_code(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    first = ""
    for line in stripped.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if candidate.startswith("#"):
            continue
        first = candidate
        break
    if not first:
        return False
    if first.lower() in {"python", "repl"}:
        return True
    if first.startswith(('"""', "'''")):
        return True
    if "=" in first:
        return True
    starters = (
        "import ",
        "from ",
        "def ",
        "class ",
        "for ",
        "while ",
        "if ",
        "try:",
        "key ",
        "snippet ",
        "summary ",
        "answer ",
        "buffer ",
        "chunks ",
        "answers ",
        "print(",
        "ctx.",
    )
    if first.startswith(starters):
        return True
    if re.match(r"^P(?:\.|\[|\(|\s*=)", first):
        return True
    if re.match(r"^(?:ask(?:_[A-Za-z0-9]+)?|llm_(?:query|batch)(?:_[A-Za-z0-9]+)?|extract_after|es_[A-Za-z0-9_]+|SHOW_VARS)\s*\(", first):
        return True
    return False


def _can_compile_python(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    try:
        compile(stripped, "<rlm-sniff>", "eval")
        return True
    except SyntaxError:
        try:
            ast.parse(stripped, mode="exec")
            return True
        except SyntaxError:
            return False


def _parse_final(text: str) -> tuple[str, str] | None:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("FINAL_VAR:"):
            return ("FINAL_VAR", stripped.split(":", 1)[1].strip())
        if stripped.startswith("FINAL:"):
            first_line = stripped.split(":", 1)[1].strip()
            remainder = "\n".join(lines[idx + 1 :]).rstrip()
            if remainder:
                return ("FINAL", f"{first_line}\n{remainder}".rstrip())
            return ("FINAL", first_line)
        if stripped.startswith("FINAL_VAR(") and stripped.endswith(")"):
            return ("FINAL_VAR", stripped[len("FINAL_VAR(") : -1].strip())
        if stripped.startswith("FINAL(") and stripped.endswith(")"):
            return ("FINAL", stripped[len("FINAL(") : -1].strip())
    return None


def _resolve_final(final: tuple[str, str], repl: PythonREPL) -> str:
    kind, value = final
    if kind == "FINAL_VAR":
        var_name = value.strip("\"'")
        resolved = repl.get(var_name)
        if resolved is None:
            raise ValueError(f"FINAL_VAR missing: {var_name}")
        return str(resolved)
    return value


def _try_resolve_final(final: tuple[str, str], repl: PythonREPL) -> str | None:
    kind, value = final
    if kind == "FINAL_VAR":
        var_name = value.strip("\"'")
        resolved = repl.get(var_name)
        if resolved is None:
            return None
        return str(resolved)
    return value


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_key(*, text: str, model: str | None, max_tokens: int, recursive: bool = False) -> str:
    model_part = model or "default"
    rec_part = "recursive" if recursive else "simple"
    return f"model={model_part}|max_tokens={max_tokens}|mode={rec_part}|text={text}"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _trace_total_tokens(trace: Trace) -> int:
    return sum(step.usage.total_tokens for step in trace.steps if step.usage)


def _trim_history(
    messages: list[dict[str, str]],
    max_tokens: int,
) -> list[dict[str, str]]:
    """Trim conversation history to fit within a token budget.

    Strategy: always keep messages[0] (system) and messages[1] (initial user
    message with full context).  Drop oldest middle turns first, keeping the
    most recent turns.
    """
    if max_tokens <= 0 or len(messages) <= 2:
        return messages

    total = sum(estimate_tokens(m.get("content", "")) for m in messages)
    if total <= max_tokens:
        return messages

    preserved_head = messages[:2]
    tail = messages[2:]

    head_tokens = sum(estimate_tokens(m.get("content", "")) for m in preserved_head)
    remaining_budget = max_tokens - head_tokens
    if remaining_budget <= 0:
        return preserved_head

    # Walk backward through tail in (assistant, user) pairs to preserve
    # role alternation.  tail starts with an assistant message and alternates
    # assistant, user, assistant, user, ...
    kept: list[dict[str, str]] = []
    accumulated = 0
    i = len(tail) - 1
    while i >= 1:
        pair = [tail[i - 1], tail[i]]
        pair_tokens = sum(estimate_tokens(m.get("content", "")) for m in pair)
        if accumulated + pair_tokens > remaining_budget:
            break
        # Append in reverse order; kept.reverse() below restores proper order
        kept.append(pair[1])
        kept.append(pair[0])
        accumulated += pair_tokens
        i -= 2

    kept.reverse()
    return preserved_head + kept


def _run_recursive_subcall(
    *,
    text: str,
    adapter: ModelAdapter,
    system_prompt: str,
    max_steps: int,
    max_tokens: int,
    depth: int,
    logger: logging.Logger,
    create_repl: Callable[[], REPLProtocol] | None = None,
    conversation_history: bool = True,
    max_history_tokens: int = 0,
    log_truncate_code: int = 2000,
) -> tuple[str, Trace]:
    """Run a mini-RLM loop for a recursive subcall.

    This implements the paper's concept of recursive subcalls where each subcall
    can itself run an RLM loop to process its portion of the context.
    """
    from .prompts import build_root_user_message

    trace = Trace(steps=[])
    repl = create_repl() if create_repl is not None else PythonREPL()
    sub_context = Context.from_text(text)

    repl.set("P", sub_context.text)
    repl.set("ctx", sub_context)

    def peek(n: int = 2000) -> str:
        return sub_context.text[:n]

    def tail(n: int = 2000) -> str:
        return sub_context.text[-n:]

    def lenp() -> int:
        return sub_context.len_chars()

    repl.set("peek", peek)
    repl.set("tail", tail)
    repl.set("lenP", lenp)

    step_id = 0

    def next_step_id() -> int:
        nonlocal step_id
        step_id += 1
        return step_id

    # Simple subcall for nested calls (non-recursive to avoid infinite depth)
    def simple_subcall(query_text: str, *, max_toks: int = 256) -> str:
        from .prompts import SUBCALL_SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": SUBCALL_SYSTEM_PROMPT},
            {"role": "user", "content": query_text},
        ]
        subcall_started = time.perf_counter()
        response = adapter.complete(messages, max_tokens=max_toks, temperature=0.0)
        trace.add(
            TraceStep(
                step_id=next_step_id(),
                kind="subcall",
                depth=depth + 1,
                prompt_summary=_truncate(query_text, 240),
                output=_truncate(response.text, log_truncate_code),
                usage=response.usage,
                elapsed=time.perf_counter() - subcall_started,
                cache_hit=False,
                input_hash=_hash_text(query_text),
                output_hash=_hash_text(response.text),
            )
        )
        return response.text

    repl.set("llm_query", simple_subcall)
    repl.set(
        "ask",
        lambda q, t, max_tokens=256: simple_subcall(
            f"Question: {q}\nSnippet:\n{t}", max_toks=max_tokens
        ),
    )

    last_stdout: str | None = None
    last_error: str | None = None
    last_state_summary: str | None = None
    repl_executed = False

    # Extract the question from the text (format: "Question: ...\nSnippet:\n...")
    query = "Answer the question based on the provided context."
    if text.startswith("Question:"):
        q_lines = text.split("\n", 1)
        query = q_lines[0].replace("Question:", "").strip()

    # Initialize conversation history for multi-turn mode
    if conversation_history:
        initial_user_msg = build_root_user_message(
            query=query,
            context_len=sub_context.len_chars(),
            repl_executed=False,
            last_stdout=None,
            last_error=None,
            step=1,
            max_steps=max_steps,
        )
        sub_history: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user_msg},
        ]

    for step in range(max_steps):
        if conversation_history:
            if step > 0:
                iter_msg = build_iteration_message(
                    last_stdout=last_stdout,
                    last_error=last_error,
                    last_state_summary=last_state_summary,
                    step=step + 1,
                    max_steps=max_steps,
                )
                sub_history.append({"role": "user", "content": iter_msg})
            if max_history_tokens > 0:
                sub_history = _trim_history(sub_history, max_history_tokens)
            messages = list(sub_history)
        else:
            user_message = build_root_user_message(
                query=query,
                context_len=sub_context.len_chars(),
                repl_executed=repl_executed,
                last_stdout=last_stdout,
                last_error=last_error,
                step=step + 1,
                max_steps=max_steps,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

        logger.debug("recursive_subcall step=%s/%s depth=%s", step + 1, max_steps, depth)
        root_started = time.perf_counter()
        response = adapter.complete(messages, max_tokens=max_tokens, temperature=0.0)
        root_elapsed = time.perf_counter() - root_started

        if conversation_history:
            sub_history.append({"role": "assistant", "content": response.text})

        prompt_summary = messages[-1]["content"] if messages else ""
        trace.add(
            TraceStep(
                step_id=next_step_id(),
                kind="root_call",
                depth=depth,
                prompt_summary=_truncate(prompt_summary, 240),
                code=_truncate(response.text, log_truncate_code),
                output=response.text,
                usage=response.usage,
                elapsed=root_elapsed,
            )
        )

        cleaned = response.text.strip()
        final = _parse_final(cleaned)
        has_fence = "```" in cleaned
        final_unfenced = None if has_fence else final

        if final_unfenced:
            resolved = _try_resolve_final(final_unfenced, repl)
            if resolved is not None:
                return resolved, trace
            # If FINAL_VAR but variable not set, treat as error
            last_error = "FINAL_VAR variable not found."
            last_state_summary = None
            continue

        code = _extract_code(cleaned)
        final_in_code = _parse_final(code)
        looks_like_code = _looks_like_code(code) or (has_fence and _can_compile_python(code))
        if final_in_code and not looks_like_code:
            resolved = _try_resolve_final(final_in_code, repl)
            if resolved is not None:
                return resolved, trace
            last_error = "FINAL_VAR variable not found."
            last_state_summary = None
            continue

        if not looks_like_code:
            last_error = "Invalid response: expected Python code or FINAL."
            last_state_summary = None
            continue

        repl_started = time.perf_counter()
        result = repl.exec(code)
        last_stdout = result.stdout
        last_error = result.error
        last_state_summary = None
        if not result.stdout and not result.error and hasattr(repl, "describe_state"):
            try:
                last_state_summary = repl.describe_state(max_items=12)
            except Exception:
                last_state_summary = None
        repl_executed = True
        trace.add(
            TraceStep(
                step_id=next_step_id(),
                kind="repl_exec",
                depth=depth,
                code=_truncate(code, log_truncate_code),
                stdout=result.stdout,
                error=result.error,
                elapsed=time.perf_counter() - repl_started,
            )
        )

        # Check for FINAL after code execution
        if final_unfenced:
            resolved = _try_resolve_final(final_unfenced, repl)
            if resolved is not None:
                return resolved, trace

    # Max steps reached, return best effort from stdout or NO_ANSWER
    if last_stdout and last_stdout.strip():
        return last_stdout.strip(), trace
    return "NO_ANSWER", trace
