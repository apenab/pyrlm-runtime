from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from .adapters.base import ModelAdapter
from .cache import CacheRecord, FileCache
from .context import Context
from .env import PythonREPL, REPLProtocol
from .policy import Policy
from .prompts import (
    BASE_SYSTEM_PROMPT,
    SUBCALL_SYSTEM_PROMPT,
    RECURSIVE_SUBCALL_SYSTEM_PROMPT,
    build_root_user_message,
)
from .trace import Trace, TraceStep


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
    # REPL backend: "python" (default) or "monty" (pydantic-monty sandbox)
    repl_backend: str = "python"

    def _create_repl(self) -> REPLProtocol:
        if self.repl_backend == "python":
            return PythonREPL()
        if self.repl_backend == "monty":
            from .env_monty import MontyREPL

            return MontyREPL()
        raise ValueError(
            f"Invalid repl_backend={self.repl_backend!r}. "
            f"Expected 'python' or 'monty'."
        )

    def run(self, query: str, context: Context) -> tuple[str, Trace]:
        logger = self.logger or logging.getLogger("pyrlm_runtime")
        policy = self.policy or Policy()
        cache = self.cache or FileCache(self.cache_dir)
        trace = Trace(steps=[])
        repl = self._create_repl()

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

        def next_step_id() -> int:
            nonlocal step_id
            step_id += 1
            return step_id

        # Select adapter for subcalls (paper-aligned: can use different/cheaper model)
        effective_subcall_adapter = self.subcall_adapter or self.adapter

        def subcall(
            text: str,
            *,
            model: str | None = None,
            max_tokens: int = 256,
            depth: int = 1,
        ) -> str:
            nonlocal subcall_made
            policy.check_subcall(depth)

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
                policy.add_subcall_tokens(cached.usage.total_tokens)
                trace.add(
                    TraceStep(
                        step_id=next_step_id(),
                        kind="subcall",
                        depth=depth,
                        prompt_summary=_truncate(text, 240),
                        usage=cached.usage,
                        cache_hit=True,
                        input_hash=input_hash,
                        output_hash=_hash_text(cached.text),
                        cache_key=cache_key,
                    )
                )
                return cached.text

            # Paper-aligned: recursive subcalls run a mini-RLM instead of single LLM call
            if self.recursive_subcalls and depth < self.max_recursion_depth:
                result_text, sub_trace = _run_recursive_subcall(
                    text=text,
                    adapter=effective_subcall_adapter,
                    system_prompt=self.recursive_subcall_system_prompt,
                    max_steps=self.recursive_subcall_max_steps,
                    max_tokens=max_tokens,
                    depth=depth,
                    logger=logger,
                    create_repl=self._create_repl,
                )
                subcall_made = True
                # Aggregate usage from sub-trace
                total_tokens = sum(
                    s.usage.total_tokens for s in sub_trace.steps if s.usage
                )
                from .adapters.base import Usage
                aggregated_usage = Usage(
                    prompt_tokens=0, completion_tokens=0, total_tokens=total_tokens
                )
                policy.add_subcall_tokens(total_tokens)
                cache.set(cache_key, CacheRecord(text=result_text, usage=aggregated_usage))
                trace.add(
                    TraceStep(
                        step_id=next_step_id(),
                        kind="recursive_subcall",
                        depth=depth,
                        prompt_summary=_truncate(text, 240),
                        usage=aggregated_usage,
                        cache_hit=False,
                        input_hash=input_hash,
                        output_hash=_hash_text(result_text),
                        cache_key=cache_key,
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
                        stdout=sub_step.stdout,
                        error=sub_step.error,
                        usage=sub_step.usage,
                        cache_hit=sub_step.cache_hit,
                        input_hash=sub_step.input_hash,
                        output_hash=sub_step.output_hash,
                        cache_key=sub_step.cache_key,
                    )
                    trace.add(sub_step_copy)
                return result_text

            # Standard subcall: single LLM call
            messages = [
                {"role": "system", "content": self.subcall_system_prompt},
                {"role": "user", "content": text},
            ]
            response = effective_subcall_adapter.complete(
                messages, max_tokens=max_tokens, temperature=0.0
            )
            subcall_made = True
            logger.debug("subcall complete depth=%s tokens=%s", depth, response.usage.total_tokens)
            policy.add_subcall_tokens(response.usage.total_tokens)
            cache.set(cache_key, CacheRecord(text=response.text, usage=response.usage))
            trace.add(
                TraceStep(
                    step_id=next_step_id(),
                    kind="subcall",
                    depth=depth,
                    prompt_summary=_truncate(text, 240),
                    usage=response.usage,
                    cache_hit=False,
                    input_hash=input_hash,
                    output_hash=_hash_text(response.text),
                    cache_key=cache_key,
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
            max_tokens: int = 256,
            chunk_size: int | None = None,
            overlap: int = 0,
            question: str | None = None,
            parallel: bool | None = None,
        ) -> list[str]:
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
                # Paper-aligned: parallel subcalls for efficiency
                unique_results: list[str | None] = [None] * len(unique_chunks)

                def process_chunk(idx: int, chunk: str) -> tuple[int, str]:
                    return idx, subcall(chunk, model=model, max_tokens=max_tokens)

                max_workers = min(self.max_concurrent_subcalls, len(unique_chunks))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(process_chunk, i, c): i
                        for i, c in enumerate(unique_chunks)
                    }
                    for future in as_completed(futures):
                        idx, result = future.result()
                        unique_results[idx] = result

                # Map results back to original order
                chunk_to_result = {
                    c: unique_results[i] for i, c in enumerate(unique_chunks)
                }
                return [chunk_to_result[c] or "" for c in prepared]
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

        def ask(question: str, text: str, *, max_tokens: int = 256) -> str:
            return subcall(f"Question: {question}\nSnippet:\n{text}", max_tokens=max_tokens)

        def ask_chunk(question: str, text: str, *, max_tokens: int = 256) -> str:
            return ask(question, text, max_tokens=max_tokens)

        def ask_chunked(
            question: str,
            chunks: object,
            *,
            max_tokens: int = 256,
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
            max_tokens: int = 256,
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
            max_tokens: int = 256,
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
        repl.set("ask", ask)
        repl.set("ask_chunk", ask_chunk)
        repl.set("ask_chunked", ask_chunked)
        repl.set("ask_chunks", ask_chunks)
        repl.set("ask_chunks_first", ask_chunks_first)
        repl.set("pick_first_answer", pick_first_answer)
        repl.set("extract_after", extract_after)

        last_stdout: str | None = None
        last_error: str | None = None
        repl_executed = False
        subcall_made = False
        invalid_responses = 0
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
                if (
                    cleaned.upper() == "NO_ANSWER"
                    and self.fallback_code
                    and not fallback_executed
                ):
                    last_error = "Auto-finalize blocked: NO_ANSWER."
                    return None
                value = cleaned
            if _can_finalize(
                require_repl=self.require_repl_before_final,
                repl_executed=repl_executed,
                require_subcall=self.require_subcall_before_final,
                subcall_made=subcall_made,
            ):
                return str(value)
            return None

        def run_fallback(reason: str) -> bool:
            nonlocal last_stdout, last_error, repl_executed, fallback_executed
            if not self.fallback_code or fallback_executed:
                return False
            logger.debug("executing fallback code reason=%s", reason)
            result = repl.exec(self.fallback_code)
            last_stdout = result.stdout
            last_error = result.error
            repl_executed = True
            fallback_executed = True
            if result.error:
                logger.debug("fallback error=%s", result.error)
            if result.stdout:
                logger.debug("fallback stdout=%s", _truncate(result.stdout, 200))
            trace.add(
                TraceStep(
                    step_id=next_step_id(),
                    kind="repl_exec",
                    depth=0,
                    code=_truncate(self.fallback_code, 800),
                    stdout=result.stdout,
                    error=result.error,
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

        while True:
            policy.check_step()
            # Get context metadata for richer user message (paper-aligned)
            ctx_meta = context.metadata()
            user_message = build_root_user_message(
                query=query,
                context_len=ctx_meta["total_length"],
                context_type=ctx_meta["context_type"],
                num_documents=ctx_meta["num_documents"],
                document_lengths=ctx_meta.get("document_lengths"),
                repl_executed=repl_executed,
                last_stdout=last_stdout,
                last_error=last_error,
                step=policy.steps,
                max_steps=policy.max_steps,
            )
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ]
            logger.debug("root_call step=%s/%s", policy.steps, policy.max_steps)
            response = self.adapter.complete(messages, max_tokens=self.max_tokens, temperature=0.0)
            policy.add_tokens(response.usage.total_tokens)
            trace.add(
                TraceStep(
                    step_id=next_step_id(),
                    kind="root_call",
                    depth=0,
                    prompt_summary=_truncate(user_message, 240),
                    code=_truncate(response.text, 800),
                    usage=response.usage,
                )
            )

            cleaned = response.text.strip()
            logger.debug("root_call output=%s", _truncate(cleaned, 200))
            final = _parse_final(cleaned)
            has_fence = "```" in cleaned
            final_unfenced = None if has_fence else final
            logger.debug(
                "root_call classify final=%s fenced=%s", bool(final_unfenced), has_fence
            )

            if final_unfenced:
                if not _can_finalize(
                    require_repl=self.require_repl_before_final,
                    repl_executed=repl_executed,
                    require_subcall=self.require_subcall_before_final,
                    subcall_made=subcall_made,
                ):
                    invalid_responses += 1
                    last_stdout = ""
                    last_error = _guard_error(
                        require_repl=self.require_repl_before_final,
                        repl_executed=repl_executed,
                        require_subcall=self.require_subcall_before_final,
                        subcall_made=subcall_made,
                    )
                    logger.debug("final blocked: %s", last_error)
                    if maybe_run_subcall_guard():
                        resolved = maybe_auto_finalize()
                        if resolved is not None:
                            return resolved, trace
                    if maybe_run_fallback_guard():
                        resolved = maybe_auto_finalize()
                        if resolved is not None:
                            return resolved, trace
                    if (
                        self.invalid_response_limit is not None
                        and invalid_responses >= self.invalid_response_limit
                    ):
                        if run_fallback("guard"):
                            resolved = maybe_auto_finalize()
                            if resolved is not None:
                                return resolved, trace
                    continue
                resolved = _try_resolve_final(final_unfenced, repl)
                if resolved is None:
                    last_stdout = ""
                    last_error = "FINAL_VAR missing in REPL; set the variable before finalizing."
                    if run_fallback("final_var_missing"):
                        resolved = maybe_auto_finalize()
                        if resolved is not None:
                            return resolved, trace
                    continue
                return resolved, trace

            code = _extract_code(cleaned)
            logger.debug("root_call extracted code=%s", _truncate(code, 200))
            final_in_code = _parse_final(code)
            if final_in_code and not _looks_like_code(code):
                if not _can_finalize(
                    require_repl=self.require_repl_before_final,
                    repl_executed=repl_executed,
                    require_subcall=self.require_subcall_before_final,
                    subcall_made=subcall_made,
                ):
                    invalid_responses += 1
                    last_stdout = ""
                    last_error = _guard_error(
                        require_repl=self.require_repl_before_final,
                        repl_executed=repl_executed,
                        require_subcall=self.require_subcall_before_final,
                        subcall_made=subcall_made,
                    )
                    logger.debug("final in code blocked: %s", last_error)
                    if maybe_run_subcall_guard():
                        resolved = maybe_auto_finalize()
                        if resolved is not None:
                            return resolved, trace
                    if maybe_run_fallback_guard():
                        resolved = maybe_auto_finalize()
                        if resolved is not None:
                            return resolved, trace
                    if (
                        self.invalid_response_limit is not None
                        and invalid_responses >= self.invalid_response_limit
                    ):
                        if run_fallback("guard"):
                            resolved = maybe_auto_finalize()
                            if resolved is not None:
                                return resolved, trace
                    continue
                resolved = _try_resolve_final(final_in_code, repl)
                if resolved is None:
                    last_stdout = ""
                    last_error = "FINAL_VAR missing in REPL; set the variable before finalizing."
                    if run_fallback("final_var_missing"):
                        resolved = maybe_auto_finalize()
                        if resolved is not None:
                            return resolved, trace
                    continue
                return resolved, trace

            if not _looks_like_code(code):
                invalid_responses += 1
                last_stdout = ""
                last_error = "Invalid response: expected Python code or FINAL."
                logger.debug("invalid response, skipping repl exec")
                if maybe_run_subcall_guard():
                    resolved = maybe_auto_finalize()
                    if resolved is not None:
                        return resolved, trace
                if maybe_run_fallback_guard():
                    resolved = maybe_auto_finalize()
                    if resolved is not None:
                        return resolved, trace
                if (
                    self.invalid_response_limit is not None
                    and invalid_responses >= self.invalid_response_limit
                ):
                    if run_fallback("invalid_response"):
                        resolved = maybe_auto_finalize()
                        if resolved is not None:
                            return resolved, trace
                continue

            logger.debug("repl exec code=%s", _truncate(code, 200))
            result = repl.exec(code)
            last_stdout = result.stdout
            last_error = result.error
            repl_executed = True
            if result.error:
                repl_errors += 1
                logger.debug("repl error=%s", result.error)
                if self.repl_error_limit is not None and repl_errors >= self.repl_error_limit:
                    if run_fallback("repl_error_limit"):
                        resolved = maybe_auto_finalize()
                        if resolved is not None:
                            return resolved, trace
            if result.stdout:
                logger.debug("repl stdout=%s", _truncate(result.stdout, 200))
            trace.add(
                TraceStep(
                    step_id=next_step_id(),
                    kind="repl_exec",
                    depth=0,
                    code=_truncate(code, 800),
                    stdout=result.stdout,
                    error=result.error,
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
                            return resolved, trace
            if maybe_run_fallback_guard():
                resolved = maybe_auto_finalize()
                if resolved is not None:
                    return resolved, trace
            resolved = maybe_auto_finalize()
            if resolved is not None:
                return resolved, trace
            if maybe_run_subcall_guard():
                resolved = maybe_auto_finalize()
                if resolved is not None:
                    return resolved, trace
            if final_unfenced and _can_finalize(
                require_repl=self.require_repl_before_final,
                repl_executed=repl_executed,
                require_subcall=self.require_subcall_before_final,
                subcall_made=subcall_made,
            ):
                resolved = _try_resolve_final(final_unfenced, repl)
                if resolved is None:
                    last_stdout = ""
                    last_error = "FINAL_VAR missing in REPL; set the variable before finalizing."
                    if run_fallback("final_var_missing"):
                        resolved = maybe_auto_finalize()
                        if resolved is not None:
                            return resolved, trace
                    continue
                return resolved, trace


def _can_finalize(
    *,
    require_repl: bool,
    repl_executed: bool,
    require_subcall: bool,
    subcall_made: bool,
) -> bool:
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
) -> str:
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
    if first.startswith(("\"\"\"", "'''")):
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
        "P",
        "ask",
        "llm_query",
        "extract_after",
    )
    return first.startswith(starters)


def _parse_final(text: str) -> tuple[str, str] | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("FINAL_VAR:"):
            return ("FINAL_VAR", stripped.split(":", 1)[1].strip())
        if stripped.startswith("FINAL:"):
            return ("FINAL", stripped.split(":", 1)[1].strip())
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


def _cache_key(
    *, text: str, model: str | None, max_tokens: int, recursive: bool = False
) -> str:
    model_part = model or "default"
    rec_part = "recursive" if recursive else "simple"
    return f"model={model_part}|max_tokens={max_tokens}|mode={rec_part}|text={text}"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


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
        response = adapter.complete(messages, max_tokens=max_toks, temperature=0.0)
        trace.add(
            TraceStep(
                step_id=next_step_id(),
                kind="subcall",
                depth=depth + 1,
                prompt_summary=_truncate(query_text, 240),
                usage=response.usage,
                cache_hit=False,
                input_hash=_hash_text(query_text),
                output_hash=_hash_text(response.text),
            )
        )
        return response.text

    repl.set("llm_query", simple_subcall)
    repl.set("ask", lambda q, t, max_tokens=256: simple_subcall(
        f"Question: {q}\nSnippet:\n{t}", max_toks=max_tokens
    ))

    last_stdout: str | None = None
    last_error: str | None = None
    repl_executed = False

    for step in range(max_steps):
        # Extract the question from the text (format: "Question: ...\nSnippet:\n...")
        query = "Answer the question based on the provided context."
        if text.startswith("Question:"):
            lines = text.split("\n", 1)
            query = lines[0].replace("Question:", "").strip()

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
        response = adapter.complete(messages, max_tokens=max_tokens, temperature=0.0)
        trace.add(
            TraceStep(
                step_id=next_step_id(),
                kind="root_call",
                depth=depth,
                prompt_summary=_truncate(user_message, 240),
                code=_truncate(response.text, 800),
                usage=response.usage,
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
            continue

        code = _extract_code(cleaned)
        final_in_code = _parse_final(code)
        if final_in_code and not _looks_like_code(code):
            resolved = _try_resolve_final(final_in_code, repl)
            if resolved is not None:
                return resolved, trace
            last_error = "FINAL_VAR variable not found."
            continue

        if not _looks_like_code(code):
            last_error = "Invalid response: expected Python code or FINAL."
            continue

        result = repl.exec(code)
        last_stdout = result.stdout
        last_error = result.error
        repl_executed = True
        trace.add(
            TraceStep(
                step_id=next_step_id(),
                kind="repl_exec",
                depth=depth,
                code=_truncate(code, 800),
                stdout=result.stdout,
                error=result.error,
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
