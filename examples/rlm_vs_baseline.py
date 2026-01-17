import logging
import os
import time
from collections import Counter

from rlm_runtime import Context, Policy, RLM
from rlm_runtime.adapters import GenericChatAdapter
from rlm_runtime.prompts import LLAMA_SYSTEM_PROMPT, TINYLLAMA_SYSTEM_PROMPT


def build_context(lines: int) -> Context:
    filler = "alpha beta gamma delta epsilon zeta eta theta iota kappa.\n"
    blocks = [filler for _ in range(lines)]
    blocks.insert(lines // 2, "The key term is: oolong.\n")
    blocks.insert(
        0,
        "RLMs treat long prompts as environment state and inspect them via code.\n",
    )
    return Context.from_text("".join(blocks))


def run_rlm(
    adapter: GenericChatAdapter,
    model: str,
    context: Context,
    query: str,
    sub_question: str,
    *,
    require_subcall: bool,
    max_steps: int,
    max_tokens: int,
    max_subcall_tokens: int | None,
    timeout: float,
    fallback_enabled: bool,
    fallback_code: str | None,
    invalid_limit: int | None,
    repl_error_limit: int | None,
    subcall_guard_steps: int | None,
) -> dict:
    is_tiny = "tinyllama" in model.lower()
    system_prompt = TINYLLAMA_SYSTEM_PROMPT if is_tiny else LLAMA_SYSTEM_PROMPT

    if not fallback_enabled:
        fallback_code = None
        invalid_limit = None
        repl_error_limit = None

    policy = Policy(
        max_steps=max_steps,
        max_subcalls=12,
        max_total_tokens=max_tokens,
        max_subcall_tokens=max_subcall_tokens,
    )
    rlm = RLM(
        adapter=adapter,
        policy=policy,
        system_prompt=system_prompt,
        require_repl_before_final=True,
        require_subcall_before_final=require_subcall,
        auto_finalize_var="key",
        invalid_response_limit=invalid_limit,
        repl_error_limit=repl_error_limit,
        fallback_code=fallback_code,
        subcall_guard_steps=subcall_guard_steps,
    )

    started = time.perf_counter()
    try:
        output, trace = rlm.run(query, context)
    except Exception as exc:  # noqa: BLE001
        return {
            "mode": "rlm",
            "output": f"ERROR: {type(exc).__name__}",
            "elapsed": time.perf_counter() - started,
            "tokens": policy.total_tokens,
            "calls": policy.subcalls + policy.steps,
            "steps": {"error": type(exc).__name__},
        }
    elapsed = time.perf_counter() - started

    counts = Counter(step.kind for step in trace.steps)
    tokens = sum(step.usage.total_tokens for step in trace.steps if step.usage is not None)
    calls = counts.get("root_call", 0) + counts.get("subcall", 0)

    return {
        "mode": "rlm",
        "output": output,
        "elapsed": elapsed,
        "tokens": tokens,
        "calls": calls,
        "steps": dict(counts),
    }


def run_baseline(
    adapter: GenericChatAdapter,
    context: Context,
    query: str,
    *,
    max_tokens: int,
) -> dict:
    prompt = (
        "Answer the question using only the provided context.\n\n"
        f"Context:\n{context.text}\n\n"
        f"Question:\n{query}\n\n"
        "Answer with only the key term value."
    )
    started = time.perf_counter()
    response = adapter.complete(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    elapsed = time.perf_counter() - started
    return {
        "mode": "baseline",
        "output": response.text.strip(),
        "elapsed": elapsed,
        "tokens": response.usage.total_tokens,
        "calls": 1,
        "steps": {"root_call": 1},
    }


def main() -> None:
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    models = os.getenv("LLM_MODELS", "qwen2.5-coder:7b, phi4-mini-reasoning:latest").split(",")
    models = [model.strip() for model in models if model.strip()]
    log_level = os.getenv("LLM_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
    timeout = float(os.getenv("LLM_TIMEOUT", "180"))
    max_steps = int(os.getenv("LLM_MAX_STEPS", "20"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "60000"))
    max_subcall_raw = os.getenv("LLM_MAX_SUBCALL_TOKENS", "").strip()
    max_subcall_tokens = int(max_subcall_raw) if max_subcall_raw else None
    require_subcall = os.getenv("LLM_REQUIRE_SUBCALL", "0") == "1"
    fallback_enabled = os.getenv("RLM_FALLBACK", "1") == "1"
    invalid_limit = int(os.getenv("RLM_INVALID_LIMIT", "1"))
    repl_error_limit = int(os.getenv("RLM_REPL_ERROR_LIMIT", "2"))
    subcall_guard_raw = os.getenv("RLM_SUBCALL_GUARD_STEPS", "").strip()
    subcall_guard_steps = int(subcall_guard_raw) if subcall_guard_raw else None
    if require_subcall and subcall_guard_steps is None:
        subcall_guard_steps = 2
    context_lines = int(os.getenv("RLM_CONTEXT_LINES", "120"))

    context = build_context(context_lines)
    sub_question = (
        "Extract the value after the literal text 'The key term is:'. "
        "Return only the value (e.g., oolong). If not present, return NO_ANSWER."
    )
    if require_subcall:
        rlm_query = (
            "Find the key term defined by 'The key term is:'. Use ask_chunks_first with "
            "the sub-question first; if it returns NO_ANSWER or None, use extract_after. "
            "Ensure you make at least one subcall. Set key and reply as FINAL_VAR: key.\n\n"
            f"Sub-question: {sub_question}"
        )
    else:
        rlm_query = (
            "Find the key term defined by 'The key term is:'. First try "
            "key = extract_after('The key term is:'). If key is None, use "
            "ask_chunks_first with the sub-question. "
            "Set key and reply as FINAL_VAR: key.\n\n"
            f"Sub-question: {sub_question}"
        )
    baseline_query = "What is the key term defined by 'The key term is:'?"

    for model in models:
        adapter = GenericChatAdapter(base_url=base_url, model=model, timeout=timeout)
        fallback_code = None
        if fallback_enabled:
            if require_subcall:
                fallback_code = (
                    "key = extract_after('The key term is:')\n"
                    "if key is None:\n"
                    f"    key = ask_chunks_first({sub_question!r}, ctx.chunk(2000))\n"
                    "if key is not None:\n"
                    f"    _ = ask({sub_question!r}, f\"The key term is: {{key}}.\")"
                )
            else:
                fallback_code = (
                    "key = extract_after('The key term is:')\n"
                    "if key is None:\n"
                    f"    key = ask_chunks_first({sub_question!r}, ctx.chunk(2000))"
                )
        rlm_result = run_rlm(
            adapter,
            model,
            context,
            rlm_query,
            sub_question,
            require_subcall=require_subcall,
            max_steps=max_steps,
            max_tokens=max_tokens,
            max_subcall_tokens=max_subcall_tokens,
            timeout=timeout,
            fallback_enabled=fallback_enabled,
            fallback_code=fallback_code,
            invalid_limit=invalid_limit if fallback_enabled else None,
            repl_error_limit=repl_error_limit if fallback_enabled else None,
            subcall_guard_steps=subcall_guard_steps,
        )
        baseline_result = run_baseline(
            adapter,
            context,
            baseline_query,
            max_tokens=256,
        )

        print("=" * 60)
        print(f"Model: {model}")
        for result in (baseline_result, rlm_result):
            print(f"- {result['mode']}: {result['output']}")
            print(
                f"  elapsed={result['elapsed']:.2f}s tokens={result['tokens']} "
                f"calls={result['calls']} steps={result['steps']}"
            )


if __name__ == "__main__":
    main()
