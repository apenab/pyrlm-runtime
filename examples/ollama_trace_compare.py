"""
Model Comparator with Detailed Tracing
======================================

PURPOSE:
    Compare multiple LLM models running the same RLM task.
    Shows the full trace to diagnose each model's behavior.

WHAT IT SHOWS:
    1. Side-by-side comparison of models (qwen, llama, tinyllama, etc.)
    2. Detailed tracing: each REPL step with executed code
    3. Metrics: tokens, time, cache hits, steps
    4. Automatic fallback for non-compliant models
    5. Optional warmup to remove initial load latency

KEY CONCEPT - Compliant vs Non-Compliant Models:
    ┌──────────────────────────────────────────────────────────────────┐
    │  COMPLIANT: Models that follow RLM instructions correctly       │
    │  - qwen2.5-coder, mistral, mixtral, deepseek-coder               │
    │  - Generate clean Python code without markdown                  │
    │  - Respect FINAL/FINAL_VAR                                      │
    │                                                                  │
    │  NON-COMPLIANT: Models that need extra help                     │
    │  - tinyllama, base llama, phi-2                                 │
    │  - May emit markdown, explanations, or ignore FINAL             │
    │  - fallback_code is applied automatically                       │
    └──────────────────────────────────────────────────────────────────┘

TRACING - What each field means:
    [step_id] kind depth=N cache=bool err=bool snippet=...

    - step_id:  Sequential step number
    - kind:     "root_call" (main call) or "subcall" (sub-LLM)
    - depth:    Recursion depth (0=root, 1=subcall, 2=sub-subcall)
    - cache:    Whether the response came from cache
    - err:      Whether execution had an error
    - snippet:  Summary of code or prompt

ENVIRONMENT VARIABLES:
    LLM_BASE_URL          Server URL (default: localhost:11434)
    LLM_MODELS            Comma-separated list of models
    LLM_TIMEOUT           Timeout per request (default: 180s)
    LLM_MAX_STEPS         Max REPL steps (default: 20)
    LLM_MAX_TOKENS        Max total tokens (default: 60000)
    LLM_WARMUP            Run warmup before measuring (default: 1)
    LLM_CACHE_BUST        Bust cache between runs (default: 1)
    LLM_REQUIRE_SUBCALL   Force at least one subcall (default: 0)
    LLM_LOG_LEVEL         Logging level (default: INFO)
    RLM_SUBCALL_GUARD_STEPS  Trigger fallback if no subcall after N steps

HOW TO RUN:
    # Single model
    LLM_MODELS=qwen2.5-coder:7b uv run python examples/ollama_trace_compare.py

    # Compare multiple models
    LLM_MODELS=qwen2.5-coder:7b,tinyllama,llama3.2:latest \
    uv run python examples/ollama_trace_compare.py

    # With detailed logging
    LLM_LOG_LEVEL=DEBUG LLM_MODELS=qwen2.5-coder:7b \
    uv run python examples/ollama_trace_compare.py

EXPECTED OUTPUT:
    ============================================================
    Model: qwen2.5-coder:7b
    Answer: oolong
    Elapsed: 2.34s
    Steps: {'root_call': 2}
    Total tokens (approx): 1250
    Cache hits: 0
    Trace preview:
      [0] root_call depth=0 cache=False err=False snippet=key = extract_after('The key term is:')
      [1] root_call depth=0 cache=False err=False snippet=FINAL_VAR: key

USEFUL FOR:
    - Evaluating which model works best for your use case
    - Diagnosing why a model fails
    - Comparing tokens/time across models
    - Verifying cache behavior
"""

import logging
import os
import time
from collections import Counter

from rlm_runtime import Context, Policy, RLM
from rlm_runtime.adapters import GenericChatAdapter
from rlm_runtime.prompts import LLAMA_SYSTEM_PROMPT, TINYLLAMA_SYSTEM_PROMPT


def run_once(
    model: str,
    base_url: str,
    query: str,
    context: Context,
    timeout: float,
    warmup: bool,
    sub_question: str,
    subcall_guard_steps: int | None,
) -> None:
    compliant_models = ("qwen", "mistral", "mixtral", "codestral", "deepseek-coder")
    adapter = GenericChatAdapter(base_url=base_url, model=model, timeout=timeout)
    max_steps = int(os.getenv("LLM_MAX_STEPS", "20"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "60000"))
    max_subcall_tokens_raw = os.getenv("LLM_MAX_SUBCALL_TOKENS", "").strip()
    max_subcall_tokens = int(max_subcall_tokens_raw) if max_subcall_tokens_raw else None
    policy = Policy(
        max_steps=max_steps,
        max_subcalls=12,
        max_total_tokens=max_tokens,
        max_subcall_tokens=max_subcall_tokens,
    )
    seed = os.getenv("LLM_CACHE_BUST_SEED", "")
    suffix = f"_{seed}" if seed else ""
    cache_dir = os.path.join(".rlm_cache", f"{model.replace('/', '_').replace(':', '_')}{suffix}")
    require_subcall = os.getenv("LLM_REQUIRE_SUBCALL", "0") == "1"
    is_compliant = any(token in model.lower() for token in compliant_models)
    is_tiny = "tinyllama" in model.lower()
    system_prompt = TINYLLAMA_SYSTEM_PROMPT if is_tiny else LLAMA_SYSTEM_PROMPT
    fallback_code = None
    if not is_compliant:
        fallback_code = (
            "key = extract_after('The key term is:')\n"
            "if key is None:\n"
            f"    key = ask_chunks_first({sub_question!r}, ctx.chunk(2000))\n"
            "if key is not None:\n"
            f"    _ = ask({sub_question!r}, f\"The key term is: {{key}}.\")"
        )
    rlm = RLM(
        adapter=adapter,
        policy=policy,
        system_prompt=system_prompt,
        require_repl_before_final=True,
        require_subcall_before_final=require_subcall,
        auto_finalize_var="key",
        cache_dir=cache_dir,
        logger=logging.getLogger("rlm_runtime"),
        invalid_response_limit=1 if not is_compliant else None,
        fallback_code=fallback_code,
        repl_error_limit=2 if not is_compliant else None,
        subcall_guard_steps=subcall_guard_steps,
    )

    if warmup:
        adapter.complete(
            [
                {"role": "system", "content": "Warmup."},
                {"role": "user", "content": "Say OK."},
            ],
            max_tokens=8,
            temperature=0.0,
        )

    started = time.perf_counter()
    output, trace = rlm.run(query, context)
    elapsed = time.perf_counter() - started

    counts = Counter(step.kind for step in trace.steps)
    total_tokens = sum(step.usage.total_tokens for step in trace.steps if step.usage is not None)
    cache_hits = sum(1 for step in trace.steps if step.cache_hit)

    print("=" * 60)
    print(f"Model: {model}")
    print(f"Answer: {output}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Steps: {dict(counts)}")
    print(f"Total tokens (approx): {total_tokens}")
    print(f"Cache hits: {cache_hits}")
    print("Trace preview:")
    for step in trace.steps:
        summary = step.prompt_summary or step.code or ""
        summary = summary.replace("\n", " ").strip()
        if len(summary) > 120:
            summary = summary[:120] + "..."
        print(
            f"  [{step.step_id}] {step.kind} depth={step.depth} "
            f"cache={step.cache_hit} err={bool(step.error)} "
            f"snippet={summary}"
        )


def main() -> None:
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    timeout = float(os.getenv("LLM_TIMEOUT", "180"))
    cache_bust = os.getenv("LLM_CACHE_BUST", "1") == "1"
    require_subcall = os.getenv("LLM_REQUIRE_SUBCALL", "0") == "1"
    warmup = os.getenv("LLM_WARMUP", "1") == "1"
    log_level = os.getenv("LLM_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
    subcall_guard_raw = os.getenv("RLM_SUBCALL_GUARD_STEPS", "").strip()
    subcall_guard_steps = int(subcall_guard_raw) if subcall_guard_raw else None
    if require_subcall and subcall_guard_steps is None:
        subcall_guard_steps = 2
    models = os.getenv("LLM_MODELS", "qwen2.5-coder:7b").split(",")
    models = [model.strip() for model in models if model.strip()]

    filler = "alpha beta gamma delta epsilon zeta eta theta iota kappa.\n"
    lines = [filler for _ in range(120)]
    lines.insert(60, "The key term is: oolong.\n")
    lines.insert(
        0,
        "RLMs treat long prompts as environment state and inspect them via code.\n",
    )
    context_text = "".join(lines)
    context = Context.from_text(context_text)
    sub_question = (
        "Extract the value after the literal text 'The key term is:'. "
        "Return only the value (e.g., oolong). If not present, return NO_ANSWER."
    )
    if require_subcall:
        query = (
            "Find the key term defined by 'The key term is:'. Use ask_chunks_first "
            "with the sub-question below; if it returns NO_ANSWER or None, use "
            "extract_after. Ensure you make at least one subcall. Set key and reply "
            "as: FINAL_VAR: key.\n\n"
            f"Sub-question: {sub_question}"
        )
    else:
        query = (
            "Find the key term defined by 'The key term is:'. First try "
            "key = extract_after('The key term is:'). If key is None, use ask_chunks "
            "with the sub-question below, then pick_first_answer to select the value. "
            "Reply as: FINAL_VAR: key.\n\n"
            f"Sub-question: {sub_question}"
        )

    for model in models:
        if cache_bust:
            os.environ["LLM_CACHE_BUST_SEED"] = str(time.time_ns())
        try:
            if os.getenv("LLM_SHOW_WARNINGS", "1") == "1":
                if not any(token in model.lower() for token in ("qwen", "mistral", "mixtral", "codestral", "phi3", "deepseek-coder")):
                    print("=" * 60)
                    print(f"Model: {model}")
                    print("WARN: model is likely non-compliant; using fallback path.")
            run_once(
                model,
                base_url,
                query,
                context,
                timeout,
                warmup,
                sub_question,
                subcall_guard_steps,
            )
        except Exception as exc:  # noqa: BLE001
            print("=" * 60)
            print(f"Model: {model}")
            print(f"ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
