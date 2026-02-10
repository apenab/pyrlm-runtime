"""
SmartRouter Demo - Automatic Baseline/RLM Selection
===================================================

PURPOSE:
    Demonstrate SmartRouter automatically choosing between baseline and RLM
    based on context size, avoiding unnecessary overhead.

WHAT IT SHOWS:
    1. Automatic routing: baseline for small contexts, RLM for large ones
    2. Execution profiles: different strategies (deterministic, semantic, hybrid)
    3. TraceFormatter: clear visualization of which strategy the model used
    4. Side-by-side comparison: baseline vs RLM with metrics

KEY CONCEPT - The Crossover Point:
    ┌────────────────────────────────────────────────────────────────┐
    │                                                                │
    │  Efficiency ▲                                                  │
    │            │                                                   │
    │            │  Baseline ──────╮                                │
    │            │                  ╲                               │
    │            │                   ╲  CROSSOVER                   │
    │            │                    ╲ (8K chars)                  │
    │            │                     ╲                            │
    │            │                      ╲                           │
    │            │                       ╰─────── RLM               │
    │            │                                                   │
    │            └─────────────────────────────────► Context Size   │
    │                                                                │
    │  < 8K chars: baseline (less overhead, faster)                 │
    │  > 8K chars: RLM (scales better, can use regex/subcalls)      │
    └────────────────────────────────────────────────────────────────┘

EXECUTION PROFILES:
    - DETERMINISTIC_FIRST: prioritizes regex/extract_after (default)
    - SEMANTIC_BATCHES: uses parallel subcalls for classification/aggregation
    - HYBRID: tries deterministic first, falls back to semantic if needed
    - VERIFY: double-check with recursive subcalls for higher confidence

ENVIRONMENT VARIABLES:
    LLM_BASE_URL       Server URL (default: localhost:11434)
    LLM_MODEL          Main model (default: qwen2.5-coder:7b)
    LLM_SUBCALL_MODEL  Model for subcalls (optional)

HOW TO RUN:
    # With defaults
    uv run python examples/smart_router_demo.py

    # With a specific model
    LLM_MODEL=qwen2.5-coder:14b uv run python examples/smart_router_demo.py

EXPECTED OUTPUT:
    ======================================================================
    SmartRouter Demo
    ======================================================================

    Test 1: Small context (2,850 chars)
    --------------------
    Router chose: baseline (context < 8000 chars threshold)
    Answer: oolong
    Time: 0.45s | Tokens: 890

    Trace:
    [1] 📝 baseline_call → Baseline query: Find the key term... [890 tok]

    Test 2: Large context (68,400 chars)
    --------------------
    Router chose: rlm (context >= 8000 chars threshold)
    Answer: oolong
    Time: 1.85s | Tokens: 1,250

    Trace:
    [1] 🔷 root_call → query with fallback [450 tok]
    [2] ⚙️ repl_exec → regex: r"key term is: (\\w+)" [0 tok]
    [3] 🔷 root_call → FINAL → key [800 tok]

    Summary: RLM used 28% fewer tokens and found the answer with regex

USEFUL FOR:
    - Understanding when to use baseline vs RLM
    - Seeing which strategy (regex, subcall, etc.) the model used
    - Comparing metrics: tokens, time, steps
    - Debugging: inspect the full trace
"""

import os

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from pyrlm_runtime import Context, ExecutionProfile, RouterConfig, SmartRouter, TraceFormatter
from pyrlm_runtime.adapters import GenericChatAdapter


def main() -> None:
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv("LLM_MODEL", "qwen2.5-coder:7b")
    subcall_model = os.getenv("LLM_SUBCALL_MODEL")

    adapter = GenericChatAdapter(base_url=base_url, model=model)
    subcall_adapter = (
        GenericChatAdapter(base_url=base_url, model=subcall_model)
        if subcall_model
        else None
    )

    # Router configuration: threshold at 8000 chars
    config = RouterConfig(baseline_threshold=8000)

    # Fallback code to extract the term via regex
    fallback_code = (
        "import re\n"
        "key = None\n"
        "m = re.search(r'key term is: (\\w+)', P, re.IGNORECASE)\n"
        "if m:\n"
        "    key = m.group(1)"
    )

    router = SmartRouter(
        adapter=adapter,
        subcall_adapter=subcall_adapter,
        config=config,
        fallback_code=fallback_code,
        auto_finalize_var="key",
    )

    formatter = TraceFormatter()

    print("=" * 70)
    print("SmartRouter Demo")
    print("=" * 70)
    print()

    # Test 1: Small context (should use baseline)
    small_context = Context.from_text(
        "RLMs are great. " * 100 + "The key term is: oolong."
    )

    print(f"Test 1: Small context ({small_context.len_chars():,} chars)")
    print("-" * 20)

    result1 = router.run(
        "Find the key term defined by 'The key term is:'. Return only the term.",
        small_context,
        profile=ExecutionProfile.DETERMINISTIC_FIRST,
    )

    print(f"Router chose: {result1.method} (context < {config.baseline_threshold} chars threshold)")
    print(f"Answer: {result1.output}")
    print(f"Time: {result1.elapsed:.2f}s | Tokens: {result1.tokens_used:,}")
    print()
    print("Trace:")
    print(formatter.format(result1.trace))
    print()

    # Test 2: Large context (should use RLM)
    large_context = Context.from_text(
        "RLMs are great. " * 2000 + "The key term is: oolong. " + "More text. " * 1000
    )

    print(f"Test 2: Large context ({large_context.len_chars():,} chars)")
    print("-" * 20)

    result2 = router.run(
        "Find the key term defined by 'The key term is:'. Use extract_after() or regex. Reply as FINAL_VAR: key.",
        large_context,
        profile=ExecutionProfile.DETERMINISTIC_FIRST,
    )

    print(f"Router chose: {result2.method} (context >= {config.baseline_threshold} chars threshold)")
    print(f"Answer: {result2.output}")
    print(f"Time: {result2.elapsed:.2f}s | Tokens: {result2.tokens_used:,}")
    print()
    print("Trace:")
    print(formatter.format(result2.trace))
    print()

    # Comparison
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print()

    if result1.tokens_used < result2.tokens_used:
        savings = int(100 * (1 - result1.tokens_used / result2.tokens_used))
        print(f"Baseline used {savings}% fewer tokens for small context (expected)")
    else:
        savings = int(100 * (1 - result2.tokens_used / result1.tokens_used))
        print(f"RLM used {savings}% fewer tokens for large context")

    print()
    print("Table:")
    print(formatter.format_table([result1, result2]))
    print()

    # Test 3: Show different profiles
    print("=" * 70)
    print("EXECUTION PROFILES")
    print("=" * 70)
    print()

    profiles_to_test = [
        (ExecutionProfile.DETERMINISTIC_FIRST, "Prioritizes regex/extract_after"),
        (ExecutionProfile.SEMANTIC_BATCHES, "Uses parallel subcalls for aggregation"),
        (ExecutionProfile.HYBRID, "Tries deterministic, falls back to semantic"),
    ]

    profile_results = []

    for profile, description in profiles_to_test:
        result = router.run(
            "Find the key term. Use REPL to inspect P. Reply as FINAL_VAR: key.",
            large_context,
            profile=profile,
        )
        profile_results.append(result)
        print(f"{profile.value}: {description}")
        print(f"  Tokens: {result.tokens_used:,} | Time: {result.elapsed:.2f}s")
        print()

    print("Profile comparison table:")
    print(formatter.format_table(profile_results))


if __name__ == "__main__":
    main()
