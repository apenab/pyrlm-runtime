"""
SmartRouter Demo - Automatic Baseline/RLM Selection
===================================================

PROPÓSITO:
    Demostrar el SmartRouter que automáticamente elige entre baseline y RLM
    basándose en el tamaño del contexto, eliminando overhead innecesario.

QUÉ DEMUESTRA:
    1. Router automático: baseline para contextos pequeños, RLM para grandes
    2. Execution profiles: diferentes estrategias (deterministic, semantic, hybrid)
    3. TraceFormatter: visualización clara de qué estrategia usó el modelo
    4. Comparación lado a lado: baseline vs RLM con métricas

CONCEPTO CLAVE - El Crossover Point:
    ┌────────────────────────────────────────────────────────────────┐
    │                                                                │
    │  Eficiencia ▲                                                  │
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
    │  < 8K chars: Baseline (menos overhead, más rápido)            │
    │  > 8K chars: RLM (escala mejor, puede usar regex/subcalls)    │
    └────────────────────────────────────────────────────────────────┘

EXECUTION PROFILES:
    - DETERMINISTIC_FIRST: Prioriza regex/extract_after (default)
    - SEMANTIC_BATCHES: Usa subcalls paralelos para clasificación/agregación
    - HYBRID: Intenta determinista primero, cae a semántico si falla
    - VERIFY: Doble-check con recursive subcalls para alta confianza

VARIABLES DE ENTORNO:
    LLM_BASE_URL       URL del servidor (default: localhost:11434)
    LLM_MODEL          Modelo principal (default: qwen2.5-coder:7b)
    LLM_SUBCALL_MODEL  Modelo para subcalls (opcional)

CÓMO EJECUTAR:
    # Con defaults
    uv run python examples/smart_router_demo.py

    # Con modelo específico
    LLM_MODEL=qwen2.5-coder:14b uv run python examples/smart_router_demo.py

OUTPUT ESPERADO:
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

    Summary: RLM used 28% fewer tokens and found answer with regex

ÚTIL PARA:
    - Entender cuándo usar baseline vs RLM
    - Ver qué estrategia (regex, subcall, etc.) usó el modelo
    - Comparar métricas: tokens, tiempo, pasos
    - Debugging: visualizar el trace completo
"""

import os

from rlm_runtime import Context, ExecutionProfile, RouterConfig, SmartRouter, TraceFormatter
from rlm_runtime.adapters import GenericChatAdapter


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

    # Configuración del router: threshold en 8000 chars
    config = RouterConfig(baseline_threshold=8000)

    # Código fallback para extraer el término con regex
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

    # Test 1: Contexto pequeño (debería usar baseline)
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

    # Test 2: Contexto grande (debería usar RLM)
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

    # Comparación
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

    # Test 3: Demostrar diferentes perfiles
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
