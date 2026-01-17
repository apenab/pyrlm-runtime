"""
Benchmark: RLM vs Baseline (Modelo Convencional)
=================================================

PROPÓSITO:
    Comparar RLM contra un enfoque "baseline" (prompt directo) para demostrar
    cuándo y por qué RLM supera a los modelos convencionales.

    Este benchmark prueba la hipótesis central del paper de MIT CSAIL:
    "RLM escala mejor con contextos grandes que los modelos tradicionales"

QUÉ DEMUESTRA:
    1. Crossover point: el tamaño de contexto donde RLM empieza a ganar
    2. Truncation effect: cómo baseline falla cuando el contexto es muy grande
    3. Optimizaciones implementadas: Fase 0 determinista, subcall paralelo
    4. Métricas fuzzy: tolerancia a typos en la evaluación (Levenshtein)

HIPÓTESIS DEL PAPER:
    ┌────────────────────────────────────────────────────────────────────┐
    │                                                                    │
    │  Precisión ▲                                                       │
    │           │      ╭──────── RLM (escala mejor)                     │
    │           │     ╱                                                  │
    │           │    ╱                                                   │
    │           │   ╱  ╲                                                 │
    │           │  ╱    ╲──────── Baseline (truncado)                   │
    │           │ ╱                                                      │
    │           │╱       CROSSOVER POINT                                │
    │           └────────────────────────────────► Tamaño contexto      │
    │                                                                    │
    │  Baseline: Pasa todo el contexto al modelo (trunca si es grande)  │
    │  RLM: Inspecciona el contexto programáticamente (sin límite)      │
    └────────────────────────────────────────────────────────────────────┘

OPTIMIZACIONES IMPLEMENTADAS (del análisis previo):
    B) Fase 0 determinista:
       - SIEMPRE intenta extract_after() ANTES de hacer subcalls
       - Reduce de ~30 subcalls a potencialmente 0-1
       - Si extract_after encuentra la respuesta, no necesita LLM

    F) Métricas fuzzy:
       - Tolera typos usando distancia Levenshtein
       - "ooloong" se considera correcto para "oolong" (distancia ≤ 2)

    Paralelo:
       - ask_chunks con parallel=True ejecuta subcalls concurrentemente
       - max_concurrent_subcalls controla el número de workers

ESTRUCTURA DEL CONTEXTO:
    - Genera N documentos con M líneas cada uno
    - La "aguja" (key term) está en el documento al 80% del total
    - Baseline trunca a 8000 chars por default → pierde la aguja en contextos grandes

VARIABLES DE ENTORNO:
    LLM_BASE_URL              URL del servidor
    LLM_MODELS                Modelos a probar (separados por coma)
    LLM_SUBCALL_MODEL         Modelo para subcalls (opcional, más pequeño/rápido)
    RLM_CONTEXT_SIZES         Tamaños de contexto a probar (default: 5,30,120 docs)
    RLM_LINES_PER_DOC         Líneas por documento (default: 8)
    RLM_KEY_DOC_RATIO         Posición de la aguja (default: 0.8 = 80%)
    BASELINE_MAX_CHARS        Límite de truncación para baseline (default: 8000)
    RLM_PARALLEL_SUBCALLS     Habilitar subcalls paralelos (default: 1)
    RLM_FALLBACK              Habilitar fallback code (default: 1)

CÓMO EJECUTAR:
    # Básico
    uv run python examples/rlm_vs_baseline.py

    # Con múltiples tamaños de contexto
    RLM_CONTEXT_SIZES=10,50,100,200 uv run python examples/rlm_vs_baseline.py

    # Con modelo de subcall separado (más eficiente)
    LLM_SUBCALL_MODEL=qwen2.5:3b uv run python examples/rlm_vs_baseline.py

OUTPUT ESPERADO:
    ============================================================
    Model: qwen2.5-coder:7b
    Subcall model: same as root
    Parallel subcalls: True max_workers=4
    Baseline max chars: 8000

    Context: docs=5 lines/doc=8 chars=2850
      baseline: oolong  elapsed=0.45s tokens=890 truncated=False contains_key=True
      rlm: oolong  elapsed=1.23s tokens=1250 steps={'root_call': 2}
      winner: baseline (fewer tokens)

    Context: docs=120 lines/doc=8 chars=68400
      baseline: I cannot find...  elapsed=0.52s tokens=920 truncated=True contains_key=False
      rlm: oolong  elapsed=2.15s tokens=1890 steps={'root_call': 2}
      winner: rlm (baseline missed key term)

    Summary:
    docs chars base_tok base_s trunc base_ok rlm_tok rlm_s rlm_ok winner
       5  2850     890   0.45 False    True    1250   1.23   True baseline (fewer tokens)
     120 68400     920   0.52  True   False    1890   2.15   True rlm (baseline missed key term)

INTERPRETACIÓN:
    - Contextos pequeños (~3K chars): Baseline gana (menos overhead)
    - Contextos grandes (>8K chars): RLM gana (baseline trunca y pierde la aguja)
    - El crossover point depende de BASELINE_MAX_CHARS y la posición de la aguja
"""

import logging
import os
import time
from collections import Counter

from rlm_runtime import Context, Policy, RLM
from rlm_runtime.adapters import GenericChatAdapter
from rlm_runtime.prompts import LLAMA_SYSTEM_PROMPT, TINYLLAMA_SYSTEM_PROMPT

KEY_MARKER = "The key term is:"
KEY_VALUE = "oolong"


def build_documents(doc_count: int, lines_per_doc: int, *, key_doc_ratio: float) -> list[str]:
    if doc_count <= 0:
        return []
    key_doc_index = max(0, min(doc_count - 1, int(doc_count * key_doc_ratio)))
    filler = "alpha beta gamma delta epsilon zeta eta theta iota kappa.\n"
    docs: list[str] = []
    for doc_idx in range(doc_count):
        lines = [filler for _ in range(lines_per_doc)]
        if doc_idx == key_doc_index:
            lines.insert(lines_per_doc // 2, f"{KEY_MARKER} {KEY_VALUE}.\n")
        if doc_idx == 0:
            lines.insert(
                0,
                "RLMs treat long prompts as environment state and inspect them via code.\n",
            )
        docs.append("".join(lines))
    return docs


def build_context(
    doc_count: int,
    lines_per_doc: int,
    *,
    key_doc_ratio: float,
    separator: str,
) -> Context:
    documents = build_documents(doc_count, lines_per_doc, key_doc_ratio=key_doc_ratio)
    return Context.from_documents(documents, separator=separator)


def run_rlm(
    adapter: GenericChatAdapter,
    subcall_adapter: GenericChatAdapter | None,
    model: str,
    context: Context,
    query: str,
    *,
    require_subcall: bool,
    max_steps: int,
    max_tokens: int,
    max_subcall_tokens: int | None,
    max_subcalls: int,
    fallback_enabled: bool,
    fallback_code: str | None,
    invalid_limit: int | None,
    repl_error_limit: int | None,
    subcall_guard_steps: int | None,
    parallel_subcalls: bool,
    max_concurrent_subcalls: int,
) -> dict:
    is_tiny = "tinyllama" in model.lower()
    system_prompt = TINYLLAMA_SYSTEM_PROMPT if is_tiny else LLAMA_SYSTEM_PROMPT

    if not fallback_enabled:
        fallback_code = None
        invalid_limit = None
        repl_error_limit = None

    policy = Policy(
        max_steps=max_steps,
        max_subcalls=max_subcalls,
        max_total_tokens=max_tokens,
        max_subcall_tokens=max_subcall_tokens,
    )
    rlm = RLM(
        adapter=adapter,
        subcall_adapter=subcall_adapter,
        policy=policy,
        system_prompt=system_prompt,
        require_repl_before_final=True,
        require_subcall_before_final=require_subcall,
        auto_finalize_var="key",
        invalid_response_limit=invalid_limit,
        repl_error_limit=repl_error_limit,
        fallback_code=fallback_code,
        subcall_guard_steps=subcall_guard_steps,
        parallel_subcalls=parallel_subcalls,
        max_concurrent_subcalls=max_concurrent_subcalls,
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
    max_context_chars: int | None,
) -> dict:
    context_text = context.text
    truncated = False
    if max_context_chars is not None and max_context_chars > 0:
        if len(context_text) > max_context_chars:
            context_text = context_text[:max_context_chars]
            truncated = True
    contains_key = KEY_MARKER in context_text
    prompt = (
        "Answer the question using only the provided context.\n\n"
        f"Context:\n{context_text}\n\n"
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
        "truncated": truncated,
        "used_chars": len(context_text),
        "context_chars": context.len_chars(),
        "contains_key": contains_key,
    }


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def is_success(output: str, *, max_distance: int = 2) -> bool:
    """Check if output contains the key value, tolerating typos.

    Handles cases like 'ooloong' (typo for 'oolong') by using Levenshtein distance.
    """
    output_lower = output.lower()
    # Exact match first
    if KEY_VALUE in output_lower:
        return True
    # Fuzzy match: check each word in output
    words = output_lower.replace(",", " ").replace(".", " ").split()
    for word in words:
        word = word.strip("\"'()[]{}:;")
        if not word:
            continue
        distance = _levenshtein(word, KEY_VALUE)
        if distance <= max_distance:
            return True
    return False


def pick_winner(baseline: dict, rlm: dict) -> str:
    baseline_ok = is_success(baseline["output"])
    rlm_ok = is_success(rlm["output"])
    if rlm_ok and not baseline_ok:
        return "rlm (baseline missed key term)"
    if baseline_ok and not rlm_ok:
        return "baseline (rlm missed key term)"
    if rlm_ok and baseline_ok:
        if rlm["tokens"] < baseline["tokens"]:
            return "rlm (fewer tokens)"
        if rlm["tokens"] > baseline["tokens"]:
            return "baseline (fewer tokens)"
        return "tie"
    return "tie"


def smart_router(
    adapter: "GenericChatAdapter",
    subcall_adapter: "GenericChatAdapter | None",
    model: str,
    context: "Context",
    query: str,
    baseline_query: str,
    *,
    baseline_max_chars: int | None,
    run_rlm_fn: callable,
    run_baseline_fn: callable,
    rlm_kwargs: dict,
) -> tuple[dict, str]:
    """A) Router baseline-first: use baseline if context fits, RLM otherwise.

    This optimization avoids the overhead of RLM when the context is small enough
    for a direct baseline call.
    """
    context_len = context.len_chars()

    # If baseline can see the full context (no truncation), try baseline first
    if baseline_max_chars is None or context_len <= baseline_max_chars:
        baseline_result = run_baseline_fn(
            adapter,
            context,
            baseline_query,
            max_tokens=256,
            max_context_chars=baseline_max_chars,
        )
        if is_success(baseline_result["output"]):
            return baseline_result, "baseline (router: context fits)"

    # Fall back to RLM for large contexts or when baseline fails
    rlm_result = run_rlm_fn(
        adapter,
        subcall_adapter,
        model,
        context,
        query,
        **rlm_kwargs,
    )
    return rlm_result, "rlm (router: context too large or baseline failed)"


def main() -> None:
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    models = os.getenv("LLM_MODELS", "qwen2.5-coder:7b").split(",")
    models = [model.strip() for model in models if model.strip()]
    log_level = os.getenv("LLM_LOG_LEVEL", "WARNING").upper()

    # Configure logging: suppress httpx noise
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("rlm_runtime").setLevel(getattr(logging, log_level, logging.WARNING))
    timeout = float(os.getenv("LLM_TIMEOUT", "180"))
    max_steps = int(os.getenv("LLM_MAX_STEPS", "20"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "60000"))
    max_subcall_raw = os.getenv("LLM_MAX_SUBCALL_TOKENS", "").strip()
    max_subcall_tokens = int(max_subcall_raw) if max_subcall_raw else None
    require_subcall = os.getenv("LLM_REQUIRE_SUBCALL", "1") == "1"
    fallback_enabled = os.getenv("RLM_FALLBACK", "1") == "1"
    invalid_limit = int(os.getenv("RLM_INVALID_LIMIT", "1"))
    repl_error_limit = int(os.getenv("RLM_REPL_ERROR_LIMIT", "2"))
    subcall_guard_raw = os.getenv("RLM_SUBCALL_GUARD_STEPS", "").strip()
    subcall_guard_steps = int(subcall_guard_raw) if subcall_guard_raw else None
    if require_subcall and subcall_guard_steps is None:
        subcall_guard_steps = 2

    sizes_raw = os.getenv("RLM_CONTEXT_SIZES", "5,30,120")
    doc_counts = [int(item) for item in sizes_raw.split(",") if item.strip()]
    lines_per_doc = int(os.getenv("RLM_LINES_PER_DOC", "8"))
    key_doc_ratio = float(os.getenv("RLM_KEY_DOC_RATIO", "0.8"))
    separator = os.getenv("RLM_DOC_SEPARATOR", "\n\n---\n\n")
    chunk_size = int(os.getenv("RLM_CHUNK_SIZE", "2000"))

    baseline_max_raw = os.getenv("BASELINE_MAX_CHARS", "8000").strip()
    baseline_max_chars = int(baseline_max_raw) if baseline_max_raw else None
    if baseline_max_chars is not None and baseline_max_chars <= 0:
        baseline_max_chars = None

    parallel_subcalls = os.getenv("RLM_PARALLEL_SUBCALLS", "1") == "1"
    max_concurrent_subcalls = int(os.getenv("RLM_MAX_CONCURRENT_SUBCALLS", "4"))
    max_subcalls_raw = os.getenv("RLM_MAX_SUBCALLS", "").strip()
    max_subcalls_override = int(max_subcalls_raw) if max_subcalls_raw else None

    use_subcall_adapter = os.getenv("RLM_USE_SUBCALL_ADAPTER", "1") == "1"
    subcall_model = os.getenv("LLM_SUBCALL_MODEL", "").strip()
    if subcall_model.lower() in {"", "same", "auto"}:
        subcall_model = ""
    subcall_base_url = os.getenv("LLM_SUBCALL_BASE_URL", base_url)
    subcall_timeout = float(os.getenv("LLM_SUBCALL_TIMEOUT", str(timeout)))

    context_variants: list[tuple[int, Context]] = []
    for doc_count in doc_counts:
        context = build_context(
            doc_count,
            lines_per_doc,
            key_doc_ratio=key_doc_ratio,
            separator=separator,
        )
        context_variants.append((doc_count, context))

    sub_question = (
        "Extract the value after the literal text 'The key term is:'. "
        "Return only the value (e.g., oolong). If not present, return NO_ANSWER."
    )
    baseline_query = "What is the key term defined by 'The key term is:'?"

    for model in models:
        adapter = GenericChatAdapter(base_url=base_url, model=model, timeout=timeout)
        subcall_adapter = None
        subcall_label = "disabled"
        if use_subcall_adapter:
            if subcall_model:
                subcall_adapter = GenericChatAdapter(
                    base_url=subcall_base_url,
                    model=subcall_model,
                    timeout=subcall_timeout,
                )
                subcall_label = subcall_model
            else:
                subcall_adapter = adapter
                subcall_label = "same as root"

        print("=" * 60)
        print(f"Model: {model}")
        print(f"Subcall model: {subcall_label}")
        print(f"Parallel subcalls: {parallel_subcalls} max_workers={max_concurrent_subcalls}")
        if baseline_max_chars is not None:
            print(f"Baseline max chars: {baseline_max_chars}")
        summary_rows: list[dict] = []

        for doc_count, context in context_variants:
            # B) Fase 0 determinista: SIEMPRE intentar extract_after PRIMERO
            # Esto reduce de ~30 subcalls a potencialmente 0-1
            query = (
                "Find the key term defined by 'The key term is:'. "
                "IMPORTANT: First try key = extract_after('The key term is:'). "
                "Only if key is None, then use "
            )
            if parallel_subcalls:
                query += (
                    f"ask_chunks with chunk_size={chunk_size} and parallel=True, "
                    "then pick_first_answer. "
                )
            else:
                query += f"ask_chunks_first with chunk_size={chunk_size}. "
            query += "Set key and reply as FINAL_VAR: key."
            query += f"\n\nSub-question: {sub_question}"

            fallback_code = None
            if fallback_enabled:
                # B) Fase 0 determinista: extract_after PRIMERO, subcalls solo si falla
                fallback_code = (
                    "key = extract_after('The key term is:')\n"
                    "if key is None:\n"
                )
                if parallel_subcalls:
                    fallback_code += (
                        f"    chunks = ctx.chunk({chunk_size})\n"
                        f"    answers = ask_chunks({sub_question!r}, chunks, parallel=True)\n"
                        "    key = pick_first_answer(answers)\n"
                    )
                else:
                    fallback_code += (
                        f"    key = ask_chunks_first({sub_question!r}, ctx.chunk({chunk_size}))\n"
                    )
                if require_subcall:
                    fallback_code += (
                        "if key is not None:\n"
                        f"    _ = ask({sub_question!r}, f\"The key term is: {{key}}.\")"
                    )

            estimated_chunks = max(1, (context.len_chars() + chunk_size - 1) // chunk_size)
            max_subcalls = max_subcalls_override or max(12, estimated_chunks + 2)

            rlm_result = run_rlm(
                adapter,
                subcall_adapter,
                model,
                context,
                query,
                require_subcall=require_subcall,
                max_steps=max_steps,
                max_tokens=max_tokens,
                max_subcall_tokens=max_subcall_tokens,
                max_subcalls=max_subcalls,
                fallback_enabled=fallback_enabled,
                fallback_code=fallback_code,
                invalid_limit=invalid_limit if fallback_enabled else None,
                repl_error_limit=repl_error_limit if fallback_enabled else None,
                subcall_guard_steps=subcall_guard_steps,
                parallel_subcalls=parallel_subcalls,
                max_concurrent_subcalls=max_concurrent_subcalls,
            )
            baseline_result = run_baseline(
                adapter,
                context,
                baseline_query,
                max_tokens=256,
                max_context_chars=baseline_max_chars,
            )

            print(
                f"Context: docs={doc_count} lines/doc={lines_per_doc} chars={context.len_chars()}"
            )
            print(
                f"  baseline: {baseline_result['output']}"
                f"  elapsed={baseline_result['elapsed']:.2f}s"
                f" tokens={baseline_result['tokens']} calls={baseline_result['calls']}"
                f" used_chars={baseline_result['used_chars']}"
                f" truncated={baseline_result['truncated']}"
                f" contains_key={baseline_result['contains_key']}"
            )
            print(
                f"  rlm: {rlm_result['output']}"
                f"  elapsed={rlm_result['elapsed']:.2f}s"
                f" tokens={rlm_result['tokens']} calls={rlm_result['calls']}"
                f" steps={rlm_result['steps']}"
            )
            winner = pick_winner(baseline_result, rlm_result)
            print(f"  winner: {winner}")

            summary_rows.append(
                {
                    "docs": doc_count,
                    "chars": context.len_chars(),
                    "baseline_tokens": baseline_result["tokens"],
                    "baseline_elapsed": baseline_result["elapsed"],
                    "baseline_truncated": baseline_result["truncated"],
                    "baseline_ok": is_success(baseline_result["output"]),
                    "rlm_tokens": rlm_result["tokens"],
                    "rlm_elapsed": rlm_result["elapsed"],
                    "rlm_ok": is_success(rlm_result["output"]),
                    "winner": winner,
                }
            )

        if summary_rows:
            print("Summary:")
            header = (
                "docs chars base_tok base_s trunc base_ok rlm_tok rlm_s rlm_ok winner"
            )
            print(header)
            for row in summary_rows:
                base_tok = row["baseline_tokens"]
                rlm_tok = row["rlm_tokens"]
                base_s = row["baseline_elapsed"]
                rlm_s = row["rlm_elapsed"]
                print(
                    f"{row['docs']:>4} {row['chars']:>5} "
                    f"{base_tok:>7} {base_s:>6.2f} "
                    f"{str(row['baseline_truncated']):>5} {str(row['baseline_ok']):>7} "
                    f"{rlm_tok:>7} {rlm_s:>6.2f} {str(row['rlm_ok']):>6} "
                    f"{row['winner']}"
                )


if __name__ == "__main__":
    main()
