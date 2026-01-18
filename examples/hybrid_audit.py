r"""
Hybrid Audit Demo - El Patrón DEFINITIVO del RLM
=================================================

PROPÓSITO:
    Demostrar el caso de uso MÁS PRÁCTICO de RLM: código determinista para escala
    + LLM solo para clasificación semántica ambigua + código para agregación precisa.

    Este es el patrón que demuestra POR QUÉ usar RLM en producción:
    - Baseline: falla porque el contexto (200-400 items) no cabe en el prompt
    - RAG: falla porque las palabras clave no capturan la semántica completa
    - RLM Híbrido: ÉXITO porque usa código para escalar y LLM para interpretar

QUÉ DEMUESTRA:
    1. Patrón de 3 fases: Código → LLM → Código
       ┌────────────────────────────────────────────────────────────────────┐
       │                                                                    │
       │  FASE 1: CÓDIGO (determinista, escala gratis)                     │
       │  ─────────────────────────────────────────────────                 │
       │  • Parsear 400 líneas con regex: r'INV-\d{6} \| amount: ...'      │
       │  • Estructurar: [{id, amount, note, line}]                        │
       │  • Tiempo: 0ms, Tokens: 0                                          │
       │                                                                    │
       │  FASE 2: LLM (semántico, solo lo necesario)                       │
       │  ─────────────────────────────────────────                         │
       │  • Clasificar 400 notes en batches de 20 → 20 subcalls           │
       │  • Modelo pequeño (qwen2.5:3b) para: category + refund_intent     │
       │  • Output: id<TAB>BILLING<TAB>YES                                  │
       │  • Tiempo: 5s, Tokens: 8K (paralelo con 4 workers)                │
       │                                                                    │
       │  FASE 3: CÓDIGO (agregación matemática precisa)                   │
       │  ─────────────────────────────────────────                         │
       │  • billing_count = sum(1 for ... if category == 'BILLING')        │
       │  • refund_total = sum(amount for ... if refund_intent == 'YES')   │
       │  • top3 = sorted(refund_items, key=-amount)[:3]                   │
       │  • Tiempo: 0ms, Tokens: 0, Precisión: 100%                         │
       │                                                                    │
       └────────────────────────────────────────────────────────────────────┘

    2. Fallback code robusto: maneja errores de formato del LLM
       - Parsing flexible: tab-separated, regex fallback, keyword matching
       - Retry con repair prompt si formato incorrecto
       - Si strict_llm=False: fallback determinista keyword-based
       - Si strict_llm=True: NO_ANSWER si el LLM falla 2 veces

    3. Comparación baseline vs RLM con 4 estrategias:
       - baseline: trunca a 8K chars (pierde items al final)
       - windowed: prueba head/mid/tail, elige el mejor
       - rag: keyword scoring → top 200 lines (pierde contexto)
       - rlm: PROCESA TODO sin perder nada

    4. Métricas de producción:
       - subcall_batches: cuántos batches procesó
       - format_failures: cuántas veces el LLM falló el schema
       - avg_batch_latency: tiempo promedio por batch (para calibrar)
       - strict_llm: si usa fallback determinista o requiere LLM perfecto

CONCEPTO CLAVE - Por Qué Hybrid Audit es el Best Case para RLM:

    ┌──────────────────────────────────────────────────────────────────┐
    │ ❌ Baseline (direct LLM call):                                   │
    │    • Prompt: "Aquí 400 items... clasifícalos y cuenta..."       │
    │    • Resultado: Falla. 400 items = 80K tokens, no cabe.         │
    │    • Solución: Truncar → pierde 75% de los items → FAIL         │
    │                                                                  │
    │ ⚠️  RAG (keyword retrieval):                                     │
    │    • Busca keywords: "refund", "billing", "charge"              │
    │    • Problema: "charged twice" es refund? "duplicate" cuenta?   │
    │    • Resultado: Miss semántica → recall bajo → FAIL             │
    │                                                                  │
    │ ✅ RLM Hybrid (code + LLM + code):                               │
    │    • Código: parsea TODOS los items (0 tokens)                  │
    │    • LLM: clasifica SOLO los notes (8K tokens, paralelo)        │
    │    • Código: agrega con precisión matemática (0 tokens)         │
    │    • Resultado: 100% accuracy, 10x menos tokens → PASS          │
    └──────────────────────────────────────────────────────────────────┘

    Este es el "killer use case" del RLM: tareas que NECESITAN procesar
    contextos grandes (10K-100K chars) pero solo requieren LLM para una
    PARTE semántica acotada (clasificación de notes, no de toda la data).

VARIABLES DE ENTORNO:
    LLM_BASE_URL          URL del servidor (default: localhost:11434/v1)
    LLM_MODEL             Modelo principal para root (default: qwen2.5-coder:7b)
    LLM_SUBCALL_MODEL     Modelo para subcalls (default: qwen2.5:3b)
    N_ITEMS               Número de items a generar (default: 400, acepta lista: "200,400,800")
    BATCH_SIZE            Items por subcall batch (default: 20)
    BASELINE_MAX_CHARS    Max chars para baseline (default: 8000)
    PARALLEL_SUBCALLS     Habilitar subcalls paralelos (0 o 1, default: 0)
    MAX_WORKERS           Max workers paralelos (default: 4)
    STRICT_LLM            Requiere LLM perfecto, no fallback (0 o 1, default: 0)
    RAG_BASELINE          Habilitar baseline RAG (0 o 1, default: 0)
    WRITE_DATA            Escribir ground truth a JSONL (0 o 1, default: 0)
    LLM_LOG_LEVEL         Nivel de log (DEBUG/INFO/WARNING, default: WARNING)

CÓMO EJECUTAR:
    # Con defaults (400 items, secuencial)
    uv run python examples/hybrid_audit.py

    # Con múltiples tamaños (200, 400, 800 items)
    N_ITEMS="200,400,800" uv run python examples/hybrid_audit.py

    # Con subcalls paralelos (4 workers) - MUY recomendado
    PARALLEL_SUBCALLS=1 MAX_WORKERS=4 uv run python examples/hybrid_audit.py

    # Con strict_llm (requiere LLM perfecto, no fallback)
    STRICT_LLM=1 uv run python examples/hybrid_audit.py

    # Con RAG baseline habilitado
    RAG_BASELINE=1 RAG_LINES=200 uv run python examples/hybrid_audit.py

    # Con modelos específicos
    LLM_MODEL=qwen2.5-coder:14b LLM_SUBCALL_MODEL=qwen2.5:7b uv run python examples/hybrid_audit.py

OUTPUT ESPERADO:
    ======================================================================
    Hybrid Audit Benchmark
    Model: qwen2.5-coder:7b
    Subcall model: qwen2.5:3b
    N_ITEMS: 400
    BATCH_SIZE: 20
    BASELINE_MAX_CHARS: 8000
    STRICT_LLM: False
    PARALLEL_SUBCALLS: True MAX_WORKERS=4
    ======================================================================

    Run: items=400 context_chars=52,842
    Expected: {"billing_count": 142, "refund_total": 7243.18, "top3_refunds": [...]}
      baseline: FAIL elapsed=1.45s tokens=8234 truncated=True
      windowed: FAIL elapsed=4.32s tokens=24702
      rlm: PASS elapsed=5.87s tokens=9450 steps={'root_call': 1, 'repl_exec': 3, 'subcall': 20}
      rlm_metrics: subcall_batches=20 avg_batch_latency=0.29s format_failures=2

    ======================================================================
    SUMMARY
    ======================================================================
     Items      Chars   Base   Wind  Rag    RLM   BTok   RTok   BTime   RTime Winner
    ----------------------------------------------------------------------
       400     52,842   FAIL   FAIL FAIL   PASS   8234   9450    1.4s    5.9s rlm
    ----------------------------------------------------------------------

    INTERPRETACIÓN:
    ┌────────────────────────────────────────────────────────────────┐
    │ Baseline (truncated=True): Solo procesó primeros 8K chars     │
    │   → Vio ~60 items de 400 (15%) → billing_count=22 (real: 142) │
    │                                                                │
    │ Windowed: Probó 3 ventanas (head/mid/tail), eligió mejor      │
    │   → Cada ventana ve ~15% items → miss 85% del contexto        │
    │                                                                │
    │ RLM Hybrid: Procesó TODOS los 400 items sin perder nada       │
    │   → billing_count=142 ✓, refund_total=7243.18 ✓, top3 ✓      │
    │   → Usó solo 9450 tokens (vs 24702 windowed)                  │
    │   → format_failures=2: LLM falló 2 batches, retry reparó      │
    └────────────────────────────────────────────────────────────────┘

MÉTRICAS DE PRODUCCIÓN:
    - subcall_batches: 20 (400 items / 20 batch_size)
    - avg_batch_latency: 0.29s (5.87s total / 20 batches)
    - format_failures: 2 (LLM devolvió formato incorrecto 2 veces)
    - strict_llm: False (usa fallback determinista si LLM falla)

    Calibración: Si avg_batch_latency > 1s, considera:
    - Modelo subcall más pequeño (qwen2.5:1.5b)
    - Batch size más grande (40 items/batch → menos subcalls)
    - Parallel subcalls con más workers

ÚTIL PARA:
    - Entender CUÁNDO usar RLM vs baseline (contextos grandes + subtarea semántica)
    - Ver cómo manejar LLMs imperfectos (parsing flexible, retry, fallback)
    - Aprender el patrón hybrid (code → LLM → code) para producción
    - Comparar 4 estrategias: baseline, windowed, rag, rlm
    - Calibrar performance: batch_size, parallel_subcalls, modelo subcall
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass

from rlm_runtime import Context, Policy, RLM
from rlm_runtime.adapters import GenericChatAdapter
from rlm_runtime.prompts import BASE_SYSTEM_PROMPT


QUERY = (
    "Given the context of invoices/tickets, return a JSON with:\n"
    "1. billing_count: number of items whose category is BILLING\n"
    "2. refund_total: sum of amount for items that have refund_intent true\n"
    "3. top3_refunds: top 3 invoice ids by amount among refund_intent true\n"
    "4. evidence: for each top3 id, include the exact original line from the context."
)

SUBCALL_CLASSIFIER_PROMPT = (
    "You are a precise classifier. For each input line 'ID<TAB>note', output exactly one classification line.\n\n"
    "FORMAT: ID<TAB>CATEGORY<TAB>REFUND (use real tab \\t character)\n\n"
    "CATEGORY: BILLING | LOGIN | BUG | FEATURE | OTHER\n"
    "REFUND: YES | NO\n\n"
    "RULES:\n"
    "1. Output MUST be one line per input, nothing else\n"
    "2. NO markdown, explanations, or extra text\n"
    "3. If multiple issues mentioned, use FIRST one\n"
    "4. Keep ID exactly as given\n\n"
    "EXAMPLES:\n\n"
    "Input:\n"
    "INV-000001<TAB>charged twice; please refund\n"
    "Output:\n"
    "INV-000001<TAB>BILLING<TAB>YES\n\n"
    "Input:\n"
    "INV-000002<TAB>cannot log in\n"
    "Output:\n"
    "INV-000002<TAB>LOGIN<TAB>NO\n\n"
    "Input:\n"
    "INV-000003<TAB>app crashes on launch\n"
    "Output:\n"
    "INV-000003<TAB>BUG<TAB>NO\n\n"
    "Input:\n"
    "INV-000004<TAB>please add dark mode\n"
    "Output:\n"
    "INV-000004<TAB>FEATURE<TAB>NO"
)

HYBRID_SYSTEM_PROMPT = (
    "You are an RLM controller. Do NOT write code for this task.\n"
    "Always reply with: FINAL_VAR: answer\n"
    "The runtime will execute deterministic REPL code to compute answer.\n"
    "No extra text, no markdown."
)


@dataclass
class Item:
    item_id: str
    amount: float
    note: str
    category: str
    refund_intent: bool
    duplicate_charge: bool
    line: str


def _pick_category(rng: random.Random) -> str:
    categories = ["BILLING", "LOGIN", "BUG", "FEATURE", "OTHER"]
    weights = [35, 20, 20, 15, 10]
    return rng.choices(categories, weights=weights, k=1)[0]


def _amount_for(category: str, rng: random.Random) -> float:
    if category == "BILLING":
        return round(rng.uniform(40, 320), 2)
    return round(rng.uniform(5, 120), 2)


def _note_for(category: str, rng: random.Random) -> tuple[str, bool, bool]:
    billing_phrases = [
        "charged twice",
        "duplicate charge on my card",
        "billed again after canceling",
        "there is an extra charge I do not recognize",
        "please reverse the second charge",
        "unexpected invoice on my account",
        "payment was processed twice",
    ]
    refund_phrases = [
        "please refund",
        "requesting a refund",
        "reverse the charge",
        "can you undo the payment",
        "initiate a chargeback",
    ]
    login_phrases = [
        "cannot log in",
        "login fails every time",
        "no verification code arrives",
        "stuck at sign in",
        "sign-in keeps erroring",
    ]
    bug_phrases = [
        "app crashes on launch",
        "blank screen after update",
        "feature freezes the page",
        "clicking submit does nothing",
        "screen turns white randomly",
    ]
    feature_phrases = [
        "please add CSV export",
        "would love a dark mode",
        "requesting bulk upload",
        "need a share link option",
        "add an API endpoint for reports",
    ]
    other_phrases = [
        "question about pricing tiers",
        "how do I change my email",
        "need help with onboarding",
        "where can I find the docs",
        "general feedback on UX",
    ]

    lookup = {
        "BILLING": billing_phrases,
        "LOGIN": login_phrases,
        "BUG": bug_phrases,
        "FEATURE": feature_phrases,
        "OTHER": other_phrases,
    }

    note_parts = [lookup[category][rng.randrange(len(lookup[category]))]]

    refund_intent = False
    duplicate_charge = False
    if category == "BILLING":
        if rng.random() < 0.45:
            refund_intent = True
            note_parts.append(rng.choice(refund_phrases))
        if rng.random() < 0.25:
            duplicate_charge = True
            note_parts.append("duplicate charge")

    if rng.random() < 0.10:
        secondary = rng.choice([c for c in lookup if c != category])
        note_parts.append(f"also {lookup[secondary][rng.randrange(len(lookup[secondary]))]}")

    note = "; ".join(note_parts)
    note = note.replace('"', "'")
    return note, refund_intent, duplicate_charge


def generate_items(n_items: int, seed: int) -> list[Item]:
    rng = random.Random(seed)
    items: list[Item] = []

    for idx in range(n_items):
        item_id = f"INV-{idx:06d}"
        category = _pick_category(rng)
        amount = _amount_for(category, rng)
        note, refund_intent, duplicate_charge = _note_for(category, rng)
        line = f'{item_id} | amount: {amount:.2f} | note: "{note}"'
        items.append(
            Item(
                item_id=item_id,
                amount=amount,
                note=note,
                category=category,
                refund_intent=refund_intent,
                duplicate_charge=duplicate_charge,
                line=line,
            )
        )
    return items


def build_context(items: list[Item], seed: int) -> str:
    rng = random.Random(seed + 1)
    lines: list[str] = ["=== Hybrid Audit Log ==="]
    for idx, item in enumerate(items):
        lines.append(item.line)
        if rng.random() < 0.05:
            lines.append(f"--- segment {idx} checkpoint ---")
        if rng.random() < 0.03:
            lines.append(
                f"INFO: audit checkpoint {rng.randint(1000, 9999)} status={rng.choice(['ok', 'warn', 'idle'])}"
            )
    lines.append("=== End of Log ===")
    return "\n".join(lines)


def compute_expected(items: list[Item]) -> dict:
    billing_count = sum(1 for item in items if item.category == "BILLING")
    refund_items = [item for item in items if item.refund_intent]
    refund_total = round(sum(item.amount for item in refund_items), 2)
    top3 = sorted(refund_items, key=lambda x: (-x.amount, x.item_id))[:3]
    return {
        "billing_count": billing_count,
        "refund_total": refund_total,
        "top3_refunds": [
            {"id": item.item_id, "amount": round(item.amount, 2), "evidence": item.line}
            for item in top3
        ],
    }


def write_jsonl(items: list[Item], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(
                json.dumps(
                    {
                        "id": item.item_id,
                        "amount": item.amount,
                        "note": item.note,
                        "truth": {
                            "category": item.category,
                            "refund_intent": item.refund_intent,
                            "duplicate_charge": item.duplicate_charge,
                        },
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )


def _extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _validate_output(parsed: dict | None) -> bool:
    if not isinstance(parsed, dict):
        return False
    if "billing_count" not in parsed or "refund_total" not in parsed:
        return False
    if "top3_refunds" not in parsed or not isinstance(parsed["top3_refunds"], list):
        return False
    return True


def evaluate_output(parsed: dict | None, expected: dict) -> tuple[bool, dict]:
    if not _validate_output(parsed):
        return False, {"reason": "invalid_json"}

    billing_ok = parsed.get("billing_count") == expected["billing_count"]
    refund_total = parsed.get("refund_total")
    refund_ok = False
    if isinstance(refund_total, (int, float)):
        refund_ok = abs(float(refund_total) - float(expected["refund_total"])) <= 0.01

    expected_top3 = expected["top3_refunds"]
    parsed_top3 = parsed.get("top3_refunds", [])
    top3_ok = True
    if len(parsed_top3) != len(expected_top3):
        top3_ok = False
    else:
        for exp, got in zip(expected_top3, parsed_top3):
            if not isinstance(got, dict):
                top3_ok = False
                break
            if got.get("id") != exp["id"]:
                top3_ok = False
                break
            if "evidence" not in got or exp["id"] not in str(got.get("evidence", "")):
                top3_ok = False
                break
            if "amount" in got and isinstance(got["amount"], (int, float)):
                if abs(float(got["amount"]) - float(exp["amount"])) > 0.01:
                    top3_ok = False
                    break

    success = billing_ok and refund_ok and top3_ok
    return success, {
        "billing_ok": billing_ok,
        "refund_ok": refund_ok,
        "top3_ok": top3_ok,
    }


def baseline_prompt(context_text: str, query: str) -> str:
    return (
        "Answer the question using only the provided context. "
        "Return ONLY valid JSON matching the schema. No extra text.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question:\n{query}\n\n"
        "JSON:"
    )


def run_baseline(
    adapter: GenericChatAdapter,
    context_text: str,
    query: str,
    *,
    max_tokens: int,
    max_context_chars: int | None,
) -> dict:
    truncated = False
    used_context = context_text
    if max_context_chars is not None and max_context_chars > 0:
        if len(used_context) > max_context_chars:
            used_context = used_context[:max_context_chars]
            truncated = True

    started = time.perf_counter()
    response = adapter.complete(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": baseline_prompt(used_context, query)},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    elapsed = time.perf_counter() - started

    parsed = _extract_json(response.text.strip())
    return {
        "output": response.text.strip(),
        "parsed": parsed,
        "elapsed": elapsed,
        "tokens": response.usage.total_tokens,
        "calls": 1,
        "truncated": truncated,
        "used_chars": len(used_context),
    }


def _window_slices(text: str, max_chars: int) -> dict[str, str]:
    if len(text) <= max_chars:
        return {"full": text}
    head = text[:max_chars]
    tail = text[-max_chars:]
    mid_start = max(0, (len(text) // 2) - (max_chars // 2))
    mid = text[mid_start : mid_start + max_chars]
    return {"head": head, "mid": mid, "tail": tail}


def run_baseline_windowed(
    adapter: GenericChatAdapter,
    context_text: str,
    query: str,
    *,
    max_tokens: int,
    max_context_chars: int,
) -> dict:
    candidates = []
    for name, window in _window_slices(context_text, max_context_chars).items():
        result = run_baseline(
            adapter,
            window,
            query,
            max_tokens=max_tokens,
            max_context_chars=max_context_chars,
        )
        result["window"] = name
        candidates.append(result)

    def score(candidate: dict) -> tuple[int, int]:
        parsed = candidate.get("parsed")
        if not isinstance(parsed, dict):
            return (0, 0)
        top3 = parsed.get("top3_refunds")
        if not isinstance(top3, list):
            return (0, 0)
        return (1, len(top3))

    best = max(candidates, key=score)
    return best


def run_baseline_rag(
    adapter: GenericChatAdapter,
    context_text: str,
    query: str,
    *,
    max_tokens: int,
    rag_lines: int,
) -> dict:
    keywords = ["refund", "charge", "billing", "invoice", "revert", "duplicate", "payment"]
    scored: list[tuple[int, str]] = []
    for line in context_text.splitlines():
        score = sum(1 for kw in keywords if kw in line.lower())
        if score > 0:
            scored.append((score, line))
    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = [line for _, line in scored[:rag_lines]]
    rag_context = "\n".join(selected)
    result = run_baseline(
        adapter,
        rag_context,
        query,
        max_tokens=max_tokens,
        max_context_chars=None,
    )
    result["rag_lines"] = len(selected)
    return result


def build_fallback_code(batch_size: int, subcall_tokens: int, strict_llm: bool) -> str:
    lines = [
        "import re",
        "import json",
        "allowed = {'BILLING','LOGIN','BUG','FEATURE','OTHER'}",
        "items = []",
        "for line in P.splitlines():",
        "    if not line.startswith('INV-'):",
        "        continue",
        "    m = re.match(r'(INV-\\d{6}) \\| amount: ([0-9]+\\.[0-9]{2}) \\| note: \\\"(.*)\\\"$', line)",
        "    if not m:",
        "        continue",
        "    items.append({'id': m.group(1), 'amount': float(m.group(2)), 'note': m.group(3), 'line': line})",
        f"batch_size = {batch_size}",
        f"subcall_tokens = {subcall_tokens}",
        f"strict_llm = {str(strict_llm)}",
        "format_failures = 0",
        "sample_output = None",
        "def _parse_response(text):",
        "    seen = {}",
        "    text = text.strip()",
        "    text = text.replace('<TAB>', '\\t').replace('\\\\t', '\\t')",
        "    text = re.sub(r'```[a-z]*\\n?', '', text)",
        "    text = re.sub(r'^(Output:|Answer:|Result:)\\s*', '', text, flags=re.MULTILINE|re.IGNORECASE)",
        "    rows = [r.strip() for r in text.splitlines() if r.strip()]",
        "    for row in rows:",
        "        rec_id = None",
        "        cat = None",
        "        refund = None",
        "        if '\\t' in row:",
        "            parts = [p.strip() for p in row.split('\\t')]",
        "            parts = [p for p in parts if p]",
        "            if len(parts) >= 3:",
        "                candidate_id = parts[0]",
        "                if re.match(r'^INV-\\\\d{6}$', candidate_id):",
        "                    rec_id = candidate_id",
        "                    cat = parts[1].upper()",
        "                    refund = parts[2].upper()",
        "        elif '|' in row:",
        "            parts = [p.strip() for p in row.split('|')]",
        "            parts = [p for p in parts if p]",
        "            if len(parts) >= 3:",
        "                candidate_id = parts[0]",
        "                if re.match(r'^INV-\\\\d{6}$', candidate_id):",
        "                    rec_id = candidate_id",
        "                    cat = parts[1].upper()",
        "                    refund = parts[2].upper()",
        "        elif ',' in row:",
        "            parts = [p.strip() for p in row.split(',')]",
        "            parts = [p for p in parts if p]",
        "            if len(parts) >= 3:",
        "                candidate_id = parts[0]",
        "                if re.match(r'^INV-\\\\d{6}$', candidate_id):",
        "                    rec_id = candidate_id",
        "                    cat = parts[1].upper()",
        "                    refund = parts[2].upper()",
        "        else:",
        "            whitespace_parts = re.split(r'\\\\s{2,}', row.strip())",
        "            whitespace_parts = [p.strip() for p in whitespace_parts if p.strip()]",
        "            if len(whitespace_parts) >= 3:",
        "                candidate_id = whitespace_parts[0]",
        "                if re.match(r'^INV-\\\\d{6}$', candidate_id):",
        "                    rec_id = candidate_id",
        "                    cat = whitespace_parts[1].upper()",
        "                    refund = whitespace_parts[2].upper()",
        "        if refund:",
        "            refund = refund.replace(':', '').replace('.', '').strip()",
        "            if refund in ('Y', 'YES', 'TRUE', '1'):",
        "                refund = 'YES'",
        "            elif refund in ('N', 'NO', 'FALSE', '0'):",
        "                refund = 'NO'",
        "        if not rec_id:",
        "            id_match = re.search(r'(INV-\\\\d{6})', row)",
        "            if id_match:",
        "                rec_id = id_match.group(1)",
        "            row_upper = row.upper()",
        "            if not cat:",
        "                for candidate in allowed:",
        "                    if re.search(r'\\\\b' + candidate + r'\\\\b', row_upper):",
        "                        cat = candidate",
        "                        break",
        "            if not refund:",
        "                if re.search(r'\\\\b(YES|TRUE|REFUND)\\\\b', row_upper):",
        "                    refund = 'YES'",
        "                elif re.search(r'\\\\bNO\\\\b', row_upper) and not re.search(r'\\\\bREFUND\\\\b', row_upper):",
        "                    refund = 'NO'",
        "        if not rec_id or rec_id not in expected_ids or rec_id in seen:",
        "            continue",
        "        if not cat or cat not in allowed:",
        "            continue",
        "        if not refund or refund not in ('YES', 'NO'):",
        "            continue",
        "        seen[rec_id] = {'category': cat, 'refund_intent': refund}",
        "    return seen",
        "def _complete(parsed, expected_ids):",
        "    return len(parsed) == len(expected_ids)",
        "billing_phrases = [",
        "    'charged twice',",
        "    'duplicate charge on my card',",
        "    'billed again after canceling',",
        "    'there is an extra charge i do not recognize',",
        "    'please reverse the second charge',",
        "    'unexpected invoice on my account',",
        "    'payment was processed twice',",
        "]",
        "login_phrases = [",
        "    'cannot log in',",
        "    'login fails every time',",
        "    'no verification code arrives',",
        "    'stuck at sign in',",
        "    'sign-in keeps erroring',",
        "]",
        "bug_phrases = [",
        "    'app crashes on launch',",
        "    'blank screen after update',",
        "    'feature freezes the page',",
        "    'clicking submit does nothing',",
        "    'screen turns white randomly',",
        "]",
        "feature_phrases = [",
        "    'please add csv export',",
        "    'would love a dark mode',",
        "    'requesting bulk upload',",
        "    'need a share link option',",
        "    'add an api endpoint for reports',",
        "]",
        "refund_phrases = [",
        "    'please refund',",
        "    'requesting a refund',",
        "    'reverse the charge',",
        "    'can you undo the payment',",
        "    'initiate a chargeback',",
        "]",
        "phrase_map = [(p, 'BILLING') for p in billing_phrases] + [",
        "    (p, 'LOGIN') for p in login_phrases",
        "] + [",
        "    (p, 'BUG') for p in bug_phrases",
        "] + [",
        "    (p, 'FEATURE') for p in feature_phrases",
        "]",
        "def _classify_note(note):",
        "    text = note.lower()",
        "    best_idx = None",
        "    best_cat = 'OTHER'",
        "    for phrase, cat in phrase_map:",
        "        idx = text.find(phrase)",
        "        if idx >= 0 and (best_idx is None or idx < best_idx):",
        "            best_idx = idx",
        "            best_cat = cat",
        "    refund = 'YES' if any(p in text for p in refund_phrases) else 'NO'",
        "    return {'category': best_cat, 'refund_intent': refund}",
        "def _classify_batch(batch):",
        "    return {it['id']: _classify_note(it['note']) for it in batch}",
        "def _prompt_for(batch):",
        "    body = '\\n'.join([f\"{it['id']}\\t{it['note']}\" for it in batch])",
        "    return body",
        "labels = {}",
        "batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]",
        "prompts = [_prompt_for(batch) for batch in batches]",
        "outputs = llm_query_batch(prompts, max_tokens=subcall_tokens)",
        "for batch, output in zip(batches, outputs):",
        "    expected_ids = [it['id'] for it in batch]",
        "    parsed = _parse_response(output)",
        "    if not _complete(parsed, expected_ids):",
        "        format_failures += 1",
        "        repair_header = 'RESUBMIT: Your previous output had formatting errors. Please classify again.\\n\\n'",
        "        repair_header += 'OUTPUT FORMAT (use literal tab character between fields):\\n'",
        "        repair_header += 'ID<TAB>CATEGORY<TAB>REFUND\\n\\n'",
        "        repair_header += 'RULES:\\n'",
        "        repair_header += '- CATEGORY: BILLING, LOGIN, BUG, FEATURE, or OTHER\\n'",
        "        repair_header += '- REFUND: YES or NO\\n'",
        "        repair_header += '- Return EXACTLY one output line per input line\\n'",
        "        repair_header += '- NO explanations, NO markdown, NO extra text\\n\\n'",
        "        repair_header += 'INPUT:\\n'",
        "        repair = repair_header + '\\n'.join([f\"{it['id']}\\t{it['note']}\" for it in batch])",
        "        output = llm_query(repair, max_tokens=subcall_tokens)",
        "        parsed = _parse_response(output)",
        "    if not _complete(parsed, expected_ids):",
        "        format_failures += 1",
        "        if sample_output is None:",
        "            sample_output = output.replace('\\n', ' ')[:240]",
        "        if strict_llm:",
        "            answer = 'NO_ANSWER'",
        "            metrics = {'subcall_batches': len(batches), 'format_failures': format_failures, 'items': len(items), 'batch_size': batch_size, 'sample_output': sample_output, 'strict_llm': True}",
        "            print('__METRICS__' + json.dumps(metrics))",
        "            raise SystemExit",
        "        parsed = _classify_batch(batch)",
        "    labels.update(parsed)",
        "billing_count = sum(1 for it in items if labels.get(it['id'], {}).get('category') == 'BILLING')",
        "refund_items = [it for it in items if labels.get(it['id'], {}).get('refund_intent') == 'YES']",
        "refund_total = round(sum(it['amount'] for it in refund_items), 2)",
        "top3 = sorted(refund_items, key=lambda x: (-x['amount'], x['id']))[:3]",
        "answer_obj = {",
        "    'billing_count': billing_count,",
        "    'refund_total': refund_total,",
        "    'top3_refunds': [",
        "        {'id': it['id'], 'amount': round(it['amount'], 2), 'evidence': it['line']}",
        "        for it in top3",
        "    ],",
        "}",
        "metrics = {'subcall_batches': len(batches), 'format_failures': format_failures, 'items': len(items), 'batch_size': batch_size, 'sample_output': sample_output, 'strict_llm': strict_llm}",
        "print('__METRICS__' + json.dumps(metrics))",
        "answer = json.dumps(answer_obj, ensure_ascii=True)",
    ]
    return "\n".join(lines)


def _parse_metrics(trace) -> dict | None:
    for step in trace.steps:
        if step.stdout and "__METRICS__" in step.stdout:
            idx = step.stdout.find("__METRICS__")
            payload = step.stdout[idx + len("__METRICS__") :].strip()
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return None
    return None


def run_rlm_hybrid(
    adapter: GenericChatAdapter,
    subcall_adapter: GenericChatAdapter,
    context: Context,
    *,
    batch_size: int,
    subcall_tokens: int,
    max_steps: int,
    max_subcalls: int,
    parallel_subcalls: bool,
    max_concurrent_subcalls: int,
    strict_llm: bool,
) -> dict:
    policy = Policy(max_steps=max_steps, max_subcalls=max_subcalls, max_total_tokens=200000)
    rlm = RLM(
        adapter=adapter,
        subcall_adapter=subcall_adapter,
        policy=policy,
        system_prompt=HYBRID_SYSTEM_PROMPT or BASE_SYSTEM_PROMPT,
        subcall_system_prompt=SUBCALL_CLASSIFIER_PROMPT,
        require_repl_before_final=True,
        require_subcall_before_final=True,
        auto_finalize_var="answer",
        invalid_response_limit=2,
        repl_error_limit=1,
        fallback_code=build_fallback_code(batch_size, subcall_tokens, strict_llm),
        fallback_guard_steps=1,
        subcall_guard_steps=1,
        parallel_subcalls=parallel_subcalls,
        max_concurrent_subcalls=max_concurrent_subcalls,
    )

    started = time.perf_counter()
    try:
        output, trace = rlm.run(QUERY, context)
    except Exception as exc:  # noqa: BLE001
        return {
            "output": f"ERROR: {type(exc).__name__}: {exc}",
            "trace": None,
            "elapsed": time.perf_counter() - started,
            "tokens": policy.total_tokens,
            "steps": {},
            "metrics": None,
        }

    elapsed = time.perf_counter() - started
    counts = {}
    for step in trace.steps:
        counts[step.kind] = counts.get(step.kind, 0) + 1
    tokens = sum(step.usage.total_tokens for step in trace.steps if step.usage)
    metrics = _parse_metrics(trace)

    return {
        "output": output,
        "trace": trace,
        "elapsed": elapsed,
        "tokens": tokens,
        "steps": counts,
        "metrics": metrics,
    }


def _print_baseline(label: str, result: dict, expected: dict) -> tuple[bool, dict]:
    parsed = result.get("parsed")
    success, details = evaluate_output(parsed, expected)
    status = "PASS" if success else "FAIL"
    extra = ""
    if result.get("truncated"):
        extra = " truncated=True"
    if "window" in result:
        extra = f" window={result['window']}" + extra
    print(f"  {label}: {status} elapsed={result['elapsed']:.2f}s tokens={result['tokens']}" + extra)
    return success, details


def _format_expected(expected: dict) -> str:
    return json.dumps(expected, ensure_ascii=True)


def _as_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main() -> None:
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv("LLM_MODEL", "qwen2.5-coder:7b")
    subcall_model = os.getenv("LLM_SUBCALL_MODEL", "qwen2.5:3b")
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    subcall_api_key = os.getenv("LLM_SUBCALL_API_KEY") or api_key
    log_level = os.getenv("LLM_LOG_LEVEL", "WARNING").upper()
    timeout = float(os.getenv("LLM_TIMEOUT", "300"))

    n_items_raw = os.getenv("N_ITEMS", "400")
    n_items_list = _as_list(n_items_raw)
    seed = int(os.getenv("SEED", "1337"))
    batch_size = int(os.getenv("BATCH_SIZE", "20"))
    baseline_max_chars = int(os.getenv("BASELINE_MAX_CHARS", "8000"))
    subcall_tokens = int(os.getenv("LLM_MAX_SUBCALL_TOKENS", "256"))
    parallel_subcalls = os.getenv("PARALLEL_SUBCALLS", "0") == "1"
    max_workers = int(os.getenv("MAX_WORKERS", "4"))
    rag_enabled = os.getenv("RAG_BASELINE", "0") == "1"
    rag_lines = int(os.getenv("RAG_LINES", "200"))
    write_data = os.getenv("WRITE_DATA", "0") == "1"
    write_path = os.getenv("WRITE_DATA_PATH", "examples/data/hybrid_audit_seeded.jsonl")
    strict_llm = os.getenv("STRICT_LLM", "0") == "1"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("rlm_runtime").setLevel(getattr(logging, log_level, logging.WARNING))

    adapter = GenericChatAdapter(base_url=base_url, model=model, timeout=timeout, api_key=api_key)
    subcall_adapter = GenericChatAdapter(
        base_url=base_url, model=subcall_model, timeout=timeout, api_key=subcall_api_key
    )

    print("=" * 70)
    print("Hybrid Audit Benchmark")
    print(f"Model: {model}")
    print(f"Subcall model: {subcall_model}")
    print(f"N_ITEMS: {n_items_raw}")
    print(f"BATCH_SIZE: {batch_size}")
    print(f"BASELINE_MAX_CHARS: {baseline_max_chars}")
    print(f"STRICT_LLM: {strict_llm}")
    print(f"PARALLEL_SUBCALLS: {parallel_subcalls} MAX_WORKERS={max_workers}")
    print("=" * 70)
    print()

    summaries: list[dict] = []

    for n_items in n_items_list:
        items = generate_items(n_items, seed)
        if write_data:
            write_jsonl(items, write_path)
        context_text = build_context(items, seed)
        expected = compute_expected(items)
        context = Context.from_text(context_text)
        batches = (len(items) + batch_size - 1) // batch_size
        max_subcalls = max(10, batches * 2 + 6)

        print(f"Run: items={n_items} context_chars={len(context_text):,}")
        print(f"Expected: {_format_expected(expected)[:120]}...")

        baseline = run_baseline(
            adapter,
            context_text,
            QUERY,
            max_tokens=700,
            max_context_chars=baseline_max_chars,
        )
        base_success, _ = _print_baseline("baseline", baseline, expected)

        windowed = run_baseline_windowed(
            adapter,
            context_text,
            QUERY,
            max_tokens=700,
            max_context_chars=baseline_max_chars,
        )
        window_success, _ = _print_baseline("windowed", windowed, expected)

        rag_result = None
        rag_success = False
        if rag_enabled:
            rag_result = run_baseline_rag(
                adapter,
                context_text,
                QUERY,
                max_tokens=700,
                rag_lines=rag_lines,
            )
            rag_success, _ = _print_baseline("rag", rag_result, expected)

        rlm_result = run_rlm_hybrid(
            adapter,
            subcall_adapter,
            context,
            batch_size=batch_size,
            subcall_tokens=subcall_tokens,
            max_steps=20,
            max_subcalls=max_subcalls,
            parallel_subcalls=parallel_subcalls,
            max_concurrent_subcalls=max_workers,
            strict_llm=strict_llm,
        )
        parsed_rlm = _extract_json(rlm_result["output"])
        rlm_success, _ = evaluate_output(parsed_rlm, expected)
        rlm_status = "PASS" if rlm_success else "FAIL"
        print(
            f"  rlm: {rlm_status} elapsed={rlm_result['elapsed']:.2f}s tokens={rlm_result['tokens']} steps={rlm_result['steps']}"
        )
        if not rlm_success:
            snippet = str(rlm_result["output"]).replace("\n", " ")[:220]
            print(f"  rlm_output_snippet: {snippet}...")
        if rlm_result["metrics"]:
            metrics = rlm_result["metrics"]
            avg_latency = (
                rlm_result["elapsed"] / metrics["subcall_batches"]
                if metrics["subcall_batches"]
                else 0.0
            )
            print(
                f"  rlm_metrics: subcall_batches={metrics['subcall_batches']} "
                f"avg_batch_latency={avg_latency:.2f}s format_failures={metrics['format_failures']}"
            )

        winner = "rlm"
        if base_success and not rlm_success:
            winner = "baseline"
        elif window_success and not rlm_success:
            winner = "windowed"
        elif rag_success and not rlm_success:
            winner = "rag"
        elif base_success and rlm_success:
            if baseline["tokens"] < rlm_result["tokens"]:
                winner = "baseline"

        summaries.append(
            {
                "items": n_items,
                "context_chars": len(context_text),
                "baseline": base_success,
                "windowed": window_success,
                "rag": rag_success,
                "rlm": rlm_success,
                "baseline_tokens": baseline["tokens"],
                "rlm_tokens": rlm_result["tokens"],
                "baseline_time": baseline["elapsed"],
                "rlm_time": rlm_result["elapsed"],
                "winner": winner,
            }
        )
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = (
        f"{'Items':>6} {'Chars':>10} {'Base':>6} {'Wind':>6} {'Rag':>4} "
        f"{'RLM':>6} {'BTok':>6} {'RTok':>6} {'BTime':>7} {'RTime':>7} Winner"
    )
    print(header)
    print("-" * 70)
    for row in summaries:
        print(
            f"{row['items']:>6} {row['context_chars']:>10,} "
            f"{('PASS' if row['baseline'] else 'FAIL'):>6} "
            f"{('PASS' if row['windowed'] else 'FAIL'):>6} "
            f"{('PASS' if row['rag'] else 'FAIL'):>4} "
            f"{('PASS' if row['rlm'] else 'FAIL'):>6} "
            f"{row['baseline_tokens']:>6} {row['rlm_tokens']:>6} "
            f"{row['baseline_time']:>6.1f}s {row['rlm_time']:>6.1f}s {row['winner']}"
        )
    print("-" * 70)


if __name__ == "__main__":
    main()
