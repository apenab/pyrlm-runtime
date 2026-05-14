# Experimentos OBLIQ-Bench × pyrlm-runtime

> 🧭 **Empieza por** [`OBLIQ-OBJETIVO.md`](OBLIQ-OBJETIVO.md) si no
> tienes el contexto del proyecto. Este doc es el tracking detallado de
> los experimentos; el objetivo y el por qué viven en el otro.
>
> Tracking doc. Última actualización: 2026-05-13

Hipótesis a contrastar: ¿pueden las dos primitivas de pyrlm-runtime
(**listwise rerank** y **RLM agentic loop**) cerrar la *retrieval-verification
asymmetry* del paper OBLIQ-Bench (Tchuindjo et al. 2026, arXiv:2605.06235)?

> **🚨 Restricción operativa.** Todo este trabajo asume que el
> Elasticsearch (o retriever equivalente) viene **dado y fijo**: no
> podemos tocar la indexación. Toda mejora tiene que venir del read
> path: query rewriting, reranking, verification, composición de pools.
> La Solución 1 del paper (anotar atributos latentes offline durante
> la indexación) está **fuera de alcance por diseño** — pero es donde
> el paper sugiere que estaría el verdadero salto cualitativo (0.43+
> NDCG vs nuestro techo ~0.2). Ver
> [`OBLIQ-BENCH-ANALISIS.md`](OBLIQ-BENCH-ANALISIS.md#-restricción-operativa-de-este-proyecto)
> para el razonamiento completo y lo que requeriría relajar esa
> restricción si en algún momento ganamos control de la indexación.

Modelo común para todas las condiciones: **gpt-5.1** (Azure OpenAI).
Dataset común: **OBLIQ-Bench math** subset (3508 docs, 151 queries).
Métrica común: **NDCG@10** y **Recall@10** sobre el qrels oficial.

---

## Matriz experimental

| # | Condición | Primera etapa | Etapa LLM | Pregunta que responde | Estado |
|---|---|---|---|---|---|
| 1 | **BM25 puro** | BM25 in-memory | — | ¿Qué tan mal le va a la primera etapa con oblique queries? | 🟢 N=151 completo |
| 2 | **BM25 + Rerank** | BM25 top-50 | ListwiseReranker | ¿El rerank rescata lo que BM25 sí encontró? | 🟢 N=151 completo |
| 3 | **Oracle + Rerank** | Pool con gold injectado | ListwiseReranker | ¿Qué tan bueno es el reranker cuando el pool sí incluye los gold? | 🟢 N=151 completo |
| 4 | **BM25 + RLM agentic loop** | RLM con tools de retrieval | RLM con subcalls | ¿Puede el loop iterativo *encontrar* los oblique gold? | 🟢 N=151 |
| 5 | **BM25 + RLM + Rerank** | RLM con tools | RLM + rerank final | ¿La combinación supera a cada parte por separado? | ⚪ no iniciado |

Leyenda: 🟢 completo  🟡 parcial  ⚪ no iniciado  ⚫ bloqueado

---

## Fase A — Validar los 3 setups existentes con N=151

**Objetivo:** Tener números robustos (no muestrales de 30) para condiciones
1, 2 y 3 antes de pasar a la Fase B. Salida: tabla publicable con CI claras.

### Resultados preliminares (N=30, oracle pool, gpt-5.1)

| Condición | NDCG@10 | Recall@10 |
|---|---:|---:|
| 1. BM25 baseline | 0.000 | 0.000 |
| 2. BM25 + rerank | 0.022 | 0.005 |
| Oracle baseline (pool barajado) | 0.320 | 0.183 |
| 3. Oracle + rerank | **0.736** | **0.509** |

**Interpretación preliminar:**
- BM25 falla casi totalmente en queries oblicuas (matemáticas analógicas).
  Confirma la tesis del paper.
- El reranker es excelente cuando el pool incluye los gold (+0.42 NDCG).
- → El cuello de botella está en la primera etapa, no en el reranker.

### Resultados finales (N=151)

**Run BM25 — completo, 2026-05-11, gpt-5.1, 464.6s (~7.7 min):**

| Condición | NDCG@10 | Recall@10 |
|---|---:|---:|
| 1. BM25 baseline | **0.0284** | 0.0287 |
| 2. BM25 + rerank | **0.0571** | 0.0510 |
| **Δ** | **+0.0287 (2.0×)** | **+0.0223 (1.78×)** |

LLM calls: ~559 reales (el summary reportó 2282 por un bug de atribución
per-query ya arreglado). Cache hits: 45 (provienen de los 30 queries del
run preliminar BM25).

**Run Oracle — completo, 2026-05-11, gpt-5.1, 478.1s (~8.0 min):**

| Condición | NDCG@10 | Recall@10 |
|---|---:|---:|
| Oracle baseline (pool barajado) | **0.2945** | 0.1840 |
| 3. Oracle + rerank | **0.7136** | 0.5503 |
| **Δ** | **+0.4191 (2.42×)** | **+0.3663 (2.99×)** |

LLM calls: 604 exactos (= 151 × 4 windows, sin race condition gracias al
fix). Cache hits: 0 (las pequeñas no-determinismos a temp=0 cambiaron
suficientes window doc_ids para invalidar las 30 entradas del run
preliminar — no afecta los resultados, solo cuánto pagamos).

### Headline para artículo / LinkedIn

| Setup | NDCG@10 | Comentario |
|---|---:|---|
| BM25 puro | 0.028 | Casi cero — confirma la tesis del paper en queries oblicuas |
| BM25 + listwise rerank gpt-5.1 | 0.057 | Duplica BM25 pero limitado por la recall del primer stage |
| Oracle (random pool) + rerank | **0.714** | Cuando los gold están en el pool, el reranker los promueve correctamente |

### Comparación con el paper (Table 3, math subset)

| Sistema | NDCG@10 (paper) | NDCG@10 (este repo, N=151) |
|---|---:|---:|
| BM25 | 0.022 / 0.029 | **0.028** ✓ (replicado) |
| LateOn 0.1B | 0.112 | — |
| Qwen3-Embed-0.6B | 0.116 | — |
| Qwen3-Embed-4B | 0.095 | — |
| Gemini-2-Embedding | 0.144 | — |
| GPT-5.2 Query Rewriter | 0.142 | — |
| GPT-5.2 Multi-Hop Agent | 0.161 | — (será condición 4) |
| Oracle GPT-5.2 Tournament | 0.279 | **0.714*** |
| Oracle Tournament+Soln | 0.434 | — |
| **BM25 + nuestro listwise rerank (gpt-5.1)** | — | **0.057** |

\* Nuestro pool "oracle" es más fácil que el del paper: usa distractores
aleatorios, mientras el paper usa los top-K duros de varios retrievers. Esto
infla nuestro número y hace la comparación no estrictamente equivalente —
hay que aclararlo en cualquier publicación.

### Interpretación final

- BM25 falla casi totalmente en queries oblicuas (matemáticas analógicas).
  Confirma la tesis del paper: replicamos NDCG@10 = 0.028 vs su 0.022/0.029.
- El reranker duplica el NDCG sobre BM25 (0.028 → 0.057), pero queda muy
  lejos de un embedding decente (Gemini 0.144). **El cuello de botella es la
  recall de la primera etapa, no el rerank.**
- El reranker brilla cuando el pool sí contiene los gold: oracle pool →
  NDCG@10 = 0.71 (+0.42 vs baseline). **La primitiva funciona; el reto es
  llevarle un pool decente.**

### Caveat importante para publicación

Nuestro pool "oracle" es **más fácil** que el del paper. Construcción:

- **Nuestro oracle**: gold positives + distractores **aleatorios** del corpus.
  El reranker tiene que distinguir gold de problemas matemáticos no
  relacionados. Tarea relativamente fácil.
- **Oracle del paper**: gold positives + top-K **duros** de varios retrievers
  (BM25, Qwen-Embed, Gemini, etc.). El reranker tiene que distinguir gold
  de los mejores falsos positivos según múltiples sistemas. Tarea mucho
  más difícil.

→ Nuestro 0.71 NDCG@10 oracle **no es directamente comparable** con el 0.28
del paper. Es una métrica del "techo" del reranker, no del rendimiento en
condiciones adversariales realistas. Hay que aclararlo en cualquier post.

### Comandos ejecutados

```bash
# Condición 1 + 2 (BM25 retriever)
uv run python examples/oblique_rerank_bench.py \
  --adapter azure --model gpt-5.1 \
  --retriever bm25 \
  --max-examples 151 --workers 4 \
  --top-n 50 --top-k 10 --window-size 20 --step 10 \
  --cache-dir .cache/oblique_rerank

# Condición 3 (Oracle retriever)
uv run python examples/oblique_rerank_bench.py \
  --adapter azure --model gpt-5.1 \
  --retriever oracle \
  --max-examples 151 --workers 4 \
  --top-n 50 --top-k 10 --window-size 20 --step 10 \
  --cache-dir .cache/oblique_rerank
```

---

## Fase B — RLM como reranker con verificación distribuida (versión B)

**Objetivo (decidido 2026-05-11):** medir si la deliberación estructurada
con subcalls paralelos supera al listwise rerank single-shot **sobre el
mismo pool BM25 top-50**. Comparación limpia: mismo input, dos rerankers
distintos.

### Por qué versión B (no A ni C)

- **A (RLM agentic puro)** replicaría el "GPT-5.2 Multi-Hop Agent" del paper.
  No aporta nada nuevo y depende de BM25 que no encuentra los oblique gold.
- **C (híbrido completo)** mezcla búsqueda iterativa + verificación + rerank.
  Demasiados grados de libertad para interpretar el resultado.
- **B (RLM-as-reranker)** es la única comparación que isolates "deliberación
  + subcalls" vs "listwise single-shot" sobre el mismo input — y explota la
  USP de pyrlm-runtime.

### Decisiones de diseño (confirmadas)

| Parámetro | Valor |
|---|---|
| Adapter root | `gpt-5.1` (Azure) |
| Adapter subcall | `gpt-5.4-mini` (Azure) |
| Pool input | BM25 top-50 (mismo que condiciones 1+2) |
| `max_steps` | 20 |
| `max_subcalls` | 100 |
| `max_tokens` root | 4096 |
| `max_tokens` subcall | 256 |
| Tool principal | `verify_relevance_batch(query, doc_ids) -> list[dict]` |
| Tool secundaria | `read_doc(doc_id) -> str` |
| Pool inicial expuesto al REPL | variable `bm25_pool: list[dict]` |
| Salida del RLM | `print(top10_ids)` (Python literal, parseable con `ast.literal_eval`) |
| Workers del bench | 2 (cada query ya usa subcalls paralelos internamente) |

**Coste estimado:** ~$40-70 USD (~2,250 root calls + ~7,500 subcalls).
**Tiempo estimado:** ~45-90 min con workers=2.

### Diseño de `verify_relevance_batch`

Función expuesta en el REPL. Recibe la query y una lista de `doc_ids`.
Para cada doc, lanza un subcall con prompt:

```
Query: {query}

Passage: {doc_content}

Question: Is this passage relevant to the query? A passage is relevant
if it shares the same proof technique, reasoning structure, or solution
approach, even if the surface topic differs.

Answer in JSON: {"relevant": true|false, "reason": "<one sentence>"}
```

Todos los subcalls van en **paralelo** (vía `parallel_subcalls` del runtime).
Devuelve `list[{"doc_id": str, "relevant": bool, "reason": str}]` para que
el root pueda combinar resultados.

### Estrategia esperada del root RLM

Algo así (el RLM lo decidirá por sí mismo):

```python
# Tiene en scope: query, bm25_pool (50 docs)
# Round 1: verificar todos en paralelo
verdicts = verify_relevance_batch(query, [d["doc_id"] for d in bm25_pool])
relevant = [v for v in verdicts if v["relevant"]]
# Si hay > 10 relevantes: rankear con razonamiento adicional
# Si hay < 10: ampliar criterio o pedir más contexto a algunos
# Construir top10_ids ordenado por confianza + razonamiento
print(top10_ids)
```

### Archivos previstos

- `examples/oblique_rlm_bench.py` — bench RLM (será nuevo, copy-paste del
  bench actual con la integración RLM)
- `src/pyrlm_runtime/rerank.py` — sin cambios; el RLM no usa `ListwiseReranker`
- Posiblemente helpers en `examples/_rlm_rerank_tools.py` para
  `verify_relevance_batch` y `read_doc` (mantener fuera de la librería
  porque son tooling de bench, no primitivas reutilizables)

---

## Implementación: estado del código

- ✅ `src/pyrlm_runtime/rerank.py` — `ListwiseReranker`, métricas
- ✅ `tests/test_rerank.py` — 20 tests verdes, incluye thread-safety
- ✅ `examples/oblique_rerank_bench.py` — bench condiciones 1, 2, 3
  - ✅ Fix `total_llm_calls` para usar el contador thread-safe directamente
    en lugar de sumar atribuciones racy por-query (los runs con `--workers
    > 1` previos sobre-cuentan ~3-4× las llamadas LLM en el summary; la
    factura real es la del contador interno).
- ✅ `docs/rerank.md` — documentación de la primitiva
- ✅ `examples/_rlm_rerank_tools.py` — `verify_relevance_batch`,
  `read_doc`, prompt del sistema y wiring via `repl_extensions`
- ✅ `examples/oblique_rlm_bench.py` — bench con `--smoke`, `--retriever
  {bm25,oracle}`, modo real Azure, logging detallado de `verify_summary`
  por query incluyendo `score_distribution` y `mean_score_gold` vs
  `mean_score_non_gold`
- ✅ Verifier paradigm: cambiado de binario a scored 1-5 — recuperó el
  gradiente. Documentado como el hallazgo central
- ✅ Run real Fase B con N=151 (oracle scored = 0.615, BM25 scored = 0.042)
- ❌ Commit final

### Matriz extendida final

| # | Condición | Pool | Reranker | Modelo | Estado |
|---|---|---|---|---|---|
| 1 | BM25 baseline | BM25 top-50 | — | — | 🟢 N=151 |
| 2 | BM25 + listwise rerank | BM25 top-50 | permutación 20-doc windows | gpt-5.1 | 🟢 N=151 |
| 3 | Oracle + listwise rerank | gold + random | permutación 20-doc windows | gpt-5.1 | 🟢 N=151 |
| 4a | Oracle + RLM verify binario | gold + random | RLM con subcalls binarios | gpt-5.4-mini | 🟢 N=5 (falló) |
| 4b | Oracle + RLM verify binario | gold + random | RLM con subcalls binarios | gpt-5.1 | 🟢 N=5 (falló) |
| 4c | **Oracle + RLM verify scored** | gold + random | RLM con subcalls scored 1-5 | gpt-5.4-mini | 🟢 N=151 → 0.615 |
| 5 | **BM25 + RLM verify scored** | BM25 top-50 | RLM con subcalls scored 1-5 | gpt-5.4-mini | 🟢 N=151 → 0.042 |
| 6 | **BM25 + RLM agentic (sin rerank)** | construido por el agente vía search() | — | gpt-5.1 | 🟢 N=151 → 0.041 |
| 7 | **Palanca 1 — multi-query (5 rewrites) + ListwiseReranker** | BM25 × 5 reformulaciones LLM, unión dedup (~108 docs) | listwise rerank single-shot | gpt-5.4-mini (rewriter) + gpt-5.1 (rerank) | 🟢 N=151 → 0.072 |
| 7t | **Palanca 1 variante — TournamentReranker** | BM25 × 5 reformulaciones, unión dedup (~108 docs) | tournament (App. C del paper) | gpt-5.4-mini (rewriter) + gpt-5.1 (rerank) | 🟢 N=30 → 0.075 (peor que sliding, hipótesis refutada) |
| 8 | **Palanca 1 v2 — + query original en fan-out** | BM25 × (5 rewrites + query original), unión dedup (~128 docs) | listwise rerank single-shot | gpt-5.4-mini (rewriter) + gpt-5.1 (rerank) | 🟢 N=151 → **0.093** — umbral 0.09 alcanzado, `QueryRewriter`+`union_pool` promovidos a `src/` |

---

## 🧠 Hallazgo interesante (para el artículo)

**Verificación binaria es categóricamente peor que permutación listwise
sobre el mismo pool oracle, incluso con el mismo dataset y el mismo
modelo de razonamiento como root.**

Mini run N=5, oracle pool, root=gpt-5.1, verifier subcall=gpt-5.4-mini:

| Query | gold en pool | gold detectados | gold rechazados | NDCG@10 |
|---|---:|---:|---:|---:|
| q01228 | 37 | 1 (3%) | 36 | 0.672 |
| q01342 | 37 | 4 (11%) | 33 | 0.786 |
| q01420 | 3 | 0 | 3 | 0.000 |
| q00844 | 13 | 0 | 13 | 0.180 |
| q01757 | 3 | 0 | 3 | 0.156 |
| **Total** | **93** | **5 (5%)** | **88** | mean 0.36 |

Comparación con listwise sobre el mismo pool oracle (N=151):
**listwise NDCG@10 = 0.714**, RLM rerank con verify binario = 0.36.

### Por qué (hipótesis a contrastar)

El listwise rerank pide al LLM **permutar** una ventana de 20 docs —
producir un orden relativo. El modelo puede colocar un gold "no estoy
100% seguro pero parece relacionado" en posición 3 sin tener que
comprometerse a un *true* binario. La información gradiente se preserva.

La verificación binaria pide al modelo un compromiso *yes/no*. Para
queries oblicuas donde la relación es sutil ("¿comparten técnica de
demostración?"), gpt-5.4-mini elige `no` por defecto en el 95% de los
casos. La información gradiente se pierde antes de llegar al ranker.

**Implicación para retrieval con LLMs:** cuando la relevancia es sutil
o latente (oblique queries del paper), pedir al LLM una *permutación* es
mucho más informativo que pedir *clasificaciones binarias paralelas*,
incluso si el segundo paradigma es más natural para descomponer en
subcalls. Es contraintuitivo desde la perspectiva de software (paralelo
parece mejor) pero crítico desde la perspectiva de información (un solo
LLM viendo 20 docs juntos extrae señal relativa que 20 LLMs viendo 1
doc cada uno pierden).

### Pregunta 1: ¿es capacidad del modelo o paradigma binario?

**Resuelto.** Test A (subcall=gpt-5.1 vs gpt-5.4-mini, mismo prompt binario):

| Verifier | detection rate gold |
|---|---:|
| gpt-5.4-mini | 5/93 (5.4%) |
| gpt-5.1 | 5/93 (5.4%) |

**Idéntico.** El modelo más capaz disponible (gpt-5.1) rechaza los gold
docs en la misma proporción que gpt-5.4-mini. No es problema de
capacidad — es el paradigma binario.

### Pregunta 2: ¿el scored 1-5 recupera el gradiente?

**Resuelto. Sí, dramáticamente — confirmado en N=151.** Test B
(subcall=gpt-5.4-mini con verifier scored 1-5 en lugar de binario,
mismo pool oracle):

| Paradigma | N | NDCG@10 | Δ vs baseline |
|---|---:|---:|---:|
| Binario (gpt-5.4-mini) | 5 | 0.359 | +0.012 |
| Binario (gpt-5.1) | 5 | 0.435 | +0.039 |
| Scored 1-5 (gpt-5.4-mini) | 5 | 0.776 | +0.342 |
| **Scored 1-5 (gpt-5.4-mini)** | **151** | **0.615** | **+0.309** |

Cambiar de binario a scored con el mismo modelo barato (gpt-5.4-mini)
multiplica el delta NDCG por **25-28x**. Con N=151 el scored
alcanza 0.615 NDCG@10 — un **86% del listwise oracle (0.714)** usando
un modelo de subcall **5x más barato** y con paralelización masiva.

### Conclusión final del hallazgo

**Cuando la relevancia es sutil (oblique queries), la calidad del
ranking depende del PARADIGMA de extracción más que del modelo:**

**Matriz final consolidada (N=151):**

Oracle pool (gold + distractores aleatorios):

| Paradigma | Modelo | NDCG@10 | Recall@10 |
|---|---|---:|---:|
| Sin LLM (oracle baseline, pool barajado) | — | 0.306 | 0.206 |
| Verify binario (N=5, no escalado) | gpt-5.4-mini | 0.36 | — |
| Verify binario (N=5, no escalado) | gpt-5.1 | 0.43 | — |
| **Verify scored 1-5** | **gpt-5.4-mini** | **0.615** | 0.482 |
| **Permutación listwise** | **gpt-5.1** | **0.714** | 0.550 |

BM25 pool (top-50 BM25):

| Paradigma | Modelo | NDCG@10 | Recall@10 |
|---|---|---:|---:|
| Sin LLM (BM25 baseline) | — | 0.028 | 0.029 |
| **Verify scored 1-5** | **gpt-5.4-mini** | **0.042** | 0.039 |
| **Permutación listwise** | **gpt-5.1** | **0.057** | 0.051 |

### Lectura honesta de los resultados

- **El listwise sobre gpt-5.1 sigue siendo el mejor reranker en NDCG
  absoluto** en ambos pools: 0.714 vs 0.615 en oracle, 0.057 vs 0.042
  en BM25.
- **El verifier scored con modelo barato alcanza el 86% del listwise en
  oracle y el 74% en BM25**, usando un subcall-model 5x más barato y
  con paralelización masiva.
- **El paradigma scored vs binario** (test A/B sobre N=5) sí muestra un
  salto cualitativo grande (+0.34 NDCG al cambiar de binario a scored
  con el mismo modelo). Ese hallazgo se sostiene como resultado de
  ablation interna del RLM-as-reranker.
- **El cuello de botella sigue siendo la primera etapa**: ambos
  paradigmas LLM sobre BM25 quedan ~10x por debajo del oracle. Confirma
  la tesis central de OBLIQ-Bench.

### Insight para el artículo (revisado tras N=151)

> "Forcing an LLM into binary relevance judgments destroys most of the
> latent-relevance signal: a 1-5 scored verifier on the same cheap model
> recovers ~25x more gold-aware NDCG than its binary counterpart on
> oblique math queries. Even so, single-shot listwise reranking on a
> frontier model remains the strongest single primitive — the scored
> verifier closes ~86% of that gap at ~5x lower per-token cost,
> suggesting a Pareto trade-off between paradigm efficiency and raw
> capability rather than a free lunch."

---

## Condición 4 — RLM agentic puro (sin rerank) — N=151

**Pregunta:** ¿La librería *antes* del trabajo en `rerank.py` ya
resolvía oblique queries con el RLM loop + tools de retrieval, o la
nueva primitiva `ListwiseReranker` es load-bearing?

**Setup:** `examples/oblique_agentic_bench.py`, root = `gpt-5.1`, tools
expuestos al REPL = `search(q, top_n)` (BM25 reformulable) +
`read_doc(doc_id)`. Sin `verify_relevance_batch`, sin pre-fetched pool,
sin reranker. Cache OFF (NoopCache). workers=1. 2026-05-12, 1429s
(~23.8 min), 0 errores.

### Resultados

| Sistema | NDCG@10 | Recall@10 | Δ NDCG vs BM25 |
|---|---:|---:|---:|
| BM25 baseline | 0.0284 | 0.0287 | — |
| **RLM agentic loop (sin rerank)** | **0.0411** | 0.0314 | **+0.0128 (1.45×)** |
| BM25 + `ListwiseReranker` | 0.0571 | 0.0510 | +0.0287 (2.0×) |

Comportamiento del agente: 5.0 búsquedas reformuladas/query, 38.3 docs
inspeccionados/query, 10.4 steps/query, **0.5 de 13.5 gold docs
"vistos" por query (~3.7%)**. El agente reformula con ahínco pero BM25
no surfacea gold incluso con reformulaciones — replica el patrón del
paper para Multi-Hop Agent.

### Lectura para la librería

Comparación directa: **librería antes vs después del trabajo en
`rerank.py`** sobre el mismo input (BM25 sobre Math-Oblique, N=151):

| Configuración | NDCG@10 |
|---|---:|
| Librería antes (loop + search/read_doc) | 0.0411 |
| Librería después (+ `ListwiseReranker`) | 0.0571 |

**La nueva primitiva añade +0.016 NDCG sobre lo que la arquitectura
agentic ya hacía** — un +39% relativo sobre la mejor configuración
"pre-rerank". Es una mejora load-bearing pero no transformadora.
También responde la inversa: el loop agentic ya aportaba +45% sobre BM25
crudo, así que la nueva primitiva no es la única forma que la librería
tiene de atacar oblique — solo la mejor.

### Caveat de comparabilidad con el paper

Nuestro Multi-Hop Agent = 0.041 NDCG@10. El "GPT-5.2 Multi-Hop Agent"
del paper = 0.161 NDCG@10. La diferencia (4×) viene de la primera
etapa: el paper usa Gemini-2-Embedding dentro de cada hop, nosotros
BM25. El agentic strategy es comparable; el retriever subyacente no.
Eso refuerza la tesis del paper: **el bottleneck es la primera etapa**.

---

---

## Condición 7t — Palanca 1 variante: TournamentReranker

**Motivación:** OBLIQ-Bench App. C describe un reranker de tipo torneo
(shuffle pool → batches → top-K survivors → recursive) y afirma que
escala a pools 300–2500 docs. La hipótesis era que sustituir
`ListwiseReranker` por `TournamentReranker` sobre el mismo pool de ~108 docs
podría mejorar la robustez.

**Implementación:** `TournamentReranker` añadido a
`src/pyrlm_runtime/rerank.py` y exportado en `__init__.py`. 9 tests
unitarios, todos verdes. Wired en `oblique_multiquery_bench.py` vía
`--reranker-mode tournament`.

### Resultados

| N | Sliding (ListwiseReranker) | Tournament (TournamentReranker) | Δ |
|---:|---:|---:|---:|
| 5 | **0.152** | 0.112 | −0.040 |
| 30 | **0.110** | 0.075 | −0.035 |

**Hipótesis refutada.** N=151 no se corrió porque N=30 ya mostraba una
diferencia consistente y negativa.

### Por qué pierde el torneo a pool size ~108

1. **Destruye el orden BM25.** El torneo aplica shuffle antes de cada
   ronda — con pool ~108, ese orden ya contiene señal real de relevancia
   que el shuffle borra. El sliding window lo preserva.
2. **Eliminación permanente.** Un doc gold que cae en un batch desfavorable
   queda eliminado para siempre. El sliding window nunca elimina — solo
   reordena.
3. **Pool size equivocado.** El torneo está diseñado para escalar a 300–2500
   docs (donde el sliding se vuelve costoso y el orden BM25 pierde precisión).
   A ~108 docs, el sliding se ejecuta en ~6 windows y conserva toda la señal.

**Conclusión:** `TournamentReranker` queda en la librería como primitiva
para pools grandes (es el diseño documentado en el paper). No es el
reranker óptimo para la Palanca 1 a pool size actual.

---

## 🎯 Siguiente paso

1. ✅ **Palanca 1 v2 completada** — NDCG=0.093, umbral alcanzado, `QueryRewriter`+`union_pool` en `src/`.
2. **Commit final:** `feat: add QueryRewriter, union_pool, ListwiseReranker, TournamentReranker + OBLIQ-Bench benchmarks`.
3. Doble-check: re-correr los experimentos clave con cache forzado OFF (ver `docs/OBLIQ-DOUBLECHECK-ROADMAP.md`).
4. Esqueleto del post LinkedIn / artículo con headline 0.093.
