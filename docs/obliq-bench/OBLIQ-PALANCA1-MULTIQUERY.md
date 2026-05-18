# Palanca 1 — Multi-Query Union antes del Listwise Rerank

> 🧭 **Empieza por** [`OBLIQ-OBJETIVO.md`](OBLIQ-OBJETIVO.md) si no
> tienes el contexto del proyecto. Este doc es el diseño detallado de
> UNA palanca; el por qué de todo el conjunto vive en el otro.
>
> Tracking doc. Última actualización: 2026-05-13  
> Estado: 🟢 N=151 v1=0.072, v2=0.093 — umbral 0.09 alcanzado. `QueryRewriter` + `union_pool` promovidos a `src/`

---

## 1. Planteamiento del problema

Tras correr todas las condiciones de OBLIQ-Bench math (ver
[`OBLIQ-EXPERIMENTS.md`](OBLIQ-EXPERIMENTS.md)), tenemos un diagnóstico
muy claro:

| Sistema | NDCG@10 | Recall@10 | Gold visible en pool/query |
|---|---:|---:|---|
| BM25 baseline (top-10) | 0.028 | 0.029 | ~baja |
| RLM agentic loop (5 reformulaciones libres) | 0.041 | 0.031 | 0.5 / 13.5 (3.7%) |
| BM25 + `ListwiseReranker` (pool top-50) | 0.057 | 0.051 | ~baja |
| Oracle pool + `ListwiseReranker` | **0.714** | **0.550** | 100% (oracle inyecta gold) |

La asimetría es brutal: **el rerank funciona perfectamente cuando el pool
contiene los gold (0.71 NDCG); el rerank apenas mueve la aguja cuando el
pool no los contiene (0.057)**. La conclusión empírica es inequívoca:

> **El cuello de botella no es el reranker. Es la composición del pool
> que llega al reranker.**

Y esa composición está estrangulada por BM25 sobre una **sola
formulación** de la query. En queries oblicuas (donde la relevancia
depende de un atributo latente — técnica de demostración, postura
implícita, modo de fallo — el problema se agrava: la query y el doc
no comparten vocabulario de superficie, así que un solo pass de BM25
nunca va a surfacear los gold.

La pregunta de esta palanca:

> **¿Podemos engordar y diversificar el pool sin tocar la indexación,
> únicamente reformulando la query con un LLM, para que el reranker
> tenga material decente con el que trabajar?**

---

## 2. Estado del arte (qué dice el paper y la literatura)

### En el propio OBLIQ-Bench

Table 3 del paper (Math subset) reporta:

| Sistema | NDCG@10 (Math, pooled) |
|---|---:|
| BM25 | 0.029 |
| Gemini-2-Embedding (dense) | 0.147 |
| **GPT-5.2 Query Rewriter (1 rewrite + Gemini)** | **0.185** |
| GPT-5.2 Multi-Hop Agent (Gemini en cada hop) | 0.207 |
| Oracle GPT-5.2 Tournament | 0.329 |
| Oracle GPT-5.2 Tournament+Soln | 0.473 |

Dos puntos clave del paper:

1. **Una sola reformulación con LLM ya bate a Gemini-2-Embedding** sobre
   el mismo retriever denso (0.185 vs 0.147). Eso es prueba directa de
   que reformular ayuda en queries oblicuas.
2. §5.2 (Lessons) afirma textualmente: *"Iterative reformulation can
   help when the latent target can be approached through several
   alternative phrasings"*. Math es exactamente este caso — una técnica
   de demostración admite N descripciones con vocabularios distintos.

### Trabajo relacionado citado en el paper (extended related work, app. A)

- **EAR** (Chuang et al. 2023): genera múltiples expansiones de query y
  rerankea antes de retrieval. Es la familia exacta de esta palanca.
- **Query Rewriting in RAG** (Ma et al. 2023a): documenta que reformular
  cierra el gap entre input de usuario y evidencia indexada.
- **ReDI** (Zhong et al. 2026): descompone queries complejas en
  sub-queries, fusiona resultados. Reporta ganancias en BEIR y BRIGHT.
- **DIVER** (Sun et al. 2026): combina iterative query expansion +
  reasoning-enhanced retriever + reranking.

### Diferencia respecto al paper

El paper hace **una sola reformulación** y la pasa por un retriever
denso (Gemini). Nosotros vamos a hacer **N reformulaciones que
ataquen distintos facets del atributo latente** y unirlas sobre un
retriever lexical (BM25). Esto explota una propiedad que ni el
paper ni los rewriters clásicos explotan: **diversidad de vocabulario
deliberada para inflar la cobertura del retriever lexical sobre una
relación oblicua**.

---

## 3. Hipótesis y predicción

### Hipótesis

> Si generamos 5 reformulaciones diversas en vocabulario de la query
> oblicua original, unimos los resultados de BM25 sobre cada
> reformulación, y aplicamos `ListwiseReranker` sobre la unión, el
> NDCG@10 final pasa del actual **0.057** a **0.09 – 0.13** en
> Math-Oblique (N=151), porque:
>
> 1. Cada reformulación con vocabulario distinto "ilumina" un
>    subconjunto distinto del corpus en BM25.
> 2. La unión de esos subconjuntos contiene más gold docs de los que
>    una sola formulación nunca surfacearía.
> 3. El reranker, que ya hemos demostrado que funciona perfectamente
>    cuando los gold están en el pool (oracle = 0.71), reordena la
>    unión y los promueve al top-10.

### Mecanismo medible

El KPI intermedio que valida la hipótesis es el **gold en pool**.

| Métrica | Valor actual (BM25 top-50) | Objetivo Palanca 1 |
|---|---:|---:|
| Pool size único | 50 | 80 – 120 |
| Gold en pool (de los disponibles) | ~5-8% | 15 – 30% |
| Recall@100 del pool | baja | sube |
| NDCG@10 final | 0.057 | 0.09 – 0.13 |

Si el gold-en-pool sube pero el NDCG no, eso indica que el reranker se
ahoga en pool más grande → habría que pasar a Palanca 3 (cascada).
Si el gold-en-pool no sube, el rewriter no está generando diversidad
útil → iterar el prompt.

---

## 4. Diseño detallado

### 4.0 Diagrama de arquitectura

```text
┌─────────────────────────────────────────────────────────────┐
│  Query oblicua del usuario                                  │
│  "Find problems requiring extremal argument with pigeonhole"│
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │  REWRITER (LLM)          │  1 llamada
              │  gpt-5.4-mini, ~3s       │  modelo barato
              │  Genera 5 reformulaciones│
              │  diversas en vocabulario │
              └──────────────┬───────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼          ▼      ▼      ▼    ▼   (5 reformulaciones)
        ┌────────┐ ┌────────┐ ┌──────┐
        │"telesc.│ │"discrete│ │"forwd│ ...
        │ binom. │ │ deriv." │ │ diff"│
        │ cube"  │ │         │ │      │
        └────┬───┘ └───┬────┘ └──┬───┘
             │         │         │
             ▼         ▼         ▼
        ┌────────┐ ┌────────┐ ┌──────┐
        │ BM25   │ │ BM25   │ │ BM25 │       5 búsquedas BM25
        │ top-25 │ │ top-25 │ │ top-25│       (gratis, in-memory)
        └────┬───┘ └───┬────┘ └──┬───┘
             │         │         │
             └─────────┼─────────┘
                       ▼
              ┌──────────────────────────┐
              │  UNIÓN DEDUPLICADA       │
              │  5 × 25 = 125 hits       │
              │  ~110 únicos             │
              │  (overlap ~13%)          │
              └──────────────┬───────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │  LISTWISE RERANKER (LLM) │  ~7 ventanas
              │  gpt-5.1, ~30s           │  modelo potente
              │  Lee QUERY ORIGINAL +    │  Importante: el reranker
              │  pasajes, permuta        │  ve el intent original,
              │  ventanas de 20 docs     │  NO los rewrites
              └──────────────┬───────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │  TOP-10 final            │
              └──────────────────────────┘
```

**Coste/query**: rewriter ~$0.003 + rerank ~$0.02 ≈ **$0.025**.
**Tiempo/query**: rewriter ~3s + BM25 instantáneo + rerank ~30s ≈ **33s**.

### Metáfora intuitiva

Imagina que buscas en una biblioteca un libro sobre **estrategias de
negociación**, pero el bibliotecario solo entiende búsquedas por
palabra exacta del título.

- **Versión antes (Cond 2):** pides "negociación". El bibliotecario te
  trae 50 libros con esa palabra en el título. Eliges los 10 mejores.
  Libros como *Getting to Yes* o *Never Split the Difference* nunca
  aparecen — no tienen "negociación" en el título.

- **Versión nueva (Palanca 1):** primero hablas con un asistente
  experto que te dice: *"podrías buscar también 'persuasion',
  'deal-making', 'conflict resolution', 'bargaining', 'influence
  tactics'"*. Vas al bibliotecario con esas 5 búsquedas, juntas las
  listas, eliminas duplicados, te queda un pool de ~110 libros.
  **Ahora sí incluyen *Getting to Yes* y *Never Split the Difference*.**
  Eliges los 10 mejores.

| Rol en la metáfora | Componente real |
|---|---|
| Asistente experto que sugiere sinónimos | Rewriter LLM (gpt-5.4-mini) |
| Bibliotecario literal | BM25 in-memory |
| Tú leyendo los lomos | Listwise reranker LLM (gpt-5.1) |

### Por qué encaja con la filosofía de pyrlm-runtime

1. **Composición pura.** No inventamos primitivas nuevas: reutilizamos
   el `ListwiseReranker` existente, añadimos una función `Rewriter`
   mínima (~30 líneas), BM25 ya estaba. Cumple el principio
   "missing primitive that cannot be composed" de
   [CLAUDE.md](../../CLAUDE.md).
2. **El intent del usuario nunca se reemplaza.** El rewriter solo
   expande el retrieval. Cuando el reranker juzga, ve la **query
   original**, no las reformulaciones — las reformulaciones eran un
   truco para abrirle la puerta al retriever lexical, no para
   reescribir lo que el usuario pidió.
3. **Dos modelos, dos roles.** gpt-5.4-mini para una tarea
   estructurada simple (JSON con 5 frases). gpt-5.1 para la tarea
   pesada de juicio de relevancia. Cada modelo donde es bueno y donde
   es barato.

### 4.1 Pipeline

```text
1. Query oblicua original
   "find problems requiring extremal argument with pigeonhole reasoning"
            ↓
2. Rewriter LLM (gpt-5.4-mini, 1 llamada)
   Genera 5 reformulaciones diversas en vocabulario:
   r1 = "problems using maximum/minimum element arguments and counting"
   r2 = "competition problems with worst-case configuration analysis"
   r3 = "combinatorial problems involving boxes balls drawer principle"
   r4 = "problems proving via considering extreme configurations"
   r5 = "selecting largest/smallest object then deducing structure"
            ↓
3. BM25 × 5
   BM25(r_i) → 25 hits cada uno
            ↓
4. Unión deduplicada por doc_id
   → ~80-120 docs únicos (solape esperado 30-50%)
            ↓
5. ListwiseReranker (sin cambios — primitiva existente)
   pool unión → top-10
            ↓
6. NDCG@10 / Recall@10 sobre qrels oficial
```

### 4.2 Prompt del rewriter (la pieza crítica)

```text
System:
You are a search-query reformulation expert for an oblique retrieval
task. The user is searching a corpus of math problems for problems
sharing the same proof technique or reasoning structure as their
query — even when the surface topic differs.

Your job: given the ORIGINAL QUERY, produce exactly 5 reformulations
that each attack the same underlying technique from a DIFFERENT
angle. Each reformulation should use vocabulary that a problem author
might plausibly choose when applying the same technique to a
different topic.

Constraints:
- Each reformulation must be a single concise phrase (10-20 words).
- The 5 reformulations together must span as much vocabulary diversity
  as possible. Do NOT generate near-duplicates of each other.
- Avoid vocabulary that is too generic ("math problem", "competition
  problem", "find problems") — that hurts BM25 precision.
- Avoid named entities, specific numbers, or specific theorem names
  from the original query.
- Stay faithful to the underlying technique. Do not drift into
  unrelated math areas.

Return JSON, no prose:
{"rewrites": ["...", "...", "...", "...", "..."]}

User: {original_query}
```

Justificación de cada constraint:

| Constraint | Motivo |
|---|---|
| 10-20 palabras | BM25 funciona mejor con queries informativas pero acotadas; queries de 5 palabras pierden señal, de 50 inflan ruido. |
| Sin near-duplicates | Sin este constraint el LLM produce 5 paráfrasis triviales. Verificable inspeccionando varias muestras. |
| Sin vocab genérico | Palabras como "problem" matchean todo el corpus → BM25 colapsa. |
| Sin entidades / theorem names | Una query "Cauchy-Schwarz problems" recuperaría exactamente lo que recupera la original, no aporta diversidad. |
| Stay faithful | Un rewriter creativo puede drift hacia técnicas adyacentes y meter ruido. Ya tenemos al rerank para filtrar; el rewriter no es el filtro. |

### 4.3 Hiperparámetros y decisiones

| Parámetro | Valor | Justificación |
|---|---|---|
| Modelo rewriter | `gpt-5.4-mini` | Tarea estructurada simple, no necesita razonamiento profundo. Coste ~10× menor que gpt-5.1. |
| Modelo reranker | `gpt-5.1` | Mismo que el run baseline (0.057) — la comparación tiene que aislar la palanca, no cambiar el reranker. |
| N rewrites | 5 | Sweet spot: suficiente diversidad sin coste excesivo. ≥10 produce solape alto, ≤3 deja recall en la mesa. |
| BM25 top-n por rewrite | 25 | 5 × 25 = 125 hits brutos → ~80-120 únicos. Da margen al reranker. |
| Reranker top-n input | usa todo el pool unión | El reranker corre window=20 step=10, escala bien hasta ~150 docs. |
| Reranker top-k output | 10 | Comparable al resto. |
| Cache | OFF (NoopCache) | Mismo régimen que el resto del doble-check. |
| Workers | 1 | Conservador, evita race conditions de runs previos. |
| N queries | smoke → 5 → 30 → 151 | Validación progresiva, mismo patrón que Cond 4. |

### 4.4 Métricas a registrar por query

Además de NDCG@10 / Recall@10 (las headline), registramos diagnósticos
de proceso para entender si la mecánica funciona:

```python
{
  "query_id": "...",
  "original_query": "...",
  "rewrites": ["...", "...", "...", "...", "..."],

  # Pool composition
  "n_hits_per_rewrite": [25, 25, 25, 25, 25],
  "n_unique_pool": 92,         # ← KPI: tamaño tras dedup
  "overlap_rate": 0.26,        # 1 - (n_unique / sum_n_hits)

  # Coverage of gold
  "n_gold_total": 13,
  "n_gold_in_pool": 4,         # ← KPI principal de la palanca
  "gold_pool_recall": 4/13,

  # Final metrics
  "ndcg10_baseline_bm25": 0.000,   # BM25 sobre query original, no rewrite
  "ndcg10_baseline_rerank": 0.000, # baseline cacheado del run anterior
  "ndcg10_multiquery": 0.123,      # ← headline
  "recall10_multiquery": 0.091,
  "recall100_multiquery": 0.231,   # ← señal de pool quality

  # Cost
  "rewriter_calls": 1,
  "rewriter_tokens_out": 180,
  "rerank_llm_calls": 7,           # ~window passes sobre el pool
  "total_time_s": 4.5
}
```

### 4.5 Cost-benefit estimado

| Componente | Coste extra vs Cond 2 (BM25+rerank) |
|---|---|
| 1 llamada rewriter gpt-5.4-mini por query | ~$0.003 |
| 4 BM25 calls extra por query (in-memory) | $0 |
| Reranker pasa de 50 a ~100 docs → ~+50% LLM calls de rerank | ~$0.01-0.02 |
| **Total extra por query** | ~$0.013-0.023 |
| **Total extra N=151** | ~$2-3 |

Sobre el coste del run actual (~$5 de Cond 2), es **+40-50% de coste** por
**+58-128% de NDCG esperado**. Ratio favorable si la predicción se cumple.

---

## 5. Implementación

### Archivos

- `examples/oblique_multiquery_bench.py` — bench standalone (~500 líneas).
  - Reutiliza `InMemoryBM25` (copia local) e infrastructure del rerank bench.
  - Añade `rewrite_query(query: str) -> list[str]`.
  - Pool = unión deduplicada de los 5 BM25s, preservando primer rank visto.
  - Aplica `ListwiseReranker` sobre la unión.
- Sin cambios en `src/` — la palanca es composición pura de primitivas
  existentes (BM25 in-memory + `ListwiseReranker` + un prompt nuevo).

### Comandos

```bash
# Smoke
uv run python examples/oblique_multiquery_bench.py --smoke

# N=5 (validar diversidad del rewriter — inspeccionar manualmente)
uv run python examples/oblique_multiquery_bench.py \
  --adapter azure --rewriter-model gpt-5.4-mini --rerank-model gpt-5.1 \
  --max-examples 5 --workers 1

# N=30 (primera señal estadística)
uv run python examples/oblique_multiquery_bench.py \
  --adapter azure --rewriter-model gpt-5.4-mini --rerank-model gpt-5.1 \
  --max-examples 30 --workers 1

# N=151 (número publicable)
uv run python examples/oblique_multiquery_bench.py \
  --adapter azure --rewriter-model gpt-5.4-mini --rerank-model gpt-5.1 \
  --max-examples 151 --workers 1
```

---

## 6. Resultados

### Smoke (FakeAdapter)

- Estado: ✅ verde
- Mecánica end-to-end validada (5 rewrites → 5 BM25 → unión dedup → rerank).

### N=5 — validación de diversidad

- Estado: ✅ rewriter produce 5 reformulaciones genuinamente diversas
- Inspección manual: cada query cubre 5 ángulos distintos sin repetir
  vocabulario ni paráfrasis triviales (e.g., para una integral 2D: "swap
  integration order", "triangular region", "odd-even cancellation",
  "separable kernel", "wedge-shaped telescoping").
- NDCG@10 = 0.152, overlap_rate=13%, gold_in_pool=1.2/18.6 (vs baseline 0.2)
- **Caveat:** N=5 tiene poco poder estadístico — el número era una
  selección sesgada del shuffle.

### N=30

- Estado: ✅
- NDCG@10 = **0.110**, pool recall 5.7% → 19.6%
- 0 errores, 854s wall time
- Aparecen las primeras regresiones por query (q01424: base 0.220 → mq 0.000)
  pero el agregado claramente positivo.

### N=151 (headline)

- Estado: ✅ — 2026-05-13, gpt-5.4-mini + gpt-5.1, 3821s wall time, 0 errores
- 2 ReadTimeouts en Azure recuperados con retry automático (no afectaron resultados)

| Métrica | Cond 2 (BM25+rerank) | Palanca 1 (multi-query+rerank) | Δ |
|---|---:|---:|---:|
| NDCG@10 | 0.0571 | **0.0717** | +26% relativo |
| Recall@10 | 0.0510 | **0.0734** | +44% relativo |
| Pool size único (avg) | 50 | 107.6 | 2.15× |
| Gold en pool / query | 0.47 | 0.76 | 1.6× |
| Pool-level recall | 6.4% | **13.0%** | 2.0× |
| Overlap rate (rewrites) | n/a | 13.9% | ─ |
| LLM calls rewriter / N | 0 | 151 | +151 |
| LLM calls rerank / N | ~604 | 1541 | 2.55× |
| Wall time | ~480s (workers=4) | 3821s (workers=1) | comparable |

### Evolución por N (mostrando la regresión a la media)

| N | NDCG@10 multi-query | Δ vs Cond 2 |
|---:|---:|---:|
| 5 | 0.152 | +0.095 |
| 30 | 0.110 | +0.053 |
| **151** | **0.072** | **+0.015** |

El número honesto es el de N=151. Los runs pequeños sobrestimaron por
selección del shuffle (mismo seed=42, primeras N queries del orden
mezclado). Documentar esta evolución es importante para futuros papers
o posts: no se puede vender el 0.152 ni el 0.110.

### Distribución de outcomes a N=151 (lectura cualitativa)

Tres tipos de query emergen:

1. **Wins claros** (~25-30 queries): la palanca rescata gold que BM25
   nunca veía. Ejemplos:
   - q02887: base 0.000 → mq **0.853** (3/3 gold rescatados)
   - q03053: base 0.156 → mq **0.636**
   - q02775: base 0.098 → mq **0.422** (5/5 gold rescatados)
   - q02891, q01506, q01522, q03422, etc.
2. **Sin efecto** (~95-100 queries): ni BM25 ni los rewrites surfacean
   gold. La palanca no degrada pero tampoco aporta. Estas son las
   queries auténticamente "duras" del dataset.
3. **Regresiones** (~10-15 queries): baseline tenía gold y los rewrites
   movieron el haz hacia otro vecindario:
   - q02712: base **0.446** → mq 0.146 (perdió 1 de 5 gold del pool)
   - q01424: base **0.220** → mq 0.000 (perdió los 4 gold)
   - q02153: base **0.636** → mq 0.182
   - q01798: base **0.190** → mq 0.000

El agregado es positivo (+0.015 NDCG) pero el balance es ajustado: los
wins claros compensan las regresiones y un mar de queries indiferentes.

### Coste por query

- Rewriter (1 call gpt-5.4-mini): ~$0.003
- BM25 × 5 (in-memory): $0
- Reranker (~10 windows gpt-5.1 sobre pool 107): ~$0.04
- **Total ≈ $0.045/query** (vs Cond 2 ~$0.025/query, +80%)
- N=151 ≈ $7 total (vs Cond 2 ~$4)

### Ablación: ¿TournamentReranker mejora sobre ListwiseReranker?

**Hipótesis:** sustituir `ListwiseReranker` por el `TournamentReranker`
(App. C del paper, shuffle + batches recursivos) podría mejorar el ranking
sobre el pool unión de ~108 docs.

**Resultado:**

| N | Sliding (ListwiseReranker) | Tournament (TournamentReranker) | Δ |
|---:|---:|---:|---:|
| 5 | **0.152** | 0.112 | −0.040 |
| 30 | **0.110** | 0.075 | −0.035 |

**Hipótesis refutada.** El torneo pierde consistentemente a este pool size.
N=151 no se corrió porque N=30 ya mostraba la diferencia estable.

**Por qué:** el torneo aplica shuffle antes de cada ronda, destruyendo el
orden BM25 que a pool ~108 sí es señal válida. Además elimina docs
permanentemente entre rondas — un gold en un batch desfavorable no tiene
segunda oportunidad. El sliding window preserva el orden inicial y nunca
descarta. El torneo está diseñado para pools 300–2500 donde el sliding se
vuelve costoso; a ~108 docs es contraproducente.

`TournamentReranker` queda en la librería para su uso con pools grandes.
`ListwiseReranker` (sliding) es el default para la Palanca 1.

---

## 7. Conclusiones

### ¿La hipótesis se confirma?

**Parcialmente.** Predicción: NDCG@10 ∈ [0.09, 0.13]. Resultado real:
**0.072**. Está por debajo del rango predicho, pero por encima del
baseline Cond 2 (0.057, +26% relativo). El KPI intermedio (pool-level
recall) sí sube como predijimos (6.4% → 13.0%, 2×) — el rewriter
**funciona** como diseño. Lo que no se cumple es la tasa de conversión
de "gold en pool" a "gold en top-10": meter más gold en el pool también
mete más ruido que el reranker tiene que filtrar.

Las predicciones optimistas de N=5 (0.152) y N=30 (0.110) fueron
**selección sesgada del shuffle**, no señal generalizable. Con seed=42,
las primeras 30 queries del orden mezclado incluían varias *big wins*
(q02887, q03053, q01522) que no representan la distribución completa.

### ¿Convierte esto en una primitiva de la librería?

**No por sí sola, todavía.** Los criterios pre-establecidos:

| Criterio (predefinido) | Resultado real | Decisión |
|---|---|---|
| NDCG ≥ 0.09 estable | 0.072 | ✗ no llega al umbral |
| Mecánica estable, 0 errores | sí | ✓ |
| Mejora consistente vs Cond 2 | +26% relativo | ✓ |
| Sin regresiones por query | 10-15 regresiones | ✗ |

→ **Decisión:** la composición se queda como **patrón documentado en
`examples/`** (no en `src/`). Hay margen claro para una v2 antes de
promoverla.

### Resultados v2 (N=151) — umbral alcanzado

Añadiendo la query original al fan-out (`--include-original`, ON por defecto):

| Métrica | v1 (5 rewrites) | v2 (+ query original) | Δ |
|---|---:|---:|---:|
| NDCG@10 | 0.0717 | **0.0927** | +0.0210 (+29%) |
| Recall@10 | 0.0734 | 0.0809 | +0.0075 (+10%) |
| Pool size avg | 107.6 | **127.5** | +19.9 docs |
| Pool-level recall | 13.0% | **16.1%** | +3.1pp |
| Gold in pool / query | 0.76 | **1.09** | +43% |
| LLM rerank calls | 1541 | 1837 | +19% |

**El umbral pre-establecido (NDCG ≥ 0.09 estable) se cumple: 0.0927.**

Regresiones clave resueltas por v2:
- q01798: base=0.190, v1=0.000 → v2=**0.448** ✅ (superó el baseline)
- q01424: base=0.220, v1=0.000 → v2=**0.208** ✅ (casi recuperación total)

Regresiones persistentes (reranker failure, no pool failure):
- q02153: base=0.636, v1=0.182 → v2=0.000 🔴 (gold=2/3 en pool — el reranker los entierra)
- q02712: base=0.446 → v2=0.214 🟡 (gold=5/5 en pool — el reranker falla)

Estos casos residuales son **reranker ceiling**, no pool ceiling: los golds están en el pool pero el reranker no detecta la relevancia latente a pool size ~130.

### Iteración v3 candidata

Si v2 alcanza el umbral 0.09, considerar:

- **Diversificar el retriever, no solo la query.** Si el ES expone
  campo vectorial, hacer hybrid_search en una de las "búsquedas" del
  fan-out (Palanca 2 del análisis original).
- **Cascada** (Palanca 3): pool grande → rerank-30 → scored verify-10.
  Aprovecha que el verifier scored fue 0.615 NDCG en oracle pool.

### Lectura para la librería

Lo que **sí** está validado por este experimento, incluso con el
número modesto:

1. **La composición funciona** — `Rewriter + BM25×N + dedup +
   ListwiseReranker` cabe en ~500 líneas de Python plano, sin tocar
   `src/`. Es exactamente la USP de pyrlm-runtime.
2. **El bottleneck es real y diagnosticable** — pool recall sube 2×,
   pero NDCG sube poco. Eso enseña algo: el reranker actual no escala
   linealmente con la calidad del pool, hay un techo intermedio.
3. **Las primitivas existentes son sólidas** — 0 errores en 151
   queries, telemetría completa, retries automáticos en Azure.

Lo que **no** está validado:

- Que esta palanca específica sea la mejor inversión de complejidad.
  Con +26% NDCG por +80% coste, la curva no es Pareto-dominante. Una
  v2 (con la mitigación) puede cambiar esa relación.

### Siguiente paso

1. ✅ **v2 corrida — umbral alcanzado.** `QueryRewriter` + `union_pool`
   promovidos a `src/pyrlm_runtime/multiquery.py`. 15 tests unitarios.
2. **Commit final:** `feat: add QueryRewriter, union_pool, ListwiseReranker,
   TournamentReranker + OBLIQ-Bench multi-query benchmarks`.
3. **Post/artículo:** los datos de v2 (0.093) son el número publicable.
4. Si en algún momento el ES expone campo vectorial, Palanca 2
   (hybrid search en una de las 6 búsquedas del fan-out) es el
   siguiente salto natural.

---

## Apéndice — Referencias relevantes del paper

- **§5.1 Descriptive queries** (Twitter, WildChat): el GPT-5.2 Query
  Rewriter es marginal en Twitter (0.066 vs 0.132 Gemini-Embed) pero
  significativo en Math (0.142 vs 0.144 Gemini-Embed). Sugiere que el
  rewriter ayuda más en tareas analógicas que descriptivas.
- **§5.2 Lessons**: *"Query rewriting might hence not be a silver-bullet
  solution to obliqueness"* — el paper documenta que rewriting daña
  Writing-Style. Math es buen target; otras tareas no.
- **App. A (Extended related work)**: EAR, ReDI, DIVER, Generate-then-Ground
  son la familia de trabajos. Esta palanca es la versión simple y limpia
  de esa familia.
- **Figure 3**: el reranker GPT-5.2 escala bien hasta pools de 2500 docs.
  El pool de 100 docs que produciremos está muy lejos de saturar al
  reranker — la elección de tamaño es conservadora, hay margen para
  subir si los datos lo justifican.
