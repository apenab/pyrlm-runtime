# OBLIQ-Bench (MIT, mayo 2026) — Análisis y relación con pyrlm-runtime

> 🧭 **Empieza por** [`OBLIQ-OBJETIVO.md`](OBLIQ-OBJETIVO.md) si no
> tienes el contexto del proyecto. Este doc es el análisis del paper
> y cómo se conecta con la librería; el objetivo de fondo está en el
> otro.

Paper: _OBLIQ-Bench: Exposing Overlooked Bottlenecks in Modern Retrievers with Latent and Implicit Queries_  
Autores: Diane Tchuindjo, Devavrat Shah, Omar Khattab — MIT  
ArXiv: 2605.06235v1

---

## Parte 1 — Análisis del paper vs pyrlm-runtime

### ¿Está relacionado con pyrlm-runtime? Sí, profundamente.

El paper describe exactamente el problema que el módulo `retrieval.py` de pyrlm-runtime enfrenta hoy. La relación es directa:

| Paper (OBLIQ-Bench)       | pyrlm-runtime                            |
| ------------------------- | ---------------------------------------- |
| BM25 / lexical search     | `ElasticsearchRetriever.search()`        |
| Dense embedding search    | `ElasticsearchRetriever.vector_search()` |
| Hybrid BM25 + dense + RRF | `ElasticsearchRetriever.hybrid_search()` |
| Multi-hop agentic search  | El loop RLM con subcalls iterativos      |

**El paper demuestra empíricamente que los tres primeros fallan** en "oblique queries" — queries donde la relevancia depende de un atributo latente del documento, no expresado en su superficie textual. Y el loop multi-hop del tipo que implementa pyrlm-runtime "ayuda solo cuando la oblicuidad se puede traducir en acciones heurísticas de búsqueda."

---

### El problema central: asimetría retrieval–verification

Hay una **asimetría retrieval–verification**: un LLM de razonamiento puede verificar fácilmente si un documento es relevante cuando se lo muestras, pero los mejores sistemas de retrieval (incluido Gemini-2-Embedding, el mejor en el benchmark) no logran _surfacear_ esos documentos en primer lugar.

Esto es exactamente lo que le pasa a un RLM usando `hybrid_search`: recupera candidatos por similitud de superficie (tokens + embeddings), pero los documentos latentemente relevantes no comparten tokens ni embeddings similares con la query — son relevantes por estructura abstracta, postura implícita, o patrón de razonamiento.

---

### Qué mejoras concretas sugiere el paper para retrieval.py

El paper propone tres mecanismos. Uno es infraestructura de indexing (fuera del scope de la librería). Los dos que sí aplican:

**Mejora 1 — `rerank()`: primitiva de reranking LLM-en-el-loop (la más impactante)**

El "Oracle GPT-5.2 Tournament" del paper alcanza NDCG@10 de 0.43 donde Gemini-2-Embedding solo llega a 0.13. Lo que hace es: recuperar un pool grande (K >> k, p.ej. 300 candidatos de múltiples sistemas) y luego aplicar razonamiento conjunto query+documento para rankear.

Actualmente `retrieval.py` no tiene esto. El RLM loop ya tiene el LLM adapter — un método `rerank(query, candidates, *, top_k)` que use el adapter para aplicar joint reasoning sería el primitive que cierra este gap. No es domain-specific: cualquier consumidor de pyrlm-runtime que enfrente queries oblicuas se beneficiaría.

**Mejora 2 — `search_large_pool()`: pool explícito para reranking downstream**

Actualmente `vector_search` tiene `num_candidates = max(top_k * 2, 50)` hardcodeado. El paper muestra que la calidad del reranker escala con el pool (Figure 3: a mayor K, mayor recall). Se necesita un método que devuelva K >> k candidatos de múltiples estrategias (BM25 + dense) sin pre-filtrar a top_k, para consumo por el reranker. Hoy no hay forma elegante de hacer esto.

---

### Lo que NO pertenece aquí

La pipeline del paper (anotar documentos con LLM, extraer atributo latente f(d), clusterar por atributo) es un pipeline de _indexing_ — lógica de dominio que pertenece en el consumidor (e.g., `banking-rlm`), no en pyrlm-runtime. La librería no debe saber qué atributo latente extraer.

---

## 🚨 Restricción operativa de este proyecto

**Asumimos que el Elasticsearch (o el retriever subyacente) ya está
construido y no podemos tocar su indexación.** Esa decisión es
deliberada y define el alcance de todo lo que sigue: cualquier
mejora tiene que venir del **read path** (cómo consultamos, cómo
combinamos resultados, cómo razonamos sobre ellos), nunca del write
path (cómo anotamos o estructuramos los documentos antes).

Lo que esta restricción **permite explorar** dentro de pyrlm-runtime:

- Reranking LLM sobre el pool devuelto (`ListwiseReranker`, ya
  implementado).
- Verification scored / binario con subcalls paralelos (RLM-as-reranker).
- Query rewriting iterativo + agentic multi-hop sobre BM25.
- Composición de pools (unión de varias estrategias de búsqueda si el
  ES expone dense + lexical).
- Cascadas tipo `BM25 → ListwiseReranker → scored verify`.

Lo que esta restricción **excluye explícitamente**:

- Re-anotar el corpus con LLMs para extraer atributos latentes.
- Re-indexar con campos nuevos derivados del contenido.
- Construir clusters por atributo (Step 3 del paper).
- Cualquier modificación del schema de Elasticsearch.

### 💡 Pero esto sería donde brillaría de verdad

**La Solución 1 del paper (anotación de atributos latentes en tiempo
de indexación) es, con mucha diferencia, el mayor salto cualitativo
posible para retrieval oblique.** Los datos del propio paper lo
confirman:

| Familia de soluciones                                                 |                            NDCG@10 (Math) |
| --------------------------------------------------------------------- | ----------------------------------------: |
| Read-path puro (BM25/dense/rerank/multi-hop)                          |                               0.03 – 0.21 |
| Read-path + oracle tournament (lo más fuerte sin tocar índice)        |                               0.28 – 0.43 |
| **Indexación con atributo latente + retrieval estándar (Solución 1)** | **arquitectura recomendada por el paper** |

El paper deja literalmente esto como su frontera abierta en la
conclusión: _"the next frontier for retrieval might need to be
architectures that make latent document attributes available at
search time"_.

**Si en algún momento se relaja la restricción operativa de este
proyecto y se gana control sobre la indexación, ése es el siguiente
salto a explorar.** Sería un proyecto en sí mismo:

1. **Pipeline de anotación** — un job que pasa cada doc del corpus por
   un LLM (gpt-5.4-mini) extrayendo el atributo latente del dominio
   (e.g., "técnica de demostración" para Math, "stance" para Twitter,
   "tipo de operativa" para banking).
2. **Indexación enriquecida** — guardar la anotación como campo
   searchable en ES junto al texto crudo.
3. **Retrieval estándar sobre campo anotado** — BM25 / dense / hybrid
   sobre la anotación, no sobre el texto.
4. **Coste** — depende del tamaño del corpus: 3.5k docs ≈ $30-50, 72k
   docs ≈ $500-1.000, millones de docs ≈ pipeline batch dedicado.

Mientras tanto, el alcance de pyrlm-runtime sobre OBLIQ es el techo
del read-path: **algo entre 0.05 y 0.20 NDCG@10 en Math-Oblique**, no
los 0.43+ que requerirían tocar la indexación. Esa cota está
documentada por los experimentos N=151 ya corridos.

---

## Parte 2 — Explicación detallada del problema y las soluciones

### El problema fundamental

Imagina que tienes un corpus de documentos y alguien hace esta búsqueda:

> _"Encuentra tweets donde el autor critica la guerra, pero de forma irónica — sin decirlo explícitamente"_

Un tweet relevante podría ser:

> _"Qué bonito que los niños vean los fuegos artificiales esta noche 🎆"_

¿Qué tienen en común la query y el tweet? **Absolutamente nada en la superficie.** No comparten palabras, no comparten semántica obvia. La relevancia está en el _subtexto_ — en lo que el tweet implica sin decir.

Esto es lo que el paper llama una **oblique query**: la relevancia depende de un atributo _latente_ del documento.

---

### ¿Por qué fallan los tres métodos actuales de retrieval.py?

#### BM25 — `search()` — búsqueda por palabras clave

Compara palabras de la query con palabras del documento. Si la query dice "guerra" e "ironía" y el tweet no dice ninguna de esas palabras, BM25 da score 0. **Falla total.**

#### Dense embeddings — `vector_search()` — búsqueda semántica

Convierte query y documento a vectores numéricos y compara por similitud coseno. Los embeddings capturan semántica general bien: "perro" está cerca de "animal". Pero **no capturan postura implícita, ironía, ni estructura abstracta de razonamiento**. El embedding del tweet de los fuegos artificiales está en el espacio semántico de "celebración", no de "crítica a la guerra".

> **Dato clave del paper:** el problema existe tanto con embeddings como sin ellos. El paper demuestra que incluso Gemini-2-Embedding (el mejor embedding disponible hoy) falla en oblique queries. Gemini alcanza NDCG@10 = 0.13 en Twitter-Conflict. Un LLM de razonamiento rerankeando sobre un pool grande llega a 0.43. La brecha es enorme incluso con los mejores embeddings.

#### Hybrid search — `hybrid_search()` — BM25 + embeddings + RRF

Combina los dos anteriores. Mejora la búsqueda general, pero si ambos métodos fallan en oblique queries, combinarlos sigue fallando. Es sumar dos ceros.

---

### ¿Por qué un LLM sí puede reconocer la relevancia?

Porque puede _razonar conjuntamente_ sobre la query y el documento. Si le muestras el tweet de los fuegos artificiales junto a la query "crítica irónica a la guerra", el LLM entiende el contexto, el tono, la ironía. Pero para hacer eso necesita _ver_ el documento — no puede hacerlo en el paso de retrieval masivo sobre un corpus de millones de documentos.

Aquí está el problema central del paper: hay una **asimetría**:

- **Verificar** relevancia (LLM lee query + documento) = fácil
- **Recuperar** el documento del corpus de 72.000 tweets = imposible con métodos actuales

---

### Las dos soluciones que el paper propone

#### Solución 1 — Extraer atributos latentes en tiempo de indexación

La idea: antes de indexar los documentos, **pasar un LLM por cada documento** para extraer el atributo latente relevante, y guardar esa anotación como campo searchable.

Ejemplo concreto:

1. Tweet: _"Qué bonito que los niños vean los fuegos artificiales 🎆"_
2. El LLM anota: `{"stance": "implicit_antiwar", "tone": "sarcastic"}`
3. Guardas esa anotación en Elasticsearch junto al tweet
4. Cuando viene la query "crítica irónica a la guerra", puedes buscar por `stance: implicit_antiwar`

**¿Aplica a pyrlm-runtime?** Esta parte es un pipeline de _indexación_ — un proceso offline que corre antes de que el RLM empiece a trabajar. pyrlm-runtime no gestiona el indexado, eso lo hace el consumidor. Así que **esta solución no pertenece en retrieval.py** — pero el paper te dice que si tienes control del pipeline de indexación en tu consumidor, esto es lo más potente.

**¿Necesitas embeddings para esto?** No necesariamente. Si tus documentos tienen anotaciones de atributos latentes como campos de texto en Elasticsearch, BM25 sobre esos campos funciona. Los embeddings ayudan para clustering de las anotaciones, pero no son obligatorios.

#### Solución 2 — Reranking LLM sobre un pool grande

Esta sí aplica directamente a pyrlm-runtime. El proceso:

1. **Recupera un pool grande** de candidatos: en vez de pedir `top_k=10`, pides `top_k=200` o `500` (aunque sean imperfectos, al menos algunos relevantes estarán en el pool)
2. **El LLM reankea** ese pool leyendo cada candidato junto a la query y puntuando relevancia con razonamiento conjunto
3. **Devuelves los top_k mejores** según el LLM

El paper llama a esto "Tournament Reranking" y muestra que funciona incluso sobre candidatos recuperados por métodos imperfectos — porque el LLM puede distinguir los relevantes cuando los _ve_.

```text
Query oblique
    ↓
hybrid_search(top_k=500)   ← pool grande, imperfecto pero incluyente
    ↓
LLM lee query + cada doc   ← razonamiento conjunto
    ↓
top_k=10 rerankeados       ← resultado final bueno
```

**¿Necesitas embeddings para esto?** No. Puedes hacer el pool grande solo con BM25, o solo con embeddings, o con hybrid. La calidad del pool mejora con embeddings, pero el reranking LLM ayuda independientemente del método de retrieval que uses para llenar el pool.

---

### Resumen práctico

| Tienes                                           | Situación actual                      | Con mejora del paper                                                             |
| ------------------------------------------------ | ------------------------------------- | -------------------------------------------------------------------------------- |
| Solo BM25 (sin embeddings)                       | Falla en queries oblicuas             | Reranking LLM sobre pool grande BM25 ayuda moderadamente                         |
| BM25 + embeddings                                | Mejor, pero sigue fallando en oblique | Reranking LLM sobre pool grande hybrid ayuda mucho más                           |
| BM25 + embeddings + atributos latentes indexados | —                                     | Solución más completa, requiere trabajo en pipeline de indexación del consumidor |

La mejora que más sentido tiene implementar en `retrieval.py` es la **Solución 2**: un método `rerank()` que tome un pool de candidatos y use el LLM del RLM loop para reankearlos. Funciona con o sin embeddings, y es un primitive genuinamente general.
