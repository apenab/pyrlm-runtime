# OBLIQ × pyrlm-runtime — Objetivo y mapa de documentos

> Doc north-star. Si vuelves al proyecto en 3 meses y has olvidado el
> contexto, **lee este primero**. Última actualización: 2026-05-13.

---

## En una frase

> Estamos midiendo, con rigor, **cuánto puede sacar pyrlm-runtime a un
> problema de retrieval moderno bajo restricciones realistas de
> producción** (un Elasticsearch que viene dado y no podemos tocar), y
> el resultado va a ser la mejor evidencia posible de que la librería
> tiene valor — no porque tenga primitivas mágicas, sino porque
> facilita componer las correctas.

---

## El objetivo, en cuatro capas

Cada capa contesta una pregunta más profunda que la anterior. Las
cuatro están vivas a la vez en cada experimento que corremos.

### Capa 1 — La pregunta inmediata del experimento

> **¿Podemos mejorar significativamente el NDCG@10 en queries oblicuas
> sin tocar la indexación de Elasticsearch?**

Una _oblique query_ (terminología de Tchuindjo et al. 2026,
[arXiv:2605.06235](https://arxiv.org/html/2605.06235)) es aquella en la que la relevancia depende de un
atributo _latente_ del documento — técnica de demostración compartida,
postura implícita en un tweet, modo de fallo en una conversación — y
no de palabras de la superficie. Los retrievers clásicos (BM25, dense
embeddings) fallan en esta clase porque buscan similitud superficial.

La restricción operativa es real: el ES viene dado, no controlamos su
schema ni su pipeline de indexación. Toda mejora tiene que venir del
**read path** (cómo consultamos, cómo combinamos, cómo razonamos).

### Capa 2 — Lo que cada experimento prueba

La progresión de experimentos construye un argumento empírico. Cada
uno sale a confirmar o refutar una hipótesis concreta.

| #   | Experimento                                    | NDCG@10 (Math, N=151) | Qué demostró                                                                                                                                                                                                |
| --- | ---------------------------------------------- | --------------------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | BM25 puro                                      |                 0.028 | Confirma la tesis del paper: lexical solo falla en oblique.                                                                                                                                                 |
| 2   | BM25 + ListwiseReranker                        |                 0.057 | El rerank ayuda 2× sobre BM25, pero está topado por lo que BM25 le entrega.                                                                                                                                 |
| 3   | Oracle pool + ListwiseReranker                 |                 0.714 | Si el pool contiene los gold, el rerank los promueve casi perfectamente. **El cuello de botella no es el reranker.**                                                                                        |
| 4c  | Oracle + RLM verify scored 1-5                 |                 0.615 | Subcalls paralelos con scored verifier alcanzan el 86% del listwise a 5× menos coste.                                                                                                                       |
| 5   | BM25 + RLM verify scored                       |                 0.042 | Mismo paradigma sobre pool malo: mismo techo. Reconfirma el bottleneck.                                                                                                                                     |
| 6   | BM25 + RLM agentic (sin rerank)                |                 0.041 | El loop con tools mínimas (search + read_doc) iguala a BM25+verify, pero no llega al rerank listwise.                                                                                                       |
| 7   | **Palanca 1 — multi-query + rerank**           |         0.072 (N=151) | **El pool recall sube 2× (6% → 13%) y el NDCG +26% sobre Cond 2, pero por debajo del rango predicho [0.09, 0.13]. La composición funciona, pero no llega al umbral para promoverla a primitiva en `src/`.** |
| 7t  | Ablación: TournamentReranker (N=30)            |                 0.075 | Peor que sliding (0.110 @ N=30). El torneo destruye el orden BM25 y elimina docs permanentemente — contraproducente a pool size ~108. Diseñado para 300-2500 docs.                                          |
| 8   | **Palanca 1 v2 — + query original en fan-out** |     **0.093** (N=151) | Umbral ≥ 0.09 alcanzado. +29% sobre v1. `QueryRewriter` + `union_pool` promovidos a `src/`.                                                                                                                 |

Resultados Palanca 1 N=151: NDCG@10 = 0.072, pool-level recall 6.4% →
13.0%, +26% relativo vs Cond 2 (0.057). Predicción 0.09–0.13 incumplida
parcialmente: N=5 (0.152) y N=30 (0.110) fueron selección sesgada del
shuffle (mismo seed), regresión a la media al pasar a la cohorte
completa. ~10-15 queries presentaron regresiones donde el rewriter
movió el haz lejos del gold ya capturado por BM25.

**Iteración v2 candidata** (sin correr): añadir la query original como
6ª búsqueda al fan-out. Coste cero (BM25 gratis), elimina por
construcción las regresiones identificadas. Esperado: 0.08–0.09.

### Capa 3 — Lo que esto dice de la librería

La pregunta original que abrió toda esta línea de trabajo:

> _"¿pyrlm-runtime es mejor después de añadir ListwiseReranker, o ya
> era igual de buena antes?"_

Tiene dos partes y la respuesta a cada una es distinta.

**Parte 1 — ¿la primitiva nueva aporta?** Sí, mensurable:

| Configuración                                                      |   NDCG@10 |
| ------------------------------------------------------------------ | --------: |
| Librería antes (loop + search/read_doc)                            |     0.041 |
| Librería después (+ ListwiseReranker)                              |     0.057 |
| Librería después (+ QueryRewriter + union_pool + ListwiseReranker) | **0.093** |

**Parte 2 — ¿dónde está realmente el valor?** En la Palanca 1 queda
visible: la mejora más grande (0.057 → ~0.11) **no viene de código
nuevo en `src/`**, viene de **componer** primitivas ya existentes
(Rewriter mínimo + BM25 × 5 + dedup + ListwiseReranker).

→ La USP real de pyrlm-runtime no es ninguna primitiva concreta. Es
que permite **componer estos patrones en ~500 líneas de Python plano,
sin tener que escribir framework**. La Palanca 1 es la prueba más
clara de esto: cero código en `src/`, mejora 2-3× sobre el reranker
solo.

### Capa 4 — La salida final del proyecto

Los entregables prácticos que justifican todo el trabajo:

1. **Un commit a la librería** — `feat: add ListwiseReranker
primitive`. Pequeño, defendible, autocontenido.
2. **Documentación honesta** — los .md de `docs/` registran qué
   aporta cada cosa y dónde está el techo. Vale internamente: si
   mañana alguien del equipo pregunta "¿por qué no probamos X?", la
   respuesta está documentada.
3. **Un post o artículo** con el mensaje doble:
   - _Sobre OBLIQ-Bench:_ BM25 + rewrite + rerank cierra ~70% del gap
     entre lexical y embedding denso, sin tocar índices.
   - _Sobre pyrlm-runtime:_ cualquier patrón de retrieval compuesto se
     escribe aquí en pocas decenas de líneas.
4. **Una base para el siguiente consumidor** — banking-rlm (u otro)
   va a tener el mismo problema: ES dado, queries con relevancia
   latente. Habremos dejado primitivas + patrones probados que ese
   proyecto puede importar directamente.

---

## La restricción que define el alcance

> **No tocamos la indexación de Elasticsearch.**

Eso excluye la Solución 1 del paper (anotar atributos latentes offline
en cada doc del corpus), que sería el mayor salto cualitativo posible
(NDCG 0.43+ vs nuestro techo ~0.2). Esa solución está documentada como
"siguiente frontera si en algún momento se relaja la restricción", no
como algo olvidado.

Detalle de la restricción y de la frontera que dejamos abierta:
[`OBLIQ-BENCH-ANALISIS.md`](OBLIQ-BENCH-ANALISIS.md#-restricción-operativa-de-este-proyecto).

---

## Siguiente frontera natural: embeddings en el retriever

El trabajo actual opera sobre **BM25 puro** como primera etapa. El salto
más grande disponible — sin tocar la indexación, solo el read path — es
sustituir o complementar BM25 con un **retriever denso**.

El paper lo confirma numéricamente:

| Primera etapa                     | NDCG@10 (paper) |     Estimado con nuestro pipeline |
| --------------------------------- | --------------: | --------------------------------: |
| BM25 solo                         |           0.029 |               0.028 ✓ (replicado) |
| Dense (Qwen3-Embed-0.6B)          |           0.116 | ~0.16–0.20 con multi-query+rerank |
| Dense (Gemini-2-Embedding)        |           0.144 | ~0.20–0.28 con multi-query+rerank |
| BM25 + nuestro multi-query+rerank |               — |                **0.093** (medido) |

**Por qué el multiplicador sería mayor con dense:** nuestro multi-query
sube el pool recall de 6.4% a 16.1% partiendo de BM25. Partiendo de un
retriever denso cuyo pool recall de base ya es ~35–50%, la misma
composición (6 búsquedas diversas + unión + reranker) llevaría el pool
recall a ~55–70%. El ListwiseReranker convertiría eso en un NDCG
estimado de 0.18–0.28 — equivalente o superior al paper.

**El path operativo** depende de qué exponga el Elasticsearch:

| Si el ES tiene…                        | Acción                                                                                                                                               |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Campo `knn` / vector field             | `es_hybrid_search` ya está en `retrieval.py` — cero código nuevo                                                                                     |
| ELSER (Elastic Learned Sparse Encoder) | BM25 semántico, se usa igual que BM25                                                                                                                |
| Solo BM25                              | Embedding externo (OpenAI, Cohere, Gemini) como re-scorer sobre el pool ya filtrado — bajo coste porque el pool tiene ~128 docs, no el corpus entero |

**La arquitectura no cambia.** `QueryRewriter` + `union_pool` +
`ListwiseReranker` son las mismas primitivas. Solo cambia el retriever
de primera etapa. Eso es exactamente el punto de la librería: la
composición es estable; los componentes son intercambiables.

---

## Mapa de documentos — desde aquí

Lee según lo que busques:

| Si buscas…                                              | Ve a                                                           |
| ------------------------------------------------------- | -------------------------------------------------------------- |
| Análisis del paper y por qué encaja con la librería     | [`OBLIQ-BENCH-ANALISIS.md`](OBLIQ-BENCH-ANALISIS.md)           |
| Tabla de todos los experimentos con sus números         | [`OBLIQ-EXPERIMENTS.md`](OBLIQ-EXPERIMENTS.md)                 |
| Diseño detallado de la Palanca 1 (rewriter+rerank)      | [`OBLIQ-PALANCA1-MULTIQUERY.md`](OBLIQ-PALANCA1-MULTIQUERY.md) |
| Cómo re-correr todo con cache OFF (publicación)         | [`OBLIQ-DOUBLECHECK-ROADMAP.md`](OBLIQ-DOUBLECHECK-ROADMAP.md) |
| La primitiva `ListwiseReranker` (API y uso)             | [`rerank.md`](rerank.md)                                       |
| Convenciones del repo (cómo añadir cosas a la librería) | [`../CLAUDE.md`](../../CLAUDE.md)                              |

---

## Estado actual (snapshot)

- ✅ Condiciones 1-3 ejecutadas (BM25, BM25+rerank, oracle+rerank) — N=151
- ✅ Condiciones 4c-5 ejecutadas (RLM verify scored, oracle y BM25) — N=151
- ✅ Condición 6 ejecutada (RLM agentic puro) — N=151
- ✅ Palanca 1 N=5, N=30 y N=151 ejecutados — headline 0.072, por debajo del rango predicho
- ✅ TournamentReranker implementado en `src/` y probado (N=5, N=30) — hipótesis refutada, sliding gana a pool ~108
- ✅ Palanca 1 v2 (+ query original al fan-out) — NDCG=0.093, umbral 0.09 alcanzado
- ✅ `QueryRewriter` + `union_pool` promovidos a `src/pyrlm_runtime/multiquery.py`, 15 tests
- ⚪ Doble-check con cache OFF en todos los runs anteriores — pendiente
- ⚪ Commit a la librería + post — pendiente

---

## Si solo lees esto: las tres frases que importan

1. El problema (queries oblicuas) es real, está bien definido, y los
   retrievers clásicos fallan en él de forma confirmada empíricamente
   (BM25 = 0.028 NDCG@10, reproducible).
2. Operando solo en el read path (sin tocar la indexación), la
   librería puede multiplicar el NDCG entre 2× y 4×+ dependiendo del
   patrón compuesto, con coste por query del orden de centavos.
3. **La aportación de pyrlm-runtime no es ningún rerank concreto —
   es que la composición de patrones de retrieval con LLMs cabe en
   código Python plano y reusable, no en framework.**
