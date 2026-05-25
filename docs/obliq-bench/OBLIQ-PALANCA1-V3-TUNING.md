# Palanca 1 v3 — Tuning experiment (n-rewrites=10, per-rewrite-top-n=50)

> Ejecutado: 2026-05-18. Referencia: Palanca 1 v2 (NDCG@10=0.0927, N=151).

---

## Hipótesis

Dos knobs de la CLI de `oblique_multiquery_bench.py` amplían el pool sin cambiar
código en `src/`:

1. **`--n-rewrites`** (5 → 10): más reformulaciones → más cobertura léxica del
   corpus.
2. **`--per-rewrite-top-n`** (25 → 50): más hits por búsqueda → pool más amplio
   antes del dedup.

Esperado: pool medio ~250–400 docs (vs ~128 en v2), pool-level recall ~25–35%,
NDCG@10 estimado 0.10–0.13.

---

## Parámetros

```bash
uv run python examples/oblique_multiquery_bench.py \
  --n-rewrites 10 \
  --per-rewrite-top-n 50 \
  --union-cap 400
```

Resto de parámetros al default (equivalente a la configuración v2):
`--subset math`, `--adapter azure`, `--rewriter-model gpt-5.4-mini`,
`--rerank-model gpt-5.1`, `--reranker-mode sliding`, `--rerank-window-size 20`,
`--rerank-step 10`, `--rerank-max-passage-chars 300`, `--workers 1`,
`--per-query-timeout 300`, `--seed 42`, sin `--cache-dir` (cache OFF).

---

## Resultados

| Métrica                    |        v2 (baseline) |          v3 (este run) |       Δ |
| -------------------------- | -------------------: | ---------------------: | ------: |
| NDCG@10                    |               0.0927 |             **0.1032** | +0.0105 |
| Recall@10                  |               0.0809 |             **0.0940** | +0.0131 |
| Pool medio                 |              127.5   |                367.3   |   +240  |
| Pool-level recall          |               16.1%  |               **26.1%**|  +10.0% |
| BM25 baseline pool recall  |                6.4%  |                  6.4%  |       — |
| Gold in pool (media)       |           ~0.82/13.5 |             1.81/13.5  |  +0.99  |
| Rerank LLM calls           |               1 837  |               5 446    |   +3.0× |
| Wall time                  |            4 412.1 s |             10 664.7 s |   +2.4× |
| Overlap rate               |              ~14%    |               32.6%    |         |
| Errors                     |                0     |                   0    |         |

---

## Decisión (regla pre-comprometida)

| Δ NDCG@10 vs v2 | Regla                                               |
| --------------- | --------------------------------------------------- |
| ≥ +0.010        | Actualizar tablas del artículo, publicar con v3     |
| +0.005–+0.010   | Footnote en v2; publicar con v2                     |
| < +0.005        | Documentar como ablación; publicar con v2 sin cambio|

**Δ = +0.0105 → regla ≥ +0.010 activada.** Las tablas del artículo se
actualizan a 0.1032.

---

## Interpretación

**¿Qué movió el resultado?**

El pool-level recall subió de 16.1% a 26.1% (+10 puntos). Eso significa que el
reranker recibió ~1.8 gold docs de media en lugar de ~0.8 — un +120% en materia
relevante disponible. El NDCG sube +11.3% relativo.

El overlap rate subió de ~14% a 32.6% porque con 10 rewrites sobre un corpus de
3508 docs, las búsquedas se empiezan a solapar en el tail léxico. El pool cap de
400 se alcanzó en la mayoría de queries (media 367.3), lo que sugiere que subir
el cap más allá de 400 podría añadir algo de recall pero a coste lineal en
ventanas de rerank — punto de rendimientos decrecientes.

**¿El coste extra merece el salto?**

- Rerank calls: 5 446 vs 1 837 → ~3× más llamadas LLM.
- Wall time: 10 665 s vs 4 412 s → ~2.4× más tiempo.
- NDCG: 0.1032 vs 0.0927 → +11.3% relativo.

Para producción: la configuración v2 (n=5, top-n=25) ofrece mejor ratio
coste/NDCG para queries individuales. La v3 tiene sentido para benchmarks o
cuando el recall es prioritario.

**¿Dónde están los límites?**

El cuello de botella sigue siendo la primera etapa (BM25). El pool-level recall
de 26.1% significa que el 74% de los gold docs nunca llegan al reranker — no
importa cuántas ventanas se usen. El salto cualitativo pendiente es un retriever
denso como primera etapa (estimado: pool recall ~50–70%, NDCG estimado 0.18–0.28
con el mismo pipeline de rerank).

---

## Cross-links

- Tabla de todos los experimentos: [`OBLIQ-EXPERIMENTS.md`](OBLIQ-EXPERIMENTS.md)
  (fila Palanca 1 v3 añadida)
- Objetivo y Capa 2: [`OBLIQ-OBJETIVO.md`](OBLIQ-OBJETIVO.md) (tabla actualizada)
- Diseño de la Palanca 1: [`OBLIQ-PALANCA1-MULTIQUERY.md`](OBLIQ-PALANCA1-MULTIQUERY.md)
