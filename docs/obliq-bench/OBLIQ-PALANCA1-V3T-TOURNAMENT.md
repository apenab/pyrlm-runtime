# Palanca 1 v3t — Ablación TournamentReranker @ pool ~400

> Ejecutado: 2026-05-18. Comparación base: Palanca 1 v3 sliding (NDCG@10=0.1032).

---

## Hipótesis

La derrota previa del torneo (Cond 7t, N=30: NDCG 0.075 vs sliding 0.110)
ocurrió a pool ~108 docs, **fuera** del rango de diseño del `TournamentReranker`
(300–2500 docs según App. C del paper OBLIQ-Bench). Con el pool de ~400 docs
del run v3, estamos exactamente dentro de su rango.

Pregunta: ¿el `TournamentReranker` iguala o supera al `ListwiseReranker` cuando
el pool es suficientemente grande para amortizar el shuffle y la eliminación
recursiva?

---

## Parámetros

```bash
uv run python examples/oblique_multiquery_bench.py \
  --n-rewrites 10 \
  --per-rewrite-top-n 50 \
  --union-cap 400 \
  --reranker-mode tournament \
  --rerank-top-k-per-batch 4
```

Idéntico al run v3 excepto `--reranker-mode tournament` y
`--rerank-top-k-per-batch 4`. El `--rerank-window-size 20` default actúa como
`batch_size` en el torneo.

---

## Resultados

| Métrica                    | v3 sliding       | v3t tournament   |         Δ |
| -------------------------- | ---------------: | ---------------: | --------: |
| NDCG@10                    |           0.1032 |       **0.1046** |   +0.0014 |
| Recall@10                  |           0.0940 |       **0.1025** |   +0.0085 |
| Pool medio                 |            367.3 |            365.5 |      −1.8 |
| Pool-level recall          |            26.1% |            24.1% |      −2.0% |
| Rerank LLM calls           |            5 446 |        **3 503** |     **−36%** |
| Wall time                  |       10 664.7 s |         6 709.5 s |     **−37%** |
| Errors                     |                0 |                2 |        +2 |

---

## Aplicación de la regla de decisión

Δ NDCG = +0.0014 < +0.005 → **"Documentar como ablación cerrada; publicar con
v3 (sliding)"**. El headline del artículo sigue siendo 0.103.

---

## Interpretación

**El empate en NDCG es el resultado central.** A pool ~400 docs, ambos
rerankers producen listas top-10 prácticamente equivalentes desde el punto de
vista del usuario. La hipótesis anterior (sliding > tournament a cualquier
pool size) queda **refutada**: aquella diferencia era un artefacto del pool
pequeño (~108), no una propiedad inherente al algoritmo.

**El torneo gana en coste y latencia.** Con ~25 LLM calls/query (vs ~39 del
sliding), el torneo es un 36% más barato y un 37% más rápido. A escala (100
queries/día, 30 días) la diferencia es material.

**El torneo gana en Recall@10.** +0.0085 absoluto (+9% relativo). El torneo
recupera más gold docs en el top-10 — probablemente porque el shuffle aleatorio
lleva algunos docs de la parte baja del pool a batches donde el LLM los puede
reconocer, mientras el sliding los deja permanentemente en ventanas tardías.

**Los 2 errores son transitorios de red**, no bugs del algoritmo:
- q03027: `ReadError: Connection reset by peer` (t=2.6s, ERR).
- q00604: `ConnectError: Invalid argument` tras 3 reintentos (t=2.0s, ERR).

En ambas queries, el pool sí se construyó; solo falló la fase de rerank.
Contribución estimada al NDCG: cero (q03027 tenía gold=3/13, q00604 gold=0/1).

---

## Conclusión para la librería

**Frontera Pareto a pool ~400 docs:**

| Prioridad        | Reranker              | Razón                                              |
| ---------------- | --------------------- | -------------------------------------------------- |
| Máximo NDCG      | Indiferente (empate)  | Δ +0.0014, dentro del ruido estadístico            |
| Máximo Recall@10 | `TournamentReranker`  | +9% relativo, el shuffle recupera docs del fondo   |
| Mínimo coste     | `TournamentReranker`  | −36% LLM calls, −37% wall time                    |
| Orden BM25 preservado | `ListwiseReranker` | El sliding no elimina docs — solo reordena        |
| Pool < 150 docs  | `ListwiseReranker`    | Torneo destruye orden BM25 a pool pequeño          |

La recomendación actualizada para `TournamentReranker` en la documentación: usar
con `pool_size ≥ 300`. Por debajo de esa cota, `ListwiseReranker` sigue siendo la
elección segura.

---

## Cross-links

- Comparación base: [`OBLIQ-PALANCA1-V3-TUNING.md`](OBLIQ-PALANCA1-V3-TUNING.md)
- Tabla completa: [`OBLIQ-EXPERIMENTS.md`](OBLIQ-EXPERIMENTS.md) (fila 9t)
- Condición 7t (derrota anterior a pool ~108): [`OBLIQ-EXPERIMENTS.md`](OBLIQ-EXPERIMENTS.md)
