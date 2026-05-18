# OBLIQ-Bench — roadmap de doble-check con cache OFF

> 🧭 **Empieza por** [`OBLIQ-OBJETIVO.md`](OBLIQ-OBJETIVO.md) si no
> tienes el contexto del proyecto. Este doc es el plan operativo de
> re-ejecución; el por qué del proyecto vive en el otro.
>
> Objetivo: re-correr todos los experimentos OBLIQ con **cache desactivado
> en todos los puntos del pipeline**, para tener números de publicación
> que reflejen exactamente lo que pagaría un usuario en su primer run, sin
> contaminación por respuestas cacheadas de runs exploratorios.

Última actualización: 2026-05-13

---

## Por qué este doble-check

Los runs previos mezclan dos fuentes de cache:

1. **`ListwiseReranker.cache`** (FileCache opt-in vía `--cache-dir` en
   `oblique_rerank_bench.py`). En el run BM25 de 2026-05-11 hubo 45
   cache hits provenientes de un run preliminar N=30; en el run Oracle
   hubo 0 hits (las pequeñas variaciones de temp=0 invalidaron las
   entradas anteriores). Impacto en métricas: nulo, pero los conteos de
   LLM calls / coste reportado están sub-estimados.
2. **`RLM.cache`** (FileCache, default en `~/.rlm_cache/`). Los runs del
   bench RLM rerank (Cond 4c/5) usaban `_NoopCache` por defecto, pero
   eso solo se confirma leyendo el código — un re-run limpio elimina la
   duda.

Para el artículo / post, queremos poder afirmar que cada NDCG@10 viene
de respuestas LLM frescas. Eso requiere cache OFF en ambos sitios.

---

## Reglas del doble-check

1. **No pasar `--cache-dir` nunca** al `oblique_rerank_bench.py`. Sin esa
   flag, `cache=None`, no se construye `FileCache`.
2. **No pasar `--use-cache` nunca** al `oblique_rlm_bench.py` ni al
   `oblique_agentic_bench.py`. Ambos instancian `_NoopCache()` por
   defecto, que hace miss en todas las llamadas.
3. **Verificar prerun** que `~/.rlm_cache/` no se está leyendo:
   `ls -la ~/.rlm_cache/ 2>/dev/null | head` — si existe, **moverlo
   temporalmente** (`mv ~/.rlm_cache ~/.rlm_cache.bak`) antes del run.
   Restaurar después si se quiere conservar.
4. **Cada run a un `--output-dir` nuevo** con timestamp claro
   (`examples/exports/oblique_doublecheck/<cond>_<timestamp>/`).
5. **Workers=1** en TODOS los runs del RLM (los runs previos con
   workers>1 tuvieron problemas y workers=1 elimina race conditions y
   no degrada apreciablemente el wall time porque cada query usa
   subcalls paralelos internos).
6. **Comparar Δ vs los runs anteriores.** Si algún NDCG@10 difiere por
   más de ±0.005 del run cacheado anterior, eso indica que el cache sí
   estaba sesgando resultados — hay que documentarlo en el artículo.

---

## Plan de ejecución (5 runs, ~2-3h total, ~$50-90)

### Pre-flight checklist

```bash
# 1. Apartar el cache global del RLM por si acaso
mv ~/.rlm_cache ~/.rlm_cache.bak 2>/dev/null

# 2. Comprobar que no hay restos de cache de los benches
ls -la .cache/oblique_rerank 2>/dev/null && \
  echo "WARN: cache dir exists — no se va a usar, pero revisar" && \
  ls .cache/oblique_rerank/

# 3. Verificar credenciales Azure
test -n "$AZURE_OPENAI_API_KEY" && test -n "$OPENAI_ENDPOINT" \
  && echo "Azure env OK" || echo "FAIL: faltan credenciales"

# 4. Crear directorio de outputs limpio
mkdir -p examples/exports/oblique_doublecheck
```

### Run 1 — Cond 1+2 (BM25 baseline + ListwiseReranker)

```bash
uv run python examples/oblique_rerank_bench.py \
  --adapter azure --model gpt-5.1 \
  --retriever bm25 \
  --max-examples 151 --workers 1 \
  --top-n 50 --top-k 10 --window-size 20 --step 10 \
  --output-dir examples/exports/oblique_doublecheck/cond12_bm25_$(date +%Y%m%d_%H%M)
# NOTA: SIN --cache-dir → cache=None, todas las llamadas frescas
```

- Espera: ~12-15 min wall time con workers=1 (vs 7.7 min con workers=4)
- Coste estimado: ~$3-5 (604 LLM calls a gpt-5.1)
- Verificación: en el summary, `total_cache_hits` debe ser 0

### Run 2 — Cond 3 (Oracle + ListwiseReranker)

```bash
uv run python examples/oblique_rerank_bench.py \
  --adapter azure --model gpt-5.1 \
  --retriever oracle \
  --max-examples 151 --workers 1 \
  --top-n 50 --top-k 10 --window-size 20 --step 10 \
  --output-dir examples/exports/oblique_doublecheck/cond3_oracle_$(date +%Y%m%d_%H%M)
```

- Espera: ~12-15 min wall time
- Coste: ~$3-5

### Run 3 — Cond 4c (Oracle + RLM verify scored)

```bash
uv run python examples/oblique_rlm_bench.py \
  --adapter azure --root-model gpt-5.1 --subcall-model gpt-5.4-mini \
  --retriever oracle \
  --max-examples 151 --workers 1 \
  --per-query-timeout 300 \
  --output-dir examples/exports/oblique_doublecheck/cond4c_oracle_rlm_$(date +%Y%m%d_%H%M)
# NOTA: SIN --use-cache → _NoopCache, todas las llamadas frescas
```

- Espera: ~45-60 min con workers=1 (cada query usa subcalls paralelos internos)
- Coste: ~$15-25 (151 root + ~151×50 subcalls)

### Run 4 — Cond 5 (BM25 + RLM verify scored)

```bash
uv run python examples/oblique_rlm_bench.py \
  --adapter azure --root-model gpt-5.1 --subcall-model gpt-5.4-mini \
  --retriever bm25 \
  --max-examples 151 --workers 1 \
  --per-query-timeout 300 \
  --output-dir examples/exports/oblique_doublecheck/cond5_bm25_rlm_$(date +%Y%m%d_%H%M)
```

- Espera: ~45-60 min
- Coste: ~$15-25

### Run 5 — Cond 6 (BM25 + RLM agentic, sin rerank)

```bash
uv run python examples/oblique_agentic_bench.py \
  --adapter azure --root-model gpt-5.1 \
  --max-examples 151 --workers 1 --per-query-timeout 300 \
  --output-dir examples/exports/oblique_doublecheck/cond6_bm25_agentic_$(date +%Y%m%d_%H%M)
```

- Espera: ~25 min (ya validado: 23.8 min en el primer N=151)
- Coste: ~$8-15

### Post-flight

```bash
# Restaurar el cache global del RLM si se apartó
mv ~/.rlm_cache.bak ~/.rlm_cache 2>/dev/null
```

---

## Validación de cada run

Después de cada run:

1. **Abrir `summary.txt` y verificar:**
   - Que `cache: DISABLED` aparece en el header (agentic / RLM rerank), o
     que `cache_hits: 0` aparece (listwise rerank).
   - `errors: 0`.
   - Wall time razonable (ver estimaciones arriba).

2. **Comparar NDCG@10 con el run cacheado anterior:**

   | Condición | Run cacheado anterior | Run doble-check | ¿Diff < 0.005? |
   |---|---:|---:|---|
   | 1. BM25 baseline | 0.0284 | ? | esperable Δ=0 (no usa LLM) |
   | 2. BM25 + listwise | 0.0571 | ? | esperable ±0.003 (temp=0) |
   | 3. Oracle + listwise | 0.7136 | ? | esperable ±0.005 |
   | 4c. Oracle + RLM scored | 0.615 | ? | esperable ±0.010 |
   | 5. BM25 + RLM scored | 0.042 | ? | esperable ±0.003 |
   | 6. BM25 + RLM agentic | 0.0411 | ? | esperable ±0.005 |

3. **Si alguna diferencia > 0.01**, abrir `per_query.jsonl` y comparar
   query a query con el run anterior. La causa probable es no-determinismo
   del modelo, no cache — pero hay que documentarlo.

---

## Tabla final consolidada (a rellenar tras el doble-check)

| # | Condición | NDCG@10 v1 (cacheado) | NDCG@10 v2 (fresh) | Δ |
|---|---|---:|---:|---:|
| 1 | BM25 baseline | 0.0284 | — | — |
| 2 | BM25 + ListwiseReranker | 0.0571 | — | — |
| 3 | Oracle + ListwiseReranker | 0.7136 | — | — |
| 4c | Oracle + RLM scored | 0.615 | — | — |
| 5 | BM25 + RLM scored | 0.042 | — | — |
| 6 | BM25 + RLM agentic | 0.0411 | — | — |

Una vez rellena, esta es la tabla del artículo / post.

---

## Riesgos y mitigaciones

- **Timeouts en Azure** (visto en runs previos con workers>1): mitigado
  con workers=1.
- **No-determinismo del modelo a temp=0**: Azure no garantiza salida bit
  a bit idéntica entre runs ni siquiera a temp=0. Esperar ±0.005 NDCG es
  realista; documentar como variabilidad inherente.
- **Coste**: ~$50-90 total. Se puede partir el doble-check en dos
  sesiones si hace falta.
- **Tiempo**: ~2-3h en serie. Se pueden lanzar Run 1+2 en una sesión y
  Run 3-5 en otra.
