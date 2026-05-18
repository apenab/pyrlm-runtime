# Listwise LLM reranker

`pyrlm_runtime.rerank` adds an LLM-based reranking stage that sits **after**
any retriever in the runtime. It addresses the *retrieval–verification
asymmetry* identified by OBLIQ-Bench (Tchuindjo et al. 2026,
arXiv:2605.06235): reasoning LLMs can easily *recognise* relevance when shown
a query–document pair, even when first-stage retrievers (BM25, dense
embeddings, hybrid) fail to *surface* the relevant documents.

## API

```python
from pyrlm_runtime import (
    ListwiseReranker,
    RerankerProtocol,
    ndcg_at_k,
    recall_at_k,
)
```

### `ListwiseReranker`

Sliding-window listwise reranker (RankGPT-style; Sun et al. 2023, *Is ChatGPT
Good at Search?*). Walks the candidate list bottom→top in overlapping
windows of `window_size` with stride `step`, asking the LLM to permute each
window's identifiers. The final order is the composition of window-level
permutations.

```python
from pyrlm_runtime import ElasticsearchRetriever, ListwiseReranker
from pyrlm_runtime.adapters import AzureOpenAIAdapter

retriever = ElasticsearchRetriever(host=..., api_key=..., index=...)
reranker = ListwiseReranker(
    AzureOpenAIAdapter(model="gpt-5.1"),
    window_size=20,
    step=10,
    max_passage_chars=300,
)

candidates = retriever.hybrid_search("query", top_k=50)   # imperfect pool of 50
top10 = reranker.rerank("query", candidates, top_k=10)    # LLM-permuted top 10
```

The reranker consumes the dict shape produced by `RetrieverProtocol.search`
(`doc_id`, `content` / `preview`, `metadata`, …) and returns the same shape
with `metadata["rerank_score"]` populated (`1 / (rank + 1)`).

### Caching

Pass a `FileCache` to memoise window-level LLM responses. The cache key
includes `(cache_namespace, query, window doc_ids, truncated passage
hashes, window_size, temperature)` — re-runs over identical inputs cost zero
LLM calls.

```python
from pyrlm_runtime import FileCache

cache = FileCache(".cache/rerank")
reranker = ListwiseReranker(
    adapter, cache=cache, cache_namespace="azure:gpt-5.1"
)
```

### Cost model

For one query with pool size `N`, window size `W`, step `S`:

```text
windows_per_query = 1 if N <= W else ceil((N - W) / S) + 1
LLM_calls_per_query ≈ windows_per_query
```

Any non-empty pool smaller than or equal to the window (`N <= W`) still
requires exactly one LLM call.

| `top_n` | `window` | `step` | LLM calls / query |
|--------:|---------:|-------:|------------------:|
| 100     | 20       | 10     | 9                 |
| 50      | 20       | 10     | 4                 |
| 50      | 10       | 5      | 9                 |
| 20      | 20       | 10     | 1                 |

### Metrics

`ndcg_at_k(ranked_doc_ids, qrels, k=10)` and
`recall_at_k(ranked_doc_ids, qrels, k=10)` accept the standard TREC-style
`qrels` dict `{doc_id: graded_relevance}`. Documents missing from `qrels`
count as zero relevance.

## Benchmark

`examples/oblique_rerank_bench.py` measures baseline vs rerank on the
OBLIQ-Bench `math` subset (or any other subset: `writing`, `twitter`,
`wildchat`, `congress`).

Smoke test without LLM credentials:

```bash
uv run python examples/oblique_rerank_bench.py \
  --adapter fake --retriever oracle --max-examples 3 --workers 1
```

Real run with OpenAI:

```bash
OPENAI_API_KEY=... uv run python examples/oblique_rerank_bench.py \
  --adapter openai --model gpt-4o-mini \
  --retriever bm25 --max-examples 30 \
  --top-n 50 --top-k 10 --window-size 20 --step 10 \
  --workers 4 --cache-dir .cache/oblique_rerank
```

Real run against your own Elasticsearch index (uses `ES_HOST`, `ES_API_KEY`,
`ES_INDEX` env vars and `hybrid_search`):

```bash
ES_HOST=... ES_API_KEY=... ES_INDEX=my-index \
OPENAI_API_KEY=... uv run python examples/oblique_rerank_bench.py \
  --adapter openai --model gpt-4o-mini \
  --retriever es --qrels-id-field source_id \
  --max-examples 50
```

Outputs (under `examples/exports/oblique_rerank_bench/run_<tag>/`):

- `metrics.json` — aggregate baseline / rerank NDCG@10 and Recall@10, plus
  `total_llm_calls`, `total_cache_hits`, `wall_time_s`.
- `per_query.jsonl` — per-query metrics + top-10 lists.
- `summary.txt` — human-readable diff.
