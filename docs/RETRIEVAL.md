# Retrieval Integration Architecture

This document explains the architecture and implementation of the retrieval layer in pyrlm-runtime, which allows the RLM to search external document indexes from within the REPL loop.

## Table of Contents

- [Overview](#overview)
- [The Problem: Context at Scale](#the-problem-context-at-scale)
- [Architecture Decision: Retrieval as REPL Functions](#architecture-decision-retrieval-as-repl-functions)
- [RetrieverProtocol](#retrieverprotocol)
- [REPL Functions](#repl-functions)
- [ElasticsearchRetriever](#elasticsearchretriever)
- [System Prompt Integration](#system-prompt-integration)
- [Usage Examples](#usage-examples)
- [Custom Retrievers](#custom-retrievers)
- [Design Decisions](#design-decisions)

## Overview

The retrieval layer turns external search indexes into **tools the model controls programmatically** inside the REPL loop. Instead of building a separate RAG pipeline, the RLM calls `es_search()`, `es_vector_search()`, or `es_hybrid_search()` the same way it calls `llm_query()` or `ctx.find()`.

This follows the same principle as the original MIT paper's BrowseComp+ experiments, which gave the RLM a `SEARCH(query)` action backed by BM25 retrieval.

## The Problem: Context at Scale

The default RLM design loads the full corpus into memory via `Context.from_documents()`. This works for small-to-medium corpora but breaks at scale:

- At **10K+ documents**, loading everything is wasteful when most queries need only 20-50 relevant documents
- At **1M documents**, the corpus cannot fit in memory or in any REPL variable
- A static query router (RAG vs RLM) misclassifies queries; the model itself is best positioned to decide when it has enough information

The retrieval layer solves this by letting the model **pull in documents on demand** from an external index.

## Architecture Decision: Retrieval as REPL Functions

```text
                    RLM with Extended REPL
                    +---------------------+
                    | Local functions:     |
                    |  peek, ctx.find,     |
                    |  llm_query, etc.     |
                    |                      |
                    | ES functions:        |
                    |  es_search,          |
                    |  es_vector_search,   |
                    |  es_hybrid_search,   |
                    |  es_get              |
                    +---------------------+
                            |
                    RetrieverProtocol
                            |
                    +-------+-------+
                    |       |       |
                   ES    Qdrant  Custom
```

Key decisions:

1. **Retrieval is a REPL function, not a separate pipeline.** The model calls `es_search()` the same way it calls `llm_query()`.
2. **No explicit query router.** Routing is emergent: for simple queries the model does one search; for complex queries it does multiple searches, cross-references, and uses subcalls.
3. **Both modes coexist.** Small corpus: `Context.from_documents()` + local inspection. Large corpus: retrieval functions. Both can be used together.
4. **Backend-agnostic.** The `RetrieverProtocol` is an interface; Elasticsearch is one implementation.

## RetrieverProtocol

All retrieval backends implement this protocol (defined in `src/pyrlm_runtime/retrieval.py`):

```python
class RetrieverProtocol(Protocol):
    def search(self, query: str, *, top_k: int = 10, filters: dict | None = None) -> list[dict]:
        """Lexical / keyword search (e.g. BM25)."""

    def vector_search(self, query: str, *, top_k: int = 10, filters: dict | None = None) -> list[dict]:
        """Embedding-based semantic search."""

    def hybrid_search(self, query: str, *, top_k: int = 10, filters: dict | None = None) -> list[dict]:
        """Hybrid lexical + semantic retrieval (e.g. RRF)."""

    def get(self, doc_id: str) -> dict:
        """Fetch full document content by id."""
```

Search methods return `list[dict]` where each dict has: `{doc_id, preview, score, metadata}`.

The `get()` method returns `{doc_id, content, metadata}`.

All return types are plain dicts (not dataclasses) to ensure compatibility with the Monty (Rust) REPL sandbox.

## REPL Functions

When a retriever is configured on the RLM instance, four functions are registered in the REPL:

| Function | Delegates to | Use case |
|----------|-------------|----------|
| `es_search(query, top_k=10, filters=None)` | `retriever.search()` | BM25 keyword matching |
| `es_vector_search(query, top_k=10, filters=None)` | `retriever.vector_search()` | Semantic similarity |
| `es_hybrid_search(query, top_k=10, filters=None)` | `retriever.hybrid_search()` | Combined BM25 + semantic (recommended) |
| `es_get(doc_id)` | `retriever.get()` | Fetch full document |

These functions are:
- Added to the REPL scaffold (protected from accidental overwrites)
- Only registered when `retriever` is not `None`
- Documented in the system prompt automatically

The `es_` prefix is a stable model-facing API. Regardless of whether the backend is Elasticsearch, Qdrant, Pinecone, or a custom implementation, the model always calls the same functions.

## ElasticsearchRetriever

The built-in Elasticsearch implementation (`src/pyrlm_runtime/retrieval.py`):

```python
from pyrlm_runtime.retrieval import ElasticsearchRetriever

retriever = ElasticsearchRetriever(
    host="https://my-cluster.es.cloud.com",
    api_key="xxx",
    index="pdf_corpus",
    content_field="content",          # Field storing document text
    vector_field="embedding",         # dense_vector field name
    embedding_model="text-embedding-3-small",  # For vector/hybrid search
    embedding_api_key=None,           # Falls back to OPENAI_API_KEY
    embedding_base_url="https://api.openai.com/v1",
    preview_length=500,               # Max chars in search previews
)
```

### Search implementations

- **`search()`**: BM25 via `bool.must.match` query
- **`vector_search()`**: kNN on the `dense_vector` field; embeds the query via an OpenAI-compatible `/embeddings` endpoint
- **`hybrid_search()`**: Combines BM25 + kNN with Elasticsearch's Reciprocal Rank Fusion (RRF)
- **`get()`**: Direct document fetch by `_id`

### Optional dependency

The `elasticsearch` Python package is lazily imported. If not installed, a clear error message is raised:

```text
ImportError: The 'elasticsearch' package is required for ElasticsearchRetriever.
Install it with: pip install elasticsearch
```

### Filters

All search methods accept a `filters` dict that maps field names to values:

```python
# Single value filter (term query)
es_search("contract", filters={"department": "legal"})

# Multi-value filter (terms query)
es_search("contract", filters={"department": ["legal", "finance"]})
```

## System Prompt Integration

When a retriever is configured, the system prompt is automatically extended with retrieval function documentation and a strategy guide. This is handled by `build_system_prompt()` in `prompts.py`.

The supplement includes:
- Function signatures and return types
- A strategy guide telling the model when to use each search type
- Best practices (search first, fetch later; refine queries; combine with `llm_query`)

When no retriever is configured, the system prompt is unchanged.

## Usage Examples

### Basic: Retriever-only mode (no local context)

```python
from pyrlm_runtime import RLM
from pyrlm_runtime.adapters import OpenAICompatAdapter
from pyrlm_runtime.retrieval import ElasticsearchRetriever

retriever = ElasticsearchRetriever(
    host="https://my-cluster.es.cloud.com",
    api_key="xxx",
    index="pdf_corpus",
    embedding_model="text-embedding-3-small",
)

adapter = OpenAICompatAdapter(model="gpt-5")
rlm = RLM(adapter=adapter, retriever=retriever)

# No context needed - the model searches the index
answer, trace = rlm.run("Who signed document X?")
```

### Hybrid: Local context + retriever

```python
# Small set of key documents loaded locally, full corpus in ES
context = Context.from_documents(key_documents)
rlm = RLM(adapter=adapter, retriever=retriever)

# Model has both ctx methods AND es_* functions
answer, trace = rlm.run("Cross-reference the summary with the full archive", context)
```

### What the model generates (example trajectory)

```python
# Step 1: Search for relevant documents
results = es_hybrid_search("february 1981 coup participants", top_k=20)
print(f"Found {len(results)} results")

# Step 2: Fetch the most relevant documents
docs = [es_get(r["doc_id"]) for r in results[:5]]

# Step 3: Analyze with sub-LLMs
people = {}
for doc in docs:
    analysis = llm_query(
        f"Extract all people and their roles:\n\n{doc['content']}"
    )
    print(analysis)

# Step 4: Targeted follow-up search
more = es_search("Tejero role", top_k=5)
```

## Custom Retrievers

Any object implementing the four methods of `RetrieverProtocol` works:

```python
class QdrantRetriever:
    def search(self, query, *, top_k=10, filters=None):
        # Your Qdrant BM25/sparse search
        return [{"doc_id": ..., "preview": ..., "score": ..., "metadata": {}}]

    def vector_search(self, query, *, top_k=10, filters=None):
        # Your Qdrant dense vector search
        ...

    def hybrid_search(self, query, *, top_k=10, filters=None):
        # Your Qdrant hybrid search
        ...

    def get(self, doc_id):
        return {"doc_id": ..., "content": ..., "metadata": {}}

rlm = RLM(adapter=adapter, retriever=QdrantRetriever())
```

For testing, use a simple in-memory retriever:

```python
class InMemoryRetriever:
    def __init__(self, docs: dict[str, str]):
        self.docs = docs

    def search(self, query, *, top_k=10, filters=None):
        return [
            {"doc_id": k, "preview": v[:200], "score": 1.0, "metadata": {}}
            for k, v in self.docs.items()
            if query.lower() in v.lower()
        ][:top_k]

    def vector_search(self, query, *, top_k=10, filters=None):
        return self.search(query, top_k=top_k, filters=filters)

    def hybrid_search(self, query, *, top_k=10, filters=None):
        return self.search(query, top_k=top_k, filters=filters)

    def get(self, doc_id):
        return {"doc_id": doc_id, "content": self.docs.get(doc_id, ""), "metadata": {}}
```

## Design Decisions

### Why plain dicts instead of dataclasses?

The Monty REPL (Rust sandbox) requires all values to be serializable. Plain dicts work natively; custom dataclasses would require special handling in the Monty serialization layer.

### Why `es_` prefix for backend-agnostic functions?

The `es_` prefix is a stable, recognizable name for the model. Changing it per-backend would require different system prompts and examples. The prefix is an abstraction — like how `git` commands work regardless of whether the remote is GitHub, GitLab, or Bitbucket.

### Why no new trace step kinds?

Retrieval calls happen inside model-generated code executed in the REPL. They're captured as part of `repl_exec` trace steps, just like `ctx.find()` calls. This keeps the trace format simple and avoids special-casing retrieval in trace analysis tools.

### Why is `context` optional?

When using retriever-only mode (large corpus indexed externally), there's no local document set to load. Requiring an empty `Context.from_text("")` would be boilerplate. Making `context` optional with a clear error when neither context nor retriever is provided is more ergonomic.

### Why lazy import for elasticsearch?

Most users won't need the Elasticsearch backend. Keeping it as an optional dependency avoids bloating the install and import time. The error message guides users to install it when needed.
