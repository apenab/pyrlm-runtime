"""Backend-agnostic retrieval interface for the RLM runtime.

The RLM REPL exposes retrieval functions (``es_search``, ``es_vector_search``,
``es_hybrid_search``, ``es_get``) that delegate to a configured retriever
implementing :class:`RetrieverProtocol`.  Elasticsearch is one such
implementation; any backend (Qdrant, Pinecone, OpenSearch, …) can provide
another.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# In-memory caches (no external dependencies)
# ---------------------------------------------------------------------------


class _LRUCache:
    """Thread-safe LRU cache with fixed capacity. No TTL — entries live forever."""

    __slots__ = ("_capacity", "_data", "_lock")

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._data[key] = value
            else:
                if len(self._data) >= self._capacity:
                    self._data.popitem(last=False)
                self._data[key] = value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class _TTLCache:
    """Thread-safe cache with per-entry TTL and fixed capacity.

    Lazy eviction: expired entries are pruned on ``get`` and ``set``.
    """

    __slots__ = ("_capacity", "_ttl", "_data", "_lock")

    def __init__(self, capacity: int, ttl: int) -> None:
        self._capacity = capacity
        self._ttl = ttl
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.monotonic() > expires_at:
                del self._data[key]
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        now = time.monotonic()
        with self._lock:
            # Lazy prune expired entries when at capacity
            if len(self._data) >= self._capacity:
                expired = [k for k, (exp, _) in self._data.items() if now > exp]
                for k in expired:
                    del self._data[k]
            # If still at capacity, evict the oldest entry
            if len(self._data) >= self._capacity:
                oldest_key = min(self._data, key=lambda k: self._data[k][0])
                del self._data[oldest_key]
            self._data[key] = (now + self._ttl, value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            now = time.monotonic()
            return sum(1 for _, (exp, _) in self._data.items() if now <= exp)


# ---------------------------------------------------------------------------
# Filter operators
# ---------------------------------------------------------------------------

_RANGE_OPS = frozenset({"gte", "gt", "lte", "lt"})


def _build_filter_clauses(
    filters: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse a user-facing *filters* dict into ES bool clauses.

    Returns ``(filter_clauses, must_not_clauses)`` so callers can place them
    in the correct bool context.

    Supported value shapes:

    * ``str | int | float | bool`` → ``term``
    * ``list``                     → ``terms``
    * ``{"gte": …, "lt": …}``     → ``range``  (any combo of gte/gt/lte/lt)
    * ``{"exists": True}``         → ``exists``
    * ``{"prefix": "…"}``          → ``prefix``
    * ``{"not": "…"}``             → ``must_not`` term
    * ``{"not": [...]}``           → ``must_not`` terms
    """
    if not filters:
        return [], []

    clauses: list[dict[str, Any]] = []
    must_not: list[dict[str, Any]] = []

    for key, value in filters.items():
        if isinstance(value, dict):
            _parse_dict_filter(key, value, clauses, must_not)
        elif isinstance(value, list):
            clauses.append({"terms": {key: value}})
        else:
            clauses.append({"term": {key: value}})

    return clauses, must_not


def _parse_dict_filter(
    key: str,
    value: dict[str, Any],
    clauses: list[dict[str, Any]],
    must_not: list[dict[str, Any]],
) -> None:
    """Route a dict-valued filter to the appropriate ES query type."""
    keys = set(value.keys())

    # Range: any combination of gte/gt/lte/lt
    if keys & _RANGE_OPS:
        range_body = {op: value[op] for op in _RANGE_OPS if op in value}
        clauses.append({"range": {key: range_body}})
        return

    # Exists
    if "exists" in keys and value["exists"]:
        clauses.append({"exists": {"field": key}})
        return

    # Prefix
    if "prefix" in keys:
        clauses.append({"prefix": {key: value["prefix"]}})
        return

    # Negation (must_not)
    if "not" in keys:
        neg = value["not"]
        if isinstance(neg, list):
            must_not.append({"terms": {key: neg}})
        else:
            must_not.append({"term": {key: neg}})
        return

    # Fallback: treat as a plain term filter (e.g. nested dict value)
    clauses.append({"term": {key: value}})


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------


def _cache_key(*parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Backend-agnostic retrieval interface.

    Every method returns plain dicts so that results are serializable
    inside the Monty (Rust) REPL sandbox.
    """

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Lexical / keyword search (e.g. BM25).

        Returns list of ``{doc_id, preview, score, metadata}``.
        """
        ...

    def vector_search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Embedding-based semantic search.

        Returns list of ``{doc_id, preview, score, metadata}``.
        """
        ...

    def hybrid_search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid lexical + semantic retrieval (e.g. RRF).

        Returns list of ``{doc_id, preview, score, metadata}``.
        """
        ...

    def get(self, doc_id: str) -> dict[str, Any]:
        """Fetch the full document by *doc_id*.

        Returns ``{doc_id, content, metadata}``.
        """
        ...


# ---------------------------------------------------------------------------
# Async Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AsyncRetrieverProtocol(Protocol):
    """Async variant of :class:`RetrieverProtocol`.

    Intended for use in async frameworks (FastAPI, etc.) — **not** for the
    Monty REPL which is synchronous.
    """

    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    async def vector_search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    async def hybrid_search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    async def get(self, doc_id: str) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Elasticsearch implementation (sync)
# ---------------------------------------------------------------------------


def _apply_bool_filter(
    body: dict[str, Any],
    path: str,
    filter_clauses: list[dict[str, Any]],
    must_not_clauses: list[dict[str, Any]],
) -> None:
    """Inject filter + must_not clauses into a bool query at *path*.

    *path* is either ``"query"`` (for BM25) or ``"knn"`` (for pre-filtering).
    """
    if not filter_clauses and not must_not_clauses:
        return

    if path == "knn":
        bool_body: dict[str, Any] = {}
        if filter_clauses:
            bool_body["filter"] = filter_clauses
        if must_not_clauses:
            bool_body["must_not"] = must_not_clauses
        body["knn"]["filter"] = {"bool": bool_body}
    else:
        # path == "query" → inject into query.bool
        if filter_clauses:
            body["query"]["bool"]["filter"] = filter_clauses
        if must_not_clauses:
            body["query"]["bool"]["must_not"] = must_not_clauses


@dataclass
class ElasticsearchRetriever:
    """Retriever backed by an Elasticsearch cluster.

    Parameters
    ----------
    host:
        Elasticsearch URL (e.g. ``https://my-cluster.es.cloud.com``).
    api_key:
        API key for the cluster.
    index:
        Name of the target index.
    content_field:
        Name of the field that stores document text (default ``"content"``).
    vector_field:
        Name of the ``dense_vector`` field (default ``"embedding"``).
    embedding_model:
        Model identifier for the OpenAI-compatible embedding API
        (e.g. ``"text-embedding-3-small"``).  Required for ``vector_search``
        and ``hybrid_search``.
    embedding_api_key:
        API key for the embedding service.  Falls back to the
        ``OPENAI_API_KEY`` environment variable.
    embedding_base_url:
        Base URL for the embedding API.  Defaults to
        ``https://api.openai.com/v1``.
    preview_length:
        Maximum character length of the ``preview`` snippet returned by
        search methods.
    cache_embeddings:
        Cache embedding vectors in-memory (LRU).  Enabled by default —
        embeddings for the same text + model never change, so this is a
        pure latency win (~400 ms saved per repeated call).
    embedding_cache_size:
        Maximum number of embedding vectors to keep in the LRU cache.
    cache_results:
        Cache search results in-memory with a TTL.  Disabled by default —
        enable when the same queries are expected within short windows.
    result_cache_size:
        Maximum number of cached result sets.
    result_cache_ttl:
        Time-to-live in seconds for cached results.
    """

    host: str
    api_key: str
    index: str
    content_field: str = "content"
    vector_field: str = "embedding"
    embedding_model: str | None = None
    embedding_api_key: str | None = None
    embedding_base_url: str = "https://api.openai.com/v1"
    preview_length: int = 500

    # Cache configuration
    cache_embeddings: bool = True
    embedding_cache_size: int = 1024
    cache_results: bool = False
    result_cache_size: int = 256
    result_cache_ttl: int = 300

    # Private — lazily initialised.
    _client: Any = field(default=None, init=False, repr=False)
    _embedding_cache: _LRUCache | None = field(default=None, init=False, repr=False)
    _result_cache: _TTLCache | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.cache_embeddings:
            self._embedding_cache = _LRUCache(self.embedding_cache_size)
        if self.cache_results:
            self._result_cache = _TTLCache(self.result_cache_size, self.result_cache_ttl)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from elasticsearch import Elasticsearch
        except ImportError as exc:
            raise ImportError(
                "The 'elasticsearch' package is required for "
                "ElasticsearchRetriever.  Install it with: "
                "pip install elasticsearch"
            ) from exc
        self._client = Elasticsearch(self.host, api_key=self.api_key)
        return self._client

    def _preview(self, text: str) -> str:
        if len(text) <= self.preview_length:
            return text
        return text[: self.preview_length] + "…"

    def _hit_to_result(self, hit: dict[str, Any]) -> dict[str, Any]:
        source = hit.get("_source", {})
        content = source.get(self.content_field, "")
        metadata = {
            k: v for k, v in source.items() if k != self.content_field and k != self.vector_field
        }
        return {
            "doc_id": hit["_id"],
            "preview": self._preview(content),
            "score": hit.get("_score", 0.0),
            "metadata": metadata,
        }

    def _embed(self, text: str) -> list[float]:
        """Embed *text* via an OpenAI-compatible ``/embeddings`` endpoint.

        Results are cached in the LRU embedding cache when enabled.
        """
        if not self.embedding_model:
            raise ValueError(
                "embedding_model must be set to use vector_search or hybrid_search"
            )

        # Check embedding cache
        if self._embedding_cache is not None:
            key = _cache_key(text, self.embedding_model)
            cached = self._embedding_cache.get(key)
            if cached is not None:
                return cached

        import os
        import urllib.request

        api_key = self.embedding_api_key or os.environ.get("OPENAI_API_KEY", "")
        url = f"{self.embedding_base_url.rstrip('/')}/embeddings"
        payload = json.dumps({"input": text, "model": self.embedding_model}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read())
        vector = body["data"][0]["embedding"]

        # Store in embedding cache
        if self._embedding_cache is not None:
            self._embedding_cache.set(key, vector)

        return vector

    def _cached_search(
        self, method: str, query: str, top_k: int, filters: dict[str, Any] | None
    ) -> list[dict[str, Any]] | None:
        """Return cached results for a search call, or *None* on miss."""
        if self._result_cache is None:
            return None
        key = _cache_key(method, query, top_k, filters)
        return self._result_cache.get(key)

    def _store_results(
        self,
        method: str,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
        results: list[dict[str, Any]],
    ) -> None:
        if self._result_cache is not None:
            key = _cache_key(method, query, top_k, filters)
            self._result_cache.set(key, results)

    def clear_cache(self) -> None:
        """Clear both embedding and result caches."""
        if self._embedding_cache is not None:
            self._embedding_cache.clear()
        if self._result_cache is not None:
            self._result_cache.clear()

    # ------------------------------------------------------------------
    # RetrieverProtocol methods
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        cached = self._cached_search("search", query, top_k, filters)
        if cached is not None:
            return cached

        es = self._get_client()
        filter_clauses, must_not_clauses = _build_filter_clauses(filters)
        body: dict[str, Any] = {
            "query": {
                "bool": {
                    "must": [{"match": {self.content_field: query}}],
                }
            },
            "size": top_k,
            "_source": True,
        }
        _apply_bool_filter(body, "query", filter_clauses, must_not_clauses)
        resp = es.search(index=self.index, body=body)
        results = [self._hit_to_result(hit) for hit in resp["hits"]["hits"]]
        self._store_results("search", query, top_k, filters, results)
        return results

    def vector_search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        cached = self._cached_search("vector_search", query, top_k, filters)
        if cached is not None:
            return cached

        es = self._get_client()
        query_vector = self._embed(query)
        filter_clauses, must_not_clauses = _build_filter_clauses(filters)
        body: dict[str, Any] = {
            "knn": {
                "field": self.vector_field,
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": max(top_k * 2, 50),
            },
            "_source": True,
        }
        _apply_bool_filter(body, "knn", filter_clauses, must_not_clauses)
        resp = es.search(index=self.index, body=body)
        results = [self._hit_to_result(hit) for hit in resp["hits"]["hits"]]
        self._store_results("vector_search", query, top_k, filters, results)
        return results

    def hybrid_search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        cached = self._cached_search("hybrid_search", query, top_k, filters)
        if cached is not None:
            return cached

        es = self._get_client()
        query_vector = self._embed(query)
        filter_clauses, must_not_clauses = _build_filter_clauses(filters)
        body: dict[str, Any] = {
            "query": {
                "bool": {
                    "must": [{"match": {self.content_field: query}}],
                }
            },
            "knn": {
                "field": self.vector_field,
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": max(top_k * 2, 50),
            },
            "rank": {"rrf": {}},
            "size": top_k,
            "_source": True,
        }
        # Apply filters to BOTH the BM25 query AND the kNN clause (pre-filtering)
        _apply_bool_filter(body, "query", filter_clauses, must_not_clauses)
        _apply_bool_filter(body, "knn", filter_clauses, must_not_clauses)
        resp = es.search(index=self.index, body=body)
        results = [self._hit_to_result(hit) for hit in resp["hits"]["hits"]]
        self._store_results("hybrid_search", query, top_k, filters, results)
        return results

    def get(self, doc_id: str) -> dict[str, Any]:
        es = self._get_client()
        resp = es.get(index=self.index, id=doc_id)
        source = resp.get("_source", {})
        content = source.get(self.content_field, "")
        metadata = {
            k: v for k, v in source.items() if k != self.content_field and k != self.vector_field
        }
        return {
            "doc_id": resp["_id"],
            "content": content,
            "metadata": metadata,
        }


# ---------------------------------------------------------------------------
# Async Elasticsearch implementation
# ---------------------------------------------------------------------------


@dataclass
class AsyncElasticsearchRetriever:
    """Async retriever backed by an Elasticsearch cluster.

    Uses ``AsyncElasticsearch`` and ``httpx.AsyncClient`` for non-blocking
    I/O.  Designed for async frameworks (FastAPI, Starlette, etc.) — **not**
    for the synchronous Monty REPL.

    Supports the same cache configuration as :class:`ElasticsearchRetriever`.
    Use as an async context manager to ensure proper resource cleanup::

        async with AsyncElasticsearchRetriever(...) as retriever:
            results = await retriever.hybrid_search("query")
    """

    host: str
    api_key: str
    index: str
    content_field: str = "content"
    vector_field: str = "embedding"
    embedding_model: str | None = None
    embedding_api_key: str | None = None
    embedding_base_url: str = "https://api.openai.com/v1"
    preview_length: int = 500

    # Cache configuration
    cache_embeddings: bool = True
    embedding_cache_size: int = 1024
    cache_results: bool = False
    result_cache_size: int = 256
    result_cache_ttl: int = 300

    # Private — lazily initialised.
    _client: Any = field(default=None, init=False, repr=False)
    _http_client: Any = field(default=None, init=False, repr=False)
    _embedding_cache: _LRUCache | None = field(default=None, init=False, repr=False)
    _result_cache: _TTLCache | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.cache_embeddings:
            self._embedding_cache = _LRUCache(self.embedding_cache_size)
        if self.cache_results:
            self._result_cache = _TTLCache(self.result_cache_size, self.result_cache_ttl)

    async def __aenter__(self) -> AsyncElasticsearchRetriever:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Release underlying connections."""
        if self._client is not None:
            await self._client.close()
            self._client = None
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from elasticsearch import AsyncElasticsearch
        except ImportError as exc:
            raise ImportError(
                "The 'elasticsearch[async]' package is required for "
                "AsyncElasticsearchRetriever.  Install it with: "
                "pip install 'elasticsearch[async]'"
            ) from exc
        self._client = AsyncElasticsearch(self.host, api_key=self.api_key)
        return self._client

    def _get_http_client(self) -> Any:
        if self._http_client is not None:
            return self._http_client
        import httpx

        self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    def _preview(self, text: str) -> str:
        if len(text) <= self.preview_length:
            return text
        return text[: self.preview_length] + "…"

    def _hit_to_result(self, hit: dict[str, Any]) -> dict[str, Any]:
        source = hit.get("_source", {})
        content = source.get(self.content_field, "")
        metadata = {
            k: v for k, v in source.items() if k != self.content_field and k != self.vector_field
        }
        return {
            "doc_id": hit["_id"],
            "preview": self._preview(content),
            "score": hit.get("_score", 0.0),
            "metadata": metadata,
        }

    async def _embed(self, text: str) -> list[float]:
        """Embed *text* via an OpenAI-compatible ``/embeddings`` endpoint (async)."""
        if not self.embedding_model:
            raise ValueError(
                "embedding_model must be set to use vector_search or hybrid_search"
            )

        # Check embedding cache
        if self._embedding_cache is not None:
            key = _cache_key(text, self.embedding_model)
            cached = self._embedding_cache.get(key)
            if cached is not None:
                return cached

        import os

        api_key = self.embedding_api_key or os.environ.get("OPENAI_API_KEY", "")
        url = f"{self.embedding_base_url.rstrip('/')}/embeddings"
        http = self._get_http_client()
        resp = await http.post(
            url,
            json={"input": text, "model": self.embedding_model},
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        resp.raise_for_status()
        body = resp.json()
        vector = body["data"][0]["embedding"]

        # Store in embedding cache
        if self._embedding_cache is not None:
            self._embedding_cache.set(key, vector)

        return vector

    def _cached_search(
        self, method: str, query: str, top_k: int, filters: dict[str, Any] | None
    ) -> list[dict[str, Any]] | None:
        if self._result_cache is None:
            return None
        key = _cache_key(method, query, top_k, filters)
        return self._result_cache.get(key)

    def _store_results(
        self,
        method: str,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
        results: list[dict[str, Any]],
    ) -> None:
        if self._result_cache is not None:
            key = _cache_key(method, query, top_k, filters)
            self._result_cache.set(key, results)

    def clear_cache(self) -> None:
        """Clear both embedding and result caches."""
        if self._embedding_cache is not None:
            self._embedding_cache.clear()
        if self._result_cache is not None:
            self._result_cache.clear()

    # ------------------------------------------------------------------
    # AsyncRetrieverProtocol methods
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        cached = self._cached_search("search", query, top_k, filters)
        if cached is not None:
            return cached

        es = self._get_client()
        filter_clauses, must_not_clauses = _build_filter_clauses(filters)
        body: dict[str, Any] = {
            "query": {
                "bool": {
                    "must": [{"match": {self.content_field: query}}],
                }
            },
            "size": top_k,
            "_source": True,
        }
        _apply_bool_filter(body, "query", filter_clauses, must_not_clauses)
        resp = await es.search(index=self.index, body=body)
        results = [self._hit_to_result(hit) for hit in resp["hits"]["hits"]]
        self._store_results("search", query, top_k, filters, results)
        return results

    async def vector_search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        cached = self._cached_search("vector_search", query, top_k, filters)
        if cached is not None:
            return cached

        es = self._get_client()
        query_vector = await self._embed(query)
        filter_clauses, must_not_clauses = _build_filter_clauses(filters)
        body: dict[str, Any] = {
            "knn": {
                "field": self.vector_field,
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": max(top_k * 2, 50),
            },
            "_source": True,
        }
        _apply_bool_filter(body, "knn", filter_clauses, must_not_clauses)
        resp = await es.search(index=self.index, body=body)
        results = [self._hit_to_result(hit) for hit in resp["hits"]["hits"]]
        self._store_results("vector_search", query, top_k, filters, results)
        return results

    async def hybrid_search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        cached = self._cached_search("hybrid_search", query, top_k, filters)
        if cached is not None:
            return cached

        es = self._get_client()
        query_vector = await self._embed(query)
        filter_clauses, must_not_clauses = _build_filter_clauses(filters)
        body: dict[str, Any] = {
            "query": {
                "bool": {
                    "must": [{"match": {self.content_field: query}}],
                }
            },
            "knn": {
                "field": self.vector_field,
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": max(top_k * 2, 50),
            },
            "rank": {"rrf": {}},
            "size": top_k,
            "_source": True,
        }
        _apply_bool_filter(body, "query", filter_clauses, must_not_clauses)
        _apply_bool_filter(body, "knn", filter_clauses, must_not_clauses)
        resp = await es.search(index=self.index, body=body)
        results = [self._hit_to_result(hit) for hit in resp["hits"]["hits"]]
        self._store_results("hybrid_search", query, top_k, filters, results)
        return results

    async def get(self, doc_id: str) -> dict[str, Any]:
        es = self._get_client()
        resp = await es.get(index=self.index, id=doc_id)
        source = resp.get("_source", {})
        content = source.get(self.content_field, "")
        metadata = {
            k: v for k, v in source.items() if k != self.content_field and k != self.vector_field
        }
        return {
            "doc_id": resp["_id"],
            "content": content,
            "metadata": metadata,
        }
