"""Backend-agnostic retrieval interface for the RLM runtime.

The RLM REPL exposes retrieval functions (``es_search``, ``es_vector_search``,
``es_hybrid_search``, ``es_get``) that delegate to a configured retriever
implementing :class:`RetrieverProtocol`.  Elasticsearch is one such
implementation; any backend (Qdrant, Pinecone, OpenSearch, …) can provide
another.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


logger = logging.getLogger(__name__)


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


def _is_rrf_license_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if "reciprocal rank fusion" in text:
        return True
    return "rrf" in text and (
        "license" in text or "non-compliant" in text or "security_exception" in text
    )


def _is_id_sort_fielddata_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "fielddata access on the _id field is disallowed" in text
        or ("_id field" in text and "fielddata" in text and "disallowed" in text)
    )


def _logical_doc_search_body(
    logical_doc_id: str,
    *,
    max_pages: int,
    include_id_sort: bool,
    page_numbers: list[int] | None = None,
) -> dict[str, Any]:
    filters: list[dict[str, Any]] = [
        {"term": {"doc_id": logical_doc_id}},
    ]
    if page_numbers:
        filters.append({"terms": {"page_num": page_numbers}})
    sort: list[dict[str, Any]] = [
        {"page_num": {"order": "asc", "missing": "_last"}},
    ]
    if include_id_sort:
        sort.append({"_id": {"order": "asc"}})
    return {
        "query": {
            "bool": {
                "filter": filters
            }
        },
        "sort": sort,
        "size": max_pages,
        "_source": True,  # full source needed for page stitching
    }


def _expanded_page_numbers(
    pages: list[int] | None,
    *,
    radius: int,
    max_pages: int,
) -> list[int] | None:
    if not pages:
        return None
    expanded: set[int] = set()
    safe_radius = max(radius, 0)
    for page in pages:
        if not isinstance(page, int):
            continue
        start = max(1, page - safe_radius)
        end = page + safe_radius
        for candidate in range(start, end + 1):
            expanded.add(candidate)
    ordered = sorted(expanded)
    if max_pages > 0:
        ordered = ordered[:max_pages]
    return ordered


def _reciprocal_rank_fuse(
    result_sets: list[list[dict[str, Any]]],
    *,
    top_k: int,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    fused_scores: dict[str, float] = {}
    fused_results: dict[str, dict[str, Any]] = {}
    first_seen: dict[str, int] = {}
    next_order = 0

    for results in result_sets:
        for rank, result in enumerate(results, start=1):
            doc_id = str(result.get("doc_id", ""))
            if not doc_id:
                continue
            if doc_id not in fused_results:
                fused_results[doc_id] = {
                    **result,
                    "metadata": dict(result.get("metadata", {})),
                }
                first_seen[doc_id] = next_order
                next_order += 1
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (rank_constant + rank)

    reranked: list[dict[str, Any]] = []
    for doc_id, result in fused_results.items():
        fused = dict(result)
        fused["score"] = fused_scores[doc_id]
        reranked.append(fused)

    reranked.sort(key=lambda item: (-item["score"], first_seen[item["doc_id"]]))
    return reranked[:top_k]


def _stitch_logical_document(
    *,
    logical_doc_id: str,
    hits: list[dict[str, Any]],
    content_field: str,
    vector_field: str,
) -> dict[str, Any]:
    """Combine page-level hits into one logical document payload."""

    def _coerce_source_pages(raw: Any) -> list[int]:
        if not isinstance(raw, list):
            return []
        return [page for page in raw if isinstance(page, int) and page > 0]

    def _page_sort_key(hit: dict[str, Any]) -> tuple[int, str]:
        source = hit.get("_source", {})
        page_num = source.get("page_num")
        if isinstance(page_num, int):
            return (page_num, str(hit.get("_id", "")))
        return (10**9, str(hit.get("_id", "")))

    ordered_hits = sorted(hits, key=_page_sort_key)
    first_source = ordered_hits[0].get("_source", {})
    metadata = {
        k: v for k, v in first_source.items() if k not in {content_field, vector_field}
    }
    metadata["doc_id"] = logical_doc_id
    expected_page_count = metadata.get("page_count")
    if not isinstance(expected_page_count, int) or expected_page_count <= 0:
        expected_page_count = None

    page_doc_ids: list[str] = []
    source_pages_seen: set[int] = set()
    parts: list[str] = []
    for hit in ordered_hits:
        source = hit.get("_source", {})
        page_doc_id = str(hit.get("_id", ""))
        if page_doc_id:
            page_doc_ids.append(page_doc_id)
        page_num = source.get("page_num")
        hit_source_pages = _coerce_source_pages(source.get("source_pages"))
        if hit_source_pages:
            source_pages_seen.update(hit_source_pages)
        elif isinstance(page_num, int):
            source_pages_seen.add(page_num)
        content = source.get(content_field, "")
        if isinstance(page_num, int):
            parts.append(f"<!-- Page {page_num} -->\n{content}")
        else:
            parts.append(str(content))

    metadata["page_doc_ids"] = page_doc_ids
    chunk_count = len(page_doc_ids)
    indexed_source_page_count = len(source_pages_seen) if source_pages_seen else chunk_count
    metadata["chunk_count"] = chunk_count
    metadata["indexed_source_page_count"] = indexed_source_page_count
    metadata["indexed_page_count"] = indexed_source_page_count
    if expected_page_count is not None:
        metadata["expected_page_count"] = expected_page_count
        metadata["page_count"] = expected_page_count
        metadata["index_incomplete"] = indexed_source_page_count < expected_page_count
    else:
        metadata["page_count"] = indexed_source_page_count
        metadata["index_incomplete"] = False
    if source_pages_seen:
        metadata["source_pages"] = sorted(source_pages_seen)

    return {
        "doc_id": logical_doc_id,
        "logical_doc_id": logical_doc_id,
        "content": "\n\n".join(parts),
        "metadata": metadata,
    }


def _disable_google_ssl_verify() -> None:
    """Disable SSL verification for Google auth/API transports.

    .. warning::
        This function monkey-patches ``requests.Session.request`` **globally**,
        affecting every HTTP request made in the current process for the lifetime
        of the interpreter.  It is triggered by setting
        ``embedding_ssl_verify=False`` on a retriever.  Only use it in
        controlled, single-tenant environments where you explicitly opt in to
        skipping TLS verification (e.g. self-signed certificates in a local
        development cluster).  Never call this in multi-tenant or
        shared-process contexts.
    """
    import requests
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    original_request = requests.Session.request

    def _request_no_verify(self: requests.Session, method: Any, url: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("verify", False)
        return original_request(self, method, url, **kwargs)

    requests.Session.request = _request_no_verify  # type: ignore[method-assign]


def _embed_vertexai_query(
    text: str,
    *,
    model: str,
    ssl_verify: bool = True,
) -> list[float]:
    """Embed a short query via Vertex AI using ADC credentials."""
    try:
        import vertexai
        from vertexai.preview.language_models import TextEmbeddingModel
    except ImportError as exc:
        raise ImportError(
            "google-cloud-aiplatform is required for Vertex AI embeddings. "
            "Install it with: pip install google-cloud-aiplatform"
        ) from exc

    if not ssl_verify:
        _disable_google_ssl_verify()

    vertexai.init()
    embedding_model = TextEmbeddingModel.from_pretrained(model)
    embeddings = embedding_model.get_embeddings([text])
    if not embeddings:
        raise RuntimeError("Vertex AI returned no embedding vectors")
    return list(embeddings[0].values)


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

        Returns list of ``{doc_id, content, preview, score, metadata}``.
        ``content`` is the full document text; ``preview`` is a truncated snippet.
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

        Returns list of ``{doc_id, content, preview, score, metadata}``.
        ``content`` is the full document text; ``preview`` is a truncated snippet.
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

        Returns list of ``{doc_id, content, preview, score, metadata}``.
        ``content`` is the full document text; ``preview`` is a truncated snippet.
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
    api_key: str = field(repr=False)
    index: str
    content_field: str = "content"
    vector_field: str = "embedding"
    embedding_provider: str = "openai"
    embedding_model: str | None = None
    embedding_api_key: str | None = field(default=None, repr=False)
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_ssl_verify: bool = True
    preview_length: int = 500

    # Proxy / TLS
    es_proxy: str | None = None
    verify_certs: bool = True

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
    _supports_rrf: bool | None = field(default=None, init=False, repr=False)
    _supports_id_secondary_sort: bool | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.cache_embeddings and self.embedding_cache_size <= 0:
            raise ValueError("embedding_cache_size must be > 0 when cache_embeddings is enabled")
        if self.cache_results:
            if self.result_cache_size <= 0:
                raise ValueError("result_cache_size must be > 0 when cache_results is enabled")
            if self.result_cache_ttl <= 0:
                raise ValueError("result_cache_ttl must be > 0 when cache_results is enabled")
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

        kwargs: dict[str, Any] = {"hosts": [self.host]}
        if self.api_key:
            kwargs["api_key"] = self.api_key

        if self.es_proxy:
            from elastic_transport import RequestsHttpNode

            proxy_url = self.es_proxy

            class _ProxyNode(RequestsHttpNode):
                def __init__(self_node, config: Any) -> None:
                    super().__init__(config)
                    self_node.session.proxies.update({
                        "http": proxy_url,
                        "https": proxy_url,
                    })

            kwargs["node_class"] = _ProxyNode
            kwargs["verify_certs"] = False
        else:
            kwargs["verify_certs"] = self.verify_certs

        kwargs["ssl_show_warn"] = False
        self._client = Elasticsearch(**kwargs)
        return self._client

    def _preview(self, text: str) -> str:
        if len(text) <= self.preview_length:
            return text
        return text[: self.preview_length] + "…"

    def _hit_to_document(self, hit: dict[str, Any]) -> dict[str, Any]:
        source = hit.get("_source", {})
        content = source.get(self.content_field, "")
        metadata = {
            k: v for k, v in source.items() if k != self.content_field and k != self.vector_field
        }
        result = {
            "doc_id": hit["_id"],
            "page_doc_id": hit["_id"],
            "content": content,
            "metadata": metadata,
        }
        logical_doc_id = metadata.get("doc_id")
        if logical_doc_id is not None:
            result["logical_doc_id"] = logical_doc_id
        page_num = metadata.get("page_num")
        if page_num is not None:
            result["page_num"] = page_num
        return result

    def _hit_to_result(self, hit: dict[str, Any]) -> dict[str, Any]:
        source = hit.get("_source", {})
        content = source.get(self.content_field, "")
        metadata = {
            k: v for k, v in source.items() if k != self.content_field and k != self.vector_field
        }
        page_doc_id = hit["_id"]
        result = {
            "doc_id": page_doc_id,
            "page_doc_id": page_doc_id,
            "content": content,
            "preview": self._preview(content),
            "score": hit.get("_score", 0.0),
            "metadata": metadata,
        }
        logical_doc_id = metadata.get("doc_id")
        if logical_doc_id is not None:
            result["logical_doc_id"] = logical_doc_id
        page_num = metadata.get("page_num")
        if page_num is not None:
            result["page_num"] = page_num
        return result

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
            key = _cache_key(
                text,
                self.embedding_provider,
                self.embedding_model,
                self.embedding_base_url,
                self.embedding_ssl_verify,
            )
            cached = self._embedding_cache.get(key)
            if cached is not None:
                return cached

        if self.embedding_provider == "vertexai":
            vector = _embed_vertexai_query(
                text,
                model=self.embedding_model,
                ssl_verify=self.embedding_ssl_verify,
            )
        else:
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
            with urllib.request.urlopen(req, timeout=30.0) as resp:
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

    def _hybrid_search_local(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        keyword_results = self.search(query, top_k=top_k, filters=filters)
        vector_results = self.vector_search(query, top_k=top_k, filters=filters)
        return _reciprocal_rank_fuse([keyword_results, vector_results], top_k=top_k)

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
        source_excludes = {"excludes": [self.vector_field]}
        body: dict[str, Any] = {
            "query": {
                "bool": {
                    "must": [{"match": {self.content_field: query}}],
                }
            },
            "size": top_k,
            "_source": source_excludes,
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
        source_excludes = {"excludes": [self.vector_field]}
        body: dict[str, Any] = {
            "knn": {
                "field": self.vector_field,
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": max(top_k * 2, 50),
            },
            "_source": source_excludes,
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

        if self._supports_rrf is False:
            results = self._hybrid_search_local(query, top_k=top_k, filters=filters)
            self._store_results("hybrid_search", query, top_k, filters, results)
            return results

        es = self._get_client()
        query_vector = self._embed(query)
        filter_clauses, must_not_clauses = _build_filter_clauses(filters)
        source_excludes = {"excludes": [self.vector_field]}
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
            "_source": source_excludes,
        }
        # Apply filters to BOTH the BM25 query AND the kNN clause (pre-filtering)
        _apply_bool_filter(body, "query", filter_clauses, must_not_clauses)
        _apply_bool_filter(body, "knn", filter_clauses, must_not_clauses)
        try:
            resp = es.search(index=self.index, body=body)
        except Exception as exc:
            if not _is_rrf_license_error(exc):
                raise
            self._supports_rrf = False
            logger.warning(
                "Elasticsearch rank.rrf is unavailable for index '%s'; falling back to local RRF fusion",
                self.index,
            )
            results = self._hybrid_search_local(query, top_k=top_k, filters=filters)
            self._store_results("hybrid_search", query, top_k, filters, results)
            return results
        self._supports_rrf = True
        results = [self._hit_to_result(hit) for hit in resp["hits"]["hits"]]
        self._store_results("hybrid_search", query, top_k, filters, results)
        return results

    def get(self, doc_id: str) -> dict[str, Any]:
        es = self._get_client()
        resp = es.get(index=self.index, id=doc_id)
        return self._hit_to_document(resp)

    def get_pages(
        self,
        logical_doc_id: str,
        *,
        pages: list[int] | None = None,
        radius: int = 0,
        max_pages: int = 20,
    ) -> list[dict[str, Any]]:
        es = self._get_client()
        include_id_sort = self._supports_id_secondary_sort is not False
        page_numbers = _expanded_page_numbers(pages, radius=radius, max_pages=max_pages)
        body = _logical_doc_search_body(
            logical_doc_id,
            max_pages=max_pages,
            include_id_sort=include_id_sort,
            page_numbers=page_numbers,
        )
        try:
            resp = es.search(index=self.index, body=body)
            if include_id_sort:
                self._supports_id_secondary_sort = True
        except Exception as exc:
            if not include_id_sort or not _is_id_sort_fielddata_error(exc):
                raise
            self._supports_id_secondary_sort = False
            logger.debug(
                "Elasticsearch _id secondary sort is unavailable for index '%s'; "
                "falling back to page_num-only sort for page retrieval",
                self.index,
            )
            body = _logical_doc_search_body(
                logical_doc_id,
                max_pages=max_pages,
                include_id_sort=False,
                page_numbers=page_numbers,
            )
            resp = es.search(index=self.index, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        return [self._hit_to_document(hit) for hit in hits]

    def get_pages_text(
        self,
        logical_doc_id: str,
        *,
        pages: list[int] | None = None,
        radius: int = 0,
        max_pages: int = 20,
    ) -> str:
        docs = self.get_pages(
            logical_doc_id,
            pages=pages,
            radius=radius,
            max_pages=max_pages,
        )
        parts: list[str] = []
        for doc in docs:
            page_num = doc.get("page_num")
            content = doc.get("content", "")
            if isinstance(page_num, int):
                parts.append(f"<!-- Page {page_num} -->\n{content}")
            else:
                parts.append(str(content))
        return "\n\n".join(parts)

    def get_logical_document(self, logical_doc_id: str, *, max_pages: int = 2000) -> dict[str, Any]:
        """Fetch and stitch all indexed pages for a logical base document id."""

        es = self._get_client()
        include_id_sort = self._supports_id_secondary_sort is not False
        body = _logical_doc_search_body(
            logical_doc_id,
            max_pages=max_pages,
            include_id_sort=include_id_sort,
            page_numbers=None,
        )
        try:
            resp = es.search(index=self.index, body=body)
            if include_id_sort:
                self._supports_id_secondary_sort = True
        except Exception as exc:
            if not include_id_sort or not _is_id_sort_fielddata_error(exc):
                raise
            self._supports_id_secondary_sort = False
            logger.debug(
                "Elasticsearch _id secondary sort is unavailable for index '%s'; "
                "falling back to page_num-only sort for logical document reconstruction",
                self.index,
            )
            body = _logical_doc_search_body(
                logical_doc_id,
                max_pages=max_pages,
                include_id_sort=False,
                page_numbers=None,
            )
            resp = es.search(index=self.index, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        if not hits:
            raise KeyError(f"logical document not found: {logical_doc_id}")
        return _stitch_logical_document(
            logical_doc_id=logical_doc_id,
            hits=hits,
            content_field=self.content_field,
            vector_field=self.vector_field,
        )

    def list_logical_doc_ids(self, *, max_docs: int = 10000) -> list[str]:
        """Return all unique ``doc_id`` values in the index.

        Uses a terms aggregation so it does not depend on RRF or vector search.
        Results are sorted alphabetically.
        """
        es = self._get_client()
        body: dict[str, Any] = {
            "size": 0,
            "aggs": {
                "doc_ids": {
                    "terms": {
                        "field": "doc_id",
                        "size": max_docs,
                    }
                }
            },
        }
        resp = es.search(index=self.index, body=body)
        buckets = resp.get("aggregations", {}).get("doc_ids", {}).get("buckets", [])
        return sorted(b["key"] for b in buckets if b.get("key"))


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

    .. note::
        Proxy and TLS certificate-verification settings (``es_proxy``,
        ``verify_certs``) are not supported by this class.  Ensure proper
        TLS configuration at the network/infrastructure level, or use
        :class:`ElasticsearchRetriever` when proxy or custom TLS settings
        are required.

    Use as an async context manager to ensure proper resource cleanup::

        async with AsyncElasticsearchRetriever(...) as retriever:
            results = await retriever.hybrid_search("query")
    """

    host: str
    api_key: str = field(repr=False)
    index: str
    content_field: str = "content"
    vector_field: str = "embedding"
    embedding_provider: str = "openai"
    embedding_model: str | None = None
    embedding_api_key: str | None = field(default=None, repr=False)
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_ssl_verify: bool = True
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
    _supports_rrf: bool | None = field(default=None, init=False, repr=False)
    _supports_id_secondary_sort: bool | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.cache_embeddings and self.embedding_cache_size <= 0:
            raise ValueError("embedding_cache_size must be > 0 when cache_embeddings is enabled")
        if self.cache_results:
            if self.result_cache_size <= 0:
                raise ValueError("result_cache_size must be > 0 when cache_results is enabled")
            if self.result_cache_ttl <= 0:
                raise ValueError("result_cache_ttl must be > 0 when cache_results is enabled")
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

    def _hit_to_document(self, hit: dict[str, Any]) -> dict[str, Any]:
        source = hit.get("_source", {})
        content = source.get(self.content_field, "")
        metadata = {
            k: v for k, v in source.items() if k != self.content_field and k != self.vector_field
        }
        result = {
            "doc_id": hit["_id"],
            "page_doc_id": hit["_id"],
            "content": content,
            "metadata": metadata,
        }
        logical_doc_id = metadata.get("doc_id")
        if logical_doc_id is not None:
            result["logical_doc_id"] = logical_doc_id
        page_num = metadata.get("page_num")
        if page_num is not None:
            result["page_num"] = page_num
        return result

    def _hit_to_result(self, hit: dict[str, Any]) -> dict[str, Any]:
        source = hit.get("_source", {})
        content = source.get(self.content_field, "")
        metadata = {
            k: v for k, v in source.items() if k != self.content_field and k != self.vector_field
        }
        page_doc_id = hit["_id"]
        result = {
            "doc_id": page_doc_id,
            "page_doc_id": page_doc_id,
            "content": content,
            "preview": self._preview(content),
            "score": hit.get("_score", 0.0),
            "metadata": metadata,
        }
        logical_doc_id = metadata.get("doc_id")
        if logical_doc_id is not None:
            result["logical_doc_id"] = logical_doc_id
        page_num = metadata.get("page_num")
        if page_num is not None:
            result["page_num"] = page_num
        return result

    async def _embed(self, text: str) -> list[float]:
        """Embed *text* via an OpenAI-compatible ``/embeddings`` endpoint (async)."""
        if not self.embedding_model:
            raise ValueError(
                "embedding_model must be set to use vector_search or hybrid_search"
            )

        # Check embedding cache
        if self._embedding_cache is not None:
            key = _cache_key(
                text,
                self.embedding_provider,
                self.embedding_model,
                self.embedding_base_url,
                self.embedding_ssl_verify,
            )
            cached = self._embedding_cache.get(key)
            if cached is not None:
                return cached

        if self.embedding_provider == "vertexai":
            vector = await asyncio.to_thread(
                _embed_vertexai_query,
                text,
                model=self.embedding_model,
                ssl_verify=self.embedding_ssl_verify,
            )
        else:
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

    async def _hybrid_search_local(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        keyword_results = await self.search(query, top_k=top_k, filters=filters)
        vector_results = await self.vector_search(query, top_k=top_k, filters=filters)
        return _reciprocal_rank_fuse([keyword_results, vector_results], top_k=top_k)

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
        source_excludes = {"excludes": [self.vector_field]}
        body: dict[str, Any] = {
            "query": {
                "bool": {
                    "must": [{"match": {self.content_field: query}}],
                }
            },
            "size": top_k,
            "_source": source_excludes,
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
        source_excludes = {"excludes": [self.vector_field]}
        body: dict[str, Any] = {
            "knn": {
                "field": self.vector_field,
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": max(top_k * 2, 50),
            },
            "_source": source_excludes,
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

        if self._supports_rrf is False:
            results = await self._hybrid_search_local(query, top_k=top_k, filters=filters)
            self._store_results("hybrid_search", query, top_k, filters, results)
            return results

        es = self._get_client()
        query_vector = await self._embed(query)
        filter_clauses, must_not_clauses = _build_filter_clauses(filters)
        source_excludes = {"excludes": [self.vector_field]}
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
            "_source": source_excludes,
        }
        _apply_bool_filter(body, "query", filter_clauses, must_not_clauses)
        _apply_bool_filter(body, "knn", filter_clauses, must_not_clauses)
        try:
            resp = await es.search(index=self.index, body=body)
        except Exception as exc:
            if not _is_rrf_license_error(exc):
                raise
            self._supports_rrf = False
            logger.warning(
                "Elasticsearch rank.rrf is unavailable for index '%s'; falling back to local RRF fusion",
                self.index,
            )
            results = await self._hybrid_search_local(query, top_k=top_k, filters=filters)
            self._store_results("hybrid_search", query, top_k, filters, results)
            return results
        self._supports_rrf = True
        results = [self._hit_to_result(hit) for hit in resp["hits"]["hits"]]
        self._store_results("hybrid_search", query, top_k, filters, results)
        return results

    async def get(self, doc_id: str) -> dict[str, Any]:
        es = self._get_client()
        resp = await es.get(index=self.index, id=doc_id)
        return self._hit_to_document(resp)

    async def get_pages(
        self,
        logical_doc_id: str,
        *,
        pages: list[int] | None = None,
        radius: int = 0,
        max_pages: int = 20,
    ) -> list[dict[str, Any]]:
        es = self._get_client()
        include_id_sort = self._supports_id_secondary_sort is not False
        page_numbers = _expanded_page_numbers(pages, radius=radius, max_pages=max_pages)
        body = _logical_doc_search_body(
            logical_doc_id,
            max_pages=max_pages,
            include_id_sort=include_id_sort,
            page_numbers=page_numbers,
        )
        try:
            resp = await es.search(index=self.index, body=body)
            if include_id_sort:
                self._supports_id_secondary_sort = True
        except Exception as exc:
            if not include_id_sort or not _is_id_sort_fielddata_error(exc):
                raise
            self._supports_id_secondary_sort = False
            logger.debug(
                "Elasticsearch _id secondary sort is unavailable for index '%s'; "
                "falling back to page_num-only sort for page retrieval",
                self.index,
            )
            body = _logical_doc_search_body(
                logical_doc_id,
                max_pages=max_pages,
                include_id_sort=False,
                page_numbers=page_numbers,
            )
            resp = await es.search(index=self.index, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        return [self._hit_to_document(hit) for hit in hits]

    async def get_pages_text(
        self,
        logical_doc_id: str,
        *,
        pages: list[int] | None = None,
        radius: int = 0,
        max_pages: int = 20,
    ) -> str:
        docs = await self.get_pages(
            logical_doc_id,
            pages=pages,
            radius=radius,
            max_pages=max_pages,
        )
        parts: list[str] = []
        for doc in docs:
            page_num = doc.get("page_num")
            content = doc.get("content", "")
            if isinstance(page_num, int):
                parts.append(f"<!-- Page {page_num} -->\n{content}")
            else:
                parts.append(str(content))
        return "\n\n".join(parts)

    async def get_logical_document(
        self, logical_doc_id: str, *, max_pages: int = 2000
    ) -> dict[str, Any]:
        es = self._get_client()
        include_id_sort = self._supports_id_secondary_sort is not False
        body = _logical_doc_search_body(
            logical_doc_id,
            max_pages=max_pages,
            include_id_sort=include_id_sort,
            page_numbers=None,
        )
        try:
            resp = await es.search(index=self.index, body=body)
            if include_id_sort:
                self._supports_id_secondary_sort = True
        except Exception as exc:
            if not include_id_sort or not _is_id_sort_fielddata_error(exc):
                raise
            self._supports_id_secondary_sort = False
            logger.debug(
                "Elasticsearch _id secondary sort is unavailable for index '%s'; "
                "falling back to page_num-only sort for logical document reconstruction",
                self.index,
            )
            body = _logical_doc_search_body(
                logical_doc_id,
                max_pages=max_pages,
                include_id_sort=False,
                page_numbers=None,
            )
            resp = await es.search(index=self.index, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        if not hits:
            raise KeyError(f"logical document not found: {logical_doc_id}")
        return _stitch_logical_document(
            logical_doc_id=logical_doc_id,
            hits=hits,
            content_field=self.content_field,
            vector_field=self.vector_field,
        )
