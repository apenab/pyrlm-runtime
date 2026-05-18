"""Multi-query retrieval primitives.

Two building blocks for the fan-out / dedup / rerank pattern:

- :class:`QueryRewriter` — generates N diverse reformulations of a query via
  a single LLM call.  The system prompt is caller-supplied so the primitive
  stays domain-agnostic.
- :func:`union_pool` — merges multiple per-rewrite retrieval result lists into
  a single deduplicated list, preserving first-occurrence order.

Typical usage::

    from pyrlm_runtime import QueryRewriter, union_pool, ListwiseReranker

    rewriter = QueryRewriter(adapter, n=5, system_prompt=MY_REWRITE_PROMPT)
    reranker = ListwiseReranker(adapter)

    rewrites = rewriter.rewrite(query) + [query]   # include original
    pools = [bm25.search(r, top_n=25) for r in rewrites]
    union = union_pool(pools)
    top10 = reranker.rerank(query, union, top_k=10)
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from typing import Any

from .adapters.base import ModelAdapter


# ---------------------------------------------------------------------------
# union_pool
# ---------------------------------------------------------------------------

_DOC_ID_KEY = "doc_id"


def union_pool(
    pools: list[list[dict[str, Any]]],
    *,
    id_key: str = _DOC_ID_KEY,
) -> list[dict[str, Any]]:
    """Merge retrieval result lists, deduplicating by *id_key*.

    The first occurrence of each document id wins — so the document keeps the
    score and position from whichever pool ranked it highest.  Output order
    follows the order of first appearance across all input pools.

    Args:
        pools: List of retrieval result lists.  Each item in a list must be a
            dict with at least the key specified by *id_key*.
        id_key: Dict key used as the document identifier.  Defaults to
            ``"doc_id"``.

    Returns:
        Deduplicated list in first-appearance order.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for pool in pools:
        for doc in pool:
            doc_id = str(doc[id_key])
            if doc_id in seen:
                continue
            seen.add(doc_id)
            out.append(doc)
    return out


# ---------------------------------------------------------------------------
# QueryRewriter
# ---------------------------------------------------------------------------

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_rewrites(text: str, n_expected: int) -> list[str]:
    match = _JSON_OBJECT_RE.search(text or "")
    if match:
        try:
            data = json.loads(match.group(0))
            rewrites = data.get("rewrites")
            if isinstance(rewrites, list):
                out = [str(r).strip() for r in rewrites if isinstance(r, str) and r.strip()]
                if out:
                    return out[:n_expected]
        except Exception:
            pass
    # Fallback: line-by-line, strip bullets / numbering
    lines = []
    for line in (text or "").splitlines():
        s = line.strip().lstrip("-*0123456789.) ").strip().strip('"').strip("'")
        if 5 <= len(s) <= 300:
            lines.append(s)
    return lines[:n_expected]


class QueryRewriter:
    """Generates N diverse reformulations of a query via a single LLM call.

    The system prompt is fully caller-supplied so this class stays
    domain-agnostic.  The LLM is expected to return a JSON object::

        {"rewrites": ["...", "...", ...]}

    If the response cannot be parsed as JSON, the class falls back to
    splitting the response on newlines.

    Args:
        adapter: Model adapter used for the rewrite call.
        n: Number of reformulations to request.
        system_prompt: System prompt sent to the LLM.  Must instruct the model
            to return ``{"rewrites": [...]}`` with exactly *n* entries.
        max_tokens: Maximum tokens for the LLM response.
        cache: Optional cache object with ``get``/``set`` interface (same
            protocol as :class:`~pyrlm_runtime.FileCache`).  When provided,
            results are cached by ``(system_prompt, query)`` hash so repeated
            calls with the same input skip the LLM.
        cache_namespace: String prepended to cache keys to avoid collisions
            across different rewriter configurations.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        n: int = 5,
        system_prompt: str,
        max_tokens: int = 400,
        cache: Any | None = None,
        cache_namespace: str = "",
    ) -> None:
        if n <= 0:
            raise ValueError("n must be > 0")
        if not system_prompt:
            raise ValueError("system_prompt must be a non-empty string")
        self._adapter = adapter
        self._n = n
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._cache = cache
        self._cache_namespace = cache_namespace
        self._calls = 0
        self._cache_hits = 0
        self._lock = threading.Lock()

    @property
    def n(self) -> int:
        return self._n

    @property
    def calls(self) -> int:
        """Total LLM calls made (excludes cache hits)."""
        with self._lock:
            return self._calls

    @property
    def cache_hits(self) -> int:
        with self._lock:
            return self._cache_hits

    def rewrite(self, query: str) -> list[str]:
        """Return up to *n* diverse reformulations of *query*.

        May return fewer than *n* entries if the LLM output cannot be parsed
        into enough valid strings.  Never raises — on complete failure returns
        an empty list.
        """
        cache_key = self._compute_cache_key(query)
        if self._cache is not None:
            record = self._cache.get(cache_key)
            if record is not None:
                with self._lock:
                    self._cache_hits += 1
                return _parse_rewrites(record.text, self._n)

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": query},
        ]
        response = self._adapter.complete(messages, max_tokens=self._max_tokens)
        with self._lock:
            self._calls += 1
        text = getattr(response, "text", None)
        if text is None:
            text = str(response)

        if self._cache is not None:
            try:
                from .cache import CacheRecord

                self._cache.set(cache_key, CacheRecord(text=text, usage=response.usage))
            except Exception:
                pass

        return _parse_rewrites(text, self._n)

    def _compute_cache_key(self, query: str) -> str:
        payload = json.dumps(
            {"ns": self._cache_namespace, "prompt": self._system_prompt, "query": query},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()
