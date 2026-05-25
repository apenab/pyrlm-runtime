"""Listwise LLM reranker for pyrlm-runtime.

Implements the sliding-window RankGPT pattern (Sun et al. 2023,
"Is ChatGPT Good at Search?"). Walks the candidate list bottom→top in
overlapping windows, asking a :class:`~pyrlm_runtime.adapters.base.ModelAdapter`
to permute each window's identifiers. The final order is the composition of
window-level permutations.

The reranker consumes the dict shape produced by
:meth:`~pyrlm_runtime.retrieval.RetrieverProtocol.search` (``doc_id``,
``content`` / ``preview``, ``metadata``, …) and returns the same shape with
``metadata["rerank_score"]`` populated.

Example
-------
>>> from pyrlm_runtime import ElasticsearchRetriever, ListwiseReranker
>>> retriever = ElasticsearchRetriever(...)
>>> reranker = ListwiseReranker(adapter, window_size=20, step=10)
>>> candidates = retriever.hybrid_search("query", top_k=50)
>>> top10 = reranker.rerank("query", candidates, top_k=10)
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from typing import Any, Protocol, runtime_checkable

from .adapters.base import ModelAdapter
from .cache import CacheRecord, FileCache

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are RankGPT, an intelligent assistant that can rank passages based "
    "on their relevance to the query."
)
_ID_PATTERN = re.compile(r"\[(\d+)\]")


@runtime_checkable
class RerankerProtocol(Protocol):
    """Backend-agnostic reranker interface."""

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        top_k: int = 10,
    ) -> list[dict[str, Any]]: ...


class ListwiseReranker:
    """Sliding-window listwise LLM reranker (RankGPT-style).

    The reranker walks ``candidates`` bottom→top in overlapping windows of
    ``window_size`` with stride ``step``. Each window is sent to the LLM as
    a single ``complete()`` call asking for a permutation of identifiers;
    the parsed permutation is applied in-place to the candidate list. After
    all windows have been processed, the head of the list reflects the
    composed ranking and ``rerank`` returns its first ``top_k`` entries.

    Parameters
    ----------
    adapter:
        Any object implementing :class:`ModelAdapter`. The reranker only
        calls ``adapter.complete(...)``.
    window_size:
        Number of candidates per LLM call. Default 20.
    step:
        Stride between consecutive windows (must satisfy ``1 <= step <=
        window_size``). Default 10 (50% overlap).
    max_passage_chars:
        Each passage's text is truncated to this many characters before
        being inserted into the prompt. Default 300.
    max_tokens:
        Token budget for the LLM permutation response. Default 200.
    temperature:
        Sampling temperature. Default 0.0 (deterministic).
    system_prompt:
        Override for the RankGPT system prompt.
    cache:
        Optional :class:`FileCache`. When provided, each window's LLM
        response is cached keyed on (namespace, query, window doc_ids,
        truncated passage hashes, window_size, temperature). Re-runs over
        identical inputs cost zero LLM calls.
    cache_namespace:
        Extra string included in the cache key. Use to differentiate
        between models when sharing a cache directory across runs.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        window_size: int = 20,
        step: int = 10,
        max_passage_chars: int = 300,
        max_tokens: int = 200,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        cache: FileCache | None = None,
        cache_namespace: str = "",
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if step <= 0:
            raise ValueError("step must be > 0")
        if step > window_size:
            raise ValueError("step must be <= window_size")
        if max_passage_chars <= 0:
            raise ValueError("max_passage_chars must be > 0")
        self._adapter = adapter
        self._window_size = window_size
        self._step = step
        self._max_passage_chars = max_passage_chars
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._cache = cache
        self._cache_namespace = cache_namespace
        # Telemetry — read by the bench script. Increments are guarded by
        # a lock so the counter stays accurate when many threads share a
        # single reranker instance (e.g. ThreadPoolExecutor in the bench).
        self.llm_calls = 0
        self.cache_hits = 0
        self._counter_lock = threading.Lock()

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Return the ``top_k`` candidates in LLM-permuted order."""
        if not candidates:
            return []
        if top_k <= 0:
            return []

        ordered: list[dict[str, Any]] = list(candidates)
        n = len(ordered)
        for start in self._window_starts(n):
            end = min(start + self._window_size, n)
            ordered[start:end] = self._rerank_window(query, ordered[start:end])

        result: list[dict[str, Any]] = []
        for rank, doc in enumerate(ordered[:top_k]):
            annotated = dict(doc)
            metadata = dict(annotated.get("metadata") or {})
            metadata["rerank_score"] = 1.0 / (rank + 1)
            annotated["metadata"] = metadata
            result.append(annotated)
        return result

    def _window_starts(self, n: int) -> list[int]:
        """Bottom-up window start indices: [N-W, N-W-S, ..., 0]."""
        if n <= self._window_size:
            return [0]
        starts: list[int] = []
        s = n - self._window_size
        while True:
            starts.append(s)
            if s == 0:
                break
            s = max(0, s - self._step)
        return starts

    def _rerank_window(
        self,
        query: str,
        window: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if len(window) <= 1:
            return list(window)

        cache_key: str | None = None
        response_text: str | None = None
        if self._cache is not None:
            cache_key = self._compute_cache_key(query, window)
            record = self._cache.get(cache_key)
            if record is not None:
                with self._counter_lock:
                    self.cache_hits += 1
                response_text = record.text

        if response_text is None:
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": self._build_user_prompt(query, window)},
            ]
            with self._counter_lock:
                self.llm_calls += 1
            response = self._adapter.complete(
                messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            response_text = response.text
            if self._cache is not None and cache_key is not None:
                self._cache.set(cache_key, CacheRecord(text=response_text, usage=response.usage))

        permutation = self._parse_permutation(response_text, len(window))
        if not permutation:
            logger.debug("Reranker got empty/malformed permutation; preserving window order")
            return list(window)
        return [window[i] for i in permutation]

    def _build_user_prompt(self, query: str, window: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        lines.append(
            f"I will provide you with {len(window)} passages, each indicated by a "
            "numerical identifier []."
        )
        lines.append(f"Rank the passages based on their relevance to the search query: {query}")
        lines.append("")
        for i, doc in enumerate(window, start=1):
            text = self._passage_text(doc).replace("\n", " ").strip()
            if len(text) > self._max_passage_chars:
                text = text[: self._max_passage_chars] + "…"
            lines.append(f"[{i}] {text}")
        lines.append("")
        lines.append(f"Search Query: {query}")
        lines.append(
            f"Rank the {len(window)} passages above based on their relevance to "
            "the search query. The passages should be listed in descending order "
            "using identifiers. Only respond with the ranking results, do not say "
            "any word or explain. Format: [2] > [4] > [1] > ..."
        )
        return "\n".join(lines)

    def _passage_text(self, doc: dict[str, Any]) -> str:
        content = doc.get("content")
        if content:
            return str(content)
        preview = doc.get("preview")
        if preview:
            return str(preview)
        return ""

    def _parse_permutation(self, text: str, n: int) -> list[int]:
        """Extract a permutation of 0..n-1 from the LLM response.

        Reads ``[d]`` tokens in order of appearance, converts to 0-based,
        dedupes, and appends any missing identifiers at the end (preserving
        their original order). Returns ``[]`` if the response contains no
        valid identifiers.
        """
        seen: set[int] = set()
        order: list[int] = []
        for match in _ID_PATTERN.finditer(text):
            idx = int(match.group(1)) - 1
            if idx < 0 or idx >= n or idx in seen:
                continue
            seen.add(idx)
            order.append(idx)
        if not order:
            return []
        for i in range(n):
            if i not in seen:
                order.append(i)
        return order

    def _compute_cache_key(self, query: str, window: list[dict[str, Any]]) -> str:
        parts: list[str] = [
            f"ns={self._cache_namespace}",
            f"w={self._window_size}",
            f"t={self._temperature}",
            f"q={query}",
        ]
        for doc in window:
            doc_id = str(doc.get("doc_id", ""))
            text = self._passage_text(doc)[: self._max_passage_chars]
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            parts.append(f"{doc_id}#{text_hash}")
        return "|".join(parts)


class TournamentReranker:
    """Tournament-style listwise LLM reranker.

    Implements the recursive-elimination ranker described by Tchuindjo
    et al. 2026 (OBLIQ-Bench, App. C). Unlike :class:`ListwiseReranker`,
    which uses overlapping sliding windows, this reranker partitions the
    pool into shuffled batches of size ``batch_size``; each batch is
    ranked by a single LLM call; the top ``top_k_per_batch`` survive to
    the next round and the rest are recorded as a tail at that depth.
    Recursion stops when the survivor pool fits in a single batch, which
    is ranked directly. The final ranking concatenates the head with the
    tails in reverse order of elimination depth (most recently eliminated
    docs come first).

    Why this matters
    ----------------
    Sliding-window reranking compares each document mainly against its
    BM25 neighbours, which is fine when relevant docs cluster near the
    top of the candidate list. On oblique queries (Tchuindjo et al.
    2026) relevant docs are scattered through the pool because the
    first-stage retriever cannot surface them by surface vocabulary.
    Tournament reranking shuffles the pool so each doc competes against
    a random sample of the entire candidate set, which preserves more of
    the relative-ranking signal when relevant docs are dispersed.

    Parameters
    ----------
    adapter:
        Any :class:`ModelAdapter`. Only ``adapter.complete(...)`` is used.
    batch_size:
        Number of candidates per LLM call. Default 20 (matches paper).
    top_k_per_batch:
        Survivors per batch. Default 4 (matches paper).
    max_passage_chars, max_tokens, temperature, system_prompt:
        Same semantics as :class:`ListwiseReranker`.
    cache, cache_namespace:
        Optional :class:`FileCache` keyed on (namespace, query, batch
        doc_ids, passage hashes, batch_size, temperature).
    shuffle_seed:
        If provided, the per-query shuffle is deterministic for that
        seed (combined with the query string). Default ``42``. Pass
        ``None`` to use OS randomness.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        batch_size: int = 20,
        top_k_per_batch: int = 4,
        max_passage_chars: int = 300,
        max_tokens: int = 200,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        cache: FileCache | None = None,
        cache_namespace: str = "",
        shuffle_seed: int | None = 42,
    ) -> None:
        if batch_size <= 1:
            raise ValueError("batch_size must be > 1")
        if top_k_per_batch <= 0:
            raise ValueError("top_k_per_batch must be > 0")
        if top_k_per_batch >= batch_size:
            raise ValueError("top_k_per_batch must be < batch_size")
        if max_passage_chars <= 0:
            raise ValueError("max_passage_chars must be > 0")
        self._adapter = adapter
        self._batch_size = batch_size
        self._top_k_per_batch = top_k_per_batch
        self._max_passage_chars = max_passage_chars
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._cache = cache
        self._cache_namespace = cache_namespace
        self._shuffle_seed = shuffle_seed
        # Shared telemetry, same shape as ListwiseReranker.
        self.llm_calls = 0
        self.cache_hits = 0
        self.rounds_run = 0
        self._counter_lock = threading.Lock()

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Return the ``top_k`` candidates in tournament-ranked order."""
        if not candidates:
            return []
        if top_k <= 0:
            return []

        import random

        if self._shuffle_seed is None:
            rng = random.Random()
        else:
            # Combine seed with query so different queries get different
            # shuffles but the same query is reproducible. Python 3.14
            # only accepts str/bytes/int/float as seeds, so encode as str.
            rng = random.Random(f"{self._shuffle_seed}::{query}")

        pool = list(candidates)
        tails: list[list[dict[str, Any]]] = []

        # Recursive rounds until pool fits in one batch.
        while len(pool) > self._batch_size:
            rng.shuffle(pool)
            new_pool: list[dict[str, Any]] = []
            new_tail: list[dict[str, Any]] = []
            for start in range(0, len(pool), self._batch_size):
                batch = pool[start : start + self._batch_size]
                if len(batch) <= self._top_k_per_batch:
                    # Too small to eliminate from — promote all.
                    new_pool.extend(batch)
                    continue
                ranked = self._rank_batch(query, batch)
                new_pool.extend(ranked[: self._top_k_per_batch])
                new_tail.extend(ranked[self._top_k_per_batch :])
            tails.append(new_tail)
            pool = new_pool
            with self._counter_lock:
                self.rounds_run += 1
            # Guard against pathological non-shrinking (shouldn't happen
            # when top_k_per_batch < batch_size, but be defensive).
            if not new_pool:
                break

        # Final-round ranking on the survivor pool.
        if len(pool) > 1:
            pool = self._rank_batch(query, pool)
            with self._counter_lock:
                self.rounds_run += 1

        # Compose final order: head + tails in reverse elimination order.
        final_order: list[dict[str, Any]] = list(pool)
        for tail in reversed(tails):
            final_order.extend(tail)

        result: list[dict[str, Any]] = []
        for rank, doc in enumerate(final_order[:top_k]):
            annotated = dict(doc)
            metadata = dict(annotated.get("metadata") or {})
            metadata["rerank_score"] = 1.0 / (rank + 1)
            annotated["metadata"] = metadata
            result.append(annotated)
        return result

    def _rank_batch(
        self,
        query: str,
        batch: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if len(batch) <= 1:
            return list(batch)

        cache_key: str | None = None
        response_text: str | None = None
        if self._cache is not None:
            cache_key = self._compute_cache_key(query, batch)
            record = self._cache.get(cache_key)
            if record is not None:
                with self._counter_lock:
                    self.cache_hits += 1
                response_text = record.text

        if response_text is None:
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": self._build_user_prompt(query, batch)},
            ]
            with self._counter_lock:
                self.llm_calls += 1
            response = self._adapter.complete(
                messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            response_text = response.text
            if self._cache is not None and cache_key is not None:
                self._cache.set(cache_key, CacheRecord(text=response_text, usage=response.usage))

        permutation = self._parse_permutation(response_text, len(batch))
        if not permutation:
            logger.debug("Tournament batch got empty/malformed permutation; preserving input order")
            return list(batch)
        return [batch[i] for i in permutation]

    # The prompt/cache helpers below are intentionally duplicated from
    # ListwiseReranker rather than shared via a base class — they are
    # simple enough that the duplication keeps the two rerankers
    # independent and easy to evolve separately.

    def _build_user_prompt(self, query: str, batch: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        lines.append(
            f"I will provide you with {len(batch)} passages, each indicated by a "
            "numerical identifier []."
        )
        lines.append(f"Rank the passages based on their relevance to the search query: {query}")
        lines.append("")
        for i, doc in enumerate(batch, start=1):
            text = self._passage_text(doc).replace("\n", " ").strip()
            if len(text) > self._max_passage_chars:
                text = text[: self._max_passage_chars] + "…"
            lines.append(f"[{i}] {text}")
        lines.append("")
        lines.append(f"Search Query: {query}")
        lines.append(
            f"Rank the {len(batch)} passages above based on their relevance to "
            "the search query. The passages should be listed in descending order "
            "using identifiers. Only respond with the ranking results, do not say "
            "any word or explain. Format: [2] > [4] > [1] > ..."
        )
        return "\n".join(lines)

    @staticmethod
    def _passage_text(doc: dict[str, Any]) -> str:
        content = doc.get("content")
        if content:
            return str(content)
        preview = doc.get("preview")
        if preview:
            return str(preview)
        return ""

    @staticmethod
    def _parse_permutation(text: str, n: int) -> list[int]:
        seen: set[int] = set()
        order: list[int] = []
        for match in _ID_PATTERN.finditer(text):
            idx = int(match.group(1)) - 1
            if idx < 0 or idx >= n or idx in seen:
                continue
            seen.add(idx)
            order.append(idx)
        if not order:
            return []
        for i in range(n):
            if i not in seen:
                order.append(i)
        return order

    def _compute_cache_key(self, query: str, batch: list[dict[str, Any]]) -> str:
        parts: list[str] = [
            f"ns={self._cache_namespace}",
            f"tour={self._batch_size}/{self._top_k_per_batch}",
            f"t={self._temperature}",
            f"q={query}",
        ]
        for doc in batch:
            doc_id = str(doc.get("doc_id", ""))
            text = self._passage_text(doc)[: self._max_passage_chars]
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            parts.append(f"{doc_id}#{text_hash}")
        return "|".join(parts)


def ndcg_at_k(
    ranked_doc_ids: list[str],
    qrels: dict[str, float],
    k: int = 10,
) -> float:
    """Normalised DCG@k with binary or graded relevance.

    ``qrels`` maps ``doc_id`` to a non-negative graded relevance. Documents
    not in ``qrels`` are treated as having relevance 0. Returns 0.0 when
    ``qrels`` is empty or contains no positive judgments.
    """
    if k <= 0 or not qrels:
        return 0.0
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        rel = float(qrels.get(doc_id, 0.0))
        if rel > 0:
            dcg += (2.0**rel - 1.0) / math.log2(rank + 1)
    ideal_rels = sorted((r for r in qrels.values() if r > 0), reverse=True)[:k]
    idcg = sum(
        (2.0**rel - 1.0) / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1)
    )
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def recall_at_k(
    ranked_doc_ids: list[str],
    qrels: dict[str, float],
    k: int = 10,
) -> float:
    """Recall@k: fraction of relevant docs (``qrels[d] > 0``) found in top-k."""
    if k <= 0:
        return 0.0
    relevant = {d for d, rel in qrels.items() if rel > 0}
    if not relevant:
        return 0.0
    top_k = set(ranked_doc_ids[:k])
    return len(top_k & relevant) / len(relevant)
