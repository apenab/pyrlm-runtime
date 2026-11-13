"""Retrieval logic behind the MCP tools.

This module does not import ``mcp``: a different protocol adapter can reuse it.
Methods raise :class:`ServiceError` subclasses with messages that are safe to
show a client. Any other exception comes from the retriever and the adapter
must sanitise it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel

from pyrlm_runtime import RetrieverProtocol

from .config import SearchMode, ServerConfig
from .schemas import DocumentResponse, Page, PagesResponse, SearchHit, SearchResponse

MAX_PAGE_NUMBER = 1_000_000
"""Largest page number ``get_pages`` accepts; pages count from 1."""

_SEARCH_METHODS: dict[SearchMode, str] = {
    "hybrid": "hybrid_search",
    "lexical": "search",
    "semantic": "vector_search",
}

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class ServiceError(Exception):
    """A failure whose message is safe to return to the client."""


class InvalidRequest(ServiceError):
    """The request breaks a server rule (unknown mode, filter key not allowed, ...)."""


class NotFound(ServiceError):
    """The requested id does not exist."""


class ScopedRetriever:
    """Wrap a retriever so every search honours the server's scope.

    Search calls get ``top_k`` clamped to ``[1, max_top_k]`` and
    ``base_filters`` merged over the caller's filters, with the base value
    winning on a shared key. ``get`` passes through unchanged.
    """

    def __init__(
        self,
        retriever: RetrieverProtocol,
        *,
        base_filters: dict[str, Any] | None = None,
        max_top_k: int = 50,
    ) -> None:
        self._retriever = retriever
        self._base_filters = dict(base_filters or {})
        self._max_top_k = max_top_k

    def clamp_top_k(self, top_k: int) -> int:
        return max(1, min(top_k, self._max_top_k))

    def _scope(self, top_k: int, filters: dict[str, Any] | None) -> dict[str, Any]:
        merged = {**(filters or {}), **self._base_filters}
        return {"top_k": self.clamp_top_k(top_k), "filters": merged or None}

    def search(
        self, query: str, *, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self._retriever.search(query, **self._scope(top_k, filters))

    def vector_search(
        self, query: str, *, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self._retriever.vector_search(query, **self._scope(top_k, filters))

    def hybrid_search(
        self, query: str, *, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self._retriever.hybrid_search(query, **self._scope(top_k, filters))

    def get(self, doc_id: str) -> dict[str, Any]:
        return self._retriever.get(doc_id)


class RetrievalService:
    """Read-only retrieval with the server's limits applied.

    Never mutates the dicts the retriever returns: a retriever with a result
    cache hands the same objects to every caller.
    """

    def __init__(self, retriever: RetrieverProtocol, config: ServerConfig) -> None:
        self._config = config
        self._scoped = ScopedRetriever(
            retriever, base_filters=config.base_filters, max_top_k=config.max_top_k
        )
        get_pages = getattr(retriever, "get_pages", None)
        self._get_pages: Callable[..., list[dict[str, Any]]] | None = (
            get_pages if callable(get_pages) else None
        )
        self._denylist = tuple(config.metadata_denylist)
        self._filterable = (
            None if config.filterable_fields is None else frozenset(config.filterable_fields)
        )

    @property
    def supports_pages(self) -> bool:
        """True when the retriever has a ``get_pages`` method."""
        return self._get_pages is not None

    def search(
        self,
        query: str,
        *,
        mode: str | None = None,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        include_content: bool = False,
    ) -> SearchResponse:
        if not query.strip():
            raise InvalidRequest("query must not be empty")
        chosen = self._resolve_mode(mode)
        self._check_filters(filters)
        top_k_applied = self._scoped.clamp_top_k(top_k)
        method = getattr(self._scoped, _SEARCH_METHODS[chosen])
        raw_hits = method(query, top_k=top_k_applied, filters=filters)
        hits = [self._to_hit(raw, include_content) for raw in raw_hits[:top_k_applied]]
        kept, truncated = _fit(hits, self._config.max_response_chars)
        return SearchResponse(
            results=kept, mode=chosen, top_k_applied=top_k_applied, truncated=truncated
        )

    def get_document(
        self, doc_id: str, *, offset: int = 0, max_chars: int | None = None
    ) -> DocumentResponse:
        limit = self._config.max_document_chars
        max_chars_applied = limit if max_chars is None else max(1, min(max_chars, limit))
        offset_applied = max(0, offset)
        raw = self._lookup(doc_id)
        text = str(raw.get("content") or "")
        end = offset_applied + max_chars_applied
        response = DocumentResponse(
            doc_id=str(raw.get("doc_id", doc_id)),
            content=text[offset_applied:end],
            offset=offset_applied,
            max_chars_applied=max_chars_applied,
            next_offset=end if end < len(text) else None,
            total_chars=len(text),
            metadata=self._metadata(raw),
        )
        if _size(response) > self._config.max_response_chars:
            response = response.model_copy(update={"metadata": {}, "truncated": True})
        return response

    def get_pages(
        self, logical_doc_id: str, *, pages: Sequence[int], radius: int = 0
    ) -> PagesResponse:
        if self._get_pages is None:
            raise InvalidRequest("this retriever does not support get_pages")
        max_pages = self._config.max_pages
        if not pages or len(pages) > max_pages:
            raise InvalidRequest(f"pages must list between 1 and {max_pages} page numbers")
        if any(not 1 <= page <= MAX_PAGE_NUMBER for page in pages):
            raise InvalidRequest(f"page numbers must be between 1 and {MAX_PAGE_NUMBER}")
        radius_applied = max(0, min(radius, max_pages))
        raw_pages = self._get_pages(
            logical_doc_id, pages=list(pages), radius=radius_applied, max_pages=max_pages
        )
        if not raw_pages:
            raise NotFound(
                f"no pages found for logical_doc_id {logical_doc_id!r}. Pass the logical_doc_id of a search hit, not its doc_id."
            )
        per_page = self._config.max_document_chars
        converted = [self._to_page(raw, per_page) for raw in raw_pages[:max_pages]]
        kept, truncated = _fit(converted, self._config.max_response_chars)
        return PagesResponse(
            logical_doc_id=logical_doc_id,
            pages=kept,
            radius_applied=radius_applied,
            max_pages_applied=max_pages,
            truncated=truncated,
        )

    def _resolve_mode(self, mode: str | None) -> SearchMode:
        available = self._config.search_modes
        if mode is None:
            return self._config.default_search_mode
        for candidate in available:
            if candidate == mode:
                return candidate
        raise InvalidRequest(
            f"search mode {mode!r} is not available; available modes: {', '.join(available)}"
        )

    def _check_filters(self, filters: dict[str, Any] | None) -> None:
        for key in filters or {}:
            if "*" in key:
                raise InvalidRequest(
                    f"filter key {key!r} is not allowed: a wildcard can name several fields"
                )
            if _touches_denied_field(key, self._denylist) or (
                self._filterable is not None and key not in self._filterable
            ):
                allowed = (
                    "any field"
                    if self._filterable is None
                    else ", ".join(sorted(self._filterable)) or "none"
                )
                raise InvalidRequest(f"filter key {key!r} is not allowed; filterable: {allowed}")

    def _lookup(self, doc_id: str) -> dict[str, Any]:
        try:
            raw = self._scoped.get(doc_id)
        except Exception as exc:
            if isinstance(exc, LookupError) or type(exc).__name__ == "NotFoundError":
                raise NotFound(f"document {doc_id!r} not found") from exc
            raise
        if not raw:
            raise NotFound(f"document {doc_id!r} not found")
        return raw

    def _metadata(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _without_paths(raw.get("metadata") or {}, self._denylist)

    def _to_hit(self, raw: dict[str, Any], include_content: bool) -> SearchHit:
        text = str(raw.get("content") or "")
        preview = raw.get("preview")
        if preview is None:
            preview = text
        content, content_truncated = None, False
        if include_content:
            content, content_truncated = _truncate(text, self._config.max_content_chars)
        logical_doc_id = raw.get("logical_doc_id")
        return SearchHit(
            doc_id=str(raw["doc_id"]),
            logical_doc_id=None if logical_doc_id is None else str(logical_doc_id),
            page_num=raw.get("page_num"),
            score=raw.get("score"),
            preview=str(preview)[: self._config.max_preview_chars],
            content=content,
            content_truncated=content_truncated,
            metadata=self._metadata(raw),
        )

    def _to_page(self, raw: dict[str, Any], limit: int) -> Page:
        content, content_truncated = _truncate(str(raw.get("content") or ""), limit)
        return Page(
            doc_id=str(raw.get("page_doc_id") or raw["doc_id"]),
            page_num=raw.get("page_num"),
            content=content,
            content_truncated=content_truncated,
            metadata=self._metadata(raw),
        )


def _touches_denied_field(key: str, denylist: Sequence[str]) -> bool:
    """True when ``key`` is a denied field, one of its subfields, or a parent of one.

    Subfields matter because Elasticsearch indexes ``field.keyword`` alongside
    ``field``: filtering on it reveals the denied value just the same.
    """
    return any(
        key == denied or key.startswith(denied + ".") or denied.startswith(key + ".")
        for denied in denylist
    )


def _without_paths(metadata: dict[str, Any], paths: Sequence[str]) -> dict[str, Any]:
    """Copy ``metadata`` without the dotted ``paths``, descending into nested dicts."""
    kept: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in paths:
            continue
        nested = [path[len(key) + 1 :] for path in paths if path.startswith(key + ".")]
        if nested and isinstance(value, dict):
            value = _without_paths(value, nested)
        kept[key] = value
    return kept


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _size(model: BaseModel) -> int:
    return len(json.dumps(model.model_dump(mode="json"), ensure_ascii=False))


def _fit(items: list[_ModelT], budget: int) -> tuple[list[_ModelT], bool]:
    """Keep the longest prefix of ``items`` whose serialised size fits ``budget``."""
    used = 0
    for index, item in enumerate(items):
        used += _size(item)
        if used > budget:
            return items[:index], True
    return items, False
