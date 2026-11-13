"""In-memory retrievers shared by the MCP server tests."""

from __future__ import annotations

from typing import Any


def _doc(unit_id: str, content: str, **metadata: Any) -> dict[str, Any]:
    return {"doc_id": unit_id, "content": content, "metadata": metadata}


DEFAULT_DOCS = [
    _doc("a-p1", "Alpha report page one about revenue growth.", doc_id="a", page_num=1),
    _doc("a-p2", "Alpha report page two about revenue costs.", doc_id="a", page_num=2),
    _doc("a-p3", "Alpha report page three closing notes.", doc_id="a", page_num=3),
    _doc("b-p1", "Beta memo about revenue in the north region.", doc_id="b", page_num=1),
]


class InMemoryRetriever:
    """Keyword retriever over a list of docs.

    Returns the same hit dicts on every call, like a retriever whose result
    cache is on, and records the arguments of the last search.
    """

    def __init__(self, docs: list[dict[str, Any]] | None = None, preview_length: int = 40) -> None:
        self.docs = [dict(doc) for doc in (docs if docs is not None else DEFAULT_DOCS)]
        self.preview_length = preview_length
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._hits = {doc["doc_id"]: self._hit(doc) for doc in self.docs}

    def _hit(self, doc: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(doc["metadata"])
        hit = {
            "doc_id": doc["doc_id"],
            "page_doc_id": doc["doc_id"],
            "content": doc["content"],
            "preview": doc["content"][: self.preview_length],
            "score": 0.0,
            "metadata": metadata,
        }
        if "doc_id" in metadata:
            hit["logical_doc_id"] = metadata["doc_id"]
        if "page_num" in metadata:
            hit["page_num"] = metadata["page_num"]
        return hit

    def _run(
        self, method: str, query: str, top_k: int, filters: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        self.calls.append((method, {"query": query, "top_k": top_k, "filters": filters}))
        terms = query.lower().split()
        scored = []
        for doc in self.docs:
            if filters and any(doc["metadata"].get(k) != v for k, v in filters.items()):
                continue
            score = sum(doc["content"].lower().count(term) for term in terms)
            if score:
                scored.append((score, doc["doc_id"]))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [self._hits[doc_id] for _, doc_id in scored[:top_k]]

    def search(
        self, query: str, *, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self._run("search", query, top_k, filters)

    def vector_search(
        self, query: str, *, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self._run("vector_search", query, top_k, filters)

    def hybrid_search(
        self, query: str, *, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self._run("hybrid_search", query, top_k, filters)

    def get(self, doc_id: str) -> dict[str, Any]:
        for doc in self.docs:
            if doc["doc_id"] == doc_id:
                return {
                    "doc_id": doc_id,
                    "content": doc["content"],
                    "metadata": dict(doc["metadata"]),
                }
        raise KeyError(doc_id)


class PagedRetriever(InMemoryRetriever):
    """Adds ``get_pages`` with the semantics of ``ElasticsearchRetriever.get_pages``."""

    def get_pages(
        self,
        logical_doc_id: str,
        *,
        pages: list[int] | None = None,
        radius: int = 0,
        max_pages: int = 20,
    ) -> list[dict[str, Any]]:
        wanted = None
        if pages:
            wanted = {p + d for p in pages for d in range(-radius, radius + 1)}
        found = [
            {
                "doc_id": doc["doc_id"],
                "page_doc_id": doc["doc_id"],
                "content": doc["content"],
                "metadata": dict(doc["metadata"]),
                "logical_doc_id": logical_doc_id,
                "page_num": doc["metadata"]["page_num"],
            }
            for doc in self.docs
            if doc["metadata"].get("doc_id") == logical_doc_id
            and (wanted is None or doc["metadata"].get("page_num") in wanted)
        ]
        found.sort(key=lambda page: page["page_num"])
        return found[:max_pages]


class FailingRetriever(InMemoryRetriever):
    """Every call raises an error whose message holds a secret-looking host."""

    def _run(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise ConnectionError("cannot reach https://es-internal.corp:9200 with key sk-SECRET")

    def get(self, doc_id: str) -> dict[str, Any]:
        raise ConnectionError("cannot reach https://es-internal.corp:9200 with key sk-SECRET")


def make_retriever() -> PagedRetriever:
    return PagedRetriever()


def make_not_a_retriever() -> object:
    return object()
