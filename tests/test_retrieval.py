"""Tests for the retrieval module (RetrieverProtocol, ElasticsearchRetriever, REPL integration)."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrlm_runtime import Context, RLM
from pyrlm_runtime.adapters import FakeAdapter
from pyrlm_runtime.retrieval import (
    AsyncElasticsearchRetriever,
    AsyncRetrieverProtocol,
    ElasticsearchRetriever,
    RetrieverProtocol,
    _build_filter_clauses,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class InMemoryRetriever:
    """Minimal retriever for testing — stores docs in a dict."""

    def __init__(self, docs: dict[str, str] | None = None) -> None:
        self.docs = docs or {}

    def search(self, query, *, top_k=10, filters=None):
        results = []
        for doc_id, content in self.docs.items():
            if query.lower() in content.lower():
                results.append({
                    "doc_id": doc_id,
                    "preview": content[:200],
                    "score": 1.0,
                    "metadata": {},
                })
        return results[:top_k]

    def vector_search(self, query, *, top_k=10, filters=None):
        # Simplified: same as search for testing
        return self.search(query, top_k=top_k, filters=filters)

    def hybrid_search(self, query, *, top_k=10, filters=None):
        return self.search(query, top_k=top_k, filters=filters)

    def get(self, doc_id):
        content = self.docs.get(doc_id, "")
        return {"doc_id": doc_id, "content": content, "metadata": {}}


class SchemaAwareRetriever(InMemoryRetriever):
    def search(self, query, *, top_k=10, filters=None):
        del query, top_k, filters
        return [
            {
                "doc_id": "page-001",
                "preview": "first preview",
                "score": 1.0,
                "metadata": {"doc_id": "logical-doc-A"},
            },
            {
                "doc_id": "page-002",
                "preview": "second preview",
                "score": 0.9,
                "metadata": {"doc_id": "logical-doc-B"},
            },
        ]


class LogicalDocFallbackRetriever(InMemoryRetriever):
    def __init__(self) -> None:
        super().__init__({
            "page-001": "Page 1 text",
            "page-002": "Page 2 text",
        })

    def search(self, query, *, top_k=10, filters=None):
        del query, top_k, filters
        return [
            {
                "doc_id": "page-001",
                "preview": "Page 1 text",
                "score": 1.0,
                "metadata": {"doc_id": "logical-doc-A", "page_num": 1},
            }
        ]

    def get(self, doc_id):
        if doc_id in self.docs:
            return {
                "doc_id": doc_id,
                "content": self.docs[doc_id],
                "metadata": {"doc_id": "logical-doc-A"},
            }
        raise KeyError(doc_id)

    def get_logical_document(self, logical_doc_id, *, max_pages=500):
        del max_pages
        if logical_doc_id != "logical-doc-A":
            raise KeyError(logical_doc_id)
        return {
            "doc_id": logical_doc_id,
            "logical_doc_id": logical_doc_id,
            "content": "Page 1 text\n\nPage 2 text",
            "metadata": {
                "doc_id": logical_doc_id,
                "page_count": 2,
                "page_doc_ids": ["page-001", "page-002"],
            },
        }


class DocSearchRetriever(InMemoryRetriever):
    def hybrid_search(self, query, *, top_k=10, filters=None):
        del query, top_k
        results = [
            {
                "doc_id": "doc-A__p7",
                "preview": "Doc A page 7",
                "score": 3.0,
                "metadata": {"doc_id": "doc-A", "page_num": 7},
            },
            {
                "doc_id": "doc-A__p8",
                "preview": "Doc A page 8",
                "score": 2.8,
                "metadata": {"doc_id": "doc-A", "page_num": 8},
            },
            {
                "doc_id": "doc-B__p2",
                "preview": "Doc B page 2",
                "score": 2.2,
                "metadata": {"doc_id": "doc-B", "page_num": 2},
            },
        ]
        doc_filter = (filters or {}).get("doc_id")
        if doc_filter:
            results = [item for item in results if item["metadata"].get("doc_id") == doc_filter]
        return results


class ExactPageRetriever(InMemoryRetriever):
    def __init__(self) -> None:
        super().__init__()
        self.pages = {
            3: "Resumen sin la partida buscada.",
            10: "| Importe neto de la cifra de\nnegocio | 982.160 | 786.063 |",
        }

    def get_logical_document(self, logical_doc_id, *, max_pages=500):
        del max_pages
        if logical_doc_id != "logical-doc-A":
            raise KeyError(logical_doc_id)
        return {
            "doc_id": logical_doc_id,
            "logical_doc_id": logical_doc_id,
            "content": (
                "<!-- Page 3 -->\nResumen sin la partida buscada.\n\n"
                "<!-- Page 10 -->\n| Importe neto de la cifra de negocios | 982.160 | 786.063 |"
            ),
            "metadata": {
                "doc_id": logical_doc_id,
                "page_count": 2,
                "page_doc_ids": ["logical-doc-A__p3", "logical-doc-A__p10"],
            },
        }

    def get_pages_text(self, logical_doc_id, *, pages=None, radius=0, max_pages=20):
        del radius, max_pages
        if logical_doc_id != "logical-doc-A":
            raise KeyError(logical_doc_id)
        wanted = set(pages or [])
        parts = []
        for page_num in sorted(wanted):
            if page_num in self.pages:
                parts.append(f"<!-- Page {page_num} -->\n{self.pages[page_num]}")
        return "\n\n".join(parts)


class MultiCandidateComparativeRetriever(InMemoryRetriever):
    def get_logical_document(self, logical_doc_id, *, max_pages=2000):
        del max_pages
        if logical_doc_id != "logical-doc-eizar":
            raise KeyError(logical_doc_id)
        return {
            "doc_id": logical_doc_id,
            "logical_doc_id": logical_doc_id,
            "content": (
                "<!-- Page 6 -->\n"
                "| 1. Importe de la cifra de negocios | 179.964.152 175.200.881 | "
                "180.073.791 174.718.771 5.355.020 |\n\n"
                "<!-- Page 72 -->\n"
                "| TOTAL | 179.964.152 | 180.073.791 |"
            ),
            "metadata": {
                "doc_id": logical_doc_id,
                "page_count": 85,
                "expected_page_count": 85,
                "indexed_page_count": 85,
                "page_doc_ids": ["logical-doc-eizar__p6", "logical-doc-eizar__p72"],
                "index_incomplete": False,
            },
        }

    def get_pages_text(self, logical_doc_id, *, pages=None, radius=0, max_pages=20):
        del radius, max_pages
        if logical_doc_id != "logical-doc-eizar":
            raise KeyError(logical_doc_id)
        texts = {
            6: "<!-- Page 6 -->\n| 1. Importe de la cifra de negocios | 179.964.152 175.200.881 | 180.073.791 174.718.771 5.355.020 |",
            72: "<!-- Page 72 -->\n| TOTAL | 179.964.152 | 180.073.791 |",
        }
        parts = []
        for page_num in pages or []:
            if page_num in texts:
                parts.append(texts[page_num])
        return "\n\n".join(parts)

    def hybrid_search(self, query, *, top_k=10, filters=None):
        del query, top_k
        if (filters or {}).get("doc_id") != "logical-doc-eizar":
            return []
        return [
            {
                "doc_id": "logical-doc-eizar__p72",
                "preview": "TOTAL 179.964.152 | 180.073.791",
                "score": 4.0,
                "metadata": {"doc_id": "logical-doc-eizar", "page_num": 72},
            },
            {
                "doc_id": "logical-doc-eizar__p6",
                "preview": "tabla ruidosa",
                "score": 3.0,
                "metadata": {"doc_id": "logical-doc-eizar", "page_num": 6},
            },
        ]


class IncompleteLogicalDocRetriever(InMemoryRetriever):
    def get_logical_document(self, logical_doc_id, *, max_pages=2000):
        del max_pages
        if logical_doc_id != "logical-doc-incomplete":
            raise KeyError(logical_doc_id)
        return {
            "doc_id": logical_doc_id,
            "logical_doc_id": logical_doc_id,
            "content": "<!-- Page 1 -->\nResumen sin la métrica.",
            "metadata": {
                "doc_id": logical_doc_id,
                "page_count": 140,
                "expected_page_count": 140,
                "indexed_page_count": 10,
                "page_doc_ids": [f"{logical_doc_id}__p{i}" for i in range(1, 11)],
                "index_incomplete": True,
            },
        }

    def hybrid_search(self, query, *, top_k=10, filters=None):
        del query, top_k, filters
        return []


# ---------------------------------------------------------------------------
# RetrieverProtocol
# ---------------------------------------------------------------------------


def test_in_memory_retriever_satisfies_protocol() -> None:
    retriever = InMemoryRetriever()
    assert isinstance(retriever, RetrieverProtocol)


def test_elasticsearch_retriever_satisfies_protocol() -> None:
    assert issubclass(ElasticsearchRetriever, RetrieverProtocol)


# ---------------------------------------------------------------------------
# ElasticsearchRetriever unit tests (mocked ES client)
# ---------------------------------------------------------------------------


def _make_es_retriever() -> ElasticsearchRetriever:
    return ElasticsearchRetriever(
        host="https://localhost:9200",
        api_key="test-key",
        index="test-index",
        embedding_model="text-embedding-3-small",
    )


def _mock_search_response(*hits: dict) -> dict:
    return {"hits": {"hits": list(hits)}}


def _make_hit(doc_id: str, content: str, score: float = 1.0, **extra) -> dict:
    source = {"content": content, **extra}
    return {"_id": doc_id, "_score": score, "_source": source}


class TestElasticsearchRetrieverSearch:
    def test_search_returns_results(self) -> None:
        retriever = _make_es_retriever()
        mock_client = MagicMock()
        mock_client.search.return_value = _mock_search_response(
            _make_hit("doc1", "Hello world", 2.5, title="Doc 1"),
            _make_hit("doc2", "Goodbye world", 1.2),
        )
        retriever._client = mock_client

        results = retriever.search("world")

        assert len(results) == 2
        assert results[0]["doc_id"] == "doc1"
        assert results[0]["preview"] == "Hello world"
        assert results[0]["score"] == 2.5
        assert results[0]["metadata"] == {"title": "Doc 1"}
        assert results[1]["doc_id"] == "doc2"

    def test_search_with_filters(self) -> None:
        retriever = _make_es_retriever()
        mock_client = MagicMock()
        mock_client.search.return_value = _mock_search_response()
        retriever._client = mock_client

        retriever.search("query", filters={"category": "legal"})

        call_body = mock_client.search.call_args[1]["body"]
        filter_clauses = call_body["query"]["bool"]["filter"]
        assert any(
            clause.get("term", {}).get("category") == "legal"
            for clause in filter_clauses
        )

    def test_search_with_list_filter(self) -> None:
        retriever = _make_es_retriever()
        mock_client = MagicMock()
        mock_client.search.return_value = _mock_search_response()
        retriever._client = mock_client

        retriever.search("query", filters={"category": ["legal", "finance"]})

        call_body = mock_client.search.call_args[1]["body"]
        filter_clauses = call_body["query"]["bool"]["filter"]
        assert any(
            clause.get("terms", {}).get("category") == ["legal", "finance"]
            for clause in filter_clauses
        )

    def test_preview_truncation(self) -> None:
        retriever = _make_es_retriever()
        retriever.preview_length = 10
        mock_client = MagicMock()
        mock_client.search.return_value = _mock_search_response(
            _make_hit("doc1", "A" * 100),
        )
        retriever._client = mock_client

        results = retriever.search("A")
        assert len(results[0]["preview"]) == 11  # 10 chars + ellipsis

    def test_get_logical_document_prefers_page_num_and_id_sort_when_supported(self) -> None:
        retriever = _make_es_retriever()
        mock_client = MagicMock()
        mock_client.search.return_value = _mock_search_response(
            {"_id": "doc__p2", "_score": 1.0, "_source": {"content": "Page 2", "page_num": 2, "doc_id": "doc", "page_count": 5}},
            {"_id": "doc__p1", "_score": 1.0, "_source": {"content": "Page 1", "page_num": 1, "doc_id": "doc", "page_count": 5}},
        )
        retriever._client = mock_client

        result = retriever.get_logical_document("doc")

        call_body = mock_client.search.call_args[1]["body"]
        assert call_body["sort"] == [
            {"page_num": {"order": "asc", "missing": "_last"}},
            {"_id": {"order": "asc"}},
        ]
        assert result["doc_id"] == "doc"
        assert result["metadata"]["page_doc_ids"] == ["doc__p1", "doc__p2"]
        assert result["metadata"]["chunk_count"] == 2
        assert result["metadata"]["indexed_source_page_count"] == 2
        assert result["metadata"]["indexed_page_count"] == 2
        assert result["metadata"]["expected_page_count"] == 5
        assert result["metadata"]["page_count"] == 5
        assert result["metadata"]["index_incomplete"] is True
        assert "Page 1" in result["content"]
        assert "Page 2" in result["content"]

    def test_get_logical_document_falls_back_when_id_sort_is_disallowed(self) -> None:
        retriever = _make_es_retriever()
        mock_client = MagicMock()
        mock_client.search.side_effect = [
            RuntimeError(
                "BadRequestError(400, 'search_phase_execution_exception', "
                "'Fielddata access on the _id field is disallowed')"
            ),
            _mock_search_response(
                {"_id": "doc__p2", "_score": 1.0, "_source": {"content": "Page 2", "page_num": 2, "doc_id": "doc", "page_count": 2}},
                {"_id": "doc__p1", "_score": 1.0, "_source": {"content": "Page 1", "page_num": 1, "doc_id": "doc", "page_count": 2}},
            ),
        ]
        retriever._client = mock_client

        result = retriever.get_logical_document("doc")

        first_call_body = mock_client.search.call_args_list[0][1]["body"]
        second_call_body = mock_client.search.call_args_list[1][1]["body"]
        assert first_call_body["sort"] == [
            {"page_num": {"order": "asc", "missing": "_last"}},
            {"_id": {"order": "asc"}},
        ]
        assert second_call_body["sort"] == [{"page_num": {"order": "asc", "missing": "_last"}}]
        assert retriever._supports_id_secondary_sort is False
        assert result["metadata"]["page_doc_ids"] == ["doc__p1", "doc__p2"]
        assert result["metadata"]["chunk_count"] == 2
        assert result["metadata"]["indexed_source_page_count"] == 2
        assert result["metadata"]["indexed_page_count"] == 2
        assert result["metadata"]["expected_page_count"] == 2
        assert result["metadata"]["index_incomplete"] is False

    def test_get_logical_document_uses_source_page_coverage_for_indexed_page_count(self) -> None:
        retriever = _make_es_retriever()
        mock_client = MagicMock()
        mock_client.search.return_value = _mock_search_response(
            {
                "_id": "doc__p1",
                "_score": 1.0,
                "_source": {
                    "content": "Pages 1-2",
                    "page_num": 1,
                    "doc_id": "doc",
                    "page_count": 5,
                    "source_pages": [1, 2],
                },
            },
            {
                "_id": "doc__p3",
                "_score": 1.0,
                "_source": {
                    "content": "Pages 3-5",
                    "page_num": 3,
                    "doc_id": "doc",
                    "page_count": 5,
                    "source_pages": [3, 4, 5],
                },
            },
        )
        retriever._client = mock_client

        result = retriever.get_logical_document("doc")

        assert result["metadata"]["chunk_count"] == 2
        assert result["metadata"]["indexed_source_page_count"] == 5
        assert result["metadata"]["indexed_page_count"] == 5
        assert result["metadata"]["source_pages"] == [1, 2, 3, 4, 5]
        assert result["metadata"]["expected_page_count"] == 5
        assert result["metadata"]["index_incomplete"] is False


class TestElasticsearchRetrieverVectorSearch:
    def test_vector_search_calls_embed(self) -> None:
        retriever = _make_es_retriever()
        mock_client = MagicMock()
        mock_client.search.return_value = _mock_search_response(
            _make_hit("doc1", "semantic result", 0.95),
        )
        retriever._client = mock_client

        with patch.object(retriever, "_embed", return_value=[0.1, 0.2, 0.3]) as mock_embed:
            results = retriever.vector_search("conceptual query")

        mock_embed.assert_called_once_with("conceptual query")
        assert len(results) == 1
        assert results[0]["doc_id"] == "doc1"

        call_body = mock_client.search.call_args[1]["body"]
        assert "knn" in call_body
        assert call_body["knn"]["query_vector"] == [0.1, 0.2, 0.3]

    def test_vector_search_with_vertexai_provider(self) -> None:
        retriever = ElasticsearchRetriever(
            host="https://localhost:9200",
            api_key="test-key",
            index="test-index",
            embedding_provider="vertexai",
            embedding_model="text-embedding-005",
        )
        mock_client = MagicMock()
        mock_client.search.return_value = _mock_search_response(
            _make_hit("doc1", "semantic result", 0.95),
        )
        retriever._client = mock_client

        with patch(
            "pyrlm_runtime.retrieval._embed_vertexai_query",
            return_value=[0.4, 0.5, 0.6],
        ) as mock_embed:
            results = retriever.vector_search("conceptual query")

        mock_embed.assert_called_once_with(
            "conceptual query",
            model="text-embedding-005",
            ssl_verify=True,
        )
        assert len(results) == 1
        call_body = mock_client.search.call_args[1]["body"]
        assert call_body["knn"]["query_vector"] == [0.4, 0.5, 0.6]


class TestElasticsearchRetrieverHybridSearch:
    def test_hybrid_search_uses_rrf(self) -> None:
        retriever = _make_es_retriever()
        mock_client = MagicMock()
        mock_client.search.return_value = _mock_search_response(
            _make_hit("doc1", "hybrid result", 0.8),
        )
        retriever._client = mock_client

        with patch.object(retriever, "_embed", return_value=[0.1, 0.2]):
            results = retriever.hybrid_search("test query")

        assert len(results) == 1
        call_body = mock_client.search.call_args[1]["body"]
        assert "rank" in call_body
        assert "rrf" in call_body["rank"]
        assert "query" in call_body
        assert "knn" in call_body

    def test_hybrid_search_falls_back_to_local_rrf_when_cluster_rejects_rrf(self) -> None:
        retriever = _make_es_retriever()
        mock_client = MagicMock()
        mock_client.search.side_effect = [
            RuntimeError(
                "AuthorizationException(403, 'security_exception', "
                "'current license is non-compliant for [Reciprocal Rank Fusion (RRF)]')"
            ),
            _mock_search_response(
                _make_hit("doc-bm25", "lexical result", 8.0),
                _make_hit("doc-shared", "shared result", 7.0),
            ),
            _mock_search_response(
                _make_hit("doc-vector", "semantic result", 0.9),
                _make_hit("doc-shared", "shared result", 0.8),
            ),
        ]
        retriever._client = mock_client

        with patch.object(retriever, "_embed", return_value=[0.1, 0.2]):
            results = retriever.hybrid_search("test query")

        assert [result["doc_id"] for result in results] == [
            "doc-shared",
            "doc-bm25",
            "doc-vector",
        ]
        assert retriever._supports_rrf is False
        assert mock_client.search.call_count == 3


class TestElasticsearchRetrieverGet:
    def test_get_returns_full_document(self) -> None:
        retriever = _make_es_retriever()
        mock_client = MagicMock()
        mock_client.get.return_value = {
            "_id": "doc1",
            "_source": {"content": "Full document text here", "title": "Doc 1"},
        }
        retriever._client = mock_client

        result = retriever.get("doc1")

        assert result["doc_id"] == "doc1"
        assert result["content"] == "Full document text here"
        assert result["metadata"] == {"title": "Doc 1"}
        mock_client.get.assert_called_once_with(index="test-index", id="doc1")


class TestElasticsearchRetrieverErrors:
    def test_missing_elasticsearch_package(self) -> None:
        retriever = _make_es_retriever()
        with patch.dict("sys.modules", {"elasticsearch": None}):
            with pytest.raises(ImportError, match="elasticsearch"):
                retriever._client = None
                retriever._get_client()

    def test_vector_search_without_embedding_model(self) -> None:
        retriever = ElasticsearchRetriever(
            host="https://localhost:9200",
            api_key="test-key",
            index="test-index",
            embedding_model=None,
        )
        mock_client = MagicMock()
        retriever._client = mock_client

        with pytest.raises(ValueError, match="embedding_model"):
            retriever.vector_search("query")


# ---------------------------------------------------------------------------
# REPL integration tests
# ---------------------------------------------------------------------------


def test_retrieval_functions_registered_in_repl() -> None:
    """When a retriever is set, es_* functions are available in the REPL."""
    retriever = InMemoryRetriever({"doc1": "Hello world"})
    adapter = FakeAdapter(
        script=[
            'results = es_search("Hello")\nprint(len(results))',
            "FINAL: 1",
        ]
    )

    runtime = RLM(adapter=adapter, retriever=retriever)
    output, trace = runtime.run("test", Context.from_text(""))

    assert output == "1"


def test_retrieval_functions_not_registered_without_retriever() -> None:
    """Without a retriever, es_* functions should not be in the REPL."""
    adapter = FakeAdapter(
        script=[
            'results = es_search("Hello")\nprint(results)',
            "FINAL: done",
        ]
    )

    runtime = RLM(adapter=adapter)
    output, trace = runtime.run("test", Context.from_text(""))

    # The REPL should error because es_search is not defined
    repl_steps = [s for s in trace.steps if s.kind == "repl_exec"]
    assert any(s.error and "es_search" in s.error for s in repl_steps)


def test_es_get_in_repl() -> None:
    """es_get() should retrieve full document content."""
    retriever = InMemoryRetriever({"doc1": "Full document content"})
    adapter = FakeAdapter(
        script=[
            'doc = es_get("doc1")\nprint(doc["content"])',
            "FINAL_VAR: doc",
        ]
    )

    runtime = RLM(adapter=adapter, retriever=retriever)
    output, trace = runtime.run("test", Context.from_text(""))

    assert "Full document content" in output


def test_es_get_text_in_repl() -> None:
    """es_get_text() should retrieve raw document text as a string."""
    retriever = InMemoryRetriever({"doc1": "Full document content"})
    adapter = FakeAdapter(
        script=[
            'text = es_get_text("doc1")\nprint(text[:4])',
            "FINAL_VAR: text",
        ]
    )

    runtime = RLM(adapter=adapter, retriever=retriever)
    output, trace = runtime.run("test", Context.from_text(""))

    assert output == "Full document content"


def test_es_get_text_accepts_logical_doc_id_in_repl() -> None:
    retriever = LogicalDocFallbackRetriever()
    adapter = FakeAdapter(
        script=[
            'text = es_get_text("logical-doc-A")\nprint(text.splitlines()[0])',
            "FINAL_VAR: text",
        ]
    )

    runtime = RLM(adapter=adapter, retriever=retriever)
    output, trace = runtime.run("test", Context.from_text(""))

    assert "Page 1 text" in output
    assert "Page 2 text" in output


def test_es_hybrid_search_in_repl() -> None:
    """es_hybrid_search() should be callable from the REPL."""
    retriever = InMemoryRetriever({
        "doc1": "The quick brown fox",
        "doc2": "A lazy dog sleeps",
    })
    adapter = FakeAdapter(
        script=[
            'results = es_hybrid_search("fox")\nprint(len(results))',
            "FINAL: 1",
        ]
    )

    runtime = RLM(adapter=adapter, retriever=retriever)
    output, trace = runtime.run("test", Context.from_text(""))

    assert output == "1"


def test_es_hybrid_doc_search_in_repl() -> None:
    retriever = DocSearchRetriever()
    adapter = FakeAdapter(
        script=[
            (
                'docs = es_hybrid_doc_search("fox", top_k=2)\n'
                'summary = [(d["logical_doc_id"], d["hit_count"]) for d in docs]\n'
                "print(summary)"
            ),
            "FINAL_VAR: docs",
        ]
    )

    runtime = RLM(adapter=adapter, retriever=retriever)
    output, trace = runtime.run("test", Context.from_text(""))

    assert "doc-A" in output
    assert "doc-B" in output


def test_es_hybrid_search_in_doc_in_repl() -> None:
    retriever = DocSearchRetriever()
    adapter = FakeAdapter(
        script=[
            (
                'pages = es_hybrid_search_in_doc("doc-A", "importe neto de la cifra de negocios", top_k=3)\n'
                'print([(p["logical_doc_id"], p["page_num"]) for p in pages])'
            ),
            "FINAL_VAR: pages",
        ]
    )

    runtime = RLM(adapter=adapter, retriever=retriever)
    output, trace = runtime.run("test", Context.from_text(""))

    assert "doc-A" in output
    assert "doc-B" not in output


def test_es_find_pages_text_in_repl() -> None:
    retriever = ExactPageRetriever()
    adapter = FakeAdapter(
        script=[
            (
                'matches = es_find_pages("logical-doc-A", ["Importe neto de la cifra de negocios"])\n'
                'text = es_find_pages_text("logical-doc-A", ["Importe neto de la cifra de negocios"])\n'
                'print(matches)\n'
                'print(text)'
            ),
            "FINAL_VAR: matches",
        ]
    )

    runtime = RLM(adapter=adapter, retriever=retriever)
    output, trace = runtime.run("test", Context.from_text(""))

    assert "10" in output
    assert "logical-doc-A__p10" in output


def test_es_extract_comparative_metric_in_repl() -> None:
    retriever = ExactPageRetriever()
    adapter = FakeAdapter(
        script=[
            (
                'row = es_extract_comparative_metric(\n'
                '    "logical-doc-A",\n'
                '    "Importe neto de la cifra de negocios",\n'
                '    aliases=["Importe de la cifra de negocios"],\n'
                ')\n'
                'print(row)\n'
                'answer = f"{row[\'recent_amount_raw\']}|{row[\'page_strategy\']}"'
            ),
            "FINAL_VAR: answer",
        ]
    )
    adapter.add_rule(
        r"You are a sub-LLM[\s\S]*Metrica objetivo: Importe neto de la cifra de negocios",
        (
            '{"entity_name":"Eurotransac","line_item_found":true,'
            '"line_item_label":"Importe neto de la cifra de negocio",'
            '"recent_year":2024,"recent_amount_raw":"982.160",'
            '"previous_year":2023,"previous_amount_raw":"786.063",'
            '"unit_hint":"Miles de euros","pages":[10],'
            '"evidence":"fila","confidence":"high","reason":"ok"}'
        ),
        regex=True,
    )

    runtime = RLM(adapter=adapter, retriever=retriever)
    output, trace = runtime.run("test", Context.from_text(""))

    assert output == "982.160|exact"


def test_comparative_metric_corpus_report_in_repl() -> None:
    retriever = ExactPageRetriever()
    adapter = FakeAdapter(
        script=[
            (
                'report = comparative_metric_corpus_report(\n'
                '    ["logical-doc-A"],\n'
                '    "Importe neto de la cifra de negocios",\n'
                '    aliases=["Importe de la cifra de negocios"],\n'
                ')\n'
                'print(report)\n'
            ),
            "FINAL_VAR: report",
        ]
    )
    adapter.add_rule(
        r"You are a sub-LLM[\s\S]*Metrica objetivo: Importe neto de la cifra de negocios",
        (
            '{"entity_name":"Eurotransac","line_item_found":true,'
            '"line_item_label":"Importe neto de la cifra de negocio",'
            '"recent_year":2024,"recent_amount_raw":"982.160",'
            '"previous_year":2023,"previous_amount_raw":"786.063",'
            '"unit_hint":"Miles de euros","pages":[10],'
            '"evidence":"fila","confidence":"high","reason":"ok"}'
        ),
        regex=True,
    )

    runtime = RLM(adapter=adapter, retriever=retriever)
    output, trace = runtime.run("test", Context.from_text(""))

    assert "1. Eurotransac" in output
    assert "Documento fuente: logical-doc-A" in output
    assert "Variación porcentual:" in output


def test_es_extract_comparative_metric_prefers_clean_single_page_candidate() -> None:
    retriever = MultiCandidateComparativeRetriever()
    adapter = FakeAdapter(
        script=[
            (
                'row = es_extract_comparative_metric(\n'
                '    "logical-doc-eizar",\n'
                '    "Importe neto de la cifra de negocios",\n'
                '    aliases=["Importe de la cifra de negocios"],\n'
                ')\n'
                'answer = f"{row[\'selected_pages\']}|{row[\'recent_amount_raw\']}|{parse_amount_text(row[\'recent_amount_raw\'])}"\n'
                'print(answer)\n'
            ),
            "FINAL_VAR: answer",
        ]
    )
    adapter.add_rule(
        r"You are a sub-LLM[\s\S]*logical-doc-eizar pages=\[6\]",
        (
            '{"entity_name":"EIZAR","line_item_found":true,'
            '"line_item_label":"Importe de la cifra de negocios",'
            '"recent_year":2024,"recent_amount_raw":"179.964.152 175.200.881",'
            '"previous_year":2023,"previous_amount_raw":"180.073.791 174.718.771 5.355.020",'
            '"unit_hint":"euros","pages":[6],"evidence":"tabla ruidosa","confidence":"low","reason":"ocr"}'
        ),
        regex=True,
    )
    adapter.add_rule(
        r"You are a sub-LLM[\s\S]*logical-doc-eizar pages=\[72\]",
        (
            '{"entity_name":"EIZAR","line_item_found":true,'
            '"line_item_label":"Importe neto de la cifra de negocios",'
            '"recent_year":2024,"recent_amount_raw":"179.964.152",'
            '"previous_year":2023,"previous_amount_raw":"180.073.791",'
            '"unit_hint":"euros","pages":[72],"evidence":"fila total","confidence":"high","reason":"ok"}'
        ),
        regex=True,
    )
    adapter.add_rule(
        r"You are a sub-LLM[\s\S]*logical-doc-eizar pages=\[(?:6, 72|72, 6)\]",
        (
            '{"entity_name":"EIZAR","line_item_found":true,'
            '"line_item_label":"Importe neto de la cifra de negocios",'
            '"recent_year":2024,"recent_amount_raw":"179.964.152 175.200.881",'
            '"previous_year":2023,"previous_amount_raw":"180.073.791 174.718.771 5.355.020",'
            '"unit_hint":"euros","pages":[6,72],"evidence":"mezcla","confidence":"low","reason":"ocr"}'
        ),
        regex=True,
    )

    runtime = RLM(adapter=adapter, retriever=retriever)
    output, trace = runtime.run("test", Context.from_text(""))

    assert output == "[72]|179.964.152|179964152"


def test_es_extract_comparative_metric_reports_incomplete_index() -> None:
    retriever = IncompleteLogicalDocRetriever()
    adapter = FakeAdapter(
        script=[
            (
                'row = es_extract_comparative_metric(\n'
                '    "logical-doc-incomplete",\n'
                '    "Importe neto de la cifra de negocios",\n'
                ')\n'
                'answer = (\n'
                '    f"{row[\'index_incomplete\']}|{row[\'indexed_page_count\']}|"\n'
                '    f"{row[\'expected_page_count\']}|{row[\'page_strategy\']}|{row[\'reason\']}"\n'
                ')\n'
                'print(answer)'
            ),
            "FINAL_VAR: answer",
        ]
    )

    runtime = RLM(adapter=adapter, retriever=retriever)
    output, trace = runtime.run("test", Context.from_text(""))

    assert output.startswith("True|10|140|no_pages|")
    assert "incomplete" in output.lower()


def test_search_result_schema_diagnostics_log_distinguishes_page_and_logical_doc_ids(
    caplog,
) -> None:
    retriever = SchemaAwareRetriever()
    adapter = FakeAdapter(
        script=[
            'results = es_search("fox")\nprint(len(results))',
            "FINAL: done",
        ]
    )

    runtime = RLM(adapter=adapter, retriever=retriever, rlm_diagnostics=True)
    with caplog.at_level(logging.DEBUG, logger="pyrlm_runtime"):
        output, trace = runtime.run("test", Context.from_text(""))

    assert output == "done"
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "search_result_schema=" in log_text
    assert '"page_doc_id": "page-001"' in log_text
    assert '"logical_doc_id": "logical-doc-A"' in log_text
    assert "first preview" not in log_text


def test_context_optional_with_retriever() -> None:
    """When retriever is set, context can be omitted."""
    retriever = InMemoryRetriever({"doc1": "Test content"})
    adapter = FakeAdapter(
        script=[
            'results = es_search("Test")\nprint(len(results))',
            "FINAL: 1",
        ]
    )

    runtime = RLM(adapter=adapter, retriever=retriever)
    output, trace = runtime.run("test")  # No context argument

    assert output == "1"


def test_empty_retrieval_context_bootstraps_prompt() -> None:
    """An empty initial retrieval context should push the model toward search."""
    retriever = InMemoryRetriever({"doc1": "Test content"})
    adapter = FakeAdapter(script=["FINAL: done"])

    runtime = RLM(adapter=adapter, retriever=retriever)
    runtime.run("test")

    initial_user_msg = next(m for m in adapter.call_log[0] if m["role"] == "user")["content"]
    assert "Total length: 0 chars" in initial_user_msg
    assert "Number of documents: 0" in initial_user_msg
    assert "current context is empty because no documents have been retrieved yet" in initial_user_msg
    assert "es_hybrid_search()" in initial_user_msg


def test_context_required_without_retriever() -> None:
    """Without retriever, omitting context raises ValueError."""
    adapter = FakeAdapter(script=["FINAL: done"])
    runtime = RLM(adapter=adapter)

    with pytest.raises(ValueError, match="context is required"):
        runtime.run("test")


def test_system_prompt_includes_retrieval_docs() -> None:
    """When retriever is set, system prompt should include retrieval function docs."""
    retriever = InMemoryRetriever()
    adapter = FakeAdapter(script=["FINAL: done"])

    runtime = RLM(adapter=adapter, retriever=retriever)
    runtime.run("test")

    # Check that the adapter received a system prompt with retrieval docs
    last_call = adapter.call_log[-1]
    system_msg = next(m for m in last_call if m["role"] == "system")
    assert "es_search" in system_msg["content"]
    assert "es_hybrid_search" in system_msg["content"]
    assert "es_hybrid_doc_search" in system_msg["content"]
    assert "es_hybrid_search_in_doc" in system_msg["content"]
    assert "es_find_pages" in system_msg["content"]
    assert "es_find_pages_text" in system_msg["content"]
    assert "es_extract_comparative_metric" in system_msg["content"]
    assert "es_get" in system_msg["content"]
    assert "es_get_text" in system_msg["content"]
    assert "page_doc_id" in system_msg["content"]
    assert "logical_doc_id" in system_msg["content"]
    assert "llm_query_json" in system_msg["content"]
    assert "llm_batch_json" in system_msg["content"]
    assert "llm_query_records" in system_msg["content"]
    assert "llm_batch_records" in system_msg["content"]
    assert "llm_extract_comparative_metric" in system_msg["content"]
    assert "parse_amount_text" in system_msg["content"]
    assert '"content"' in system_msg["content"]


def test_system_prompt_excludes_retrieval_docs_without_retriever() -> None:
    """Without retriever, system prompt should NOT include retrieval docs."""
    adapter = FakeAdapter(script=["FINAL: done"])

    runtime = RLM(adapter=adapter)
    runtime.run("test", Context.from_text("some text"))

    last_call = adapter.call_log[-1]
    system_msg = next(m for m in last_call if m["role"] == "system")
    assert "es_search" not in system_msg["content"]


def test_retrieval_with_llm_query_integration() -> None:
    """The model can combine es_search with llm_query for deep analysis."""
    retriever = InMemoryRetriever({
        "doc1": "Contract signed by Alice on Jan 1",
        "doc2": "Contract signed by Bob on Feb 2",
    })

    adapter = FakeAdapter(
        script=[
            (
                'results = es_search("signed")\n'
                'docs = [es_get(r["doc_id"]) for r in results]\n'
                'texts = [d["content"] for d in docs]\n'
                'summary = llm_query("Who signed? " + " | ".join(texts))\n'
                'answer = summary'
            ),
            "FINAL_VAR: answer",
        ]
    )
    adapter.add_rule("You are a sub-LLM", "Alice and Bob signed contracts")

    runtime = RLM(adapter=adapter, retriever=retriever)
    output, trace = runtime.run("Who signed the contracts?")

    assert "Alice and Bob" in output


# ---------------------------------------------------------------------------
# Advanced filter tests
# ---------------------------------------------------------------------------


class TestBuildFilterClauses:
    def test_empty_filters(self) -> None:
        clauses, must_not = _build_filter_clauses(None)
        assert clauses == []
        assert must_not == []

    def test_term_filter(self) -> None:
        clauses, must_not = _build_filter_clauses({"status": "published"})
        assert clauses == [{"term": {"status": "published"}}]
        assert must_not == []

    def test_terms_filter(self) -> None:
        clauses, must_not = _build_filter_clauses({"tags": ["a", "b"]})
        assert clauses == [{"terms": {"tags": ["a", "b"]}}]
        assert must_not == []

    def test_range_filter_gte_lte(self) -> None:
        clauses, _ = _build_filter_clauses({"date": {"gte": "2024-01-01", "lte": "2024-12-31"}})
        assert len(clauses) == 1
        assert clauses[0] == {"range": {"date": {"gte": "2024-01-01", "lte": "2024-12-31"}}}

    def test_range_filter_gt_lt(self) -> None:
        clauses, _ = _build_filter_clauses({"price": {"gt": 10, "lt": 100}})
        assert clauses[0] == {"range": {"price": {"gt": 10, "lt": 100}}}

    def test_exists_filter(self) -> None:
        clauses, _ = _build_filter_clauses({"description": {"exists": True}})
        assert clauses == [{"exists": {"field": "description"}}]

    def test_prefix_filter(self) -> None:
        clauses, _ = _build_filter_clauses({"title": {"prefix": "fin"}})
        assert clauses == [{"prefix": {"title": "fin"}}]

    def test_not_filter_scalar(self) -> None:
        clauses, must_not = _build_filter_clauses({"category": {"not": "draft"}})
        assert clauses == []
        assert must_not == [{"term": {"category": "draft"}}]

    def test_not_filter_list(self) -> None:
        clauses, must_not = _build_filter_clauses({"category": {"not": ["draft", "archived"]}})
        assert clauses == []
        assert must_not == [{"terms": {"category": ["draft", "archived"]}}]

    def test_combined_filters(self) -> None:
        filters = {
            "status": "published",
            "date": {"gte": "2024-01-01"},
            "category": {"not": "draft"},
        }
        clauses, must_not = _build_filter_clauses(filters)
        assert len(clauses) == 2  # term + range
        assert len(must_not) == 1  # not


class TestAdvancedFiltersInSearch:
    def test_range_filter_in_search(self) -> None:
        retriever = _make_es_retriever()
        mock_client = MagicMock()
        mock_client.search.return_value = _mock_search_response()
        retriever._client = mock_client

        retriever.search("query", filters={"date": {"gte": "2024-01-01"}})

        call_body = mock_client.search.call_args[1]["body"]
        filter_clauses = call_body["query"]["bool"]["filter"]
        assert any("range" in c for c in filter_clauses)

    def test_must_not_filter_in_search(self) -> None:
        retriever = _make_es_retriever()
        mock_client = MagicMock()
        mock_client.search.return_value = _mock_search_response()
        retriever._client = mock_client

        retriever.search("query", filters={"category": {"not": "draft"}})

        call_body = mock_client.search.call_args[1]["body"]
        must_not = call_body["query"]["bool"]["must_not"]
        assert must_not == [{"term": {"category": "draft"}}]

    def test_filters_in_knn_pre_filtering(self) -> None:
        """Filters must be applied inside knn clause (pre-filtering)."""
        retriever = _make_es_retriever()
        mock_client = MagicMock()
        mock_client.search.return_value = _mock_search_response()
        retriever._client = mock_client

        with patch.object(retriever, "_embed", return_value=[0.1, 0.2]):
            retriever.vector_search("query", filters={"status": "published"})

        call_body = mock_client.search.call_args[1]["body"]
        knn_filter = call_body["knn"]["filter"]["bool"]
        assert "filter" in knn_filter
        assert any(
            c.get("term", {}).get("status") == "published"
            for c in knn_filter["filter"]
        )

    def test_must_not_in_knn_pre_filtering(self) -> None:
        retriever = _make_es_retriever()
        mock_client = MagicMock()
        mock_client.search.return_value = _mock_search_response()
        retriever._client = mock_client

        with patch.object(retriever, "_embed", return_value=[0.1, 0.2]):
            retriever.vector_search("query", filters={"category": {"not": "draft"}})

        call_body = mock_client.search.call_args[1]["body"]
        knn_filter = call_body["knn"]["filter"]["bool"]
        assert "must_not" in knn_filter
        assert knn_filter["must_not"] == [{"term": {"category": "draft"}}]


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------


class TestEmbeddingCache:
    def test_embedding_cache_hit(self) -> None:
        retriever = _make_es_retriever()
        retriever._client = MagicMock()

        # Manually populate embedding cache
        from pyrlm_runtime.retrieval import _cache_key

        key = _cache_key(
            "hello",
            "openai",
            "text-embedding-3-small",
            "https://api.openai.com/v1",
            True,
        )
        retriever._embedding_cache.set(key, [0.1, 0.2, 0.3])

        # _embed should return cached value without making HTTP call
        result = retriever._embed("hello")
        assert result == [0.1, 0.2, 0.3]

    def test_embedding_cache_disabled(self) -> None:
        retriever = ElasticsearchRetriever(
            host="https://localhost:9200",
            api_key="test-key",
            index="test-index",
            embedding_model="text-embedding-3-small",
            cache_embeddings=False,
        )
        assert retriever._embedding_cache is None

    def test_embedding_cache_stores_after_compute(self) -> None:
        retriever = _make_es_retriever()
        mock_client = MagicMock()
        mock_client.search.return_value = _mock_search_response()
        retriever._client = mock_client

        assert len(retriever._embedding_cache) == 0

        with patch("urllib.request.urlopen") as mock_urlopen:
            import json

            response_data = json.dumps({"data": [{"embedding": [0.5, 0.6]}]}).encode()
            mock_response = MagicMock()
            mock_response.read.return_value = response_data
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = retriever._embed("test text")

        assert result == [0.5, 0.6]
        assert len(retriever._embedding_cache) == 1

        # Second call should hit cache (no HTTP call)
        with patch("urllib.request.urlopen") as mock_urlopen:
            result2 = retriever._embed("test text")
            mock_urlopen.assert_not_called()

        assert result2 == [0.5, 0.6]


class TestResultCache:
    def test_result_cache_disabled_by_default(self) -> None:
        retriever = _make_es_retriever()
        assert retriever._result_cache is None

    def test_result_cache_hit(self) -> None:
        retriever = ElasticsearchRetriever(
            host="https://localhost:9200",
            api_key="test-key",
            index="test-index",
            embedding_model="text-embedding-3-small",
            cache_results=True,
        )
        mock_client = MagicMock()
        mock_client.search.return_value = _mock_search_response(
            _make_hit("doc1", "Hello", 1.0),
        )
        retriever._client = mock_client

        # First call: cache miss → hits ES
        results1 = retriever.search("hello")
        assert mock_client.search.call_count == 1

        # Second call: cache hit → no ES call
        results2 = retriever.search("hello")
        assert mock_client.search.call_count == 1
        assert results1 == results2

    def test_result_cache_different_queries(self) -> None:
        retriever = ElasticsearchRetriever(
            host="https://localhost:9200",
            api_key="test-key",
            index="test-index",
            cache_results=True,
        )
        mock_client = MagicMock()
        mock_client.search.return_value = _mock_search_response()
        retriever._client = mock_client

        retriever.search("query1")
        retriever.search("query2")
        assert mock_client.search.call_count == 2

    def test_result_cache_different_filters(self) -> None:
        retriever = ElasticsearchRetriever(
            host="https://localhost:9200",
            api_key="test-key",
            index="test-index",
            cache_results=True,
        )
        mock_client = MagicMock()
        mock_client.search.return_value = _mock_search_response()
        retriever._client = mock_client

        retriever.search("query", filters={"a": 1})
        retriever.search("query", filters={"a": 2})
        assert mock_client.search.call_count == 2

    def test_clear_cache(self) -> None:
        retriever = ElasticsearchRetriever(
            host="https://localhost:9200",
            api_key="test-key",
            index="test-index",
            embedding_model="text-embedding-3-small",
            cache_results=True,
        )
        mock_client = MagicMock()
        mock_client.search.return_value = _mock_search_response()
        retriever._client = mock_client

        retriever.search("hello")
        assert mock_client.search.call_count == 1

        retriever.clear_cache()

        retriever.search("hello")
        assert mock_client.search.call_count == 2

    def test_get_is_not_cached(self) -> None:
        """es_get() should never be cached — user wants latest document."""
        retriever = ElasticsearchRetriever(
            host="https://localhost:9200",
            api_key="test-key",
            index="test-index",
            cache_results=True,
        )
        mock_client = MagicMock()
        mock_client.get.return_value = {
            "_id": "doc1",
            "_source": {"content": "text"},
        }
        retriever._client = mock_client

        retriever.get("doc1")
        retriever.get("doc1")
        assert mock_client.get.call_count == 2


# ---------------------------------------------------------------------------
# LRU / TTL cache unit tests
# ---------------------------------------------------------------------------


class TestLRUCache:
    def test_basic_get_set(self) -> None:
        from pyrlm_runtime.retrieval import _LRUCache

        cache = _LRUCache(capacity=3)
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") is None

    def test_eviction(self) -> None:
        from pyrlm_runtime.retrieval import _LRUCache

        cache = _LRUCache(capacity=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_access_refreshes_order(self) -> None:
        from pyrlm_runtime.retrieval import _LRUCache

        cache = _LRUCache(capacity=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # refresh "a"
        cache.set("c", 3)  # should evict "b" (not "a")
        assert cache.get("a") == 1
        assert cache.get("b") is None

    def test_clear(self) -> None:
        from pyrlm_runtime.retrieval import _LRUCache

        cache = _LRUCache(capacity=10)
        cache.set("a", 1)
        cache.clear()
        assert len(cache) == 0


class TestTTLCache:
    def test_basic_get_set(self) -> None:
        from pyrlm_runtime.retrieval import _TTLCache

        cache = _TTLCache(capacity=10, ttl=60)
        cache.set("a", 1)
        assert cache.get("a") == 1

    def test_expiration(self) -> None:
        from pyrlm_runtime.retrieval import _TTLCache

        cache = _TTLCache(capacity=10, ttl=0)  # 0-second TTL
        cache.set("a", 1)
        # Entry expired immediately
        import time

        time.sleep(0.01)
        assert cache.get("a") is None

    def test_capacity_eviction(self) -> None:
        from pyrlm_runtime.retrieval import _TTLCache

        cache = _TTLCache(capacity=2, ttl=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # evicts oldest
        assert len(cache) == 2

    def test_clear(self) -> None:
        from pyrlm_runtime.retrieval import _TTLCache

        cache = _TTLCache(capacity=10, ttl=60)
        cache.set("a", 1)
        cache.clear()
        assert len(cache) == 0


# ---------------------------------------------------------------------------
# Async retriever tests
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine synchronously (no pytest-asyncio needed)."""
    import asyncio

    return asyncio.run(coro)


class TestAsyncElasticsearchRetriever:
    def test_satisfies_async_protocol(self) -> None:
        assert issubclass(AsyncElasticsearchRetriever, AsyncRetrieverProtocol)

    def test_async_search(self) -> None:
        async def _test():
            retriever = AsyncElasticsearchRetriever(
                host="https://localhost:9200",
                api_key="test-key",
                index="test-index",
            )
            mock_client = AsyncMock()
            mock_client.search.return_value = _mock_search_response(
                _make_hit("doc1", "Hello world", 2.0),
            )
            retriever._client = mock_client

            results = await retriever.search("world")

            assert len(results) == 1
            assert results[0]["doc_id"] == "doc1"
            mock_client.search.assert_called_once()

        _run(_test())

    def test_async_vector_search(self) -> None:
        async def _test():
            retriever = AsyncElasticsearchRetriever(
                host="https://localhost:9200",
                api_key="test-key",
                index="test-index",
                embedding_model="text-embedding-3-small",
            )
            mock_client = AsyncMock()
            mock_client.search.return_value = _mock_search_response(
                _make_hit("doc1", "semantic", 0.9),
            )
            retriever._client = mock_client

            async def fake_embed(text):
                return [0.1, 0.2]

            retriever._embed = fake_embed
            results = await retriever.vector_search("query")

            assert len(results) == 1
            call_body = mock_client.search.call_args[1]["body"]
            assert "knn" in call_body

        _run(_test())

    def test_async_hybrid_search(self) -> None:
        async def _test():
            retriever = AsyncElasticsearchRetriever(
                host="https://localhost:9200",
                api_key="test-key",
                index="test-index",
                embedding_model="text-embedding-3-small",
            )
            mock_client = AsyncMock()
            mock_client.search.return_value = _mock_search_response(
                _make_hit("doc1", "hybrid", 0.8),
            )
            retriever._client = mock_client

            async def fake_embed(text):
                return [0.1, 0.2]

            retriever._embed = fake_embed
            results = await retriever.hybrid_search("query")

            assert len(results) == 1
            call_body = mock_client.search.call_args[1]["body"]
            assert "rank" in call_body
            assert "rrf" in call_body["rank"]

        _run(_test())

    def test_async_hybrid_search_falls_back_to_local_rrf(self) -> None:
        async def _test():
            retriever = AsyncElasticsearchRetriever(
                host="https://localhost:9200",
                api_key="test-key",
                index="test-index",
                embedding_model="text-embedding-3-small",
            )
            mock_client = AsyncMock()
            mock_client.search.side_effect = [
                RuntimeError(
                    "AuthorizationException(403, 'security_exception', "
                    "'current license is non-compliant for [Reciprocal Rank Fusion (RRF)]')"
                ),
                _mock_search_response(
                    _make_hit("doc-bm25", "lexical result", 8.0),
                    _make_hit("doc-shared", "shared result", 7.0),
                ),
                _mock_search_response(
                    _make_hit("doc-vector", "semantic result", 0.9),
                    _make_hit("doc-shared", "shared result", 0.8),
                ),
            ]
            retriever._client = mock_client

            async def fake_embed(text):
                return [0.1, 0.2]

            retriever._embed = fake_embed
            results = await retriever.hybrid_search("query")

            assert [result["doc_id"] for result in results] == [
                "doc-shared",
                "doc-bm25",
                "doc-vector",
            ]
            assert retriever._supports_rrf is False
            assert mock_client.search.call_count == 3

        _run(_test())

    def test_async_get(self) -> None:
        async def _test():
            retriever = AsyncElasticsearchRetriever(
                host="https://localhost:9200",
                api_key="test-key",
                index="test-index",
            )
            mock_client = AsyncMock()
            mock_client.get.return_value = {
                "_id": "doc1",
                "_source": {"content": "Full text", "title": "T1"},
            }
            retriever._client = mock_client

            result = await retriever.get("doc1")

            assert result["doc_id"] == "doc1"
            assert result["content"] == "Full text"
            assert result["metadata"] == {"title": "T1"}

        _run(_test())

    def test_async_context_manager(self) -> None:
        async def _test():
            async with AsyncElasticsearchRetriever(
                host="https://localhost:9200",
                api_key="test-key",
                index="test-index",
            ) as retriever:
                assert retriever is not None

        _run(_test())

    def test_async_result_cache(self) -> None:
        async def _test():
            retriever = AsyncElasticsearchRetriever(
                host="https://localhost:9200",
                api_key="test-key",
                index="test-index",
                cache_results=True,
            )
            mock_client = AsyncMock()
            mock_client.search.return_value = _mock_search_response(
                _make_hit("doc1", "cached", 1.0),
            )
            retriever._client = mock_client

            await retriever.search("hello")
            await retriever.search("hello")
            assert mock_client.search.call_count == 1

        _run(_test())

    def test_async_filters(self) -> None:
        async def _test():
            retriever = AsyncElasticsearchRetriever(
                host="https://localhost:9200",
                api_key="test-key",
                index="test-index",
            )
            mock_client = AsyncMock()
            mock_client.search.return_value = _mock_search_response()
            retriever._client = mock_client

            await retriever.search("q", filters={"date": {"gte": "2024-01-01"}})

            call_body = mock_client.search.call_args[1]["body"]
            filter_clauses = call_body["query"]["bool"]["filter"]
            assert any("range" in c for c in filter_clauses)

        _run(_test())


# ---------------------------------------------------------------------------
# Async retriever rejection
# ---------------------------------------------------------------------------


def test_rlm_rejects_async_retriever() -> None:
    """RLM.run() should raise TypeError when given an async retriever."""
    retriever = AsyncElasticsearchRetriever(
        host="https://localhost:9200",
        api_key="test-key",
        index="test-index",
    )
    adapter = FakeAdapter(script=["FINAL: done"])
    runtime = RLM(adapter=adapter, retriever=retriever)

    with pytest.raises(TypeError, match="async retriever"):
        runtime.run("test")


# ---------------------------------------------------------------------------
# Cache validation
# ---------------------------------------------------------------------------


def test_invalid_embedding_cache_size_raises() -> None:
    with pytest.raises(ValueError, match="embedding_cache_size"):
        ElasticsearchRetriever(
            host="https://localhost:9200",
            api_key="test-key",
            index="test-index",
            cache_embeddings=True,
            embedding_cache_size=0,
        )


def test_invalid_result_cache_size_raises() -> None:
    with pytest.raises(ValueError, match="result_cache_size"):
        ElasticsearchRetriever(
            host="https://localhost:9200",
            api_key="test-key",
            index="test-index",
            cache_results=True,
            result_cache_size=0,
        )


def test_invalid_result_cache_ttl_raises() -> None:
    with pytest.raises(ValueError, match="result_cache_ttl"):
        ElasticsearchRetriever(
            host="https://localhost:9200",
            api_key="test-key",
            index="test-index",
            cache_results=True,
            result_cache_ttl=0,
        )


def test_async_invalid_embedding_cache_size_raises() -> None:
    with pytest.raises(ValueError, match="embedding_cache_size"):
        AsyncElasticsearchRetriever(
            host="https://localhost:9200",
            api_key="test-key",
            index="test-index",
            cache_embeddings=True,
            embedding_cache_size=-1,
        )


def test_async_invalid_result_cache_ttl_raises() -> None:
    with pytest.raises(ValueError, match="result_cache_ttl"):
        AsyncElasticsearchRetriever(
            host="https://localhost:9200",
            api_key="test-key",
            index="test-index",
            cache_results=True,
            result_cache_ttl=-5,
        )


# ---------------------------------------------------------------------------
# Sensitive fields not in repr
# ---------------------------------------------------------------------------


def test_api_key_not_in_repr() -> None:
    retriever = ElasticsearchRetriever(
        host="https://localhost:9200",
        api_key="super-secret-key",
        index="test-index",
        embedding_api_key="another-secret",
    )
    r = repr(retriever)
    assert "super-secret-key" not in r
    assert "another-secret" not in r


# ---------------------------------------------------------------------------
# Monty REPL retrieval tests
# ---------------------------------------------------------------------------

try:
    from pyrlm_runtime.env_monty import MONTY_AVAILABLE
except ImportError:
    MONTY_AVAILABLE = False


@pytest.mark.skipif(not MONTY_AVAILABLE, reason="pydantic-monty not installed")
def test_retrieval_functions_registered_in_repl_monty() -> None:
    """When a retriever is set, es_* functions are available in the Monty REPL."""
    retriever = InMemoryRetriever({"doc1": "Hello world"})
    adapter = FakeAdapter(
        script=[
            'results = es_search("Hello")\nprint(len(results))',
            "FINAL: 1",
        ]
    )

    runtime = RLM(adapter=adapter, retriever=retriever, repl_backend="monty")
    output, trace = runtime.run("test", Context.from_text(""))

    assert output == "1"


@pytest.mark.skipif(not MONTY_AVAILABLE, reason="pydantic-monty not installed")
def test_es_get_in_repl_monty() -> None:
    """es_get() should retrieve full document content in the Monty REPL."""
    retriever = InMemoryRetriever({"doc1": "Full document content"})
    adapter = FakeAdapter(
        script=[
            'doc = es_get("doc1")\nprint(doc["content"])',
            "FINAL_VAR: doc",
        ]
    )

    runtime = RLM(adapter=adapter, retriever=retriever, repl_backend="monty")
    output, trace = runtime.run("test", Context.from_text(""))

    assert "Full document content" in output
