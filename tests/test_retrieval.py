"""Tests for the retrieval module (RetrieverProtocol, ElasticsearchRetriever, REPL integration)."""

from __future__ import annotations

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
    assert "es_get" in system_msg["content"]


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

        key = _cache_key("hello", "text-embedding-3-small")
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
            import io
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
