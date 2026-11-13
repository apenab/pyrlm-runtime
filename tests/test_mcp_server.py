from __future__ import annotations

import pytest

pytest.importorskip("mcp")

from _mcp_fakes import FailingRetriever, InMemoryRetriever, PagedRetriever  # noqa: E402
from mcp import Client  # noqa: E402

from pyrlm_runtime.mcp_server import ConfigError, ServerConfig, build_server  # noqa: E402

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _client(retriever, **config):
    return Client(build_server(retriever=retriever, config=ServerConfig(**config)))


def _error_text(result) -> str:
    assert result.is_error is True
    return result.content[0].text


class TestCapabilities:
    def test_nothing_configured_refuses_to_build(self):
        with pytest.raises(ConfigError, match="nothing to publish"):
            build_server(retriever=None)

    async def test_plain_retriever_publishes_search_and_get_document(self):
        async with _client(InMemoryRetriever()) as client:
            tools = (await client.list_tools()).tools
            instructions = client.instructions

        assert sorted(tool.name for tool in tools) == ["get_document", "search"]
        assert "get_pages" not in instructions
        assert "untrusted data" in instructions

    async def test_retriever_with_get_pages_also_publishes_get_pages(self):
        async with _client(PagedRetriever()) as client:
            tools = (await client.list_tools()).tools
            instructions = client.instructions

        assert sorted(tool.name for tool in tools) == ["get_document", "get_pages", "search"]
        assert "get_pages" in instructions

    async def test_every_tool_is_read_only_idempotent_with_an_output_schema(self):
        async with _client(PagedRetriever()) as client:
            tools = (await client.list_tools()).tools

        for tool in tools:
            assert tool.annotations.read_only_hint is True
            assert tool.annotations.idempotent_hint is True
            assert tool.output_schema is not None

    async def test_search_description_names_only_configured_modes(self):
        async with _client(InMemoryRetriever(), search_modes=("lexical",)) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}

        description = tools["search"].description
        assert "`lexical`" in description
        assert "`semantic`" not in description
        assert "`hybrid`" not in description


class TestSearchTool:
    async def test_returns_structured_hits_without_content_by_default(self):
        async with _client(PagedRetriever()) as client:
            result = await client.call_tool("search", {"query": "revenue", "top_k": 2})

        assert result.is_error is False
        payload = result.structured_content
        assert payload["top_k_applied"] == 2
        assert payload["mode"] == "hybrid"
        assert len(payload["results"]) == 2
        assert all(hit["content"] is None for hit in payload["results"])

    async def test_include_content_returns_text(self):
        async with _client(PagedRetriever()) as client:
            result = await client.call_tool("search", {"query": "north", "include_content": True})

        assert result.structured_content["results"][0]["content"].startswith("Beta memo")

    async def test_unavailable_mode_is_a_tool_error_listing_available_modes(self):
        async with _client(InMemoryRetriever(), search_modes=("lexical",)) as client:
            result = await client.call_tool("search", {"query": "revenue", "mode": "semantic"})

        assert "available modes: lexical" in _error_text(result)

    async def test_invalid_argument_type_is_a_tool_error(self):
        async with _client(InMemoryRetriever()) as client:
            result = await client.call_tool("search", {"query": "revenue", "top_k": "many"})

        assert "top_k" in _error_text(result)

    async def test_backend_failure_is_sanitised(self):
        async with _client(FailingRetriever()) as client:
            result = await client.call_tool("search", {"query": "revenue"})

        text = _error_text(result)
        assert "ConnectionError" in text
        assert "es-internal" not in text
        assert "SECRET" not in text


class TestReadTools:
    async def test_get_document_pages_through_long_text(self):
        docs = [{"doc_id": "d", "content": "y" * 30, "metadata": {}}]
        async with _client(InMemoryRetriever(docs), max_document_chars=20) as client:
            first = await client.call_tool("get_document", {"doc_id": "d"})
            second = await client.call_tool(
                "get_document", {"doc_id": "d", "offset": first.structured_content["next_offset"]}
            )

        assert first.structured_content["next_offset"] == 20
        assert second.structured_content["content"] == "y" * 10
        assert second.structured_content["next_offset"] is None

    async def test_get_document_unknown_id_is_a_tool_error(self):
        async with _client(InMemoryRetriever()) as client:
            result = await client.call_tool("get_document", {"doc_id": "missing"})

        assert "not found" in _error_text(result)

    async def test_get_pages_with_a_hit_doc_id_explains_which_id_to_use(self):
        async with _client(PagedRetriever()) as client:
            hit = (await client.call_tool("search", {"query": "growth"})).structured_content[
                "results"
            ][0]
            wrong = await client.call_tool(
                "get_pages", {"logical_doc_id": hit["doc_id"], "pages": [1]}
            )
            right = await client.call_tool(
                "get_pages", {"logical_doc_id": hit["logical_doc_id"], "pages": [hit["page_num"]]}
            )

        assert "not its doc_id" in _error_text(wrong)
        assert right.is_error is False
        assert [page["doc_id"] for page in right.structured_content["pages"]] == ["a-p1"]
