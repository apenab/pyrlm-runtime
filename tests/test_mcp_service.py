from __future__ import annotations

import copy

import pytest

pytest.importorskip("mcp")

from _mcp_fakes import InMemoryRetriever, PagedRetriever  # noqa: E402

from pyrlm_runtime.mcp_server.config import ServerConfig  # noqa: E402
from pyrlm_runtime.mcp_server.service import (  # noqa: E402
    InvalidRequest,
    NotFound,
    RetrievalService,
    ScopedRetriever,
)


def _service(retriever=None, **config) -> RetrievalService:
    return RetrievalService(retriever or PagedRetriever(), ServerConfig(**config))


class TestSearch:
    def test_top_k_is_clamped_and_reported(self):
        retriever = InMemoryRetriever()
        service = _service(retriever, max_top_k=3)

        response = service.search("revenue", top_k=500)

        assert response.top_k_applied == 3
        assert retriever.calls[-1][1]["top_k"] == 3
        assert len(response.results) == 3
        assert service.search("revenue", top_k=0).top_k_applied == 1

    def test_hits_carry_preview_but_no_content_by_default(self):
        response = _service().search("revenue")

        hit = response.results[0]
        assert hit.preview
        assert hit.content is None
        assert hit.logical_doc_id in {"a", "b"}

    def test_include_content_truncates_without_mutating_retriever_hits(self):
        long_text = "revenue " * 100
        retriever = InMemoryRetriever([{"doc_id": "x", "content": long_text, "metadata": {}}])
        before = copy.deepcopy(retriever.search("revenue"))

        response = _service(retriever, max_content_chars=50).search("revenue", include_content=True)

        hit = response.results[0]
        assert hit.content == long_text[:50]
        assert hit.content_truncated is True
        assert retriever.search("revenue") == before

    def test_response_cap_drops_hits_from_the_tail(self):
        docs = [{"doc_id": f"d{i}", "content": "revenue", "metadata": {}} for i in range(10)]
        service = _service(InMemoryRetriever(docs), max_response_chars=500)

        response = service.search("revenue", top_k=10)

        assert response.truncated is True
        assert 0 < len(response.results) < 10
        assert [hit.doc_id for hit in response.results] == [
            f"d{i}" for i in range(len(response.results))
        ]

    def test_default_mode_is_the_first_configured_mode(self):
        retriever = InMemoryRetriever()
        response = _service(retriever, search_modes=("lexical", "hybrid")).search("revenue")

        assert response.mode == "lexical"
        assert retriever.calls[-1][0] == "search"

    def test_mode_maps_to_the_retriever_method(self):
        retriever = InMemoryRetriever()
        service = _service(retriever)

        for mode, method in (
            ("hybrid", "hybrid_search"),
            ("lexical", "search"),
            ("semantic", "vector_search"),
        ):
            service.search("revenue", mode=mode)
            assert retriever.calls[-1][0] == method

    def test_unavailable_mode_lists_available_modes(self):
        service = _service(search_modes=("lexical",))

        with pytest.raises(InvalidRequest, match="available modes: lexical"):
            service.search("revenue", mode="semantic")

    def test_empty_query_is_rejected(self):
        with pytest.raises(InvalidRequest):
            _service().search("   ")

    def test_denylisted_filter_key_is_rejected(self):
        service = _service(metadata_denylist=("secret_field",))

        with pytest.raises(InvalidRequest, match="secret_field"):
            service.search("revenue", filters={"secret_field": {"prefix": "12"}})

    def test_filterable_fields_restrict_filter_keys(self):
        retriever = InMemoryRetriever()
        service = _service(retriever, filterable_fields=("doc_id",))

        with pytest.raises(InvalidRequest, match="page_num"):
            service.search("revenue", filters={"page_num": 1})
        response = service.search("revenue", filters={"doc_id": "b"})
        assert [hit.doc_id for hit in response.results] == ["b-p1"]

    def test_denylisted_metadata_is_removed_from_hits(self):
        docs = [{"doc_id": "x", "content": "revenue", "metadata": {"ssn": "1", "year": 2020}}]
        retriever = InMemoryRetriever(docs)

        response = _service(retriever, metadata_denylist=("ssn",)).search("revenue")

        assert response.results[0].metadata == {"year": 2020}
        assert retriever.search("revenue")[0]["metadata"] == {"ssn": "1", "year": 2020}


class TestScopedRetriever:
    def test_base_filters_win_over_client_filters(self):
        retriever = InMemoryRetriever()
        scoped = ScopedRetriever(retriever, base_filters={"doc_id": "a"}, max_top_k=5)

        scoped.search("revenue", top_k=99, filters={"doc_id": "b", "page_num": 1})

        assert retriever.calls[-1][1] == {
            "query": "revenue",
            "top_k": 5,
            "filters": {"doc_id": "a", "page_num": 1},
        }

    def test_base_filters_apply_through_the_service(self):
        response = _service(base_filters={"doc_id": "b"}).search("revenue", filters={"doc_id": "a"})

        assert {hit.logical_doc_id for hit in response.results} == {"b"}


class TestGetDocument:
    def test_slices_long_documents(self):
        text = "x" * 25
        retriever = InMemoryRetriever([{"doc_id": "d", "content": text, "metadata": {}}])
        service = _service(retriever, max_document_chars=10)

        first = service.get_document("d")
        last = service.get_document("d", offset=20)

        assert (first.content, first.next_offset, first.total_chars) == ("x" * 10, 10, 25)
        assert (last.content, last.next_offset) == ("x" * 5, None)

    def test_max_chars_and_offset_are_clamped(self):
        service = _service(max_document_chars=10)

        response = service.get_document("a-p1", offset=-5, max_chars=1000)

        assert response.offset == 0
        assert response.max_chars_applied == 10

    def test_unknown_id_is_not_found(self):
        with pytest.raises(NotFound, match="nope"):
            _service().get_document("nope")


class TestGetPages:
    def test_reads_pages_of_a_logical_document(self):
        response = _service().get_pages("a", pages=[2], radius=1)

        assert [page.page_num for page in response.pages] == [1, 2, 3]
        assert response.pages[0].doc_id == "a-p1"
        assert response.radius_applied == 1

    def test_hit_doc_id_instead_of_logical_id_explains_the_mistake(self):
        with pytest.raises(NotFound, match="logical_doc_id of a search hit, not its doc_id"):
            _service().get_pages("a-p1", pages=[1])

    def test_radius_is_clamped_to_max_pages(self):
        response = _service(max_pages=2).get_pages("a", pages=[1], radius=50)

        assert response.radius_applied == 2
        assert len(response.pages) == 2

    def test_empty_page_list_is_rejected(self):
        with pytest.raises(InvalidRequest):
            _service().get_pages("a", pages=[])

    def test_retriever_without_get_pages_is_rejected(self):
        service = _service(InMemoryRetriever())

        assert service.supports_pages is False
        with pytest.raises(InvalidRequest):
            service.get_pages("a", pages=[1])


class TestDenylistReach:
    def test_subfield_and_parent_of_a_denied_field_are_rejected_as_filters(self):
        service = _service(metadata_denylist=("ssn", "owner.tax_id"))

        for key in ("ssn.keyword", "owner"):
            with pytest.raises(InvalidRequest, match="not allowed"):
                service.search("revenue", filters={key: {"prefix": "1"}})

    def test_dotted_entry_removes_nested_metadata_everywhere(self):
        docs = [
            {
                "doc_id": "x-p1",
                "content": "revenue",
                "metadata": {"doc_id": "x", "page_num": 1, "owner": {"tax_id": "9", "name": "Ann"}},
            }
        ]
        retriever = PagedRetriever(docs)
        service = _service(retriever, metadata_denylist=("owner.tax_id",))

        outputs = [
            service.search("revenue").results[0].metadata,
            service.get_document("x-p1").metadata,
            service.get_pages("x", pages=[1]).pages[0].metadata,
        ]

        for metadata in outputs:
            assert metadata["owner"] == {"name": "Ann"}
        assert retriever.search("revenue")[0]["metadata"]["owner"]["tax_id"] == "9"


class TestGetPagesLimits:
    def test_pages_are_truncated_and_the_response_is_capped(self):
        docs = [
            {"doc_id": f"d-p{n}", "content": "z" * 300, "metadata": {"doc_id": "d", "page_num": n}}
            for n in range(1, 6)
        ]
        service = _service(PagedRetriever(docs), max_document_chars=100, max_response_chars=400)

        response = service.get_pages("d", pages=[3], radius=2)

        assert all(page.content == "z" * 100 and page.content_truncated for page in response.pages)
        assert response.truncated is True
        assert 0 < len(response.pages) < 5

    @pytest.mark.parametrize("page", [0, -3, 1_000_001, 10**30])
    def test_page_numbers_outside_the_valid_range_are_rejected(self, page):
        retriever = PagedRetriever()

        with pytest.raises(InvalidRequest, match="between 1 and 1000000"):
            _service(retriever).get_pages("a", pages=[1, page])

    def test_more_page_numbers_than_max_pages_is_rejected(self):
        with pytest.raises(InvalidRequest, match="between 1 and 2"):
            _service(max_pages=2).get_pages("a", pages=[1, 2, 3])


class TestWildcardFilterKeys:
    @pytest.mark.parametrize("key", ["ss*", "*", "owner.*", "*.keyword"])
    def test_a_key_that_can_name_several_fields_is_rejected(self, key):
        service = _service(metadata_denylist=("ssn",))

        with pytest.raises(InvalidRequest, match="wildcard"):
            service.search("revenue", filters={key: {"exists": True}})


class TestUnboundedRetrieverFields:
    def test_a_long_retriever_preview_is_capped(self):
        docs = [{"doc_id": "x", "content": "revenue", "metadata": {}}]
        retriever = InMemoryRetriever(docs)
        retriever._hits["x"]["preview"] = "p" * 5000

        hit = _service(retriever, max_preview_chars=100).search("revenue").results[0]

        assert hit.preview == "p" * 100
        assert retriever._hits["x"]["preview"] == "p" * 5000

    def test_get_document_leaves_out_metadata_that_breaks_the_response_cap(self):
        docs = [{"doc_id": "x", "content": "short", "metadata": {"blob": "m" * 5000}}]
        service = _service(InMemoryRetriever(docs), max_response_chars=1000)

        response = service.get_document("x")

        assert (response.content, response.metadata, response.truncated) == ("short", {}, True)

    def test_get_document_keeps_metadata_that_fits(self):
        response = _service().get_document("a-p1")

        assert response.metadata == {"doc_id": "a", "page_num": 1}
        assert response.truncated is False
