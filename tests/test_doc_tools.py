"""Tests for doctools/tools.py — create_doc_tools, resolve_doc_path, search_corpus_structured,
and i18n label support."""

from __future__ import annotations

import pytest

from pyrlm_runtime.doctools import DEFAULT_LABELS, create_doc_tools
from pyrlm_runtime.doctools.cache import DocumentCache
from pyrlm_runtime.doctools.policy import DocumentPolicy
from pyrlm_runtime.doctools.schema import DocInfo, PageInfo


# ---------------------------------------------------------------------------
# Minimal mock implementations
# ---------------------------------------------------------------------------


def _make_doc(
    doc_id: str = "doc1",
    path: str = "/pdfs/doc1.pdf",
    title: str = "Annual Report 2024",
    is_scanned_prob: float = 0.0,
    summary: str = "Annual financial report for 2024.",
) -> DocInfo:
    return DocInfo(
        doc_id=doc_id,
        path=path,
        file_hash="abc123",
        file_size=100_000,
        modified_at="2024-01-01T00:00:00Z",
        title=title,
        num_pages=10,
        language="en",
        is_scanned_prob=is_scanned_prob,
        has_tables_prob=0.8,
        parser_hint="pymupdf4llm",
        summary=summary,
        pages=[
            PageInfo(
                page_num=0,
                char_count=2000,
                text_extractable=True,
                headings=["Balance Sheet"],
                table_count=2,
                flags=[],
            )
        ],
    )


class FakeIndexStore:
    def __init__(self, docs: list[DocInfo]) -> None:
        self._docs = {d.doc_id: d for d in docs}
        self._by_path = {d.path: d for d in docs}

    def search(self, query: str, top_k: int = 10) -> list[DocInfo]:
        return list(self._docs.values())[:top_k]

    def get(self, doc_id: str) -> DocInfo | None:
        return self._docs.get(doc_id)

    def get_by_path(self, path: str) -> DocInfo | None:
        return self._by_path.get(path)

    def list_all(self) -> list[DocInfo]:
        return list(self._docs.values())


class FakeIndexStoreEmpty(FakeIndexStore):
    def __init__(self) -> None:
        super().__init__([])

    def search(self, query: str, top_k: int = 10) -> list[DocInfo]:
        return []


class FakePageReader:
    def read_pages(self, path: str, pages: list[int]) -> dict[int, str]:
        return {p: f"Content of page {p}" for p in pages}

    def extract_table(self, path: str, page: int, table_num: int) -> str:
        return "| col1 | col2 |\n| val1 | val2 |"


def _make_tools(tmp_path, docs: list[DocInfo] | None = None, labels: dict | None = None):
    store = FakeIndexStore(docs or [_make_doc()])
    reader = FakePageReader()
    cache = DocumentCache(cache_dir=tmp_path)
    policy = DocumentPolicy()
    return create_doc_tools(store, reader, cache, policy, labels=labels)


def _make_empty_tools(tmp_path, labels: dict | None = None):
    store = FakeIndexStoreEmpty()
    reader = FakePageReader()
    cache = DocumentCache(cache_dir=tmp_path)
    policy = DocumentPolicy()
    return create_doc_tools(store, reader, cache, policy, labels=labels)


# ---------------------------------------------------------------------------
# resolve_doc_path tests
# ---------------------------------------------------------------------------


class TestResolveDocPath:
    def test_exact_path_match(self, tmp_path):
        tools = _make_tools(tmp_path)
        result = tools["resolve_doc_path"]("/pdfs/doc1.pdf")
        assert result == "/pdfs/doc1.pdf"

    def test_exact_doc_id_match(self, tmp_path):
        tools = _make_tools(tmp_path)
        result = tools["resolve_doc_path"]("doc1")
        assert result == "/pdfs/doc1.pdf"

    def test_search_fallback(self, tmp_path):
        # When exact match fails, falls back to search (FakeIndexStore always returns docs)
        tools = _make_tools(tmp_path)
        result = tools["resolve_doc_path"]("Annual Report")
        assert result == "/pdfs/doc1.pdf"

    def test_not_found_returns_error_string(self, tmp_path):
        tools = _make_empty_tools(tmp_path)
        result = tools["resolve_doc_path"]("nonexistent document")
        assert isinstance(result, str)
        assert "nonexistent document" in result

    def test_result_is_string_not_dict(self, tmp_path):
        """Regression: search_corpus() returning a string caused AttributeError when
        the LLM called .get("path") on it. resolve_doc_path() is the correct alternative."""
        tools = _make_tools(tmp_path)
        result = tools["resolve_doc_path"]("Annual Report")
        assert isinstance(result, str)
        # Confirm the old bug: .get() on search_corpus() result raises AttributeError
        corpus_result = tools["search_corpus"]("Annual Report")
        with pytest.raises(AttributeError):
            corpus_result.get("path")  # type: ignore[union-attr]
        # But resolve_doc_path returns a usable path directly
        assert result == "/pdfs/doc1.pdf"


# ---------------------------------------------------------------------------
# search_corpus_structured tests
# ---------------------------------------------------------------------------


class TestSearchCorpusStructured:
    def test_returns_list_of_dicts(self, tmp_path):
        tools = _make_tools(tmp_path)
        result = tools["search_corpus_structured"]("annual report")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)

    def test_dict_has_required_keys(self, tmp_path):
        tools = _make_tools(tmp_path)
        result = tools["search_corpus_structured"]("annual report")
        doc = result[0]
        for key in ("title", "path", "doc_id", "num_pages", "summary"):
            assert key in doc

    def test_path_accessible_via_get(self, tmp_path):
        tools = _make_tools(tmp_path)
        result = tools["search_corpus_structured"]("annual report")
        path = result[0].get("path")
        assert path == "/pdfs/doc1.pdf"

    def test_empty_results(self, tmp_path):
        tools = _make_empty_tools(tmp_path)
        result = tools["search_corpus_structured"]("nothing")
        assert result == []

    def test_summary_truncated_to_200_chars(self, tmp_path):
        doc = _make_doc(summary="x" * 500)
        tools = _make_tools(tmp_path, docs=[doc])
        result = tools["search_corpus_structured"]("anything")
        assert len(result[0]["summary"]) == 200


# ---------------------------------------------------------------------------
# i18n / labels tests
# ---------------------------------------------------------------------------


class TestLabels:
    def test_default_labels_are_english(self, tmp_path):
        tools = _make_empty_tools(tmp_path)
        result = tools["list_pdfs"]()
        assert "No documents indexed" in result

    def test_custom_labels_override(self, tmp_path):
        custom = {"no_docs_indexed": "Keine Dokumente vorhanden."}
        tools = _make_empty_tools(tmp_path, labels=custom)
        result = tools["list_pdfs"]()
        assert "Keine Dokumente vorhanden." in result

    def test_partial_labels_merge_with_defaults(self, tmp_path):
        # Only override "scanned" — rest should still be English
        custom = {"scanned": "gescannt"}
        doc = _make_doc(is_scanned_prob=0.9)
        tools = _make_tools(tmp_path, docs=[doc], labels=custom)
        result = tools["list_pdfs"]()
        assert "gescannt" in result
        # English default for other strings still in effect
        assert "Available documents" in result

    def test_default_labels_dict_exported(self):
        assert isinstance(DEFAULT_LABELS, dict)
        assert "no_docs_indexed" in DEFAULT_LABELS
        assert "scanned" in DEFAULT_LABELS
        assert DEFAULT_LABELS["scanned"] == "scanned"
        assert DEFAULT_LABELS["digital"] == "digital"


# ---------------------------------------------------------------------------
# Return type contract tests
# ---------------------------------------------------------------------------


class TestReturnTypes:
    def test_list_pdfs_returns_string(self, tmp_path):
        tools = _make_tools(tmp_path)
        result = tools["list_pdfs"]()
        assert isinstance(result, str)

    def test_search_corpus_returns_string(self, tmp_path):
        tools = _make_tools(tmp_path)
        result = tools["search_corpus"]("report")
        assert isinstance(result, str)

    def test_search_corpus_structured_returns_list(self, tmp_path):
        tools = _make_tools(tmp_path)
        result = tools["search_corpus_structured"]("report")
        assert isinstance(result, list)

    def test_resolve_doc_path_always_returns_string(self, tmp_path):
        tools = _make_tools(tmp_path)
        result = tools["resolve_doc_path"]("anything")
        assert isinstance(result, str)

    def test_all_eight_tools_present(self, tmp_path):
        tools = _make_tools(tmp_path)
        expected = {
            "list_pdfs", "get_pdf_info", "read_pdf_pages", "extract_table",
            "search_in_pdf", "search_corpus", "search_corpus_structured", "resolve_doc_path",
        }
        assert expected.issubset(set(tools.keys()))
