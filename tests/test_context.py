import pytest

from pyrlm_runtime import Context


def test_len_and_slice() -> None:
    ctx = Context.from_text("hello world")
    assert ctx.len_chars() == 11
    assert ctx.slice(0, 5) == "hello"
    assert ctx.slice(-5, 5) == "hello"
    assert ctx.slice(6, 50) == "world"
    assert ctx.slice(10, 5) == ""


def test_find_literal_and_regex() -> None:
    ctx = Context.from_text("aba abb aab")
    assert ctx.find("ab")[:2] == [(0, 2, "ab"), (4, 6, "ab")]
    matches = ctx.find(r"a.b", regex=True, max_matches=10)
    assert matches[0][2] == "abb"


def test_chunking() -> None:
    ctx = Context.from_text("abcdefghij")
    chunks = ctx.chunk(4, overlap=1)
    assert chunks[0] == (0, 4, "abcd")
    assert chunks[1] == (3, 7, "defg")
    assert chunks[-1][2] == "ghij"

    with pytest.raises(ValueError):
        ctx.chunk(0)
    with pytest.raises(ValueError):
        ctx.chunk(4, overlap=4)


def test_from_documents() -> None:
    """Test creating context from a list of documents."""
    docs = ["Document 1 content", "Document 2 content", "Document 3 content"]
    ctx = Context.from_documents(docs)

    assert ctx.context_type == "document_list"
    assert ctx.num_documents() == 3
    assert ctx.get_document(0) == "Document 1 content"
    assert ctx.get_document(1) == "Document 2 content"
    assert ctx.get_document(2) == "Document 3 content"
    # Text should contain all documents with separator
    assert "Document 1 content" in ctx.text
    assert "Document 2 content" in ctx.text
    assert "Document 3 content" in ctx.text


def test_chunk_documents() -> None:
    """Test chunking by documents rather than characters."""
    docs = ["Doc A", "Doc B", "Doc C", "Doc D", "Doc E"]
    ctx = Context.from_documents(docs)

    chunks = ctx.chunk_documents(docs_per_chunk=2)
    assert len(chunks) == 3  # 2 + 2 + 1
    assert chunks[0] == (0, 2, ["Doc A", "Doc B"])
    assert chunks[1] == (2, 4, ["Doc C", "Doc D"])
    assert chunks[2] == (4, 5, ["Doc E"])


def test_metadata() -> None:
    """Test metadata generation for context."""
    # String context
    ctx = Context.from_text("Hello world")
    meta = ctx.metadata()
    assert meta["context_type"] == "string"
    assert meta["total_length"] == 11
    assert meta["num_documents"] == 1
    assert "document_lengths" not in meta

    # Document list context
    docs = ["Short", "A bit longer document"]
    ctx_docs = Context.from_documents(docs)
    meta_docs = ctx_docs.metadata()
    assert meta_docs["context_type"] == "document_list"
    assert meta_docs["num_documents"] == 2
    assert meta_docs["document_lengths"] == [5, 21]


# ── ctx.page() tests ────────────────────────────────────────────────


_SAMPLE_DOC = (
    "Header text\n"
    "<!-- Page 1 -->\nPage one content here.\n"
    "<!-- Page 2 -->\nPage two content here.\n"
    "<!-- Page 3 -->\nPage three — the last page.\n"
)


def test_page_basic_extraction() -> None:
    ctx = Context.from_text(_SAMPLE_DOC)
    assert ctx.page(1) == "\nPage one content here.\n"
    assert ctx.page(2) == "\nPage two content here.\n"


def test_page_last_page_extends_to_end() -> None:
    ctx = Context.from_text(_SAMPLE_DOC)
    result = ctx.page(3)
    assert result is not None
    assert "Page three" in result
    # Last page should extend to end of text, not stop at a marker.
    assert result == "\nPage three — the last page.\n"


def test_page_not_found_returns_none() -> None:
    ctx = Context.from_text(_SAMPLE_DOC)
    assert ctx.page(99) is None
    assert ctx.page(0) is None


def test_page_no_markers_returns_none() -> None:
    ctx = Context.from_text("Plain text with no page markers at all.")
    assert ctx.page(1) is None


def test_page_custom_marker_pattern() -> None:
    text = "[P1]First\n[P2]Second\n[P3]Third"
    ctx = Context.from_text(text)
    result = ctx.page(2, marker_pattern=r"\[P(\d+)\]")
    assert result == "Second\n"


def test_page_regression_doc1_p49_q11() -> None:
    """Regression: the broad ask() found the answer on Page 48.

    Previously ctx.find() for the distinctive phrase returned [] due to
    OCR issues, causing a keyword fallback to the wrong page.  ctx.page(48)
    should return the correct content directly.
    """
    # Build a document that mimics the real structure
    pages = []
    for i in range(1, 51):
        pages.append(f"<!-- Page {i} -->\nContent of page {i}.")
        if i == 48:
            pages[-1] += "\nInversiones inmobiliarias — Nota 8 detail here."
    doc = "\n".join(pages)
    ctx = Context.from_text(doc)

    result = ctx.page(48)
    assert result is not None
    assert "Inversiones inmobiliarias" in result
    # Must NOT contain content from page 49
    assert "Content of page 49" not in result
