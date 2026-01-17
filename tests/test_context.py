import pytest

from rlm_runtime import Context


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
