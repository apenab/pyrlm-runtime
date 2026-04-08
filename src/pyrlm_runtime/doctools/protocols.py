from __future__ import annotations

from typing import Protocol, runtime_checkable

from .schema import DocInfo


@runtime_checkable
class PageReaderProtocol(Protocol):
    """Reads pages from PDF files on demand.

    Implementations live in downstream projects (e.g. banking-rlm) and use
    whichever PDF library is appropriate (pymupdf4llm, docling, PyMuPDF, etc.).
    All methods return plain strings so results are usable inside the REPL.
    """

    def read_pages(self, path: str, pages: list[int]) -> dict[int, str]:
        """Read the given page numbers (0-indexed) from a PDF.

        Returns a mapping of page_num → markdown/text content.
        Missing pages are silently omitted from the dict.
        """
        ...

    def extract_table(self, path: str, page: int, table_num: int) -> str:
        """Extract a specific table from a PDF page as a markdown string.

        ``page`` and ``table_num`` are both 0-indexed.
        Returns an empty string if no table is found at that position.
        """
        ...


@runtime_checkable
class DocIndexStoreProtocol(Protocol):
    """Backend-agnostic document index.

    Stores structural metadata (DocInfo) for a corpus of PDFs.
    For Phase 1 the implementation is JSON-file based; later can be ES.
    """

    def search(self, query: str, top_k: int = 10) -> list[DocInfo]:
        """Text search over title + summary + headings."""
        ...

    def get(self, doc_id: str) -> DocInfo | None:
        """Retrieve a document by its doc_id."""
        ...

    def get_by_path(self, path: str) -> DocInfo | None:
        """Retrieve a document by its file path."""
        ...

    def list_all(self) -> list[DocInfo]:
        """Return all indexed documents."""
        ...
