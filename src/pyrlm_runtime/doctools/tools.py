"""Factory for doc tool functions injected into the RLM REPL scaffold."""

from __future__ import annotations

from typing import Callable

from .cache import DocumentCache
from .policy import DocumentPolicy, MaxPDFsExceeded, MaxPagesExceeded, MaxTablesExceeded
from .protocols import DocIndexStoreProtocol, PageReaderProtocol
from .schema import DocInfo

DEFAULT_LABELS: dict[str, str] = {
    "no_docs_found_query": "No documents found for query: '{query}'",
    "docs_found_query": "Documents found for '{query}' ({count}):",
    "no_docs_indexed": "No documents indexed. The index is empty.",
    "docs_available": "Available documents ({count}):",
    "and_more": "... and {count} more.",
    "error_listing": "Error listing PDFs: {error}",
    "doc_not_found": "Document not found: '{id}'. Use list_pdfs() to see available documents.",
    "scanned": "scanned",
    "digital": "digital",
    "pages_label": "pages",
    "tables_label": "tables",
    "error_info": "Error getting info for '{id}': {error}",
    "error_reading": "Error reading '{path}': {error}",
    "truncated": "[...truncated, {count} more chars]",
    "doc_limit": "[DOCUMENT LIMIT]: {error}",
    "table_limit": "[TABLE LIMIT]: {error}",
    "table_not_found": "Table {table_num} not found on page {page}.",
    "error_table": "Error extracting table from '{path}' page {page}: {error}",
    "not_found_in_cache": (
        "'{query}' not found in cached pages of '{title}'. "
        "Use read_pdf_pages() first to cache the content."
    ),
    "found_in_doc": "'{query}' in {title} — {count} page(s):",
    "error_search_in": "Error searching in '{path}': {error}",
    "results_for": "Results for '{query}' ({count} documents):",
    "no_results_for": "No documents found for: '{query}'",
    "error_corpus_search": "Error in corpus search: {error}",
    "resolve_not_found": "Document not found: '{query}'",
}


def _fmt_doc(doc: DocInfo, index: int, labels: dict[str, str]) -> str:
    tables = sum(p.table_count for p in doc.pages)
    scanned = labels["scanned"] if doc.is_scanned_prob > 0.5 else labels["digital"]
    pages_label = labels["pages_label"]
    tables_label = labels["tables_label"]
    return (
        f"{index + 1}. {doc.title} ({doc.num_pages} {pages_label}, {tables} {tables_label}, {scanned})\n"
        f"   path: {doc.path}\n"
        f"   id: {doc.doc_id}"
    )


def _resolve_doc(
    path_or_id: str, index_store: DocIndexStoreProtocol
) -> DocInfo | None:
    doc = index_store.get_by_path(path_or_id)
    if doc is None:
        doc = index_store.get(path_or_id)
    return doc


def create_doc_tools(
    index_store: DocIndexStoreProtocol,
    page_reader: PageReaderProtocol,
    cache: DocumentCache,
    policy: DocumentPolicy,
    labels: dict[str, str] | None = None,
) -> dict[str, Callable]:
    """Create all doc tool functions for REPL injection.

    Returns a dict of {function_name: callable} ready to merge into the RLM
    scaffold alongside peek(), llm_query(), es_search(), etc.

    All functions return strings — never raise exceptions (errors are returned
    as informative strings so the LLM can adapt).

    The exception is ``search_corpus_structured`` and ``resolve_doc_path``, which
    return structured data for programmatic use in multi-step workflows.
    """
    lbl: dict[str, str] = {**DEFAULT_LABELS, **(labels or {})}

    def list_pdfs(query: str | None = None) -> str:
        """List available PDFs in the corpus, optionally filtered by query.

        Returns a formatted string with document metadata.
        Use this to discover which PDFs are available before reading them.
        """
        try:
            if query:
                docs = index_store.search(query, top_k=20)
                if not docs:
                    return lbl["no_docs_found_query"].format(query=query)
                header = lbl["docs_found_query"].format(query=query, count=len(docs)) + "\n"
            else:
                docs = index_store.list_all()
                if not docs:
                    return lbl["no_docs_indexed"]
                header = lbl["docs_available"].format(count=len(docs)) + "\n"

            lines = [header]
            for i, doc in enumerate(docs[:50]):
                lines.append(_fmt_doc(doc, i, lbl))
            if len(docs) > 50:
                lines.append("\n" + lbl["and_more"].format(count=len(docs) - 50))
            return "\n".join(lines)
        except Exception as exc:
            return lbl["error_listing"].format(error=exc)

    def get_pdf_info(path_or_id: str) -> str:
        """Get detailed metadata and page structure for a PDF.

        Args:
            path_or_id: File path or doc_id from list_pdfs()

        Returns a formatted string with document metadata and per-page details.
        """
        try:
            doc = _resolve_doc(path_or_id, index_store)
            if doc is None:
                return lbl["doc_not_found"].format(id=path_or_id)

            lines = [
                f"## {doc.title}",
                f"- path: {doc.path}",
                f"- id: {doc.doc_id}",
                f"- pages: {doc.num_pages}",
                f"- size: {doc.file_size:,} bytes",
                f"- language: {doc.language}",
                f"- scanned: {doc.is_scanned_prob:.0%}",
                f"- parser hint: {doc.parser_hint}",
                f"- summary: {doc.summary[:300]}{'...' if len(doc.summary) > 300 else ''}",
                "",
                "### Pages:",
            ]
            for p in doc.pages:
                headings_str = ", ".join(p.headings[:3]) if p.headings else "—"
                flags_str = f" [{', '.join(p.flags)}]" if p.flags else ""
                lines.append(
                    f"  Page {p.page_num + 1}: {p.char_count} chars, "
                    f"{p.table_count} tables, headings: [{headings_str}]{flags_str}"
                )
            return "\n".join(lines)
        except Exception as exc:
            return lbl["error_info"].format(id=path_or_id, error=exc)

    def read_pdf_pages(
        path: str,
        pages: list[int] | None = None,
        max_chars: int = 16000,
    ) -> str:
        """Read specific pages from a PDF on demand.

        Args:
            path: File path (use get_pdf_info() or list_pdfs() to get paths)
            pages: List of 0-indexed page numbers. If None, reads first 3 pages.
            max_chars: Maximum characters to return (truncated if exceeded)

        Returns the page content as text/markdown, with page markers.
        Check the cache first — repeated reads of the same page are free.
        """
        try:
            doc = _resolve_doc(path, index_store)
            if doc is None:
                return lbl["doc_not_found"].format(id=path)

            if pages is None:
                pages = list(range(min(3, doc.num_pages)))

            policy.check_pdf_open(doc.path)
            policy.check_page_read(len(pages))

            parts: list[str] = []
            uncached: list[int] = []

            for p in pages:
                cached = cache.get_page(doc.file_hash, p)
                if cached is not None:
                    parts.append(f"<!-- Page {p + 1} -->\n{cached}")
                else:
                    uncached.append(p)

            if uncached:
                read_result = page_reader.read_pages(doc.path, uncached)
                for p in uncached:
                    content = read_result.get(p, "")
                    if content:
                        cache.set_page(doc.file_hash, p, content, doc.parser_hint)
                    parts.append(f"<!-- Page {p + 1} -->\n{content}")

            combined = "\n\n".join(parts)
            if len(combined) > max_chars:
                combined = combined[:max_chars] + "\n\n" + lbl["truncated"].format(count=len(combined) - max_chars)
            return combined

        except (MaxPDFsExceeded, MaxPagesExceeded) as exc:
            return lbl["doc_limit"].format(error=exc)
        except Exception as exc:
            return lbl["error_reading"].format(path=path, error=exc)

    def extract_table(path: str, page: int, table_num: int = 0) -> str:
        """Extract a specific table from a PDF page as markdown.

        Args:
            path: File path
            page: 0-indexed page number
            table_num: 0-indexed table number within the page

        Returns the table formatted as a markdown string.
        """
        try:
            doc = _resolve_doc(path, index_store)
            if doc is None:
                return lbl["doc_not_found"].format(id=path)

            policy.check_pdf_open(doc.path)
            policy.check_table_extraction()

            cached = cache.get_table(doc.file_hash, page, table_num)
            if cached is not None:
                return cached

            result = page_reader.extract_table(doc.path, page, table_num)
            if result:
                cache.set_table(doc.file_hash, page, table_num, result)
            else:
                result = lbl["table_not_found"].format(table_num=table_num, page=page + 1)
            return result

        except MaxTablesExceeded as exc:
            return lbl["table_limit"].format(error=exc)
        except (MaxPDFsExceeded, MaxPagesExceeded) as exc:
            return lbl["doc_limit"].format(error=exc)
        except Exception as exc:
            return lbl["error_table"].format(path=path, page=page + 1, error=exc)

    def search_in_pdf(path: str, query: str, top_k: int = 5) -> str:
        """Search for text within a specific PDF using cached page content.

        Args:
            path: File path or doc_id
            query: Text to search for (case-insensitive substring match)
            top_k: Maximum number of matching pages to return

        Returns matching page snippets with context.
        """
        try:
            doc = _resolve_doc(path, index_store)
            if doc is None:
                return lbl["doc_not_found"].format(id=path)

            ql = query.lower()
            hits: list[tuple[int, str]] = []

            for page_info in doc.pages:
                p = page_info.page_num
                content = cache.get_page(doc.file_hash, p)
                if content is None:
                    continue
                cl = content.lower()
                idx = cl.find(ql)
                if idx != -1:
                    snippet = content[max(0, idx - 100): idx + 300]
                    hits.append((p, snippet))
                if len(hits) >= top_k:
                    break

            if not hits:
                return lbl["not_found_in_cache"].format(query=query, title=doc.title)

            lines = [lbl["found_in_doc"].format(query=query, title=doc.title, count=len(hits)) + "\n"]
            for p, snippet in hits:
                lines.append(f"--- Page {p + 1} ---\n...{snippet}...")
            return "\n".join(lines)

        except Exception as exc:
            return lbl["error_search_in"].format(path=path, error=exc)

    def search_corpus(query: str, top_k: int = 10) -> str:
        """Search across all indexed documents by metadata (title, summary, headings).

        This searches the document INDEX (fast, no PDF reading), not the full text.
        For full-text search within a document, use search_in_pdf() after reading pages.

        Args:
            query: Search terms
            top_k: Maximum results to return

        Returns ranked list of matching documents.
        """
        try:
            docs = index_store.search(query, top_k=top_k)
            if not docs:
                return lbl["no_results_for"].format(query=query)

            lines = [lbl["results_for"].format(query=query, count=len(docs)) + "\n"]
            for i, doc in enumerate(docs):
                tables = sum(p.table_count for p in doc.pages)
                lines.append(
                    f"{i + 1}. {doc.title}\n"
                    f"   {doc.num_pages} pages · {tables} tables · {doc.summary[:150]}...\n"
                    f"   path: {doc.path}"
                )
            return "\n".join(lines)
        except Exception as exc:
            return lbl["error_corpus_search"].format(error=exc)

    def search_corpus_structured(query: str, top_k: int = 10) -> list[dict]:
        """Search documents and return structured results for programmatic use.

        Unlike search_corpus(), this returns a list of dicts — NOT a formatted string.
        Use this when you need to iterate over results or access fields like path/doc_id.

        Args:
            query: Search terms
            top_k: Maximum results to return

        Returns:
            list of dicts with keys: title, path, doc_id, num_pages, summary
        """
        docs = index_store.search(query, top_k=top_k)
        return [
            {
                "title": d.title,
                "path": d.path,
                "doc_id": d.doc_id,
                "num_pages": d.num_pages,
                "summary": d.summary[:200],
            }
            for d in docs
        ]

    def resolve_doc_path(query_or_title: str) -> str:
        """Resolve a document title, query, or ID to its file path.

        Returns the file path as a plain string suitable for passing directly to
        read_pdf_pages(), extract_table(), or search_in_pdf(). Falls back to a
        fuzzy search if the exact path/id is not found.

        Do NOT call .get() on the result — it is always a plain string.

        Args:
            query_or_title: File path, doc_id, or title/query to search for

        Returns:
            File path string on success, or an error message string if not found.
        """
        doc = _resolve_doc(query_or_title, index_store)
        if doc is not None:
            return doc.path
        docs = index_store.search(query_or_title, top_k=1)
        if docs:
            return docs[0].path
        return lbl["resolve_not_found"].format(query=query_or_title)

    return {
        "list_pdfs": list_pdfs,
        "get_pdf_info": get_pdf_info,
        "read_pdf_pages": read_pdf_pages,
        "extract_table": extract_table,
        "search_in_pdf": search_in_pdf,
        "search_corpus": search_corpus,
        "search_corpus_structured": search_corpus_structured,
        "resolve_doc_path": resolve_doc_path,
    }
