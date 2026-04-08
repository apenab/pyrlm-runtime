DOC_TOOLS_PROMPT_SUPPLEMENT = """

## Document Tools (on-demand PDF access)

You have direct access to the PDF corpus via these functions:

list_pdfs(query=None)
    List all available PDFs. Pass a query string to filter by relevance.
    Returns: formatted list with title, page count, table count, path.

get_pdf_info(path_or_id)
    Get detailed metadata and page structure for a document.
    Shows: per-page char counts, table counts, detected headings, flags.
    Use this to identify which pages contain tables before reading them.

read_pdf_pages(path, pages=None, max_chars=16000)
    Read specific pages on demand. pages is a list of 0-indexed page numbers.
    If pages=None, reads the first 3 pages.
    Results are cached — re-reading the same page is free.
    Returns: page content as text/markdown with <!-- Page N --> markers.

extract_table(path, page, table_num=0)
    Extract a specific table from a page as a markdown table.
    page and table_num are both 0-indexed.
    Results are cached.

search_in_pdf(path, query, top_k=5)
    Search for text within a document using cached page content.
    Only searches pages already in cache — call read_pdf_pages() first.

search_corpus(query, top_k=10)
    Search across all indexed documents by title/summary/headings (fast).
    Does NOT read PDFs — searches the structural index only.
    Returns a FORMATTED STRING — do NOT call .get() or iterate over it.

search_corpus_structured(query, top_k=10)
    Same search, but returns list[dict] with keys: title, path, doc_id, num_pages, summary.
    Use this when you need programmatic access to document metadata.
    Example: docs = search_corpus_structured("annual report"); path = docs[0]["path"]

resolve_doc_path(query_or_title)
    Resolve a document title, query, or ID to its file path string.
    Use this to get a path for read_pdf_pages(), extract_table(), or search_in_pdf().
    Example: path = resolve_doc_path("Acme Corp"); pages = read_pdf_pages(path, [0, 1])

DOCUMENT ACCESS STRATEGY:
1. Use list_pdfs() or search_corpus() to find relevant documents (for display/logging)
   Use search_corpus_structured() or resolve_doc_path() when you need paths programmatically
2. Use get_pdf_info() to identify pages with tables (table_count > 0)
3. Use read_pdf_pages() to read specific pages — prefer targeted pages over full docs
4. Use extract_table() when you need a specific table as structured data
5. Use llm_query() to analyze extracted page content
6. Use search_in_pdf() to locate where specific terms appear within a cached doc

IMPORTANT: search_corpus() and list_pdfs() return FORMATTED STRINGS.
Do NOT call .get("path") or any dict methods on their results.
Use resolve_doc_path(title) or search_corpus_structured(query) for programmatic access.

BUDGET AWARENESS:
- You have a limited number of pages you can read per query.
- Always use get_pdf_info() first to identify the right pages — avoid reading entire documents.
- Prefer extract_table() over reading full pages when you only need tabular data.
- Once a page is cached, reading it again is free (no budget cost).

IMPORTANT: These tools read the ORIGINAL PDF files with high fidelity.
Tables are extracted with full structure — do NOT try to parse HTML tables from the text.
"""
