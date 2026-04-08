from __future__ import annotations

from pydantic import BaseModel


class PageInfo(BaseModel):
    page_num: int
    char_count: int
    text_extractable: bool
    headings: list[str]
    table_count: int
    flags: list[str]  # e.g. "multi_column", "low_ocr_confidence"


class DocInfo(BaseModel):
    doc_id: str
    path: str
    file_hash: str          # SHA256 of file bytes
    file_size: int
    modified_at: str        # ISO 8601 timestamp
    title: str
    num_pages: int
    language: str           # "es", "en", etc.
    is_scanned_prob: float  # 0.0 (digital) to 1.0 (fully scanned)
    has_tables_prob: float
    parser_hint: str        # "pymupdf4llm" | "docling" | "ocr_fallback"
    summary: str            # First ~500 chars of page 1 text
    pages: list[PageInfo]
