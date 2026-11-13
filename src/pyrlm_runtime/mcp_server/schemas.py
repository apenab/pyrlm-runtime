"""Output models of the MCP tools.

These models are the stable contract with MCP clients: each tool publishes the
JSON schema of its model as ``outputSchema`` and returns an instance of it as
structured content.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    """One ranked hit returned by the ``search`` tool."""

    doc_id: str = Field(description="Id of one indexed unit, such as a page. Pass to get_document.")
    logical_doc_id: str | None = Field(
        default=None, description="Id of the document the unit belongs to. Pass to get_pages."
    )
    page_num: int | None = None
    score: float | None = None
    preview: str
    content: str | None = Field(
        default=None, description="Full text, present only with include_content=true."
    )
    content_truncated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Result of the ``search`` tool."""

    results: list[SearchHit]
    mode: str = Field(description="Search mode that ran.")
    top_k_applied: int = Field(description="Number of hits requested after clamping.")
    truncated: bool = Field(
        default=False, description="True when hits were dropped to fit the response size cap."
    )


class DocumentResponse(BaseModel):
    """Result of the ``get_document`` tool: one slice of one indexed unit."""

    doc_id: str
    content: str
    offset: int = Field(description="Character offset of the slice after clamping.")
    max_chars_applied: int = Field(description="Slice length limit after clamping.")
    next_offset: int | None = Field(
        description="Offset of the next slice, or null when the slice reaches the end."
    )
    total_chars: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    truncated: bool = Field(
        default=False,
        description="True when metadata was left out to fit the response size cap.",
    )


class Page(BaseModel):
    """One page returned by the ``get_pages`` tool."""

    doc_id: str = Field(description="Id of the indexed page. Pass to get_document to read more.")
    page_num: int | None = None
    content: str
    content_truncated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class PagesResponse(BaseModel):
    """Result of the ``get_pages`` tool."""

    logical_doc_id: str
    pages: list[Page]
    radius_applied: int
    max_pages_applied: int
    truncated: bool = Field(
        default=False, description="True when pages were dropped to fit the response size cap."
    )
