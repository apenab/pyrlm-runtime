from __future__ import annotations

import threading
from dataclasses import dataclass, field


class DocumentPolicyError(RuntimeError):
    pass


class MaxPDFsExceeded(DocumentPolicyError):
    pass


class MaxPagesExceeded(DocumentPolicyError):
    pass


class MaxTablesExceeded(DocumentPolicyError):
    pass


@dataclass
class DocumentPolicy:
    """Budget tracker for on-demand PDF access.

    Separate from Policy (which tracks LLM token budgets). This tracks
    document I/O budgets: how many PDFs, pages, and tables can be accessed
    during a single RLM run.

    Thread-safe — same pattern as pyrlm_runtime.policy.Policy.
    """

    max_pdfs_per_query: int = 20
    max_pages_per_pdf: int = 50
    max_total_pages: int = 100
    max_table_extractions: int = 30

    # Runtime counters (reset per run via reset())
    pdfs_opened: int = 0
    pages_read: int = 0
    tables_extracted: int = 0

    _opened_paths: set[str] = field(default_factory=set, init=False, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def check_pdf_open(self, path: str) -> None:
        """Increment pdfs_opened on first access of a path, or raise MaxPDFsExceeded."""
        with self._lock:
            if path in self._opened_paths:
                return  # already counted
            if self.pdfs_opened >= self.max_pdfs_per_query:
                raise MaxPDFsExceeded(
                    f"Reached limit of {self.max_pdfs_per_query} PDFs per query"
                )
            self.pdfs_opened += 1
            self._opened_paths.add(path)

    def check_page_read(self, count: int = 1) -> None:
        """Add ``count`` to pages_read, raising MaxPagesExceeded if over budget."""
        with self._lock:
            if self.pages_read + count > self.max_total_pages:
                raise MaxPagesExceeded(
                    f"Reached limit of {self.max_total_pages} total pages per query"
                )
            self.pages_read += count

    def check_table_extraction(self) -> None:
        """Increment tables_extracted or raise MaxTablesExceeded."""
        with self._lock:
            if self.tables_extracted >= self.max_table_extractions:
                raise MaxTablesExceeded(
                    f"Reached limit of {self.max_table_extractions} table extractions per query"
                )
            self.tables_extracted += 1

    def get_counters(self) -> dict[str, int]:
        """Return current runtime counters (for logging / display)."""
        with self._lock:
            return {
                "pdfs_opened": self.pdfs_opened,
                "pages_read": self.pages_read,
                "tables_extracted": self.tables_extracted,
            }

    def reset(self) -> None:
        """Reset all runtime counters (call before each RLM run if reusing policy)."""
        with self._lock:
            self.pdfs_opened = 0
            self.pages_read = 0
            self.tables_extracted = 0
            self._opened_paths.clear()
