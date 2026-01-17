from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List, Tuple


@dataclass(frozen=True)
class Context:
    """Deterministic text context with safe inspection helpers.

    Supports both single text and list of documents (for multi-document tasks
    like BrowseComp+). When initialized with documents, `text` is the concatenated
    version and `documents` holds the original list.
    """

    text: str
    documents: Tuple[str, ...] = field(default_factory=tuple)
    context_type: str = "string"

    @classmethod
    def from_text(cls, text: str) -> "Context":
        return cls(text=text, documents=(text,), context_type="string")

    @classmethod
    def from_documents(
        cls, documents: List[str], *, separator: str = "\n\n---\n\n"
    ) -> "Context":
        """Create context from a list of documents (e.g., for BrowseComp+)."""
        if not documents:
            return cls(text="", documents=(), context_type="document_list")
        text = separator.join(documents)
        return cls(
            text=text,
            documents=tuple(documents),
            context_type="document_list",
        )

    def len_chars(self) -> int:
        return len(self.text)

    def num_documents(self) -> int:
        """Return number of documents (1 for plain text context)."""
        return len(self.documents) if self.documents else 1

    def get_document(self, index: int) -> str | None:
        """Get a specific document by index."""
        if not self.documents or index < 0 or index >= len(self.documents):
            return None
        return self.documents[index]

    def document_lengths(self) -> List[int]:
        """Return character lengths of each document."""
        if not self.documents:
            return [len(self.text)]
        return [len(doc) for doc in self.documents]

    def slice(self, start: int, end: int) -> str:
        length = len(self.text)
        safe_start = max(0, min(start, length))
        safe_end = max(0, min(end, length))
        if safe_start >= safe_end:
            return ""
        return self.text[safe_start:safe_end]

    def find(
        self,
        pattern: str,
        *,
        regex: bool = False,
        max_matches: int = 20,
        case_sensitive: bool = True,
        flags: int | None = None,
    ) -> List[Tuple[int, int, str]]:
        if max_matches <= 0:
            return []
        if pattern == "":
            return []
        results: List[Tuple[int, int, str]] = []
        if regex or flags is not None:
            re_flags = flags if flags is not None else (0 if case_sensitive else re.IGNORECASE)
            for match in re.finditer(pattern, self.text, re_flags):
                results.append((match.start(), match.end(), match.group(0)))
                if len(results) >= max_matches:
                    break
            return results

        start = 0
        haystack = self.text if case_sensitive else self.text.lower()
        needle = pattern if case_sensitive else pattern.lower()
        while len(results) < max_matches:
            idx = haystack.find(needle, start)
            if idx == -1:
                break
            end = idx + len(needle)
            results.append((idx, end, self.text[idx:end]))
            start = end
        return results

    def chunk(self, size: int, overlap: int = 0) -> List[Tuple[int, int, str]]:
        if size <= 0:
            raise ValueError("size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= size:
            raise ValueError("overlap must be < size")

        chunks: List[Tuple[int, int, str]] = []
        length = len(self.text)
        start = 0
        while start < length:
            end = min(length, start + size)
            chunks.append((start, end, self.text[start:end]))
            if end >= length:
                break
            start = end - overlap
        return chunks

    def chunk_documents(
        self, docs_per_chunk: int = 10
    ) -> List[Tuple[int, int, List[str]]]:
        """Chunk by documents rather than characters (for document lists).

        Returns list of (start_doc_idx, end_doc_idx, docs_in_chunk).
        """
        if not self.documents:
            return [(0, 1, [self.text])]

        if docs_per_chunk <= 0:
            raise ValueError("docs_per_chunk must be > 0")

        chunks: List[Tuple[int, int, List[str]]] = []
        num_docs = len(self.documents)
        start = 0
        while start < num_docs:
            end = min(num_docs, start + docs_per_chunk)
            chunks.append((start, end, list(self.documents[start:end])))
            start = end
        return chunks

    def metadata(self) -> dict:
        """Return metadata about the context for the system prompt."""
        meta: dict = {
            "context_type": self.context_type,
            "total_length": self.len_chars(),
            "num_documents": self.num_documents(),
        }
        # Only include document_lengths for multi-document contexts
        if self.context_type == "document_list" and len(self.documents) > 1:
            meta["document_lengths"] = self.document_lengths()
        return meta
