from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Tuple


@dataclass(frozen=True)
class Context:
    """Deterministic text context with safe inspection helpers."""

    text: str

    @classmethod
    def from_text(cls, text: str) -> "Context":
        return cls(text=text)

    def len_chars(self) -> int:
        return len(self.text)

    def slice(self, start: int, end: int) -> str:
        length = len(self.text)
        safe_start = max(0, min(start, length))
        safe_end = max(0, min(end, length))
        if safe_start >= safe_end:
            return ""
        return self.text[safe_start:safe_end]

    def find(
        self, pattern: str, *, regex: bool = False, max_matches: int = 20
    ) -> List[Tuple[int, int, str]]:
        if max_matches <= 0:
            return []
        if pattern == "":
            return []
        results: List[Tuple[int, int, str]] = []
        if regex:
            for match in re.finditer(pattern, self.text):
                results.append((match.start(), match.end(), match.group(0)))
                if len(results) >= max_matches:
                    break
            return results

        start = 0
        while len(results) < max_matches:
            idx = self.text.find(pattern, start)
            if idx == -1:
                break
            end = idx + len(pattern)
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
