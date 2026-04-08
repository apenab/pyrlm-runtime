from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class DocumentCache:
    """Write-through file cache for on-demand PDF reading results.

    Directory structure:
        {cache_dir}/{file_hash}/
            page_001.md
            page_001.meta.json   # {"parser": str, "timestamp": str, "char_count": int}
            page_003_table_0.md

    Keys are file hashes (SHA256) so cache entries are automatically
    invalidated when a PDF changes (its hash changes).
    """

    def __init__(self, cache_dir: str | Path = ".rlm_cache/docs") -> None:
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Page cache
    # ------------------------------------------------------------------

    def _page_path(self, file_hash: str, page: int) -> Path:
        return self.root / file_hash / f"page_{page:03d}.md"

    def _page_meta_path(self, file_hash: str, page: int) -> Path:
        return self.root / file_hash / f"page_{page:03d}.meta.json"

    def get_page(self, file_hash: str, page: int) -> str | None:
        path = self._page_path(file_hash, page)
        with self._lock:
            if path.exists():
                self._hits += 1
                return path.read_text(encoding="utf-8")
            self._misses += 1
            return None

    def set_page(self, file_hash: str, page: int, content: str, parser: str) -> None:
        path = self._page_path(file_hash, page)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        meta = {
            "parser": parser,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "char_count": len(content),
        }
        self._page_meta_path(file_hash, page).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Table cache
    # ------------------------------------------------------------------

    def _table_path(self, file_hash: str, page: int, table_num: int) -> Path:
        return self.root / file_hash / f"page_{page:03d}_table_{table_num}.md"

    def get_table(self, file_hash: str, page: int, table_num: int) -> str | None:
        path = self._table_path(file_hash, page, table_num)
        with self._lock:
            if path.exists():
                self._hits += 1
                return path.read_text(encoding="utf-8")
            self._misses += 1
            return None

    def set_table(self, file_hash: str, page: int, table_num: int, content: str) -> None:
        path = self._table_path(file_hash, page, table_num)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, int]:
        with self._lock:
            total_size = sum(
                f.stat().st_size for f in self.root.rglob("*.md") if f.exists()
            )
            cached_pages = sum(
                1 for f in self.root.rglob("page_???.md") if f.exists()
            )
            return {
                "hits": self._hits,
                "misses": self._misses,
                "cached_pages": cached_pages,
                "total_size_bytes": total_size,
            }
