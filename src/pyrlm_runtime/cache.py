from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .adapters.base import Usage


@dataclass(frozen=True)
class CacheRecord:
    text: str
    usage: Usage


class FileCache:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def get(self, key: str) -> CacheRecord | None:
        path = self._path(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            # Tolerate a corrupt or half-written entry (e.g. an interrupted
            # write under parallel_subcalls) by treating it as a cache miss
            # instead of crashing the run.
            return None
        usage = Usage.from_dict(data.get("usage", {}))
        return CacheRecord(text=data.get("text", ""), usage=usage)

    def set(self, key: str, record: CacheRecord) -> None:
        path = self._path(key)
        payload: dict[str, Any] = {
            "text": record.text,
            "usage": record.usage.to_dict(),
        }
        # Write atomically: a unique temp file in the same directory followed by
        # an atomic rename. This keeps readers (and concurrent writers under
        # parallel_subcalls) from ever observing a partially written entry.
        fd, tmp_name = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=True, indent=2))
            os.replace(tmp_name, path)
        except BaseException:
            # Never leave a stray temp file behind on failure.
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
