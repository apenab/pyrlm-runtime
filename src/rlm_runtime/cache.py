from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
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
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        usage = Usage.from_dict(data.get("usage", {}))
        return CacheRecord(text=data.get("text", ""), usage=usage)

    def set(self, key: str, record: CacheRecord) -> None:
        path = self._path(key)
        payload: dict[str, Any] = {
            "text": record.text,
            "usage": record.usage.to_dict(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
