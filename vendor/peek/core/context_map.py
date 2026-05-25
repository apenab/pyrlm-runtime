"""Section-indexed, line-addressable context map.

Each item is rendered as a single line: ``[<slug>-NNNNN] <content>`` under a
section heading ``## <SECTION NAME>``. Item IDs are stable across edits so the
Cartographer can reference them by ID.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from peek._io import load_prompt
from peek.core.types import SECTION_SLUG, SECTIONS, Item, Operation

_ITEM_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$")
_SLUG_TAIL = re.compile(r"-(\d+)$")


def normalize_section(name: str) -> str:
    return name.lower().strip().replace(" ", "_").replace("-", "_").rstrip(":")


@dataclass
class ContextMap:
    text: str

    @classmethod
    def initial(cls) -> ContextMap:
        return cls(load_prompt("initial_context_map.txt").rstrip() + "\n")

    @classmethod
    def from_file(cls, path: str | Path) -> ContextMap:
        return cls(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.text, encoding="utf-8")

    def items(self) -> list[Item]:
        out: list[Item] = []
        section = "general"
        for line in self.text.splitlines():
            stripped = line.strip()
            if stripped.startswith("##"):
                section = normalize_section(stripped.lstrip("#").strip())
                continue
            m = _ITEM_RE.match(stripped)
            if m:
                out.append(Item(id=m.group(1), section=section, content=m.group(2).strip()))
        return out

    def item_ids(self) -> list[str]:
        return [it.id for it in self.items()]

    def next_id_int(self) -> int:
        n = 0
        for it in self.items():
            m = _SLUG_TAIL.search(it.id)
            if m:
                n = max(n, int(m.group(1)))
        return n + 1

    def apply(self, operations: list[Operation]) -> ContextMap:
        if not operations:
            return ContextMap(self.text)

        lines = self.text.splitlines()
        deletes: set[str] = set()
        replaces: dict[str, str] = {}
        adds: list[tuple[str, str]] = []  # (section, line)
        next_id = self.next_id_int()

        for op in operations:
            if op.type == "DELETE" and op.item_id:
                deletes.add(op.item_id)
            elif op.type == "REPLACE" and op.item_id and op.content:
                replaces[op.item_id] = op.content
            elif op.type == "ADD" and op.section and op.content:
                section = normalize_section(op.section)
                slug = SECTION_SLUG.get(section, section[:4])
                line = f"[{slug}-{next_id:05d}] {op.content}"
                next_id += 1
                adds.append((section, line))

        out_lines: list[str] = []
        current: str | None = None

        def flush(section: str | None) -> None:
            if section is None:
                return
            picks = [ln for s, ln in adds if s == section]
            if picks:
                out_lines.extend(picks)
                adds[:] = [(s, ln) for s, ln in adds if s != section]

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("##"):
                flush(current)
                if out_lines and out_lines[-1] != "":
                    out_lines.append("")
                current = normalize_section(stripped.lstrip("#").strip())
                out_lines.append(line)
                continue

            m = _ITEM_RE.match(stripped)
            if m:
                bid = m.group(1)
                if bid in deletes:
                    continue
                if bid in replaces:
                    out_lines.append(f"[{bid}] {replaces[bid]}")
                    continue
            out_lines.append(line)

        flush(current)

        # Leftover ADDs (sections not present): append at the end under their headers.
        if adds:
            buckets: dict[str, list[str]] = {}
            for s, ln in adds:
                buckets.setdefault(s, []).append(ln)
            for section in [s for s in SECTIONS if s in buckets] + [
                s for s in buckets if s not in SECTIONS
            ]:
                if out_lines and out_lines[-1] != "":
                    out_lines.append("")
                out_lines.append(f"## {section.upper().replace('_', ' ')}")
                out_lines.extend(buckets[section])

        return ContextMap(_collapse_blank_lines("\n".join(out_lines)) + "\n")


def _collapse_blank_lines(text: str) -> str:
    out: list[str] = []
    for line in text.split("\n"):
        if not line.strip():
            if out and not out[-1].strip():
                continue
        out.append(line)
    return "\n".join(out).rstrip()
