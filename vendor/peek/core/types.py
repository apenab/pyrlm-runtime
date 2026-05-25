from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ItemTag = Literal["helpful", "harmful", "neutral", "stale"]
OpType = Literal["ADD", "DELETE", "REPLACE"]

SECTIONS: tuple[str, ...] = (
    "context_roadmap",
    "context_understanding",
    "domain_constants",
    "parsing_schema",
    "error_patterns",
    "reusable_results",
)

SECTION_SLUG: dict[str, str] = {
    "context_roadmap": "cr",
    "context_understanding": "cu",
    "domain_constants": "dc",
    "parsing_schema": "ps",
    "error_patterns": "ep",
    "reusable_results": "rr",
}


@dataclass(frozen=True)
class Item:
    id: str
    section: str
    content: str


@dataclass(frozen=True)
class Operation:
    type: OpType
    section: str | None = None
    item_id: str | None = None
    content: str | None = None


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass
class DistillerOutput:
    raw: str
    diagnosis: str
    item_tags: dict[str, ItemTag] = field(default_factory=dict)
    cache_candidates: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


@dataclass
class CartographerOutput:
    raw: str
    operations: list[Operation] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
