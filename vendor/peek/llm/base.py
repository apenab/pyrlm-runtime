from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from peek.core.types import Usage


@runtime_checkable
class LMClient(Protocol):
    """Minimal LM interface used by Peek.

    Implementations must record per-call usage so :meth:`last_usage` returns
    the token counts from the most recent ``completion`` call.
    """

    def completion(self, messages: list[dict[str, Any]]) -> str: ...

    def last_usage(self) -> Usage: ...
