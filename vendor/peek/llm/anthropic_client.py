from __future__ import annotations

import os
from typing import Any

from peek.core.types import Usage


class AnthropicClient:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "AnthropicClient requires `anthropic`. "
                "Install with: pip install peek-ai[anthropic]"
            ) from e

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )
        self._last = Usage()

    def completion(self, messages: list[dict[str, Any]]) -> str:
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        chat = [m for m in messages if m.get("role") != "system"]
        kwargs: dict[str, Any] = {"max_tokens": self.max_tokens}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        resp = self._client.messages.create(model=self.model, messages=chat, **kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self._last = Usage(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
            )
        text_blocks = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
        return "".join(text_blocks)

    def last_usage(self) -> Usage:
        return self._last
