"""OpenAI-compatible client. Works with the OpenAI API, Together AI, vLLM, and
any other endpoint that speaks the OpenAI chat-completions protocol.
"""

from __future__ import annotations

import os
from typing import Any

from peek.core.types import Usage


class OpenAIClient:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        extra: dict[str, Any] | None = None,
    ):
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "OpenAIClient requires `openai`. Install with: pip install peek-ai[openai]"
            ) from e

        self.model = model
        self.temperature = temperature
        self.extra = dict(extra or {})
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )
        self._last = Usage()

    def completion(self, messages: list[dict[str, Any]]) -> str:
        kwargs: dict[str, Any] = dict(self.extra)
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self._last = Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
        return resp.choices[0].message.content or ""

    def last_usage(self) -> Usage:
        return self._last
