from __future__ import annotations

import os

from .base import ModelAdapter, ModelResponse
from .generic_chat import GenericChatAdapter


class OpenAICompatAdapter(ModelAdapter):
    """Minimal OpenAI-compatible chat adapter using httpx."""

    def __init__(self, *, model: str, base_url: str | None = None) -> None:
        base = (
            base_url
            or os.getenv("LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self._adapter = GenericChatAdapter(base_url=base, model=model, api_key=api_key)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ModelResponse:
        return self._adapter.complete(messages, max_tokens=max_tokens, temperature=temperature)
