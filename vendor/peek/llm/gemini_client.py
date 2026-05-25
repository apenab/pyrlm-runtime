from __future__ import annotations

import os
from typing import Any

from peek.core.types import Usage


class GeminiClient:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float | None = None,
    ):
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "GeminiClient requires `google-genai`. "
                "Install with: pip install peek-ai[gemini]"
            ) from e

        self.model = model
        self.temperature = temperature
        self._client = genai.Client(
            api_key=api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
        )
        self._last = Usage()

    def completion(self, messages: list[dict[str, Any]]) -> str:
        from google.genai import types as gt

        system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system") or None
        contents: list[Any] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                continue
            contents.append(
                gt.Content(
                    role="user" if role in (None, "user") else "model",
                    parts=[gt.Part.from_text(text=m["content"])],
                )
            )

        config: dict[str, Any] = {}
        if system:
            config["system_instruction"] = system
        if self.temperature is not None:
            config["temperature"] = self.temperature

        resp = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=gt.GenerateContentConfig(**config) if config else None,
        )
        usage = getattr(resp, "usage_metadata", None)
        if usage is not None:
            self._last = Usage(
                input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            )
        return resp.text or ""

    def last_usage(self) -> Usage:
        return self._last
