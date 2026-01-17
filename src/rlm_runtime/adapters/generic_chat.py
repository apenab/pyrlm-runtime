from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import httpx

from .base import ModelAdapter, ModelResponse, Usage, estimate_usage

PayloadBuilder = Callable[[list[dict[str, str]], int, float, str | None], dict[str, Any]]
ResponseParser = Callable[[dict[str, Any]], tuple[str, Usage | None]]


def default_payload_builder(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    model: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if model:
        payload["model"] = model
    return payload


def default_response_parser(data: dict[str, Any]) -> tuple[str, Usage | None]:
    content = data["choices"][0]["message"]["content"]
    usage_data = data.get("usage")
    usage = Usage.from_dict(usage_data) if usage_data else None
    return content, usage


class GenericChatAdapter(ModelAdapter):
    """Schema-configurable chat adapter for OpenAI-compatible endpoints."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        base_url: str | None = None,
        path: str = "/chat/completions",
        model: str | None = None,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        payload_builder: PayloadBuilder | None = None,
        response_parser: ResponseParser | None = None,
        timeout: float = 60.0,
    ) -> None:
        if endpoint is None:
            if not base_url:
                raise ValueError("endpoint or base_url is required")
            endpoint = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.payload_builder = payload_builder or default_payload_builder
        self.response_parser = response_parser or default_response_parser
        self.headers = {"Content-Type": "application/json"}
        if headers:
            self.headers.update(headers)
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ModelResponse:
        payload = self.payload_builder(messages, max_tokens, temperature, self.model)
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.endpoint, json=payload, headers=self.headers)
            response.raise_for_status()
            data = response.json()

        content, usage = self.response_parser(data)
        if usage is None:
            prompt = "\n".join(msg.get("content", "") for msg in messages)
            usage = estimate_usage(prompt, content)
        return ModelResponse(text=content, usage=usage)
