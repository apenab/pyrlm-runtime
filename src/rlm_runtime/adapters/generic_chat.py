from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from .base import ModelAdapter, ModelResponse, Usage, estimate_usage

logger = logging.getLogger(__name__)

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
    """Schema-configurable chat adapter for OpenAI-compatible endpoints.

    Supports automatic retry with exponential backoff for transient errors
    (HTTP 429, 500, 502, 503, 504).
    """

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
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
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
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

    def _should_retry(self, status_code: int) -> bool:
        """Check if the error is retryable (transient server errors)."""
        return status_code in {429, 500, 502, 503, 504}

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter."""
        import random
        delay = self.retry_base_delay * (2 ** attempt)
        delay = min(delay, self.retry_max_delay)
        # Add jitter (0.5x to 1.5x)
        jitter = 0.5 + random.random()
        return delay * jitter

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ModelResponse:
        payload = self.payload_builder(messages, max_tokens, temperature, self.model)
        last_error: httpx.HTTPStatusError | None = None

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        self.endpoint, json=payload, headers=self.headers
                    )
                    response.raise_for_status()
                    data = response.json()

                content, usage = self.response_parser(data)
                if usage is None:
                    prompt = "\n".join(msg.get("content", "") for msg in messages)
                    usage = estimate_usage(prompt, content)
                return ModelResponse(text=content, usage=usage)

            except httpx.HTTPStatusError as e:
                last_error = e
                if not self._should_retry(e.response.status_code):
                    raise
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        "HTTP %d error, retrying in %.1fs (attempt %d/%d)",
                        e.response.status_code,
                        delay,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(delay)

        # All retries exhausted
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unexpected state: no response and no error")
