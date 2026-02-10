from __future__ import annotations

import json
import logging
import random
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
    (HTTP 429, 500, 502, 503, 504) and network/timeout errors.

    Can be used as a context manager for explicit resource cleanup::

        with GenericChatAdapter(base_url="http://localhost:11434/v1") as adapter:
            response = adapter.complete(messages)
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
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if headers:
            self._headers.update(headers)
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self._client = httpx.Client(timeout=self.timeout)

    # Keep backward-compatible property for code that reads self.headers
    @property
    def headers(self) -> dict[str, str]:
        return self._headers

    def close(self) -> None:
        """Close the underlying HTTP client and release connections."""
        self._client.close()

    def __enter__(self) -> GenericChatAdapter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _should_retry(self, status_code: int) -> bool:
        """Check if the error is retryable (transient server errors)."""
        return status_code in {429, 500, 502, 503, 504}

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter."""
        delay = self.retry_base_delay * (2**attempt)
        delay = min(delay, self.retry_max_delay)
        # Add jitter (0.5x to 1.5x)
        jitter = 0.5 + random.random()
        return delay * jitter

    def _wait(self, seconds: float) -> None:
        """Sleep for the given duration. Extracted for testability."""
        time.sleep(seconds)

    def _send_request(
        self,
        payload: dict[str, Any],
        messages: list[dict[str, str]],
    ) -> ModelResponse:
        """Execute a single HTTP request and parse the response."""
        started = time.monotonic()
        response = self._client.post(
            self.endpoint, json=payload, headers=self._headers
        )
        elapsed = time.monotonic() - started
        logger.debug(
            "HTTP %d from %s in %.2fs",
            response.status_code,
            self.endpoint,
            elapsed,
        )
        response.raise_for_status()

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Malformed JSON response from {self.endpoint}: {e}"
            ) from e

        try:
            content, usage = self.response_parser(data)
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(
                f"Unexpected response structure from {self.endpoint}: {e}"
            ) from e

        if usage is None:
            prompt = "\n".join(msg.get("content", "") for msg in messages)
            usage = estimate_usage(prompt, content)
        return ModelResponse(text=content, usage=usage, model_id=self.model)

    def _handle_retryable_error(self, error: Exception, attempt: int) -> None:
        """Log and wait before retrying; no-op when retries are exhausted.

        When ``attempt < max_retries``, logs a warning and sleeps before the
        next attempt.  When retries are exhausted the method simply returns,
        allowing ``complete()`` to re-raise ``last_error``.
        """
        if attempt >= self.max_retries:
            return
        delay = self._calculate_delay(attempt)
        if isinstance(error, httpx.HTTPStatusError):
            logger.warning(
                "HTTP %d error, retrying in %.1fs (attempt %d/%d)",
                error.response.status_code,
                delay,
                attempt + 1,
                self.max_retries,
            )
        else:
            logger.warning(
                "%s, retrying in %.1fs (attempt %d/%d)",
                type(error).__name__,
                delay,
                attempt + 1,
                self.max_retries,
            )
        self._wait(delay)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ModelResponse:
        payload = self.payload_builder(messages, max_tokens, temperature, self.model)
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return self._send_request(payload, messages)
            except httpx.HTTPStatusError as e:
                if not self._should_retry(e.response.status_code):
                    raise
                last_error = e
                self._handle_retryable_error(e, attempt)
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                self._handle_retryable_error(e, attempt)

        # All retries exhausted
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unexpected state: no response and no error")
