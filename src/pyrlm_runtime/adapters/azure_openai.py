from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlparse

import httpx

from .base import ModelAdapter, ModelResponse
from .generic_chat import GenericChatAdapter

logger = logging.getLogger(__name__)


def _azure_payload_builder(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
    model: str | None,
) -> dict[str, object]:
    """Classic deployment-URL style: model is in the URL, not the body."""
    del model
    payload: dict[str, object] = {
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    return payload


def _azure_v1_payload_builder(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
    model: str | None,
) -> dict[str, object]:
    """OpenAI v1-compat style: model goes in the request body."""
    payload: dict[str, object] = {
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if model:
        payload["model"] = model
    return payload


def _is_v1_compat_endpoint(parsed_path: str) -> bool:
    """Return True when the endpoint path already contains a meaningful subpath.

    Examples that are v1-compat:
      /openai/v1/          -> True
      /openai/v1           -> True
      /v1/                 -> True
    Classic deployment (no subpath):
      (empty)  /           -> False
    """
    path = parsed_path.strip("/")
    return bool(path)


class AzureOpenAIAdapter(ModelAdapter):
    """Azure OpenAI chat adapter.

    Supports two endpoint styles detected automatically from ``OPENAI_ENDPOINT``:

    * **Classic deployment** (no path in endpoint URL):
      ``https://<resource>.openai.azure.com``
      → builds ``/openai/deployments/<model>/chat/completions?api-version=…``
      → model name is part of the URL; not included in the request body.

    * **v1-compat / serverless** (endpoint URL already contains a subpath):
      ``https://<resource>.openai.azure.com/openai/v1/``
      → appends ``chat/completions`` to the given base path.
      → model name is included in the request body (OpenAI-compatible style).
    """

    def __init__(
        self,
        *,
        model: str,
        api_version: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        api_key = re.sub(r"\s+", "", os.getenv("AZURE_OPENAI_API_KEY") or "")
        endpoint = re.sub(r"\s+", "", os.getenv("OPENAI_ENDPOINT") or "")
        account_name = re.sub(r"\s+", "", os.getenv("AZURE_ACCOUNT_NAME") or "")
        self.model = re.sub(r"\s+", "", model)
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21"

        if not api_key:
            raise EnvironmentError("Missing AZURE_OPENAI_API_KEY")
        if endpoint:
            azure_endpoint = endpoint.rstrip("/")
        elif account_name:
            azure_endpoint = f"https://{account_name}.openai.azure.com"
        else:
            raise EnvironmentError("Set OPENAI_ENDPOINT or AZURE_ACCOUNT_NAME")

        parsed = urlparse(azure_endpoint)
        base_origin = (
            f"{parsed.scheme}://{parsed.netloc}"
            if (parsed.scheme and parsed.netloc)
            else azure_endpoint
        )

        if _is_v1_compat_endpoint(parsed.path):
            # v1-compat: use the full path already present, just add /chat/completions
            self.endpoint = f"{azure_endpoint}/chat/completions"
            payload_builder = _azure_v1_payload_builder
        else:
            # Classic deployment: build the standard Azure deployment URL
            self.endpoint = (
                f"{base_origin}/openai/deployments/{self.model}/chat/completions"
                f"?api-version={self.api_version}"
            )
            payload_builder = _azure_payload_builder

        self._adapter = GenericChatAdapter(
            endpoint=self.endpoint,
            model=self.model,
            headers={"api-key": api_key},
            payload_builder=payload_builder,
            timeout=timeout,
        )
        # Populated on first detection of a temperature-related 400.
        # Avoids a wasted round-trip on every subsequent call.
        # None  → omit the field entirely (unsupported_parameter)
        # float → send that fixed value (unsupported_value, e.g. 1.0)
        self._temperature_unsupported = False
        self._temperature_fallback: float | None = 1.0

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float | None = 0.0,
    ) -> ModelResponse:
        effective_temp = self._temperature_fallback if self._temperature_unsupported else temperature
        try:
            return self._adapter.complete(
                messages, max_tokens=max_tokens, temperature=effective_temp
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 400:
                raise
            try:
                err = exc.response.json().get("error", {})
                code = err.get("code", "")
                param = err.get("param", "")
                msg = err.get("message", "")
            except Exception:
                raise exc
            is_temperature_error = param == "temperature" or "temperature" in msg
            if not is_temperature_error:
                raise
            if code == "unsupported_parameter":
                # Model does not accept the temperature field at all — omit it.
                fallback: float | None = None
            elif code == "unsupported_value":
                # Model accepts temperature but only the default (1.0).
                fallback = 1.0
            else:
                raise
            logger.info(
                "Model %s rejected temperature=%s (code=%s); retrying with temperature=%s",
                self.model,
                effective_temp,
                code,
                fallback,
            )
            self._temperature_unsupported = True
            self._temperature_fallback = fallback
            return self._adapter.complete(
                messages, max_tokens=max_tokens, temperature=fallback
            )

    def close(self) -> None:
        self._adapter.close()
