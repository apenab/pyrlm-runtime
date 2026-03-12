from __future__ import annotations

import os
import re
from urllib.parse import urlparse

from .base import ModelAdapter, ModelResponse
from .generic_chat import GenericChatAdapter


def _azure_payload_builder(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    model: str | None,
) -> dict[str, object]:
    """Classic deployment-URL style: model is in the URL, not the body."""
    del model
    return {
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }


def _azure_v1_payload_builder(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    model: str | None,
) -> dict[str, object]:
    """OpenAI v1-compat style: model goes in the request body."""
    payload: dict[str, object] = {
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }
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

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ModelResponse:
        return self._adapter.complete(messages, max_tokens=max_tokens, temperature=temperature)

    def close(self) -> None:
        self._adapter.close()
