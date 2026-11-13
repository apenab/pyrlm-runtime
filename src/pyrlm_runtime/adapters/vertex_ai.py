"""Vertex AI Gemini adapter for pyrlm_runtime.

This adapter calls Gemini on Google Cloud Vertex AI through the ``google-genai``
SDK, implementing the ModelAdapter protocol.

Requirements:
    - google-genai >= 1.66.0
    - Valid GCP authentication (Application Default Credentials or service account)

Example:
    from pyrlm_runtime.adapters.vertex_ai import VertexAIAdapter

    adapter = VertexAIAdapter(
        project_id="my-gcp-project",
        location="us-central1",
        model="gemini-2.5-pro"
    )

    response = adapter.complete(
        [{"role": "user", "content": "Hello!"}],
        max_tokens=100,
        temperature=0.0
    )
"""

from __future__ import annotations

import logging
import os
import ssl
import time
import warnings
from typing import Any

import httpx
from google import genai
from google.genai import errors, types

from .base import ModelResponse, Usage

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = frozenset({429, 503, 504})


class VertexAIAdapter:
    """Vertex AI Gemini adapter implementing ModelAdapter protocol.

    This adapter converts OpenAI-style message format to Gemini's format
    and handles retry logic for common API errors.

    Attributes:
        project_id: GCP project ID
        location: GCP region (default: "us-central1")
        model_name: Gemini model name (default: "gemini-2.5-pro")
        timeout: Per-call timeout in seconds, on any thread
        max_retries: Maximum number of attempts for transient errors
        api_transport: Kept for compatibility. google-genai is REST only.
        client: The ``google.genai.Client`` the adapter calls
    """

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model: str = "gemini-2.5-pro",
        timeout: int = 600,
        max_retries: int = 3,
        api_transport: str = "rest",
        logger_instance: logging.Logger | None = None,
    ):
        """Initialize Vertex AI adapter.

        Args:
            project_id: GCP project ID
            location: GCP region for Vertex AI
            model: Gemini model identifier
            timeout: Per-call timeout in seconds, on any thread. It is sent as
                ``HttpOptions(timeout=...)``: the HTTP client applies it to the
                connect and to every read, so a server that stops replying fails
                within it, and Vertex receives it as ``X-Server-Timeout``. A
                response that keeps sending bytes can still run longer.
                ``None`` or ``<= 0`` disables it.
            max_retries: Max attempts for 429/503/504 errors and timeouts
            api_transport: Accepted for compatibility. The google-genai SDK
                speaks REST only, which honors ``HTTPS_PROXY`` and a custom CA
                bundle (``SSL_CERT_FILE``, or ``REQUESTS_CA_BUNDLE`` when
                ``SSL_CERT_FILE`` is unset). Any value other than ``"rest"``
                emits a ``DeprecationWarning`` and REST is used.
            logger_instance: Optional custom logger
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_transport = api_transport
        self.logger = logger_instance or logger

        if api_transport != "rest":
            warnings.warn(
                f"VertexAIAdapter(api_transport={api_transport!r}) is ignored: the "
                "google-genai SDK supports REST only.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
            http_options=_ca_bundle_http_options(),
        )
        self.logger.info(
            f"Initialized VertexAI adapter: project={project_id}, "
            f"location={location}, model={model}"
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ModelResponse:
        """Generate completion using Gemini model.

        Args:
            messages: List of message dicts with "role" and "content" keys.
                     Supported roles: "system", "user", "assistant"
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)

        Returns:
            ModelResponse with text, usage, and model_id

        Raises:
            TimeoutError: The last attempt timed out.
            Exception: On other API errors, after retries for retryable ones.
        """
        self.logger.debug(
            f"complete() called: messages={len(messages)}, "
            f"max_tokens={max_tokens}, temperature={temperature}"
        )

        contents = self._convert_messages(messages)
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            http_options=types.HttpOptions(timeout=self._timeout_ms()),
        )

        attempts = max(1, self.max_retries)
        for attempt in range(1, attempts + 1):
            start_time = time.time()
            try:
                response = self.client.models.generate_content(
                    model=self.model_name, contents=contents, config=config
                )
            except Exception as exc:
                is_timeout = isinstance(exc, (httpx.TimeoutException, TimeoutError))
                retryable = is_timeout or (
                    isinstance(exc, errors.APIError) and exc.code in _RETRYABLE_STATUS_CODES
                )
                if retryable and attempt < attempts:
                    backoff_delay = 2**attempt
                    self.logger.warning(
                        f"Retryable error (attempt {attempt}/{attempts}): {exc!r}. "
                        f"Retrying in {backoff_delay}s..."
                    )
                    time.sleep(backoff_delay)
                    continue
                self.logger.error(f"Completion failed: {exc!r}")
                if is_timeout:
                    raise TimeoutError(
                        f"VertexAI generate_content timed out after {self.timeout}s"
                    ) from exc
                raise

            elapsed = time.time() - start_time
            text = self._extract_text(response)
            usage = self._extract_usage(response, text, messages)
            # The RLM root loop relies on meta["finish_reason"] to detect
            # truncation; without it the truncation-handling branch is dead.
            meta = self._build_meta(response, text)
            self.logger.info(f"Completion successful: {usage.total_tokens} tokens, {elapsed:.2f}s")
            return ModelResponse(text=text, usage=usage, model_id=self.model_name, meta=meta)

        raise AssertionError("unreachable: the loop returns or raises")

    def _timeout_ms(self) -> int | None:
        if self.timeout is None or self.timeout <= 0:
            return None
        return max(1, round(self.timeout * 1000))

    def _extract_text(self, response: Any) -> str:
        """Join the answer text parts of the first candidate with newlines.

        Thinking parts (``part.thought``) and parts without text (for example a
        function call) are skipped. A response with no candidates or no parts,
        as with ``finish_reason=MAX_TOKENS``, yields ``""``.
        """
        candidates = response.candidates or []
        if not candidates:
            return ""
        content = candidates[0].content
        parts = (content.parts if content else None) or []
        if not parts:
            finish = candidates[0].finish_reason
            if finish is not None:
                self.logger.warning(
                    f"Empty response parts (finish_reason={finish}), returning empty text"
                )
            return ""
        return "\n".join(p.text for p in parts if p.text and not p.thought)

    def _build_meta(self, response: Any, text: str) -> dict[str, Any]:
        """Build ModelResponse.meta from a Gemini response.

        The RLM root loop reads meta["finish_reason"] (normalized to the
        OpenAI-style vocabulary, e.g. "length"/"stop") to drive truncation
        handling, plus content_kind and reasoning_present for diagnostics.
        This mirrors the contract the OpenAI-compatible adapters provide.

        Returns a dict with:
            finish_reason: normalized reason ("length" for MAX_TOKENS, "stop"
                for STOP, otherwise the lowercased Gemini enum name, e.g.
                "safety"). None when there are no candidates.
            content_kind: "text" when visible text was produced, else "empty".
            reasoning_present: True when any response part is a thinking part.
        """
        finish_reason = None
        reasoning_present = False
        candidates = response.candidates or []
        if candidates:
            finish_reason = self._normalize_finish_reason(candidates[0].finish_reason)
            content = candidates[0].content
            parts = (content.parts if content else None) or []
            reasoning_present = any(p.thought for p in parts)

        return {
            "finish_reason": finish_reason,
            "content_kind": "text" if text else "empty",
            "reasoning_present": reasoning_present,
        }

    @staticmethod
    def _normalize_finish_reason(raw: Any) -> str | None:
        """Map a Gemini FinishReason to the loop's OpenAI-style vocabulary.

        Gemini uses MAX_TOKENS where OpenAI uses "length", and STOP where
        OpenAI uses "stop". The RLM loop only special-cases "length", so the
        mapping must be applied here rather than passing the raw enum through.
        Other reasons (SAFETY, RECITATION, ...) pass through lowercased.
        """
        if raw is None:
            return None
        name = getattr(raw, "name", None) or str(raw)
        mapping = {"MAX_TOKENS": "length", "STOP": "stop"}
        return mapping.get(name, name.lower())

    def _convert_messages(self, messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Convert OpenAI-style messages to Gemini format.

        OpenAI roles: system, user, assistant
        Gemini roles: user, model

        System messages are prepended to the first user message, which keeps
        the prompt the model sees identical to earlier releases of this adapter.

        Args:
            messages: List with "role" and "content" keys

        Returns:
            List of Gemini content dicts with "role" and "parts"

        Gemini format:
            {"role": "user", "parts": [{"text": "..."}]}
            {"role": "model", "parts": [{"text": "..."}]}
        """
        system_instructions = []
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instructions.append(content)
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
            else:
                self.logger.warning(f"Unknown role '{role}', treating as user")
                contents.append({"role": "user", "parts": [{"text": content}]})

        if system_instructions and contents:
            combined_instruction = "\n\n".join(system_instructions)
            for content_msg in contents:
                if content_msg["role"] == "user":
                    original_text = content_msg["parts"][0]["text"]
                    content_msg["parts"][0]["text"] = f"{combined_instruction}\n\n{original_text}"
                    break

        return contents

    def _extract_usage(self, response: Any, text: str, messages: list[dict[str, str]]) -> Usage:
        """Extract token usage from ``response.usage_metadata``.

        Gemini 2.5 thinking models report reasoning tokens in
        ``thoughts_token_count``, which ``candidates_token_count`` excludes and
        ``total_token_count`` includes. They are folded into completion tokens
        so prompt + completion == total and reasoning cost is not undercounted.
        Falls back to a word-count estimate when there is no usage metadata.
        """
        usage_meta = response.usage_metadata
        if usage_meta:
            prompt_tokens = usage_meta.prompt_token_count or 0
            completion_tokens = (usage_meta.candidates_token_count or 0) + (
                usage_meta.thoughts_token_count or 0
            )
            total_tokens = usage_meta.total_token_count or prompt_tokens + completion_tokens
            return Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        prompt_tokens = self._estimate_tokens(messages)
        completion_tokens = self._estimate_tokens(text)
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _estimate_tokens(self, text: Any) -> int:
        """Estimate token count using simple word-based heuristic.

        Args:
            text: String or list of message dicts

        Returns:
            Estimated token count (word count)
        """
        if isinstance(text, str):
            return len(text.split())
        if isinstance(text, list):
            return sum(len(str(msg.get("content", "")).split()) for msg in text)
        return len(str(text).split())


def _ca_bundle_http_options() -> types.HttpOptions | None:
    """google-genai reads ``SSL_CERT_FILE`` but not ``REQUESTS_CA_BUNDLE``, which
    the requests-based SDK this adapter used before honored. Keep honoring it.
    """
    bundle = os.environ.get("REQUESTS_CA_BUNDLE")
    if not bundle or os.environ.get("SSL_CERT_FILE"):
        return None
    ctx = ssl.create_default_context(cafile=bundle, capath=os.environ.get("SSL_CERT_DIR"))
    return types.HttpOptions(client_args={"verify": ctx})
