"""Vertex AI Gemini adapter for pyrlm_runtime.

This adapter integrates Google Cloud Vertex AI's Gemini models with pyrlm_runtime,
implementing the ModelAdapter protocol to provide LLM completion capabilities.

Requirements:
    - google-cloud-aiplatform >= 1.60.0
    - vertexai >= 1.60.0
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
import time
from typing import Any

from google.cloud import aiplatform
from vertexai.generative_models import GenerationConfig, GenerativeModel

from .base import ModelResponse, Usage

logger = logging.getLogger(__name__)


class VertexAIAdapter:
    """Vertex AI Gemini adapter implementing ModelAdapter protocol.

    This adapter converts OpenAI-style message format to Gemini's format
    and handles retry logic for common API errors.

    Attributes:
        project_id: GCP project ID
        location: GCP region (default: "us-central1")
        model: Gemini model name (default: "gemini-2.5-pro")
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts for transient errors
    """

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model: str = "gemini-2.5-pro",
        timeout: int = 600,
        max_retries: int = 3,
        logger_instance: logging.Logger | None = None,
    ):
        """Initialize Vertex AI adapter.

        Args:
            project_id: GCP project ID
            location: GCP region for Vertex AI
            model: Gemini model identifier
            timeout: Request timeout in seconds
            max_retries: Max retry attempts for 429/503/504 errors
            logger_instance: Optional custom logger
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logger_instance or logger

        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        self.logger.info(
            f"Initialized VertexAI adapter: project={project_id}, "
            f"location={location}, model={model}"
        )

        # Initialize Gemini model
        self._init_model()

    def _init_model(self):
        """Initialize or reinitialize the Gemini model instance."""
        self.model = GenerativeModel(self.model_name)
        self.logger.debug(f"Created GenerativeModel instance: {self.model_name}")

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
            Exception: On API errors after retries exhausted
        """
        self.logger.debug(
            f"complete() called: messages={len(messages)}, "
            f"max_tokens={max_tokens}, temperature={temperature}"
        )

        # Convert OpenAI-style messages to Gemini format
        contents = self._convert_messages(messages)

        # Configure generation
        generation_config = GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        # Attempt generation with retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                # Generate content
                response = self.model.generate_content(
                    contents,
                    generation_config=generation_config,
                )

                elapsed = time.time() - start_time

                # Extract text
                text = response.text if response.text else ""

                # Extract usage metadata if available
                usage = self._extract_usage(response, text, messages)

                self.logger.info(
                    f"Completion successful: {usage.total_tokens} tokens, {elapsed:.2f}s"
                )

                return ModelResponse(
                    text=text,
                    usage=usage,
                    model_id=self.model_name,
                )

            except Exception as exc:
                last_error = exc
                error_msg = str(exc).lower()

                # Check if error is retryable
                if any(code in error_msg for code in ["429", "503", "504", "timeout"]):
                    backoff_delay = 2 ** (attempt + 1)  # Exponential backoff
                    self.logger.warning(
                        f"Retryable error (attempt {attempt + 1}/{self.max_retries}): {exc}. "
                        f"Retrying in {backoff_delay}s..."
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(backoff_delay)
                        continue

                # Non-retryable error or retries exhausted
                self.logger.error(f"Completion failed: {exc}", exc_info=True)
                raise

        # Should not reach here, but handle exhausted retries
        raise last_error

    def _convert_messages(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-style messages to Gemini format.

        OpenAI roles: system, user, assistant
        Gemini roles: user, model

        System messages are prepended to the first user message to ensure
        compatibility across all Vertex AI SDK versions.

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
                # Collect system messages to prepend to first user message
                system_instructions.append(content)
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                # Gemini uses "model" role for assistant messages
                contents.append({"role": "model", "parts": [{"text": content}]})
            else:
                self.logger.warning(f"Unknown role '{role}', treating as user")
                contents.append({"role": "user", "parts": [{"text": content}]})

        # Prepend system instructions to first user message
        if system_instructions and contents:
            combined_instruction = "\n\n".join(system_instructions)
            # Find first user message
            for i, content_msg in enumerate(contents):
                if content_msg["role"] == "user":
                    original_text = content_msg["parts"][0]["text"]
                    contents[i]["parts"][0]["text"] = (
                        f"{combined_instruction}\n\n{original_text}"
                    )
                    break

        return contents

    def _extract_usage(
        self, response: Any, text: str, messages: list[dict[str, str]]
    ) -> Usage:
        """Extract token usage from Gemini response.

        Gemini API provides usage metadata in response.usage_metadata.
        Falls back to estimation if not available.

        Args:
            response: Gemini API response object
            text: Generated text
            messages: Original messages for prompt token estimation

        Returns:
            Usage object with token counts
        """
        try:
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage_meta = response.usage_metadata
                prompt_tokens = getattr(usage_meta, "prompt_token_count", 0)
                completion_tokens = getattr(usage_meta, "candidates_token_count", 0)
                total_tokens = getattr(usage_meta, "total_token_count", 0)

                # If total not provided, calculate it
                if total_tokens == 0:
                    total_tokens = prompt_tokens + completion_tokens

                return Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
        except Exception as exc:
            self.logger.debug(f"Could not extract usage metadata: {exc}")

        # Fallback: estimate tokens
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
