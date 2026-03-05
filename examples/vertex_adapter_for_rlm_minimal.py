"""Vertex AI adapter compatible with rlm-minimal's OpenAIClient protocol.

This module provides adapters that integrate Google Cloud Vertex AI's Gemini models
with the rlm-minimal library, which expects an OpenAIClient-compatible interface.

Key Differences from pyrlm_runtime adapter:
    - Returns str (not ModelResponse)
    - Accepts messages as str or list[dict]
    - No Usage metadata in return value

Example:
    from vertex_adapter_for_rlm_minimal import InstrumentedVertexAIClient
    from rlm.rlm_repl import RLM_REPL

    client = InstrumentedVertexAIClient(
        project_id="my-project",
        model="gemini-2.5-pro"
    )

    rlm = RLM_REPL(
        client=client,
        recursive_client=client,
        max_iterations=20
    )

    output = rlm.completion(context="...", query="...")
"""

from __future__ import annotations

import logging
import time
from typing import Any

from google.cloud import aiplatform
from vertexai.generative_models import GenerationConfig, GenerativeModel

logger = logging.getLogger(__name__)


class VertexAIClientForRLMMinimal:
    """Vertex AI adapter compatible with rlm-minimal's OpenAIClient protocol.

    This adapter mimics the OpenAIClient interface expected by rlm-minimal:
        - completion(messages: str | list[dict], max_tokens, **kwargs) -> str

    Attributes:
        project_id: GCP project ID
        location: GCP region
        model: Gemini model name
        logger: Logger instance

    Note:
        Unlike pyrlm_runtime's VertexAIAdapter, this returns plain str
        and does not expose Usage metadata directly.
    """

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model: str = "gemini-2.5-pro",
        logger_instance: logging.Logger | None = None,
    ):
        """Initialize Vertex AI client for rlm-minimal.

        Args:
            project_id: GCP project ID
            location: GCP region
            model: Gemini model identifier
            logger_instance: Optional custom logger
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model
        self.logger = logger_instance or logger

        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        self.logger.debug(
            f"VertexAIClientForRLMMinimal initialized: project={project_id}, model={model}"
        )

        # Initialize Gemini model
        self._init_model()

    def _init_model(self):
        """Initialize Gemini model instance."""
        self.model = GenerativeModel(self.model_name)
        self.logger.debug(f"Created GenerativeModel: {self.model_name}")

    def completion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        """Generate completion (OpenAIClient-compatible signature).

        Args:
            messages: Either a prompt string or list of message dicts
            max_tokens: Maximum tokens to generate (default: 2048)
            **kwargs: Additional parameters (temperature, etc.)

        Returns:
            Generated text as string

        Raises:
            Exception: On API errors
        """
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = max_tokens or 2048

        # Convert input to Gemini format
        if isinstance(messages, str):
            # Direct string prompt
            prompt = messages
            self.logger.debug(f"String prompt: {len(prompt)} chars")
        else:
            # Message list - convert to Gemini format
            contents = self._convert_messages(messages)
            self.logger.debug(f"Message list: {len(messages)} messages")
            # For single user message, simplify to string
            if len(contents) == 1 and contents[0].get("role") == "user":
                prompt = contents[0]["parts"][0]["text"]
            else:
                # Multi-turn: use contents directly
                prompt = contents

        # Configure generation
        generation_config = GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        try:
            # Generate content (system instruction already embedded in messages)
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
            )

            text = response.text if response.text else ""
            self.logger.debug(f"Generated {len(text)} chars")
            return text

        except Exception as exc:
            self.logger.error(f"Completion failed: {exc}", exc_info=True)
            raise

    def _convert_messages(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-style messages to Gemini format.

        System messages are prepended to the first user message for compatibility.

        Args:
            messages: List with "role" and "content" keys

        Returns:
            List of Gemini content dicts with "role" and "parts"
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

        # Prepend system instructions to first user message
        if system_instructions and contents:
            combined_instruction = "\n\n".join(system_instructions)
            for i, content_msg in enumerate(contents):
                if content_msg["role"] == "user":
                    original_text = content_msg["parts"][0]["text"]
                    contents[i]["parts"][0]["text"] = (
                        f"{combined_instruction}\n\n{original_text}"
                    )
                    break

        return contents


class InstrumentedVertexAIClient(VertexAIClientForRLMMinimal):
    """Instrumented wrapper that tracks token usage and call metrics.

    This subclass wraps VertexAIClientForRLMMinimal and tracks all LLM calls,
    enabling token counting and latency analysis for benchmarking purposes.

    Additional Attributes:
        total_tokens_used: Cumulative estimated tokens across all calls
        call_count: Total number of completion() calls
        call_log: List of dicts with per-call metrics

    Example:
        client = InstrumentedVertexAIClient(project_id="my-project")
        output = client.completion("Hello")
        print(f"Tokens used: {client.total_tokens_used}")
        print(f"Call log: {client.call_log}")
    """

    def __init__(self, *args, **kwargs):
        """Initialize instrumented client with tracking state."""
        super().__init__(*args, **kwargs)
        self.total_tokens_used = 0
        self.call_count = 0
        self.call_log: list[dict[str, Any]] = []
        self.logger.debug("InstrumentedVertexAIClient: tracking enabled")

    def completion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        """Generate completion with automatic metrics tracking.

        Args:
            messages: Prompt string or message list
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated text string

        Side Effects:
            Updates total_tokens_used, call_count, and call_log
        """
        start_time = time.time()

        # Call parent's completion method
        response = super().completion(messages, max_tokens, **kwargs)

        elapsed = time.time() - start_time

        # Estimate tokens
        prompt_tokens = self._estimate_tokens(messages)
        completion_tokens = self._estimate_tokens(response)
        total = prompt_tokens + completion_tokens

        # Update tracking state
        self.total_tokens_used += total
        self.call_count += 1
        self.call_log.append(
            {
                "call_index": self.call_count,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total,
                "elapsed": elapsed,
                "timestamp": start_time,
            }
        )

        self.logger.debug(
            f"Call {self.call_count}: {total} tokens, {elapsed:.2f}s, "
            f"cumulative: {self.total_tokens_used} tokens"
        )

        return response

    def _estimate_tokens(self, text: Any) -> int:
        """Estimate token count using simple word-based heuristic.

        Args:
            text: String, list of dicts (messages), or other

        Returns:
            Estimated token count
        """
        if isinstance(text, str):
            return len(text.split())
        if isinstance(text, list):
            # List of message dicts
            return sum(len(str(msg.get("content", "")).split()) for msg in text)
        return len(str(text).split())

    def reset_metrics(self):
        """Reset all tracking metrics to zero."""
        self.total_tokens_used = 0
        self.call_count = 0
        self.call_log = []
        self.logger.debug("Metrics reset")
