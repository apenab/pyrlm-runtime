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
import signal
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

import vertexai
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
        api_transport: SDK transport, "rest" (default) or "grpc"
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
            timeout: Request timeout in seconds
            max_retries: Max retry attempts for 429/503/504 errors
            api_transport: SDK transport to use. ``"rest"`` (the default) uses
                HTTP/REST, which honors ``HTTPS_PROXY`` and the system CA bundle
                (``REQUESTS_CA_BUNDLE`` / ``SSL_CERT_FILE``) — required behind
                corporate proxies with a self-signed TLS certificate — and is
                immune to the gRPC pollset deadlock that ``_call_timeout_guard``
                works around. ``"grpc"`` restores the previous transport.

                NOTE: the transport is configured via ``vertexai.init``, which
                writes process-global SDK state. Do not mix transports across
                multiple adapters in the same process — the last ``init`` wins.
            logger_instance: Optional custom logger
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_transport = api_transport
        self.logger = logger_instance or logger

        # Initialize Vertex AI. vertexai.init (not aiplatform.init) is what
        # configures GenerativeModel, and it accepts api_transport so REST can
        # be selected. A single init call sets project/location/transport.
        vertexai.init(project=project_id, location=location, api_transport=api_transport)
        self.logger.info(
            f"Initialized VertexAI adapter: project={project_id}, "
            f"location={location}, model={model}, transport={api_transport}"
        )

        # Initialize Gemini model
        self._init_model()

    def _init_model(self):
        """Initialize or reinitialize the Gemini model instance."""
        self.model = GenerativeModel(self.model_name)
        self.logger.debug(f"Created GenerativeModel instance: {self.model_name}")

    @contextmanager
    def _call_timeout_guard(self) -> Iterator[None]:
        """Wall-clock guard around a Vertex generate_content call.

        Uses SIGALRM on Unix main thread to interrupt the underlying gRPC
        ``poll()`` syscall. The motivation is real: long-running bench loops
        on Vertex regularly deadlock inside gRPC's pollset_work waiting on a
        condition that never arrives, with the main thread blocked in
        ``poll()`` indefinitely. The SDK provides no per-call timeout, so we
        impose one here.

        No-op when not on Unix main thread (signals can't be delivered);
        the call may still hang there, but that's the same behavior as
        before this change. Most importantly, the bench (RLM main loop)
        runs on the main thread on macOS/Linux, which is exactly where the
        deadlock was observed.
        """
        t = self.timeout
        can_alarm = (
            t is not None
            and t > 0
            and hasattr(signal, "SIGALRM")
            and hasattr(signal, "setitimer")
            and threading.current_thread() is threading.main_thread()
        )
        if not can_alarm:
            yield
            return

        # If an outer SIGALRM/ITIMER_REAL timer is already armed (e.g. a nested
        # Vertex call higher up the stack), do not arm a nested one: re-arming
        # would cancel the outer timer when this block exits. Let the outer
        # timeout cover this call instead. (The PythonREPL exec guard uses
        # ITIMER_PROF, a different timer, so it does not interfere here.)
        if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
            yield
            return

        def _on_alarm(signum: int, frame: Any) -> None:  # noqa: ARG001
            raise TimeoutError(f"VertexAI generate_content exceeded {t}s (gRPC deadlock guard)")

        old = signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, float(t))
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)

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

                # Wall-clock guard around generate_content: the vertexai SDK
                # does not surface a per-call timeout, and its underlying gRPC
                # pollset can deadlock on stale connections. Without this guard
                # the loop hangs in poll() indefinitely. The guard re-inits the
                # model on timeout so subsequent calls start from a clean
                # client.
                with self._call_timeout_guard():
                    response = self.model.generate_content(
                        contents,
                        generation_config=generation_config,
                    )

                elapsed = time.time() - start_time

                # Extract text — response.text raises ValueError when:
                # - finish_reason=MAX_TOKENS with zero output parts
                # - response has multiple text parts (code block + FINAL_VAR sentinel)
                text = self._extract_text(response)

                # Extract usage metadata if available
                usage = self._extract_usage(response, text, messages)

                # Build response meta (finish_reason, content_kind, reasoning_present).
                # The RLM root loop relies on meta["finish_reason"] to detect
                # truncation; without it the truncation-handling branch is dead.
                meta = self._build_meta(response, text)

                self.logger.info(
                    f"Completion successful: {usage.total_tokens} tokens, {elapsed:.2f}s"
                )

                return ModelResponse(
                    text=text,
                    usage=usage,
                    model_id=self.model_name,
                    meta=meta,
                )

            except TimeoutError as exc:
                last_error = exc
                self.logger.warning(
                    "VertexAI call timed out after %ss (attempt %d/%d); "
                    "re-initializing model to recover from stale gRPC state.",
                    self.timeout,
                    attempt + 1,
                    self.max_retries,
                )
                self._init_model()
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue
                self.logger.error("VertexAI call timed out after all retries")
                raise

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

    def _extract_text(self, response: Any) -> str:
        """Extract text from Gemini response, handling edge cases.

        response.text raises ValueError when:
        - finish_reason=MAX_TOKENS with no output parts (empty response)
        - response has multiple text parts (code block + FINAL_VAR sentinel)

        In the multi-part case we join all text parts; in the empty case we
        return "" so the caller can treat it as an empty completion.
        """
        try:
            text = response.text
            return text if text else ""
        except ValueError:
            pass

        # Fallback: extract parts manually from candidates
        try:
            candidates = response.candidates
            if not candidates:
                return ""
            parts = candidates[0].content.parts if candidates[0].content else []
            if not parts:
                finish = getattr(candidates[0], "finish_reason", None)
                if finish is not None:
                    self.logger.warning(
                        f"Empty response parts (finish_reason={finish}), returning empty text"
                    )
                return ""
            # Join all answer text parts (handles multi-part responses). Read each
            # part defensively: a non-text part (e.g. a function_call) can raise
            # when its .text is accessed, and one bad part must not discard the
            # good text parts alongside it. Skip thinking parts (part.thought is
            # True on Gemini 2.5 thinking models) so the reasoning summary does
            # not leak into the visible answer text.
            texts = []
            for p in parts:
                try:
                    if getattr(p, "thought", False):
                        continue
                    if hasattr(p, "text") and p.text:
                        texts.append(p.text)
                except Exception:
                    continue
            return "\n".join(texts)
        except Exception as exc:
            self.logger.warning(f"Could not extract text from response parts: {exc}")
            return ""

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
        try:
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                finish_reason = self._normalize_finish_reason(
                    getattr(candidates[0], "finish_reason", None)
                )
                content = getattr(candidates[0], "content", None)
                parts = getattr(content, "parts", []) if content else []
                reasoning_present = any(getattr(p, "thought", False) for p in parts)
        except Exception as exc:
            self.logger.debug(f"Could not build response meta: {exc}")

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
                    contents[i]["parts"][0]["text"] = f"{combined_instruction}\n\n{original_text}"
                    break

        return contents

    def _extract_usage(self, response: Any, text: str, messages: list[dict[str, str]]) -> Usage:
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

                # Gemini 2.5 thinking models report reasoning tokens separately
                # in thoughts_token_count; candidates_token_count excludes them
                # while total_token_count includes them. Fold thoughts into
                # completion so prompt + completion == total and the cost of
                # reasoning is not undercounted.
                thoughts_tokens = getattr(usage_meta, "thoughts_token_count", 0) or 0
                completion_tokens += thoughts_tokens

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
