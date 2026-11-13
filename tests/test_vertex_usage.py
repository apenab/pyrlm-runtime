"""Regression tests for VertexAIAdapter._extract_usage token accounting.

Gemini 2.5 thinking models report reasoning tokens in ``thoughts_token_count``,
which ``candidates_token_count`` excludes but ``total_token_count`` includes.
The adapter must fold thoughts into completion_tokens so the reasoning cost is
not undercounted and prompt + completion == total. Responses are real
``google.genai`` types parsed from wire-format JSON, and the adapter is built
with ``__new__`` so no network or GCP credentials are touched.
"""

from __future__ import annotations

import logging

import pytest

try:
    from google.genai import types

    from pyrlm_runtime.adapters.vertex_ai import VertexAIAdapter

    VERTEX_AVAILABLE = True
except ImportError:  # pragma: no cover - google-genai missing
    VERTEX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not VERTEX_AVAILABLE, reason="google-genai not installed")


def _adapter() -> "VertexAIAdapter":
    adapter = VertexAIAdapter.__new__(VertexAIAdapter)
    adapter.logger = logging.getLogger("test_vertex_usage")
    return adapter


def _response(usage: dict | None):
    payload = {"usageMetadata": usage} if usage is not None else {}
    return types.GenerateContentResponse.model_validate(payload)


def test_usage_includes_thinking_tokens() -> None:
    resp = _response(
        {
            "promptTokenCount": 100,
            "candidatesTokenCount": 20,
            "thoughtsTokenCount": 80,
            "totalTokenCount": 200,
        }
    )
    usage = _adapter()._extract_usage(resp, text="ans", messages=[])
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 100  # 20 answer + 80 thinking
    assert usage.total_tokens == 200


def test_usage_without_thoughts_field() -> None:
    resp = _response({"promptTokenCount": 50, "candidatesTokenCount": 10, "totalTokenCount": 60})
    usage = _adapter()._extract_usage(resp, text="ans", messages=[])
    assert usage.completion_tokens == 10
    assert usage.total_tokens == 60


def test_usage_without_candidates_count() -> None:
    """A MAX_TOKENS response can omit candidatesTokenCount; the reported counts
    must still be used instead of falling back to the word-count estimate."""
    resp = _response({"promptTokenCount": 50, "thoughtsTokenCount": 30, "totalTokenCount": 80})
    usage = _adapter()._extract_usage(resp, text="", messages=[])
    assert usage.prompt_tokens == 50
    assert usage.completion_tokens == 30
    assert usage.total_tokens == 80


def test_usage_falls_back_to_estimate_without_metadata() -> None:
    usage = _adapter()._extract_usage(
        _response(None), text="two words", messages=[{"role": "user", "content": "a b c"}]
    )
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (3, 2, 5)
