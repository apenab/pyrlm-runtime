"""Regression test for VertexAIAdapter._extract_usage thinking-token accounting.

Gemini 2.5 thinking models report reasoning tokens in ``thoughts_token_count``,
which ``candidates_token_count`` excludes but ``total_token_count`` includes.
The adapter must fold thoughts into completion_tokens so the reasoning cost is
not undercounted and prompt + completion == total. Uses ``__new__`` so no
network or GCP credentials are touched.
"""

from __future__ import annotations

import logging

import pytest

try:
    from pyrlm_runtime.adapters.vertex_ai import VertexAIAdapter

    VERTEX_AVAILABLE = True
except Exception:  # pragma: no cover - vertex deps missing
    VERTEX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not VERTEX_AVAILABLE, reason="vertexai not installed")


def _adapter() -> "VertexAIAdapter":
    adapter = VertexAIAdapter.__new__(VertexAIAdapter)
    adapter.logger = logging.getLogger("test_vertex_usage")
    return adapter


class _UsageMeta:
    def __init__(self, prompt: int, candidates: int, thoughts: int, total: int) -> None:
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.thoughts_token_count = thoughts
        self.total_token_count = total


class _Response:
    def __init__(self, usage_metadata: object) -> None:
        self.usage_metadata = usage_metadata


def test_usage_includes_thinking_tokens() -> None:
    # prompt=100, answer=20, thinking=80, total=200
    resp = _Response(_UsageMeta(prompt=100, candidates=20, thoughts=80, total=200))
    usage = _adapter()._extract_usage(resp, text="ans", messages=[])
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 100  # 20 answer + 80 thinking
    assert usage.total_tokens == 200


def test_usage_without_thoughts_field() -> None:
    """No thoughts_token_count attribute → completion is just candidates."""

    class _NoThoughts:
        prompt_token_count = 50
        candidates_token_count = 10
        total_token_count = 60

    usage = _adapter()._extract_usage(_Response(_NoThoughts()), text="ans", messages=[])
    assert usage.completion_tokens == 10
    assert usage.total_tokens == 60
