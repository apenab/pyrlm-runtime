"""Regression tests for VertexAIAdapter._extract_text.

The RLM routinely gets multi-part responses (code block + FINAL_VAR sentinel),
empty ones (finish_reason=MAX_TOKENS), and thinking parts from Gemini 2.5.
``_extract_text`` must join the answer parts, skip the rest, and never raise.
Responses are real ``google.genai`` types parsed from wire-format JSON, and the
adapter is built with ``__new__`` so no network or GCP credentials are touched.
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
    adapter.logger = logging.getLogger("test_vertex")
    return adapter


def _response(parts: list[dict] | None, finish_reason: str | None = "STOP"):
    candidate: dict = {}
    if parts is not None:
        candidate["content"] = {"role": "model", "parts": parts}
    if finish_reason:
        candidate["finishReason"] = finish_reason
    return types.GenerateContentResponse.model_validate({"candidates": [candidate]})


def test_extract_text_simple() -> None:
    assert _adapter()._extract_text(_response([{"text": "hello world"}])) == "hello world"


def test_extract_text_empty_text_returns_empty_string() -> None:
    assert _adapter()._extract_text(_response([{"text": ""}])) == ""


def test_extract_text_multipart_joins_parts() -> None:
    resp = _response([{"text": "print(42)"}, {"text": "FINAL_VAR: ans"}])
    assert _adapter()._extract_text(resp) == "print(42)\nFINAL_VAR: ans"


def test_extract_text_max_tokens_no_parts_returns_empty() -> None:
    assert _adapter()._extract_text(_response(None, finish_reason="MAX_TOKENS")) == ""


def test_extract_text_no_candidates_returns_empty() -> None:
    resp = types.GenerateContentResponse.model_validate({})
    assert _adapter()._extract_text(resp) == ""


def test_extract_text_skips_non_text_part_keeps_good_parts() -> None:
    """A function_call part must not discard the text parts around it."""
    resp = _response([{"text": "good output"}, {"functionCall": {"name": "f", "args": {}}}])
    assert _adapter()._extract_text(resp) == "good output"


def test_extract_text_skips_thought_part() -> None:
    """The thinking summary of a Gemini 2.5 model must not leak into the answer."""
    resp = _response([{"text": "internal reasoning...", "thought": True}, {"text": "the answer"}])
    assert _adapter()._extract_text(resp) == "the answer"
