"""Regression tests for VertexAIAdapter._build_meta / _normalize_finish_reason.

The RLM root loop reads ``response.meta["finish_reason"]`` (normalized to the
OpenAI-style vocabulary, e.g. "length") to drive truncation handling. The
Vertex adapter must populate ``meta`` with a normalized finish_reason, plus
``content_kind`` and ``reasoning_present``, or that branch is dead for Vertex.
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
    adapter.logger = logging.getLogger("test_vertex_meta")
    return adapter


def _response(parts: list[dict], finish_reason: str):
    candidate = {"content": {"role": "model", "parts": parts}, "finishReason": finish_reason}
    return types.GenerateContentResponse.model_validate({"candidates": [candidate]})


def test_normalize_max_tokens_to_length() -> None:
    """Gemini MAX_TOKENS must map to the loop's "length" so the
    empty-content-with-truncation branch in the RLM loop fires."""
    assert VertexAIAdapter._normalize_finish_reason(types.FinishReason.MAX_TOKENS) == "length"


def test_normalize_stop_to_stop() -> None:
    assert VertexAIAdapter._normalize_finish_reason(types.FinishReason.STOP) == "stop"


def test_normalize_other_passthrough_lowercased() -> None:
    assert VertexAIAdapter._normalize_finish_reason(types.FinishReason.SAFETY) == "safety"


def test_normalize_none() -> None:
    assert VertexAIAdapter._normalize_finish_reason(None) is None


def test_build_meta_max_tokens() -> None:
    meta = _adapter()._build_meta(_response([], "MAX_TOKENS"), text="")
    assert meta["finish_reason"] == "length"
    assert meta["content_kind"] == "empty"
    assert meta["reasoning_present"] is False


def test_build_meta_content_kind_text() -> None:
    meta = _adapter()._build_meta(_response([{"text": "ans"}], "STOP"), text="ans")
    assert meta["finish_reason"] == "stop"
    assert meta["content_kind"] == "text"


def test_build_meta_reasoning_present() -> None:
    resp = _response([{"text": "thinking", "thought": True}, {"text": "ans"}], "STOP")
    assert _adapter()._build_meta(resp, text="ans")["reasoning_present"] is True


def test_build_meta_no_candidates() -> None:
    resp = types.GenerateContentResponse.model_validate({})
    assert _adapter()._build_meta(resp, text="")["finish_reason"] is None
