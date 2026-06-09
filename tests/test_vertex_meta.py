"""Regression tests for VertexAIAdapter._build_meta / _normalize_finish_reason.

The RLM root loop reads ``response.meta["finish_reason"]`` (normalized to the
OpenAI-style vocabulary, e.g. "length") to drive truncation handling. The
Vertex adapter must populate ``meta`` with a normalized finish_reason, plus
``content_kind`` and ``reasoning_present`` — otherwise that branch is dead for
Vertex. These tests build fakes and call the helpers on an un-initialized
instance (``__new__``) so they never touch the network or GCP credentials.
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
    adapter.logger = logging.getLogger("test_vertex_meta")
    return adapter


class _FinishReason:
    """Enum-like Gemini FinishReason with a ``.name`` attribute."""

    def __init__(self, name: str) -> None:
        self.name = name


class _Part:
    def __init__(self, text: str = "", thought: bool = False) -> None:
        self.text = text
        self.thought = thought


class _Content:
    def __init__(self, parts: list[_Part]) -> None:
        self.parts = parts


class _Candidate:
    def __init__(self, parts: list[_Part], finish_reason: object = None) -> None:
        self.content = _Content(parts)
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, candidates: list[_Candidate] | None = None) -> None:
        self.candidates = candidates if candidates is not None else []


def test_normalize_max_tokens_to_length() -> None:
    """Gemini MAX_TOKENS must map to the loop's "length" so the
    empty-content-with-truncation branch in the RLM loop fires."""
    assert VertexAIAdapter._normalize_finish_reason(_FinishReason("MAX_TOKENS")) == "length"


def test_normalize_stop_to_stop() -> None:
    assert VertexAIAdapter._normalize_finish_reason(_FinishReason("STOP")) == "stop"


def test_normalize_other_passthrough_lowercased() -> None:
    assert VertexAIAdapter._normalize_finish_reason(_FinishReason("SAFETY")) == "safety"


def test_normalize_none() -> None:
    assert VertexAIAdapter._normalize_finish_reason(None) is None


def test_build_meta_max_tokens() -> None:
    resp = _Response(candidates=[_Candidate([], finish_reason=_FinishReason("MAX_TOKENS"))])
    meta = _adapter()._build_meta(resp, text="")
    assert meta["finish_reason"] == "length"
    assert meta["content_kind"] == "empty"
    assert meta["reasoning_present"] is False


def test_build_meta_content_kind_text() -> None:
    resp = _Response(candidates=[_Candidate([_Part("ans")], finish_reason=_FinishReason("STOP"))])
    meta = _adapter()._build_meta(resp, text="ans")
    assert meta["finish_reason"] == "stop"
    assert meta["content_kind"] == "text"


def test_build_meta_reasoning_present() -> None:
    resp = _Response(
        candidates=[
            _Candidate(
                [_Part("thinking", thought=True), _Part("ans")],
                finish_reason=_FinishReason("STOP"),
            )
        ]
    )
    meta = _adapter()._build_meta(resp, text="ans")
    assert meta["reasoning_present"] is True


def test_build_meta_no_candidates() -> None:
    meta = _adapter()._build_meta(_Response(candidates=[]), text="")
    assert meta["finish_reason"] is None
