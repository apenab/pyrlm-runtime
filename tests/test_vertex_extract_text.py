"""Regression tests for VertexAIAdapter._extract_text.

The vertexai ``response.text`` accessor raises ``ValueError`` when the response
has multiple parts (code block + FINAL_VAR sentinel, which the RLM routinely
emits) or zero parts (finish_reason=MAX_TOKENS). ``_extract_text`` must degrade
gracefully in those cases instead of letting the ValueError propagate.

These tests call ``_extract_text`` on an un-initialized instance (``__new__``)
so they never touch the network or GCP credentials.
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
    """Build an adapter without running __init__ (no network / GCP creds)."""
    adapter = VertexAIAdapter.__new__(VertexAIAdapter)
    adapter.logger = logging.getLogger("test_vertex")
    return adapter


class _Part:
    def __init__(self, text: str) -> None:
        self.text = text


class _Content:
    def __init__(self, parts: list[_Part]) -> None:
        self.parts = parts


class _Candidate:
    def __init__(self, parts: list[_Part], finish_reason: object = None) -> None:
        self.content = _Content(parts)
        self.finish_reason = finish_reason


class _Response:
    """Fake Gemini response. ``text`` raises ValueError when ``raises`` is set."""

    def __init__(
        self,
        *,
        text: str | None = None,
        raises: bool = False,
        candidates: list[_Candidate] | None = None,
    ) -> None:
        self._text = text
        self._raises = raises
        self.candidates = candidates if candidates is not None else []

    @property
    def text(self) -> str:
        if self._raises:
            raise ValueError("Multiple content parts are not supported")
        return self._text or ""


def test_extract_text_simple() -> None:
    """Single-part response returns its text unchanged."""
    assert _adapter()._extract_text(_Response(text="hello world")) == "hello world"


def test_extract_text_empty_returns_empty_string() -> None:
    """response.text == '' returns '' (not None)."""
    assert _adapter()._extract_text(_Response(text="")) == ""


def test_extract_text_multipart_joins_parts() -> None:
    """When response.text raises (multi-part), parts are joined — no exception."""
    resp = _Response(
        raises=True,
        candidates=[_Candidate([_Part("print(42)"), _Part("FINAL_VAR: ans")])],
    )
    assert _adapter()._extract_text(resp) == "print(42)\nFINAL_VAR: ans"


def test_extract_text_max_tokens_no_parts_returns_empty() -> None:
    """finish_reason=MAX_TOKENS with no parts returns '' instead of raising."""
    resp = _Response(raises=True, candidates=[_Candidate([], finish_reason="MAX_TOKENS")])
    assert _adapter()._extract_text(resp) == ""


def test_extract_text_no_candidates_returns_empty() -> None:
    """No candidates at all returns '' instead of raising."""
    assert _adapter()._extract_text(_Response(raises=True, candidates=[])) == ""


class _RaisingPart:
    """A non-text part (e.g. a function_call) whose ``.text`` raises."""

    @property
    def text(self) -> str:
        raise ValueError("Part has no text (function_call)")


def test_extract_text_skips_bad_part_keeps_good_parts() -> None:
    """Regression: one part that raises on ``.text`` access must not discard the
    surrounding good text parts. Before the fix the whole list comprehension
    raised and the fallback returned '' — losing the real output."""
    resp = _Response(
        raises=True,
        candidates=[_Candidate([_Part("good output"), _RaisingPart()])],
    )
    assert _adapter()._extract_text(resp) == "good output"
