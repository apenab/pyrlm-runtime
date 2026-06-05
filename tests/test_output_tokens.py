"""Tests for the output-token cap rename and its deprecated aliases.

``RLM.max_tokens`` / ``subcall_max_tokens`` were renamed to
``max_output_tokens`` / ``subcall_max_output_tokens``. The old names remain as
deprecated aliases that still work but emit a ``DeprecationWarning``.
"""

from __future__ import annotations

import warnings

import pytest

from pyrlm_runtime import Context, RLM
from pyrlm_runtime.adapters import FakeAdapter, ModelResponse, Usage


class _MaxTokensRecordingAdapter:
    """Adapter that records the max_tokens passed to each completion call."""

    def __init__(self, text: str = "FINAL: done") -> None:
        self._text = text
        self.max_tokens_calls: list[int] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ModelResponse:
        self.max_tokens_calls.append(max_tokens)
        return ModelResponse(text=self._text, usage=Usage(1, 1, 2))


def test_new_defaults() -> None:
    """The renamed fields carry the raised defaults (4096 / 1024)."""
    rlm = RLM(adapter=FakeAdapter(script=["FINAL: done"]))
    assert rlm.max_output_tokens == 4096
    assert rlm.subcall_max_output_tokens == 1024


def test_max_tokens_alias_warns_and_maps() -> None:
    """The deprecated max_tokens alias warns and populates max_output_tokens."""
    with pytest.warns(DeprecationWarning, match="max_tokens is deprecated"):
        rlm = RLM(adapter=FakeAdapter(script=["FINAL: done"]), max_tokens=777)
    assert rlm.max_output_tokens == 777


def test_subcall_max_tokens_alias_warns_and_maps() -> None:
    """The deprecated subcall_max_tokens alias warns and maps to the new field."""
    with pytest.warns(DeprecationWarning, match="subcall_max_tokens is deprecated"):
        rlm = RLM(adapter=FakeAdapter(script=["FINAL: done"]), subcall_max_tokens=333)
    assert rlm.subcall_max_output_tokens == 333


def test_new_names_do_not_warn() -> None:
    """Using the new names must not emit a deprecation warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        RLM(
            adapter=FakeAdapter(script=["FINAL: done"]),
            max_output_tokens=2048,
            subcall_max_output_tokens=512,
        )


def test_root_call_uses_max_output_tokens() -> None:
    """Regression: the root LLM call must pass max_output_tokens as max_tokens
    (ensures the internal rename was applied, not just the field renamed)."""
    adapter = _MaxTokensRecordingAdapter()
    rlm = RLM(adapter=adapter, max_output_tokens=1234)
    rlm.run("Q?", Context.from_text("ctx"))
    assert adapter.max_tokens_calls, "adapter was never called"
    assert adapter.max_tokens_calls[0] == 1234


def test_alias_flows_through_to_root_call() -> None:
    """The deprecated alias still drives the actual per-call output cap."""
    adapter = _MaxTokensRecordingAdapter()
    with pytest.warns(DeprecationWarning):
        rlm = RLM(adapter=adapter, max_tokens=999)
    rlm.run("Q?", Context.from_text("ctx"))
    assert adapter.max_tokens_calls[0] == 999
