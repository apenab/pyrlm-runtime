"""Tests for tiktoken-based token counting and context-limit lookup."""

from __future__ import annotations

from pyrlm_runtime.policy import count_tokens, get_context_limit


def test_count_tokens_empty() -> None:
    assert count_tokens([]) == 0


def test_count_tokens_with_tiktoken_grows_with_content() -> None:
    short = [{"role": "user", "content": "hello world"}]
    long = [{"role": "user", "content": "hello world " * 50}]
    n_short = count_tokens(short, "gpt-4o")
    n_long = count_tokens(long, "gpt-4o")
    assert n_short > 0
    assert n_long > n_short


def test_count_tokens_fallback_without_model() -> None:
    # No model_name => per-message len//4 estimate (40 // 4 == 10).
    assert count_tokens([{"role": "user", "content": "a" * 40}]) == 10


def test_count_tokens_non_openai_model_approximates() -> None:
    # Claude has no native tiktoken encoding; falls back to cl100k_base.
    n = count_tokens([{"role": "user", "content": "hello world"}], "claude-3-5-sonnet")
    assert n > 0


def test_count_tokens_handles_non_string_content() -> None:
    # content may be a list of parts or None; must not raise.
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "user", "content": None},
    ]
    assert count_tokens(msgs, "gpt-4o") >= 0


def test_get_context_limit_exact() -> None:
    assert get_context_limit("gpt-4") == 8_192


def test_get_context_limit_longest_substring_match() -> None:
    # "gpt-4o-mini" (longer key) must win over "gpt-4o" / "gpt-4".
    assert get_context_limit("gpt-4o-mini-2024-07-18") == 128_000


def test_get_context_limit_unknown_defaults() -> None:
    assert get_context_limit("some-unknown-model") == 128_000
    assert get_context_limit("") == 128_000
