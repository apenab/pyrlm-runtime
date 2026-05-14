"""Tests for pyrlm_runtime.multiquery (QueryRewriter + union_pool)."""

from __future__ import annotations

import pytest

from pyrlm_runtime import QueryRewriter, union_pool
from pyrlm_runtime.adapters import FakeAdapter


SYSTEM_PROMPT = 'Return {"rewrites": ["a", "b", "c"]}'


# ---------------------------------------------------------------------------
# union_pool
# ---------------------------------------------------------------------------


def test_union_pool_basic_dedup():
    pool_a = [{"doc_id": "1", "score": 0.9}, {"doc_id": "2", "score": 0.8}]
    pool_b = [{"doc_id": "2", "score": 0.5}, {"doc_id": "3", "score": 0.4}]
    result = union_pool([pool_a, pool_b])
    assert [d["doc_id"] for d in result] == ["1", "2", "3"]


def test_union_pool_first_occurrence_wins():
    pool_a = [{"doc_id": "x", "score": 0.9}]
    pool_b = [{"doc_id": "x", "score": 0.1}]
    result = union_pool([pool_a, pool_b])
    assert len(result) == 1
    assert result[0]["score"] == 0.9


def test_union_pool_empty_inputs():
    assert union_pool([]) == []
    assert union_pool([[], []]) == []


def test_union_pool_single_pool():
    pool = [{"doc_id": "a"}, {"doc_id": "b"}]
    assert union_pool([pool]) == pool


def test_union_pool_custom_id_key():
    pool_a = [{"id": "1", "text": "hello"}]
    pool_b = [{"id": "2", "text": "world"}]
    result = union_pool([pool_a, pool_b], id_key="id")
    assert len(result) == 2


def test_union_pool_preserves_first_appearance_order():
    pool_a = [{"doc_id": "c"}, {"doc_id": "a"}]
    pool_b = [{"doc_id": "b"}, {"doc_id": "a"}]
    result = union_pool([pool_a, pool_b])
    assert [d["doc_id"] for d in result] == ["c", "a", "b"]


# ---------------------------------------------------------------------------
# QueryRewriter
# ---------------------------------------------------------------------------


def _make_rewriter(response: str, n: int = 3) -> QueryRewriter:
    adapter = FakeAdapter(script=[response] * 10)
    return QueryRewriter(adapter, n=n, system_prompt=SYSTEM_PROMPT)


def test_rewriter_parses_json():
    rw = _make_rewriter('{"rewrites": ["foo", "bar", "baz"]}')
    result = rw.rewrite("some query")
    assert result == ["foo", "bar", "baz"]
    assert rw.calls == 1


def test_rewriter_truncates_to_n():
    rw = _make_rewriter('{"rewrites": ["a", "b", "c", "d", "e"]}', n=3)
    result = rw.rewrite("q")
    assert len(result) == 3


def test_rewriter_fallback_on_bad_json():
    rw = _make_rewriter("- option one\n- option two\n- option three")
    result = rw.rewrite("q")
    assert len(result) == 3
    assert result[0] == "option one"


def test_rewriter_empty_response_returns_empty():
    rw = _make_rewriter("")
    result = rw.rewrite("q")
    assert result == []


def test_rewriter_calls_counter_increments():
    rw = _make_rewriter('{"rewrites": ["x"]}', n=1)
    rw.rewrite("q1")
    rw.rewrite("q2")
    assert rw.calls == 2


def test_rewriter_invalid_n_raises():
    with pytest.raises(ValueError, match="n must be > 0"):
        QueryRewriter(FakeAdapter(script=[]), n=0, system_prompt=SYSTEM_PROMPT)


def test_rewriter_empty_system_prompt_raises():
    with pytest.raises(ValueError, match="system_prompt"):
        QueryRewriter(FakeAdapter(script=[]), n=3, system_prompt="")


def test_rewriter_n_property():
    rw = _make_rewriter('{"rewrites": []}', n=7)
    assert rw.n == 7


def test_rewriter_thread_safety():
    import threading

    rw = _make_rewriter('{"rewrites": ["a", "b", "c"]}')
    errors: list[Exception] = []

    def worker():
        try:
            rw.rewrite("q")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert rw.calls == 10
