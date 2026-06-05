from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from pyrlm_runtime import FileCache
from pyrlm_runtime.adapters import FakeAdapter
from pyrlm_runtime.rerank import (
    ListwiseReranker,
    RerankerProtocol,
    TournamentReranker,
    ndcg_at_k,
    recall_at_k,
)


def _make_candidates(n: int, *, text_field: str = "content") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(n):
        doc: dict[str, Any] = {
            "doc_id": f"d{i}",
            text_field: f"passage text number {i}",
            "score": float(n - i),
            "metadata": {"i": i},
        }
        out.append(doc)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_ndcg_at_k_perfect_order() -> None:
    qrels = {"a": 3.0, "b": 2.0, "c": 1.0}
    score = ndcg_at_k(["a", "b", "c"], qrels, k=3)
    assert score == pytest.approx(1.0)


def test_ndcg_at_k_reverse_order_below_perfect() -> None:
    qrels = {"a": 3.0, "b": 2.0, "c": 1.0}
    perfect = ndcg_at_k(["a", "b", "c"], qrels, k=3)
    reversed_ = ndcg_at_k(["c", "b", "a"], qrels, k=3)
    assert 0.0 < reversed_ < perfect


def test_ndcg_at_k_empty_qrels() -> None:
    assert ndcg_at_k(["a"], {}, k=10) == 0.0


def test_ndcg_at_k_missing_docs_count_as_zero() -> None:
    qrels = {"a": 1.0}
    # Single relevant doc at rank 1 → perfect NDCG
    assert ndcg_at_k(["a", "x", "y"], qrels, k=3) == pytest.approx(1.0)
    # Single relevant doc at rank 2 → DCG=1/log2(3), IDCG=1/log2(2)=1
    expected = (1.0 / math.log2(3)) / 1.0
    assert ndcg_at_k(["x", "a", "y"], qrels, k=3) == pytest.approx(expected)


def test_recall_at_k_basic() -> None:
    qrels = {"a": 1.0, "b": 1.0, "c": 0.0, "d": 2.0}  # 3 relevant
    assert recall_at_k(["a", "b", "d"], qrels, k=3) == pytest.approx(1.0)
    assert recall_at_k(["a", "x", "y"], qrels, k=3) == pytest.approx(1.0 / 3.0)
    assert recall_at_k(["x", "y", "z"], qrels, k=3) == 0.0


def test_recall_at_k_no_relevant() -> None:
    assert recall_at_k(["a"], {"a": 0.0}, k=10) == 0.0


# ---------------------------------------------------------------------------
# ListwiseReranker — windowing & parsing
# ---------------------------------------------------------------------------


def test_listwise_rerank_single_window() -> None:
    adapter = FakeAdapter(script=["[3] > [1] > [5] > [2] > [4]"])
    reranker = ListwiseReranker(adapter, window_size=5, step=5)
    candidates = _make_candidates(5)
    out = reranker.rerank("q", candidates, top_k=5)
    assert [d["doc_id"] for d in out] == ["d2", "d0", "d4", "d1", "d3"]
    # rerank_score is monotonically decreasing (1, 1/2, 1/3, ...)
    scores = [d["metadata"]["rerank_score"] for d in out]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.5)
    # Telemetry
    assert reranker.llm_calls == 1
    assert reranker.cache_hits == 0


def test_listwise_rerank_top_k_truncation() -> None:
    adapter = FakeAdapter(script=["[1] > [2] > [3] > [4] > [5]"])
    reranker = ListwiseReranker(adapter, window_size=5, step=5)
    out = reranker.rerank("q", _make_candidates(5), top_k=2)
    assert len(out) == 2
    assert [d["doc_id"] for d in out] == ["d0", "d1"]


def test_listwise_rerank_sliding_window_call_count() -> None:
    # 25 candidates, window=10, step=5 → starts = [15, 10, 5, 0] → 4 calls
    identity = "[1] > [2] > [3] > [4] > [5] > [6] > [7] > [8] > [9] > [10]"
    adapter = FakeAdapter(script=[identity] * 4)
    reranker = ListwiseReranker(adapter, window_size=10, step=5)
    out = reranker.rerank("q", _make_candidates(25), top_k=10)
    assert reranker.llm_calls == 4
    # Identity permutations preserve order
    assert [d["doc_id"] for d in out] == [f"d{i}" for i in range(10)]


def test_listwise_rerank_sliding_window_reorders_top() -> None:
    # 12 candidates, window=10, step=5 → starts = [2, 0]
    # First window reranks indices 2..11; second window reranks 0..9.
    # First call returns permutation that promotes position 11 (d11) to head of window.
    # Second call returns permutation that promotes that same doc to head of list.
    adapter = FakeAdapter(
        script=[
            "[10] > [9] > [8] > [7] > [6] > [5] > [4] > [3] > [2] > [1]",
            "[10] > [9] > [8] > [7] > [6] > [5] > [4] > [3] > [2] > [1]",
        ]
    )
    reranker = ListwiseReranker(adapter, window_size=10, step=5)
    out = reranker.rerank("q", _make_candidates(12), top_k=3)
    # After first window (positions 2..11 reversed): d0, d1, d11, d10, d9, d8, d7, d6, d5, d4, d3, d2
    # After second window (positions 0..9 reversed): d4, d5, d6, d7, d8, d9, d10, d11, d1, d0, d3, d2
    assert [d["doc_id"] for d in out] == ["d4", "d5", "d6"]


def test_listwise_rerank_malformed_response_preserves_order() -> None:
    adapter = FakeAdapter(script=["I cannot rank these documents."])
    reranker = ListwiseReranker(adapter, window_size=5, step=5)
    out = reranker.rerank("q", _make_candidates(5), top_k=5)
    assert [d["doc_id"] for d in out] == [f"d{i}" for i in range(5)]
    assert reranker.llm_calls == 1


def test_listwise_rerank_partial_response_pads_missing_ids() -> None:
    # LLM only emits 3 of 5 identifiers; the rest preserve original order.
    adapter = FakeAdapter(script=["[5] > [1] > [3]"])
    reranker = ListwiseReranker(adapter, window_size=5, step=5)
    out = reranker.rerank("q", _make_candidates(5), top_k=5)
    # Expected: [d4, d0, d2, d1, d3] (missing ids 2, 4 → original-order indices 1, 3 → d1, d3)
    assert [d["doc_id"] for d in out] == ["d4", "d0", "d2", "d1", "d3"]


def test_listwise_rerank_out_of_range_ids_filtered() -> None:
    adapter = FakeAdapter(script=["[2] > [99] > [1] > [0]"])
    reranker = ListwiseReranker(adapter, window_size=3, step=3)
    out = reranker.rerank("q", _make_candidates(3), top_k=3)
    # [99] is out of range, [0] is out of range (1-based), [2] and [1] valid → d1, d0, plus missing d2.
    assert [d["doc_id"] for d in out] == ["d1", "d0", "d2"]


def test_listwise_rerank_preview_fallback() -> None:
    # Candidates lack 'content'; they have 'preview' instead.
    candidates = _make_candidates(3, text_field="preview")
    adapter = FakeAdapter(script=["[3] > [2] > [1]"])
    reranker = ListwiseReranker(adapter, window_size=3, step=3)
    out = reranker.rerank("q", candidates, top_k=3)
    assert [d["doc_id"] for d in out] == ["d2", "d1", "d0"]
    # Verify the prompt actually included the preview text
    user_msg = adapter.call_log[0][-1]["content"]
    assert "passage text number 0" in user_msg


def test_listwise_rerank_protocol_runtime_check() -> None:
    reranker = ListwiseReranker(FakeAdapter(script=[]))
    assert isinstance(reranker, RerankerProtocol)


def test_listwise_rerank_empty_candidates() -> None:
    reranker = ListwiseReranker(FakeAdapter(script=[]))
    assert reranker.rerank("q", [], top_k=10) == []
    assert reranker.llm_calls == 0


def test_listwise_rerank_invalid_params() -> None:
    adapter = FakeAdapter(script=[])
    with pytest.raises(ValueError):
        ListwiseReranker(adapter, window_size=0)
    with pytest.raises(ValueError):
        ListwiseReranker(adapter, step=0)
    with pytest.raises(ValueError):
        ListwiseReranker(adapter, window_size=5, step=10)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_listwise_rerank_cache_hit_skips_llm(tmp_path: Path) -> None:
    cache = FileCache(tmp_path)
    candidates = _make_candidates(5)
    # First run populates cache.
    adapter1 = FakeAdapter(script=["[5] > [4] > [3] > [2] > [1]"])
    r1 = ListwiseReranker(adapter1, window_size=5, step=5, cache=cache)
    out1 = r1.rerank("q", candidates, top_k=5)
    assert r1.llm_calls == 1
    assert r1.cache_hits == 0

    # Second run with adapter that would fail if called.
    adapter2 = FakeAdapter(script=[])
    r2 = ListwiseReranker(adapter2, window_size=5, step=5, cache=cache)
    out2 = r2.rerank("q", candidates, top_k=5)
    assert r2.llm_calls == 0
    assert r2.cache_hits == 1
    assert [d["doc_id"] for d in out1] == [d["doc_id"] for d in out2]


def test_listwise_rerank_counters_thread_safe() -> None:
    """Many threads sharing one reranker must observe an accurate llm_calls."""
    n_queries = 50
    # Each query needs its own script entry; FakeAdapter consumes them in order.
    identity = "[1] > [2] > [3]"
    adapter = FakeAdapter(script=[identity] * n_queries)
    reranker = ListwiseReranker(adapter, window_size=3, step=3)
    candidates = _make_candidates(3)

    def run_one(_: int) -> None:
        reranker.rerank("q", candidates, top_k=3)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run_one, range(n_queries)))

    # Each rerank does exactly one LLM call (single window of 3 candidates).
    assert reranker.llm_calls == n_queries
    assert reranker.cache_hits == 0


def test_listwise_rerank_cache_namespace_isolates(tmp_path: Path) -> None:
    cache = FileCache(tmp_path)
    candidates = _make_candidates(3)
    adapter_a = FakeAdapter(script=["[3] > [2] > [1]"])
    ListwiseReranker(
        adapter_a, window_size=3, step=3, cache=cache, cache_namespace="model-a"
    ).rerank("q", candidates, top_k=3)

    # Different namespace → cache miss → adapter_b must be called.
    adapter_b = FakeAdapter(script=["[1] > [2] > [3]"])
    r_b = ListwiseReranker(adapter_b, window_size=3, step=3, cache=cache, cache_namespace="model-b")
    out = r_b.rerank("q", candidates, top_k=3)
    assert r_b.llm_calls == 1
    assert [d["doc_id"] for d in out] == ["d0", "d1", "d2"]


# ---------------------------------------------------------------------------
# TournamentReranker
# ---------------------------------------------------------------------------


def test_tournament_rerank_single_batch_no_recursion() -> None:
    """Pool that fits in one batch → one LLM call, no elimination rounds."""
    adapter = FakeAdapter(script=["[3] > [1] > [2]"])
    reranker = TournamentReranker(adapter, batch_size=20, top_k_per_batch=4)
    candidates = _make_candidates(3)
    out = reranker.rerank("q", candidates, top_k=3)
    assert [d["doc_id"] for d in out] == ["d2", "d0", "d1"]
    assert reranker.llm_calls == 1
    assert reranker.rounds_run == 1


def test_tournament_rerank_two_rounds() -> None:
    """Pool > batch_size → at least one elimination round + final ranking."""
    # 30 docs, batch_size=10, top_k_per_batch=2:
    #   Round 1: 3 batches × top-2 = 6 survivors, 24 in tails
    #   Round 2 (final): 1 batch of 6, ranked directly
    # Total LLM calls: 3 + 1 = 4
    n_calls_expected = 4
    perm = " > ".join(f"[{i + 1}]" for i in range(10))
    adapter = FakeAdapter(script=[perm] * n_calls_expected)
    reranker = TournamentReranker(adapter, batch_size=10, top_k_per_batch=2, shuffle_seed=42)
    candidates = _make_candidates(30)
    out = reranker.rerank("q", candidates, top_k=10)
    assert len(out) == 10
    assert reranker.llm_calls == n_calls_expected
    assert reranker.rounds_run == 2


def test_tournament_rerank_empty() -> None:
    adapter = FakeAdapter(script=[])
    reranker = TournamentReranker(adapter)
    assert reranker.rerank("q", [], top_k=10) == []
    assert reranker.llm_calls == 0


def test_tournament_rerank_top_k_zero() -> None:
    adapter = FakeAdapter(script=[])
    reranker = TournamentReranker(adapter)
    assert reranker.rerank("q", _make_candidates(5), top_k=0) == []


def test_tournament_rerank_top_k_truncation() -> None:
    adapter = FakeAdapter(script=["[5] > [4] > [3] > [2] > [1]"])
    reranker = TournamentReranker(adapter, batch_size=20, top_k_per_batch=4)
    out = reranker.rerank("q", _make_candidates(5), top_k=3)
    assert [d["doc_id"] for d in out] == ["d4", "d3", "d2"]


def test_tournament_rerank_invalid_params() -> None:
    adapter = FakeAdapter(script=[])
    with pytest.raises(ValueError):
        TournamentReranker(adapter, batch_size=1)
    with pytest.raises(ValueError):
        TournamentReranker(adapter, batch_size=20, top_k_per_batch=0)
    with pytest.raises(ValueError):
        TournamentReranker(adapter, batch_size=20, top_k_per_batch=20)
    with pytest.raises(ValueError):
        TournamentReranker(adapter, max_passage_chars=0)


def test_tournament_rerank_deterministic_with_seed() -> None:
    """Same seed + same query + same input → identical output."""
    perm = " > ".join(f"[{i + 1}]" for i in range(10))
    adapter_a = FakeAdapter(script=[perm] * 10)
    adapter_b = FakeAdapter(script=[perm] * 10)
    candidates = _make_candidates(25)
    a = TournamentReranker(adapter_a, batch_size=10, top_k_per_batch=3, shuffle_seed=7)
    b = TournamentReranker(adapter_b, batch_size=10, top_k_per_batch=3, shuffle_seed=7)
    out_a = a.rerank("q", candidates, top_k=10)
    out_b = b.rerank("q", candidates, top_k=10)
    assert [d["doc_id"] for d in out_a] == [d["doc_id"] for d in out_b]


def test_tournament_rerank_protocol_runtime_check() -> None:
    adapter = FakeAdapter(script=[])
    assert isinstance(TournamentReranker(adapter), RerankerProtocol)


def test_tournament_rerank_cache_hit_skips_llm(tmp_path: Path) -> None:
    cache = FileCache(tmp_path)
    adapter_a = FakeAdapter(script=["[3] > [1] > [2]"])
    r1 = TournamentReranker(
        adapter_a,
        batch_size=20,
        top_k_per_batch=4,
        cache=cache,
        cache_namespace="t",
    )
    candidates = _make_candidates(3)
    out1 = r1.rerank("q", candidates, top_k=3)

    adapter_b = FakeAdapter(script=[])  # no responses available
    r2 = TournamentReranker(
        adapter_b,
        batch_size=20,
        top_k_per_batch=4,
        cache=cache,
        cache_namespace="t",
    )
    out2 = r2.rerank("q", candidates, top_k=3)
    assert [d["doc_id"] for d in out2] == [d["doc_id"] for d in out1]
    assert r2.llm_calls == 0
    assert r2.cache_hits == 1
