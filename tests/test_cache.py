"""Regression tests for FileCache robustness and cache-key model separation.

Covers two fixes:

* R2 — FileCache must be crash-safe and atomic so a corrupt/half-written entry
  (possible under ``parallel_subcalls``) degrades to a cache miss instead of
  raising, and concurrent writers never observe a partial file.
* R1 — the subcall cache key must fold in the *effective* model identity so
  entries served by different adapters/models never collide in a shared cache
  directory (e.g. when changing ``subcall_adapter``).
"""

from __future__ import annotations

import threading
from pathlib import Path

from pyrlm_runtime import Context, FileCache, RLM
from pyrlm_runtime.adapters import FakeAdapter
from pyrlm_runtime.adapters.base import Usage
from pyrlm_runtime.cache import CacheRecord
from pyrlm_runtime.rlm import _adapter_identity, _cache_key


def _rec(text: str = "hi") -> CacheRecord:
    return CacheRecord(text=text, usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3))


# --------------------------------------------------------------------------- R2


def test_set_get_roundtrip_and_no_tmp_leftovers(tmp_path: Path) -> None:
    cache = FileCache(root=tmp_path)
    cache.set("k", _rec("hello"))

    got = cache.get("k")
    assert got is not None
    assert got.text == "hello"
    assert got.usage.total_tokens == 3

    # Atomic write must not leave temp files behind.
    assert list(tmp_path.glob("*.tmp")) == []


def test_get_missing_key_returns_none(tmp_path: Path) -> None:
    assert FileCache(root=tmp_path).get("never-written") is None


def test_get_tolerates_corrupt_entry(tmp_path: Path) -> None:
    cache = FileCache(root=tmp_path)
    cache.set("k", _rec("good"))

    # Simulate a half-written / corrupt entry.
    cache._path("k").write_text("{ this is not valid json", encoding="utf-8")

    # Must degrade to a miss, not raise.
    assert cache.get("k") is None


def test_concurrent_writes_same_key_never_corrupt(tmp_path: Path) -> None:
    cache = FileCache(root=tmp_path)
    payload = "x" * 5000
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(40):
                cache.set("hot", _rec(payload))
                cache.get("hot")  # readers must never see a partial file
        except BaseException as exc:  # noqa: BLE001 - surface any failure
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    final = cache.get("hot")
    assert final is not None and final.text == payload
    assert list(tmp_path.glob("*.tmp")) == []


# --------------------------------------------------------------------------- R1


def test_adapter_identity_prefers_model() -> None:
    assert _adapter_identity(FakeAdapter(model="model-x")) == "model-x"


def test_adapter_identity_falls_back_to_class_name() -> None:
    class NoModelAdapter:
        pass

    assert _adapter_identity(NoModelAdapter()) == "NoModelAdapter"


def test_adapter_identity_prefers_model_name_over_object_model() -> None:
    """Vertex-like adapter: ``model`` is a non-string object, the real id lives
    in ``model_name``. Identity must use the string, never str() the object."""

    class VertexLike:
        def __init__(self) -> None:
            self.model_name = "gemini-2.5-pro"
            self.model = object()  # a GenerativeModel-like opaque object

    assert _adapter_identity(VertexLike()) == "gemini-2.5-pro"


def test_adapter_identity_skips_non_string_model_id() -> None:
    class Weird:
        model_id = 123  # not a string → must be skipped

    assert _adapter_identity(Weird()) == "Weird"


def test_cache_key_varies_by_model() -> None:
    a = _cache_key(text="same", model="model-a", max_tokens=64)
    b = _cache_key(text="same", model="model-b", max_tokens=64)
    assert a != b


def test_subcall_cache_separates_by_model(tmp_path: Path) -> None:
    """Two runs over the same prompt but different subcall models must not
    collide in a shared cache dir. Before R1 both keyed on ``model=default``,
    so the second run would wrongly return the first model's answer."""
    cache = FileCache(root=tmp_path)
    ctx = Context.from_text("irrelevant body")

    def root() -> FakeAdapter:
        # Generate code that makes a subcall, then finalize with its result.
        return FakeAdapter(script=["answer = llm_query('What is X?')", "FINAL_VAR: answer"])

    sub_a = FakeAdapter(model="model-a")
    sub_a.add_rule("What is X?", "ANSWER_FROM_A")
    out_a, _ = RLM(adapter=root(), subcall_adapter=sub_a, cache=cache).run("q", ctx)
    assert out_a == "ANSWER_FROM_A"

    sub_b = FakeAdapter(model="model-b")
    sub_b.add_rule("What is X?", "ANSWER_FROM_B")
    out_b, _ = RLM(adapter=root(), subcall_adapter=sub_b, cache=cache).run("q", ctx)
    assert out_b == "ANSWER_FROM_B"  # would be ANSWER_FROM_A if keys collided


def test_subcall_cache_hits_for_same_model(tmp_path: Path) -> None:
    """Same model + same prompt must still hit the cache on the second run."""
    cache = FileCache(root=tmp_path)
    ctx = Context.from_text("irrelevant body")

    def root() -> FakeAdapter:
        return FakeAdapter(script=["answer = llm_query('What is X?')", "FINAL_VAR: answer"])

    sub1 = FakeAdapter(model="model-a")
    sub1.add_rule("What is X?", "CACHED_ANSWER")
    out1, _ = RLM(adapter=root(), subcall_adapter=sub1, cache=cache).run("q", ctx)
    assert out1 == "CACHED_ANSWER"

    # Second adapter, same model id, but NO rule and NO script: it can only
    # succeed if the answer is served from the cache.
    sub2 = FakeAdapter(model="model-a")
    out2, trace = RLM(adapter=root(), subcall_adapter=sub2, cache=cache).run("q", ctx)
    assert out2 == "CACHED_ANSWER"
    assert any(s.cache_hit for s in trace.steps)
    assert sub2.call_log == []  # adapter never invoked — pure cache hit
