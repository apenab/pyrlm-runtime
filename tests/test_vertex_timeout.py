"""Regression tests for the VertexAIAdapter per-call timeout.

A call to a server that never replies must fail within ``timeout`` on any
thread, including worker threads where no signal-based guard can run (an MCP
server calls ``complete`` from one).
"""

from __future__ import annotations

import threading
import time

import pytest

try:
    from pyrlm_runtime.adapters import vertex_ai

    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not VERTEX_AVAILABLE, reason="google-genai not installed")

TIMEOUT_S = 0.3


def _timed_call(adapter) -> dict:
    outcome: dict = {}
    start = time.monotonic()
    try:
        adapter.complete([{"role": "user", "content": "hi"}])
    except TimeoutError as exc:
        outcome["exc"] = exc
    outcome["elapsed"] = time.monotonic() - start
    return outcome


def test_worker_thread_call_times_out(hang_server, vertex_base_url) -> None:
    vertex_base_url(hang_server[0])
    adapter = vertex_ai.VertexAIAdapter(project_id="p", timeout=TIMEOUT_S, max_retries=1)

    outcome: dict = {}
    worker = threading.Thread(target=lambda: outcome.update(_timed_call(adapter)), daemon=True)
    worker.start()
    worker.join(TIMEOUT_S + 3)

    assert not worker.is_alive(), "complete() hung in a worker thread past its timeout"
    assert "exc" in outcome
    assert outcome["elapsed"] < TIMEOUT_S + 2


def test_main_thread_call_times_out(hang_server, vertex_base_url) -> None:
    vertex_base_url(hang_server[0])
    adapter = vertex_ai.VertexAIAdapter(project_id="p", timeout=TIMEOUT_S, max_retries=1)

    outcome = _timed_call(adapter)

    assert "exc" in outcome
    assert outcome["elapsed"] < TIMEOUT_S + 2


def test_timeout_is_sent_to_the_server(hang_server, vertex_base_url) -> None:
    url, received = hang_server
    vertex_base_url(url)
    adapter = vertex_ai.VertexAIAdapter(project_id="p", timeout=TIMEOUT_S, max_retries=1)

    _timed_call(adapter)

    assert b"x-server-timeout: 1" in received[0].lower()


def test_timed_out_call_is_retried(hang_server, vertex_base_url, monkeypatch) -> None:
    url, received = hang_server
    vertex_base_url(url)
    monkeypatch.setattr(vertex_ai.time, "sleep", lambda s: None)
    adapter = vertex_ai.VertexAIAdapter(project_id="p", timeout=TIMEOUT_S, max_retries=2)

    outcome = _timed_call(adapter)

    assert "exc" in outcome
    assert len(received) == 2
