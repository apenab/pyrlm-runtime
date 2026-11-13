from __future__ import annotations

import socket
import threading
import time

import pytest

try:
    import requests
    from google.auth.credentials import AnonymousCredentials
    from google.cloud.aiplatform import initializer

    from pyrlm_runtime.adapters import vertex_ai

    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not VERTEX_AVAILABLE, reason="vertexai not installed")

TIMEOUT_S = 0.3


@pytest.fixture
def hang_endpoint():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    conns: list[socket.socket] = []

    def accept_forever() -> None:
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            conns.append(conn)

    threading.Thread(target=accept_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.getsockname()[1]}"
    srv.close()
    for conn in conns:
        conn.close()


@pytest.fixture
def isolated_vertex_config(monkeypatch):
    # vertexai.init is the old config's bound method, so rebind it as well.
    config = initializer._Config()
    monkeypatch.setattr(initializer, "global_config", config)
    monkeypatch.setattr(vertex_ai.vertexai, "init", config.init)


def test_worker_thread_call_times_out(hang_endpoint, isolated_vertex_config) -> None:
    vertex_ai.vertexai.init(
        project="p",
        location="us-central1",
        api_transport="rest",
        api_endpoint=hang_endpoint,
        credentials=AnonymousCredentials(),
    )
    adapter = vertex_ai.VertexAIAdapter(project_id="p", timeout=TIMEOUT_S, max_retries=1)

    outcome: dict = {}

    def call() -> None:
        start = time.monotonic()
        try:
            adapter.complete([{"role": "user", "content": "hi"}])
        except requests.exceptions.Timeout as exc:
            outcome["exc"] = exc
        outcome["elapsed"] = time.monotonic() - start

    worker = threading.Thread(target=call, daemon=True)
    worker.start()
    worker.join(TIMEOUT_S + 3)

    assert not worker.is_alive(), "complete() hung in a worker thread past its timeout"
    assert "exc" in outcome
    assert outcome["elapsed"] < TIMEOUT_S + 2
