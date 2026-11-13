from __future__ import annotations

import socket
import threading

import pytest


@pytest.fixture
def hang_server():
    """A TCP server that accepts connections and never replies.

    Yields ``(url, received)``, where ``received`` collects the first bytes of
    each connection.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    conns: list[socket.socket] = []
    received: list[bytes] = []

    def accept_forever() -> None:
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            conns.append(conn)
            conn.settimeout(2)
            try:
                received.append(conn.recv(4096))
            except OSError:
                pass

    threading.Thread(target=accept_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.getsockname()[1]}", received
    srv.close()
    for conn in conns:
        conn.close()


@pytest.fixture
def vertex_base_url(monkeypatch):
    """Point every ``VertexAIAdapter`` built afterwards at a local base URL.

    Returns a setter taking the URL, or ``None`` to keep the real Vertex host
    (for proxy tests). The client gets a static fake token, so no GCP
    credentials are needed.
    """
    try:
        from google import genai
        from google.genai import types
        from google.oauth2.credentials import Credentials
    except ImportError:
        pytest.skip("google-genai not installed")

    real_client = genai.Client

    def install(url: str | None) -> None:
        def client(**kwargs):
            opts = kwargs.pop("http_options", None) or types.HttpOptions()
            if url is not None:
                opts = opts.model_copy(update={"base_url": url})
            return real_client(credentials=Credentials(token="fake"), http_options=opts, **kwargs)

        monkeypatch.setattr(genai, "Client", client)

    return install
