"""Regression tests for the VertexAIAdapter transport.

google-genai speaks REST only. Behind a corporate proxy the adapter must honor
``HTTPS_PROXY`` and a custom CA bundle from ``SSL_CERT_FILE`` or
``REQUESTS_CA_BUNDLE``. Callers that still pass ``api_transport`` must keep
working. Every test talks to a local server, never to Vertex.
"""

from __future__ import annotations

import datetime
import http.server
import ipaddress
import json
import ssl
import threading
import warnings

import pytest

try:
    from google.genai import errors

    from pyrlm_runtime.adapters import vertex_ai

    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not VERTEX_AVAILABLE, reason="google-genai not installed")

_REPLY = {
    "candidates": [
        {"content": {"role": "model", "parts": [{"text": "pong"}]}, "finishReason": "STOP"}
    ],
    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
}


def _self_signed_cert(tmp_path):
    pytest.importorskip("cryptography")
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


@pytest.fixture
def tls_server(tmp_path):
    """An HTTPS server with a self-signed certificate that answers like Gemini."""
    cert_path, key_path = _self_signed_cert(tmp_path)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps(_REPLY).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_path, key_path)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"https://127.0.0.1:{srv.server_address[1]}", cert_path
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def clean_env(monkeypatch):
    for var in (
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "NO_PROXY",
        "no_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.mark.parametrize("env_var", ["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"])
def test_custom_ca_bundle_is_trusted(tls_server, vertex_base_url, clean_env, env_var) -> None:
    url, cert_path = tls_server
    vertex_base_url(url)
    clean_env.setenv(env_var, str(cert_path))

    response = vertex_ai.VertexAIAdapter(project_id="p", timeout=5).complete(
        [{"role": "user", "content": "ping"}]
    )

    assert response.text == "pong"
    assert response.meta["finish_reason"] == "stop"


def test_untrusted_certificate_is_rejected(tls_server, vertex_base_url, clean_env) -> None:
    vertex_base_url(tls_server[0])

    with pytest.raises(Exception, match="CERTIFICATE_VERIFY_FAILED"):
        vertex_ai.VertexAIAdapter(project_id="p", timeout=5, max_retries=1).complete(
            [{"role": "user", "content": "ping"}]
        )


def test_https_proxy_is_honored(hang_server, vertex_base_url, clean_env) -> None:
    proxy_url, received = hang_server
    vertex_base_url(None)
    clean_env.setenv("HTTPS_PROXY", proxy_url)

    with pytest.raises(TimeoutError):
        vertex_ai.VertexAIAdapter(project_id="p", timeout=0.3, max_retries=1).complete(
            [{"role": "user", "content": "ping"}]
        )

    assert received[0].startswith(b"CONNECT us-central1-aiplatform.googleapis.com:443")


def test_default_transport_does_not_warn(vertex_base_url) -> None:
    vertex_base_url(None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        adapter = vertex_ai.VertexAIAdapter(project_id="p")
    assert adapter.api_transport == "rest"


def test_grpc_transport_warns_and_still_builds(vertex_base_url) -> None:
    vertex_base_url(None)
    with pytest.warns(DeprecationWarning, match="REST only"):
        adapter = vertex_ai.VertexAIAdapter(project_id="p", api_transport="grpc")
    assert adapter.api_transport == "grpc"


class _ScriptedModels:
    def __init__(self, outcomes: list) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def generate_content(self, **kwargs):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _scripted_adapter(monkeypatch, vertex_base_url, outcomes: list):
    vertex_base_url(None)
    monkeypatch.setattr(vertex_ai.time, "sleep", lambda s: None)
    adapter = vertex_ai.VertexAIAdapter(project_id="p", max_retries=3)
    models = _ScriptedModels(outcomes)
    monkeypatch.setattr(adapter, "client", type("C", (), {"models": models})())
    return adapter, models


@pytest.mark.parametrize("code", [429, 503, 504])
def test_retryable_status_is_retried(monkeypatch, vertex_base_url, code) -> None:
    from google.genai import types

    ok = types.GenerateContentResponse.model_validate(_REPLY)
    adapter, models = _scripted_adapter(
        monkeypatch, vertex_base_url, [errors.APIError(code, {"error": {"message": "x"}}), ok]
    )

    assert adapter.complete([{"role": "user", "content": "ping"}]).text == "pong"
    assert models.calls == 2


def test_client_error_is_not_retried(monkeypatch, vertex_base_url) -> None:
    adapter, models = _scripted_adapter(
        monkeypatch, vertex_base_url, [errors.APIError(400, {"error": {"message": "bad"}})]
    )

    with pytest.raises(errors.APIError):
        adapter.complete([{"role": "user", "content": "ping"}])
    assert models.calls == 1


def test_lazy_export_from_adapters_package() -> None:
    """VertexAIAdapter is exported lazily from pyrlm_runtime.adapters."""
    from pyrlm_runtime.adapters import VertexAIAdapter

    assert VertexAIAdapter is vertex_ai.VertexAIAdapter
