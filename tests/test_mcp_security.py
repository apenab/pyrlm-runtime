from __future__ import annotations

import pytest

pytest.importorskip("mcp")

from _mcp_fakes import PagedRetriever  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from pyrlm_runtime.mcp_server import (  # noqa: E402
    ConfigError,
    ServerConfig,
    build_asgi_app,
    build_server,
)
from pyrlm_runtime.mcp_server.security import (  # noqa: E402
    check_http_startup,
    is_loopback,
    sanitize_error,
)

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1"},
    },
}
MCP_HEADERS = {"accept": "application/json, text/event-stream"}


@pytest.mark.parametrize(
    "host, expected",
    [
        ("127.0.0.1", True),
        ("127.0.0.2", True),
        ("localhost", True),
        ("::1", True),
        ("[::1]", True),
        ("0.0.0.0", False),
        ("::", False),
        ("", False),
        ("mcp.example.com", False),
        ("10.0.0.5", False),
    ],
)
def test_is_loopback(host, expected):
    assert is_loopback(host) is expected


class TestStartupGuard:
    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "", "mcp.example.com"])
    def test_non_loopback_without_token_refuses_to_start(self, host):
        config = ServerConfig(transport="streamable-http", host=host, allowed_hosts=("x",))

        with pytest.raises(ConfigError, match="BEARER_TOKEN"):
            check_http_startup(config, bearer_token=None)

    def test_non_loopback_without_allowed_hosts_refuses_to_start(self):
        config = ServerConfig(transport="streamable-http", host="0.0.0.0")

        with pytest.raises(ConfigError, match="ALLOWED_HOSTS"):
            check_http_startup(config, bearer_token="tok")

    def test_non_loopback_with_token_and_allowlist_starts(self):
        config = ServerConfig(transport="streamable-http", host="0.0.0.0", allowed_hosts=("x",))

        check_http_startup(config, bearer_token="tok")

    def test_explicit_unauthenticated_opt_in_starts(self):
        config = ServerConfig(
            transport="streamable-http",
            host="0.0.0.0",
            allowed_hosts=("x",),
            allow_unauthenticated=True,
        )

        check_http_startup(config, bearer_token=None)

    def test_loopback_bind_with_a_remote_allowed_host_needs_a_token(self):
        config = ServerConfig(transport="streamable-http", allowed_hosts=("mcp.example.com",))

        with pytest.raises(ConfigError, match="BEARER_TOKEN"):
            check_http_startup(config, bearer_token=None)
        check_http_startup(config, bearer_token="tok")

    def test_loopback_without_token_starts(self):
        check_http_startup(ServerConfig(transport="streamable-http"), bearer_token=None)

    def test_build_asgi_app_applies_the_guard(self):
        config = ServerConfig(transport="streamable-http", host="0.0.0.0", allowed_hosts=("x",))
        server = build_server(retriever=PagedRetriever(), config=config)

        with pytest.raises(ConfigError):
            build_asgi_app(server, config)


def _http_client(bearer_token=None, **config):
    config = ServerConfig(transport="streamable-http", **config)
    server = build_server(retriever=PagedRetriever(), config=config)
    app = build_asgi_app(server, config, bearer_token=bearer_token)
    return TestClient(app, base_url="http://127.0.0.1:8000")


class TestBearerAuth:
    def test_health_needs_no_token(self):
        with _http_client(bearer_token="tok") as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @pytest.mark.parametrize("header", [None, "Bearer wrong", "tok", "Basic dG9r"])
    def test_missing_or_wrong_token_is_401(self, header):
        headers = dict(MCP_HEADERS)
        if header is not None:
            headers["authorization"] = header
        with _http_client(bearer_token="tok") as client:
            response = client.post("/mcp", json=INITIALIZE, headers=headers)

        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"

    @pytest.mark.parametrize("header", ["Bearer tok", "bearer tok", "BEARER  tok", "Bearer tok "])
    def test_correct_token_reaches_the_server(self, header):
        with _http_client(bearer_token="tok") as client:
            response = client.post(
                "/mcp", json=INITIALIZE, headers={**MCP_HEADERS, "authorization": header}
            )

        assert response.status_code == 200
        assert '"serverInfo"' in response.text

    def test_well_known_paths_are_404_not_401(self):
        with _http_client(bearer_token="tok") as client:
            response = client.get("/.well-known/oauth-protected-resource")

        assert response.status_code == 404

    def test_no_token_configured_on_loopback_serves_without_auth(self):
        with _http_client() as client:
            response = client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS)

        assert response.status_code == 200


class TestTransportSecurity:
    def test_any_loopback_bind_address_accepts_its_own_host_header(self):
        config = ServerConfig(transport="streamable-http", host="127.0.0.2")
        server = build_server(retriever=PagedRetriever(), config=config)
        app = build_asgi_app(server, config)
        with TestClient(app, base_url="http://127.0.0.2:8000") as client:
            response = client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS)

        assert response.status_code == 200

    def test_unlisted_host_header_is_rejected_on_loopback(self):
        with _http_client() as client:
            response = client.post(
                "/mcp", json=INITIALIZE, headers={**MCP_HEADERS, "host": "evil.example"}
            )

        assert response.status_code == 421

    def test_allowed_hosts_are_enforced_on_non_loopback(self):
        with _http_client(
            bearer_token="tok", host="0.0.0.0", allowed_hosts=("mcp.example.com",)
        ) as client:
            auth = {**MCP_HEADERS, "authorization": "Bearer tok"}
            allowed = client.post(
                "/mcp", json=INITIALIZE, headers={**auth, "host": "mcp.example.com"}
            )
            denied = client.post("/mcp", json=INITIALIZE, headers={**auth, "host": "evil.example"})

        assert allowed.status_code == 200
        assert denied.status_code == 421


def test_sanitize_error_keeps_only_the_exception_type():
    text = sanitize_error(ConnectionError("https://es-internal.corp:9200 key=sk-SECRET"))

    assert "ConnectionError" in text
    assert "es-internal" not in text
    assert "SECRET" not in text


def test_mounted_app_keeps_auth_and_an_open_health_route():
    import contextlib

    from starlette.applications import Starlette
    from starlette.routing import Mount

    config = ServerConfig(transport="streamable-http")
    server = build_server(retriever=PagedRetriever(), config=config)
    mcp_app = build_asgi_app(server, config, bearer_token="tok")

    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with server.session_manager.run():
            yield

    app = Starlette(routes=[Mount("/rag", app=mcp_app)], lifespan=lifespan)
    auth = {**MCP_HEADERS, "authorization": "Bearer tok"}
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        health = client.get("/rag/health")
        anonymous = client.post("/rag/mcp", json=INITIALIZE, headers=MCP_HEADERS)
        authorised = client.post("/rag/mcp", json=INITIALIZE, headers=auth)

    assert (health.status_code, anonymous.status_code, authorised.status_code) == (200, 401, 200)
