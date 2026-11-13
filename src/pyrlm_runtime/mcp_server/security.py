"""Network security of the MCP server: start-up guard, bearer auth, error text.

:func:`build_asgi_app` is the supported way to serve over HTTP, on its own or
mounted in another ASGI app. :func:`run_server` wraps it and also serves stdio.
"""

from __future__ import annotations

import hmac
import ipaddress
import json

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import ConfigError, ServerConfig
from .stdio import serve_stdio

HEALTH_PATH = "/health"

_LOOPBACK_HOSTS = ("127.0.0.1:*", "localhost:*", "[::1]:*")
_LOOPBACK_ORIGINS = ("http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*")


def sanitize_error(exc: BaseException) -> str:
    """Client-facing text for an unexpected failure: the exception type only.

    The message of a backend exception can carry hostnames, index names or
    keys, so it goes to the server log and never to the client.
    """
    return f"the retrieval backend failed ({type(exc).__name__}); details are in the server log"


def is_loopback(host: str) -> bool:
    """True for ``localhost`` and loopback IP literals; false for anything else.

    ``0.0.0.0``, ``::``, the empty string and every other hostname bind or may
    resolve beyond the local machine, so they count as non-loopback.
    """
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _host_part(allowed_host: str) -> str:
    if allowed_host.startswith("["):
        return allowed_host[1:].partition("]")[0]
    return allowed_host.partition(":")[0]


def check_http_startup(config: ServerConfig, bearer_token: str | None) -> None:
    """Refuse to serve HTTP on a non-loopback host without auth or a Host allowlist.

    Raises:
        ConfigError: the host is not loopback and either no bearer token is set
            (unless ``allow_unauthenticated``) or ``allowed_hosts`` is empty.
    """
    remote_names = [name for name in config.allowed_hosts if not is_loopback(_host_part(name))]
    if is_loopback(config.host) and not remote_names:
        return
    if not bearer_token and not config.allow_unauthenticated:
        exposure = (
            f"non-loopback host {config.host!r}"
            if not is_loopback(config.host)
            else f"allowed hosts {', '.join(remote_names)} (a reverse proxy can reach it)"
        )
        raise ConfigError(
            f"refusing to serve on {exposure} without "
            "PYRLM_MCP_BEARER_TOKEN; set PYRLM_MCP_ALLOW_UNAUTHENTICATED=1 if a gateway "
            "in front of the server authenticates clients"
        )
    if not is_loopback(config.host) and not config.allowed_hosts:
        raise ConfigError(
            f"PYRLM_MCP_ALLOWED_HOSTS is required on non-loopback host {config.host!r}: "
            "list the Host header values clients use, such as mcp.example.com or "
            "mcp.example.com:*"
        )


def transport_security(config: ServerConfig) -> TransportSecuritySettings:
    """Host and Origin allowlists for the SDK's DNS-rebinding protection.

    Always enabled: the SDK disables the protection when the host is not
    loopback and no settings are passed.
    """
    hosts = list(config.allowed_hosts)
    origins = list(config.allowed_origins)
    if is_loopback(config.host):
        bound = f"[{config.host}]" if ":" in config.host and "[" not in config.host else config.host
        hosts += [*_LOOPBACK_HOSTS, f"{bound}:*"]
        origins += [*_LOOPBACK_ORIGINS, f"http://{bound}:*"]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=hosts, allowed_origins=origins
    )


class BearerAuthMiddleware:
    """Require ``Authorization: Bearer <token>`` on every HTTP request.

    ``/health`` stays open, ``/.well-known/*`` answers 404 so OAuth-capable
    clients do not start an OAuth flow, and lifespan events pass through so the
    app can be mounted in another ASGI app.
    """

    def __init__(self, app: ASGIApp, token: str) -> None:
        if not token:
            raise ValueError("token must not be empty")
        self._app = app
        self._token = token.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self._app(scope, receive, send)
            return
        if scope["type"] != "http":
            await send({"type": "websocket.close", "code": 1008})
            return
        path: str = scope["path"]
        root_path: str = scope.get("root_path", "")
        if root_path and path.startswith(root_path):
            path = path[len(root_path) :]
        if path == HEALTH_PATH:
            await self._app(scope, receive, send)
            return
        if path.startswith("/.well-known/"):
            await _reply(send, 404, {"error": "not_found"})
            return
        header = dict(scope.get("headers") or []).get(b"authorization", b"")
        scheme, _, supplied = header.strip().partition(b" ")
        if scheme.lower() != b"bearer" or not hmac.compare_digest(supplied.strip(), self._token):
            await _reply(send, 401, {"error": "unauthorized"}, ((b"www-authenticate", b"Bearer"),))
            return
        await self._app(scope, receive, send)


async def _reply(
    send: Send, status: int, body: dict[str, str], headers: tuple[tuple[bytes, bytes], ...] = ()
) -> None:
    payload = json.dumps(body).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
                *headers,
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload})


def build_asgi_app(
    server: MCPServer, config: ServerConfig, *, bearer_token: str | None = None
) -> ASGIApp:
    """Return the streamable-HTTP ASGI app with every security layer applied.

    The app serves MCP at ``/mcp`` and an unauthenticated ``/health``. It
    enforces the start-up guard (:func:`check_http_startup`), the SDK's Host and
    Origin allowlists, and the bearer token when one is given. To mount it in
    another app, run ``server.session_manager.run()`` in that app's lifespan, as
    the SDK documents for mounted servers.

    Raises:
        ConfigError: the start-up guard refuses the configuration.
    """
    check_http_startup(config, bearer_token)
    app: ASGIApp = server.streamable_http_app(
        transport_security=transport_security(config), host=config.host
    )
    if bearer_token:
        app = BearerAuthMiddleware(app, bearer_token)
    return app


def run_server(server: MCPServer, config: ServerConfig, *, bearer_token: str | None = None) -> None:
    """Serve ``server`` on the transport ``config`` names, blocking until it stops.

    ``stdio`` first calls :func:`~pyrlm_runtime.mcp_server.stdio.claim_stdio`,
    so from then on only the protocol reaches the real stdout. Call
    ``claim_stdio()`` yourself before building the retriever to cover start-up
    output too; the CLI does. ``streamable-http`` serves :func:`build_asgi_app`
    with uvicorn on ``config.host:config.port``.

    Raises:
        ConfigError: the start-up guard refuses the configuration.
    """
    if config.transport == "stdio":
        serve_stdio(server)
        return
    import uvicorn

    app = build_asgi_app(server, config, bearer_token=bearer_token)
    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level.lower())
