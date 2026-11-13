"""MCP server for pyrlm-runtime: publishes retrieval as MCP tools.

Needs the ``mcp`` extra: ``pip install "pyrlm-runtime[mcp]"``. Importing
``pyrlm_runtime`` does not import this package.
"""

from __future__ import annotations

try:
    import mcp  # noqa: F401
except ImportError as exc:
    raise ImportError(
        'pyrlm_runtime.mcp_server needs the mcp extra: pip install "pyrlm-runtime[mcp]"'
    ) from exc

from .config import ConfigError, Secrets, ServerConfig, build_retriever
from .security import build_asgi_app, run_server
from .server import build_server

__all__ = [
    "ConfigError",
    "Secrets",
    "ServerConfig",
    "build_asgi_app",
    "build_retriever",
    "build_server",
    "run_server",
]
