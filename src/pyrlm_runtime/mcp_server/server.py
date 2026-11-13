"""MCP adapter: publishes the retrieval service as MCP tools.

Tools are published by capability (see ``_TOOL_MATRIX``): a tool whose
requirements the configuration does not meet is not registered, and the
server instructions only mention registered tools.
"""

from __future__ import annotations

import enum
import logging
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from pyrlm_runtime import RetrieverProtocol, __version__

from . import prompts
from .config import ConfigError, ServerConfig
from .schemas import DocumentResponse, PagesResponse, SearchResponse
from .security import sanitize_error
from .service import RetrievalService, ServiceError

logger = logging.getLogger("pyrlm_runtime.mcp_server")


class Capability(enum.Enum):
    """What the server was given, which decides the tools it publishes."""

    RETRIEVER = "retriever"
    PAGE_READER = "page_reader"


_READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)


@dataclass(frozen=True)
class _ToolSpec:
    name: str
    needs: frozenset[Capability]
    annotations: ToolAnnotations
    instruction: str


_TOOL_MATRIX: tuple[_ToolSpec, ...] = (
    _ToolSpec("search", frozenset({Capability.RETRIEVER}), _READ_ONLY, prompts.SEARCH_INSTRUCTION),
    _ToolSpec(
        "get_document",
        frozenset({Capability.RETRIEVER}),
        _READ_ONLY,
        prompts.GET_DOCUMENT_INSTRUCTION,
    ),
    _ToolSpec(
        "get_pages",
        frozenset({Capability.RETRIEVER, Capability.PAGE_READER}),
        _READ_ONLY,
        prompts.GET_PAGES_INSTRUCTION,
    ),
)


def build_server(
    *,
    retriever: RetrieverProtocol | None,
    config: ServerConfig | None = None,
) -> MCPServer:
    """Build an MCP server that publishes the tools ``retriever`` can serve.

    ``search`` and ``get_document`` need a retriever; ``get_pages`` also needs
    the retriever to have a ``get_pages`` method. Serve the result with
    :func:`run_server`, or mount :func:`build_asgi_app` in an ASGI app; calling
    the SDK's ``run`` or ``streamable_http_app`` directly skips the security
    checks.

    Raises:
        ConfigError: no tool can be published.
    """
    config = config or ServerConfig()
    capabilities: set[Capability] = set()
    service: RetrievalService | None = None
    if retriever is not None:
        service = RetrievalService(retriever, config)
        capabilities.add(Capability.RETRIEVER)
        if service.supports_pages:
            capabilities.add(Capability.PAGE_READER)

    published = [spec for spec in _TOOL_MATRIX if spec.needs <= capabilities]
    if not published or service is None:
        raise ConfigError(
            "nothing to publish: configure a retriever (PYRLM_MCP_ES_HOST or "
            "PYRLM_MCP_RETRIEVER_FACTORY)"
        )

    server = MCPServer(
        config.name,
        instructions=prompts.server_instructions([spec.instruction for spec in published]),
        version=__version__,
        log_level=config.log_level,
    )
    handlers = _retrieval_handlers(service, config)
    for spec in published:
        handler, description = handlers[spec.name]
        server.add_tool(
            handler,
            name=spec.name,
            description=description,
            annotations=spec.annotations,
            structured_output=True,
        )

    @server.custom_route("/health", methods=["GET"])
    async def health(_: Request) -> Response:
        return JSONResponse({"status": "ok"})

    return server


def _retrieval_handlers(
    service: RetrievalService, config: ServerConfig
) -> dict[str, tuple[Callable[..., Any], str]]:
    def search(
        query: str,
        mode: Literal["hybrid", "lexical", "semantic"] | None = None,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        include_content: bool = False,
    ) -> SearchResponse:
        with _tool_call("search"):
            return service.search(
                query, mode=mode, top_k=top_k, filters=filters, include_content=include_content
            )

    # The schema's mode enum lists only the offered modes. The annotation is
    # replaced with a type object, which the SDK uses as is.
    search.__annotations__["mode"] = Literal[tuple(config.search_modes)] | None  # ty: ignore[invalid-type-form]

    def get_document(
        doc_id: str, offset: int = 0, max_chars: int | None = None
    ) -> DocumentResponse:
        with _tool_call("get_document"):
            return service.get_document(doc_id, offset=offset, max_chars=max_chars)

    def get_pages(logical_doc_id: str, pages: list[int], radius: int = 0) -> PagesResponse:
        with _tool_call("get_pages"):
            return service.get_pages(logical_doc_id, pages=pages, radius=radius)

    return {
        "search": (search, prompts.search_description(config.search_modes)),
        "get_document": (get_document, prompts.GET_DOCUMENT_DESCRIPTION),
        "get_pages": (get_pages, prompts.GET_PAGES_DESCRIPTION),
    }


@contextmanager
def _tool_call(tool: str) -> Generator[None]:
    started = time.perf_counter()
    status = "ok"
    try:
        yield
    except ServiceError as exc:
        status = "rejected"
        raise ToolError(str(exc)) from None
    except Exception as exc:
        status = "failed"
        logger.warning("tool %s failed", tool, exc_info=exc)
        raise ToolError(sanitize_error(exc)) from None
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info("tool=%s status=%s duration_ms=%.0f", tool, status, elapsed_ms)
