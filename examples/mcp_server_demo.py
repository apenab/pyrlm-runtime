"""Run the pyrlm-runtime MCP server as a subprocess and drive it with a real MCP client.

The server gets an in-memory retriever through ``PYRLM_MCP_RETRIEVER_FACTORY``,
so no Elasticsearch is needed.

    uv run python examples/mcp_server_demo.py stdio
    uv run python examples/mcp_server_demo.py http

``stdio`` launches ``python -m pyrlm_runtime.mcp_server`` and talks over its
stdin and stdout. ``http`` launches it on a free loopback port with a bearer
token, then connects once with the token and once without.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import anyio
import httpx
import httpx2
from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

EXAMPLES_DIR = str(Path(__file__).resolve().parent)
SERVER_ENV = {
    "PYTHONPATH": EXAMPLES_DIR,
    "PYRLM_MCP_RETRIEVER_FACTORY": "mcp_server_demo:make_retriever",
    "PYRLM_MCP_LOG_LEVEL": "WARNING",
}

PAGES = {
    "report": [
        "Quarterly report. Revenue grew 12 percent on strong northern sales.",
        "Costs rose 4 percent, driven by logistics and new hires.",
        "Outlook: management expects revenue growth to continue next year.",
    ],
    "memo": ["Internal memo: the northern warehouse opens in March."],
}


class DemoRetriever:
    """Keyword retriever over a few paged documents."""

    def __init__(self) -> None:
        self.units = [
            {
                "doc_id": f"{name}-p{number}",
                "content": text,
                "metadata": {"doc_id": name, "page_num": number},
            }
            for name, pages in PAGES.items()
            for number, text in enumerate(pages, start=1)
        ]

    def _hit(self, unit: dict[str, Any], score: float) -> dict[str, Any]:
        return {
            "doc_id": unit["doc_id"],
            "content": unit["content"],
            "preview": unit["content"][:40],
            "score": score,
            "metadata": dict(unit["metadata"]),
            "logical_doc_id": unit["metadata"]["doc_id"],
            "page_num": unit["metadata"]["page_num"],
        }

    def search(
        self, query: str, *, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        terms = query.lower().split()
        scored = []
        for unit in self.units:
            if filters and any(unit["metadata"].get(k) != v for k, v in filters.items()):
                continue
            score = sum(unit["content"].lower().count(term) for term in terms)
            if score:
                scored.append((score, unit))
        scored.sort(key=lambda item: -item[0])
        return [self._hit(unit, float(score)) for score, unit in scored[:top_k]]

    vector_search = search
    hybrid_search = search

    def get(self, doc_id: str) -> dict[str, Any]:
        for unit in self.units:
            if unit["doc_id"] == doc_id:
                return dict(unit)
        raise KeyError(doc_id)

    def get_pages(
        self,
        logical_doc_id: str,
        *,
        pages: list[int] | None = None,
        radius: int = 0,
        max_pages: int = 20,
    ) -> list[dict[str, Any]]:
        wanted = {p + d for p in pages or [] for d in range(-radius, radius + 1)}
        found = [
            {**unit, "page_doc_id": unit["doc_id"], "page_num": unit["metadata"]["page_num"]}
            for unit in self.units
            if unit["metadata"]["doc_id"] == logical_doc_id
            and (not wanted or unit["metadata"]["page_num"] in wanted)
        ]
        return found[:max_pages]


def make_retriever() -> DemoRetriever:
    return DemoRetriever()


def _show(label: str, result: Any) -> None:
    if result.is_error:
        print(f"{label}: ERROR {result.content[0].text}")
    else:
        print(f"{label}: {result.structured_content}")


async def exercise(client: Client) -> None:
    tools = (await client.list_tools()).tools
    print("tools:", sorted(tool.name for tool in tools))
    search = await client.call_tool("search", {"query": "revenue growth", "top_k": 2})
    _show("search", search)
    hit = search.structured_content["results"][0]
    _show("get_document", await client.call_tool("get_document", {"doc_id": hit["doc_id"]}))
    _show(
        "get_pages",
        await client.call_tool(
            "get_pages", {"logical_doc_id": hit["logical_doc_id"], "pages": [2], "radius": 0}
        ),
    )
    _show(
        "get_pages with a hit doc_id",
        await client.call_tool("get_pages", {"logical_doc_id": hit["doc_id"], "pages": [1]}),
    )
    _show("search with bad top_k", await client.call_tool("search", {"query": "x", "top_k": "?"}))


async def run_stdio() -> None:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "pyrlm_runtime.mcp_server"], env=SERVER_ENV
    )
    async with Client(params) as client:
        await exercise(client)


def _describe(exc: BaseException) -> str:
    while isinstance(exc, BaseExceptionGroup):
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def run_http() -> None:
    port = _free_port()
    token = "demo-token"
    env = {
        **os.environ,
        **SERVER_ENV,
        "PYRLM_MCP_TRANSPORT": "streamable-http",
        "PYRLM_MCP_PORT": str(port),
        "PYRLM_MCP_BEARER_TOKEN": token,
    }
    process = subprocess.Popen([sys.executable, "-m", "pyrlm_runtime.mcp_server"], env=env)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                print("health:", httpx.get(f"{base}/health").json())
                break
            except httpx.TransportError:
                time.sleep(0.1)
        try:
            async with Client(f"{base}/mcp") as client:
                await client.list_tools()
            print("client without token: connected (unexpected)")
        except Exception as exc:
            print("client without token: refused,", _describe(exc))
        http_client = httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"})
        async with (
            http_client,
            Client(streamable_http_client(f"{base}/mcp", http_client=http_client)) as client,
        ):
            await exercise(client)
    finally:
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    anyio.run(run_http if mode == "http" else run_stdio)
