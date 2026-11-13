from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

pytest.importorskip("mcp")

import anyio  # noqa: E402
from mcp import Client, StdioServerParameters  # noqa: E402

pytestmark = pytest.mark.anyio

TESTS_DIR = Path(__file__).resolve().parent

NOISY_FACTORY = textwrap.dedent(
    """
    import os
    import sys

    from _mcp_fakes import PagedRetriever


    def noise(stage):
        print(f"NOISE-print-{stage}")
        os.write(1, f"NOISE-fd-{stage}\\n".encode())
        sys.stdout.write(f"NOISE-write-{stage} ")
        sys.stdout.flush()


    noise("import")


    class NoisyRetriever(PagedRetriever):
        def _run(self, *args, **kwargs):
            noise("tool")
            return super()._run(*args, **kwargs)

        def close(self):
            noise("close")


    def make():
        noise("build")
        return NoisyRetriever()
    """
)
STAGES = ("import", "build", "tool", "close")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def server_env(tmp_path):
    (tmp_path / "noisy_factory.py").write_text(NOISY_FACTORY)
    return {
        "PYTHONPATH": os.pathsep.join([str(tmp_path), str(TESTS_DIR)]),
        "PYRLM_MCP_RETRIEVER_FACTORY": "noisy_factory:make",
        "PYRLM_MCP_LOG_LEVEL": "WARNING",
    }


def _rpc(method, params=None, id=None):
    message = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        message["params"] = params
    if id is not None:
        message["id"] = id
    return (json.dumps(message) + "\n").encode()


def test_stdout_carries_only_protocol_messages_for_the_whole_process(server_env):
    process = subprocess.Popen(
        [sys.executable, "-m", "pyrlm_runtime.mcp_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, **server_env},
    )
    lines: queue.Queue[bytes] = queue.Queue()
    threading.Thread(
        target=lambda: [lines.put(line) for line in iter(process.stdout.readline, b"")],
        daemon=True,
    ).start()

    def send(payload: bytes) -> None:
        process.stdin.write(payload)
        process.stdin.flush()

    try:
        send(
            _rpc(
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
                id=1,
            )
        )
        first = lines.get(timeout=30)
        send(_rpc("notifications/initialized"))
        send(_rpc("tools/call", {"name": "search", "arguments": {"query": "revenue"}}, id=2))
        second = lines.get(timeout=30)
        process.stdin.close()
        process.wait(timeout=30)
    finally:
        process.kill()
    rest = []
    while not lines.empty():
        rest.append(lines.get())
    stderr = process.stderr.read().decode()

    stdout_lines = [first, second, *rest]
    messages = [json.loads(line) for line in stdout_lines]
    assert [message.get("id") for message in messages] == [1, 2]
    assert messages[1]["result"]["isError"] is False
    for stage in STAGES:
        for channel in ("print", "write", "fd"):
            assert f"NOISE-{channel}-{stage}" in stderr


async def test_official_client_completes_a_session_with_a_noisy_retriever(server_env):
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "pyrlm_runtime.mcp_server"], env=server_env
    )
    # Unprotected stdout made the client's first request time out (about 10 s)
    # before it fell back; a clean channel finishes in well under a second.
    with anyio.fail_after(5):
        async with Client(params) as client:
            tools = (await client.list_tools()).tools
            result = await client.call_tool("search", {"query": "revenue"})

    assert sorted(tool.name for tool in tools) == ["get_document", "get_pages", "search"]
    assert result.is_error is False
