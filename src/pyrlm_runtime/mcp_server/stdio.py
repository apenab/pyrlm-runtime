"""Keep the stdio protocol channel away from everything else in the process.

On stdio, file descriptors 0 and 1 carry the MCP protocol. Any other byte on
fd 1 (a ``print``, a library banner, a C extension or a child process that
inherits the descriptor) corrupts the stream. :func:`claim_stdio` moves the
protocol to private descriptors once and points fd 0 at the null device and
fd 1 at stderr for the rest of the process, so stray reads see EOF and stray
writes land on stderr.
"""

from __future__ import annotations

import io
import os
import threading
from dataclasses import dataclass
from typing import TextIO

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.stdio import stdio_server


@dataclass(frozen=True)
class StdioWire:
    """Text streams over the private duplicates of the original fd 0 and fd 1."""

    stdin: TextIO
    stdout: TextIO


_wire: StdioWire | None = None
_lock = threading.Lock()


def _private_duplicate(fd: int) -> int:
    """Duplicate ``fd`` above the standard range, closed on exec so children never inherit it."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        duplicate = os.dup(fd)
        if duplicate <= 2:
            os.close(duplicate)
            raise OSError(f"duplicate of fd {fd} landed in the standard range") from None
        return duplicate
    return fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 3)


def claim_stdio() -> StdioWire:
    """Move the protocol channel off fd 0 and fd 1 for the rest of the process.

    Idempotent: later calls return the first claim. Call it before any code
    that might write to stdout runs. Output buffered in ``sys.stdout`` before
    the claim drains to stderr, never to the protocol.

    Raises:
        OSError: fd 0 or fd 1 is closed or cannot be duplicated.
    """
    global _wire
    with _lock:
        if _wire is None:
            wire_in = _private_duplicate(0)
            wire_out = _private_duplicate(1)
            null = os.open(os.devnull, os.O_RDONLY)
            try:
                os.dup2(null, 0)
            finally:
                os.close(null)
            os.dup2(2, 1)
            _wire = StdioWire(
                stdin=io.TextIOWrapper(
                    os.fdopen(wire_in, "rb", closefd=False), encoding="utf-8", errors="replace"
                ),
                stdout=io.TextIOWrapper(os.fdopen(wire_out, "wb", closefd=False), encoding="utf-8"),
            )
        return _wire


def serve_stdio(server: MCPServer) -> None:
    """Serve ``server`` over the claimed stdio channel until the client disconnects."""
    wire = claim_stdio()
    anyio.run(_serve, server, wire)


async def _serve(server: MCPServer, wire: StdioWire) -> None:
    # The SDK has no public entry point that takes explicit streams; this is the
    # body of MCPServer.run_stdio_async with the claimed streams passed in.
    lowlevel = server._lowlevel_server
    stdin = anyio.wrap_file(wire.stdin)
    stdout = anyio.wrap_file(wire.stdout)
    async with stdio_server(stdin, stdout) as (read_stream, write_stream):
        await lowlevel.run(read_stream, write_stream, lowlevel.create_initialization_options())
