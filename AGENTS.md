# AGENTS.md - Project conventions for pyrlm-runtime

## Project overview

Minimal runtime for Recursive Language Models (RLMs). Python 3.12+, MIT license.

## Commands

- **Install deps**: `uv sync`
- **Run all tests**: `uv run pytest`
- **Run specific test file**: `uv run pytest tests/test_env_monty.py -v`
- **Run single test**: `uv run pytest tests/test_file.py::TestClass::test_name -v`
- **Lint**: `uv run ruff check src/ tests/`
- **Format**: `uv run ruff format src/ tests/`
- **Build**: `uv build`


## Code style

- Formatter/linter: **ruff** (line-length=100, target py312)
- Quote style: double quotes
- Type hints: use `from __future__ import annotations` in all files
- Commit messages: conventional commits (`feat:`, `fix:`, `refactor:`, etc.)
- **Never** add `Co-Authored-By` lines to commits
- **Always run `uv run ruff check src/ tests/` before committing** — fix any errors before creating the commit

## Testing conventions

- Framework: **pytest**
- Tests that require `pydantic-monty`: use `@pytest.mark.skipif(not MONTY_AVAILABLE, ...)`
- Use `FakeAdapter` (from `pyrlm_runtime.adapters`) for RLM integration tests
- Tests that require the `mcp` extra: `pytest.importorskip("mcp")` at module top; the dev group installs it

### Regression tests rule

When fixing a bug or correcting a behavior, always add a regression test that would fail if the bug were reintroduced. Propose the test proactively without waiting for the user to ask.

## REPL backends

Two interchangeable backends via `REPLProtocol`:

- `"python"` (default): PythonREPL using `exec()` with whitelist sandbox
- `"monty"`: MontyREPL using pydantic-monty (Rust interpreter, secure)

Both expose: `exec(code) -> ExecResult`, `get(name)`, `set(name, value)`

## MCP server

`src/pyrlm_runtime/mcp_server/` is an optional MCP server behind the `mcp`
extra (`pip install "pyrlm-runtime[mcp]"`). Rules:

- It imports only the public API of `pyrlm_runtime`, like an external consumer.
- `service.py` holds the logic and never imports `mcp`; `server.py` is the thin
  MCP adapter.
- Tools are published by capability through the table in `server.py`.
- Importing `pyrlm_runtime` must not import `mcp` (`tests/test_mcp_imports.py`).
- On stdio, stdout is the protocol channel: log to stderr and never run a
  `PythonREPL` in the server process.

Design: `docs/MCP-SERVER-RFC.md`. User guide: `docs/MCP.md`.

## Key patterns

- **Variable capture in MontyREPL**: AST-based detection of assignments, append capture dict, extract from result
- **Object proxy for MontyREPL**: complex objects (e.g. Context) registered via `_register_object` -- methods become external functions with `{name}__{method}` naming, AST rewrites `ctx.method()` -> `ctx__method()`
- Every change in the Architecture or core logic must be accompanied by updates to the documentation inside the /docs directory.
