# CLAUDE.md - Project conventions for pyrlm-runtime

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

## Project structure

```
src/pyrlm_runtime/
  rlm.py          # Core RLM loop (run, subcalls, recursive subcalls)
  env.py          # PythonREPL (exec-based sandbox), REPLProtocol, ExecResult
  env_monty.py    # MontyREPL (pydantic-monty Rust sandbox)
  context.py      # Context dataclass (text, documents, find, chunk, slice)
  policy.py       # Policy (step/token limits)
  prompts.py      # System prompts and user message builder
  trace.py        # Trace/TraceStep for execution logging
  cache.py        # File-based cache for subcall results
  router.py       # Router for model selection
  adapters/       # LLM adapters (base, openai_compat, generic_chat, fake)
tests/
  test_*.py       # One test file per module
```

## Code style

- Formatter/linter: **ruff** (line-length=100, target py312)
- Quote style: double quotes
- Type hints: use `from __future__ import annotations` in all files
- Commit messages: conventional commits (`feat:`, `fix:`, `refactor:`, etc.)

## Testing conventions

- Framework: **pytest**
- Tests that require `pydantic-monty`: use `@pytest.mark.skipif(not MONTY_AVAILABLE, ...)`
- Use `FakeAdapter` (from `pyrlm_runtime.adapters`) for RLM integration tests

### Regression tests rule

When fixing a bug or correcting a behavior, always add a regression test that would fail if the bug were reintroduced. Propose the test proactively without waiting for the user to ask.

## REPL backends

Two interchangeable backends via `REPLProtocol`:
- `"python"` (default): PythonREPL using `exec()` with whitelist sandbox
- `"monty"`: MontyREPL using pydantic-monty (Rust interpreter, secure)

Both expose: `exec(code) -> ExecResult`, `get(name)`, `set(name, value)`

## Key patterns

- **Variable capture in MontyREPL**: AST-based detection of assignments, append capture dict, extract from result
- **Object proxy for MontyREPL**: complex objects (e.g. Context) registered via `_register_object` -- methods become external functions with `{name}__{method}` naming, AST rewrites `ctx.method()` -> `ctx__method()`
