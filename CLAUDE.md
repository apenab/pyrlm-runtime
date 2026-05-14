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
  retrieval.py    # ElasticsearchRetriever: hybrid search, page-aware ops, RRF
  adapters/       # LLM adapters (base, openai_compat, generic_chat, fake)
  doctools/
    tools.py      # Six doc tools: list_pdfs, get_pdf_info, read_pdf_pages,
                  #   extract_table, search_in_pdf, search_corpus
    policy.py     # DocumentPolicy: per-query PDF/page/table budgets
    cache.py      # Write-through file cache for PDF page reads
    protocols.py  # PageReaderProtocol, DocIndexStoreProtocol
    schema.py     # DocInfo, PageInfo Pydantic models
    prompts.py    # System prompt supplement for doc tools
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

## Design constraints

### This library is generic — domain logic belongs in consumers

pyrlm-runtime provides primitives: RLM loop, REPL sandbox, Context, retrieval, doc tools, policy enforcement. It must not contain domain-specific logic (financial extraction, Spanish text parsing, banking heuristics, etc.). That belongs in consumers like `banking-rlm`.

### When changes to this library are appropriate

- A **missing primitive** that cannot be composed from existing ones in a consumer
- A **general-purpose improvement** that benefits any RLM consumer, not just one
- A **bug fix** in existing behavior
- An **additive change** — do not break existing interfaces

### Be conservative with retrieval.py and doctools/

These are the most used integration points. Before modifying:
1. Check whether the gap can be solved with existing filter DSL or function composition
2. If a new function is needed, ensure it follows the existing naming and return-value conventions
3. Add tests that use `FakeAdapter` to verify the new behavior without hitting real ES or PDFs
