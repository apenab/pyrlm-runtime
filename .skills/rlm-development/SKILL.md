---
name: rlm-development
description: Guide for developing and contributing to the pyrlm-runtime project. Use when modifying core modules, adding features, fixing bugs, or understanding the architecture of the Recursive Language Model runtime.
license: MIT
metadata:
  author: apenab
  version: "1.0"
---

# RLM Runtime Development Guide

## Project Overview

`pyrlm-runtime` is a minimal runtime for Recursive Language Models (RLMs) based on the MIT CSAIL paper. Instead of feeding full context into an LLM, RLMs treat large context as **environment state** in a Python REPL. The LLM writes code to inspect, chunk, and recursively query sub-LLMs on smaller pieces.

## Architecture

```
src/pyrlm_runtime/
├── rlm.py              # Core RLM loop (entry point: RLM.run())
├── context.py           # Immutable Context dataclass with inspection helpers
├── env.py               # Sandboxed PythonREPL for code execution
├── policy.py            # Execution limits (steps, tokens, subcalls, recursion)
├── cache.py             # File-based subcall memoization (.rlm_cache/)
├── trace.py             # Execution recording (TraceStep, Trace)
├── router.py            # SmartRouter: auto baseline vs RLM selection
├── prompts.py           # System prompts (BASE, SUBCALL, RECURSIVE, LLAMA, QWEN)
└── adapters/
    ├── base.py          # ModelAdapter protocol, Usage, ModelResponse
    ├── openai_compat.py # OpenAI-compatible API wrapper
    ├── generic_chat.py  # HTTP-based generic chat adapter
    └── fake.py          # Deterministic adapter for testing
```

## Key Design Patterns

### 1. Protocol-Based Adapters

All model integrations implement the `ModelAdapter` protocol:

```python
class ModelAdapter(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ModelResponse: ...
```

When adding a new adapter, implement this protocol. Return `ModelResponse(text=..., usage=Usage(...))`.

### 2. Frozen Dataclasses for Immutability

`Context`, `Usage`, `ModelResponse`, and `ExecResult` are frozen dataclasses. This ensures determinism and thread safety. Never make these mutable.

### 3. Sandboxed REPL

`PythonREPL` restricts execution to safe builtins and allowed modules (`re`, `math`, `json`, `textwrap`). When modifying the REPL:
- Never add `os`, `sys`, `subprocess`, or network modules to allowed imports
- Keep the stdout limit (default 4000 chars) to prevent memory issues
- The `_RegexProxy` wraps `re` to auto-coerce Context/list/tuple inputs to strings

### 4. The RLM Loop (rlm.py)

The core execution loop in `RLM.run()`:
1. Initialize REPL with context as `P` (string) and `ctx` (Context object)
2. Inject helper functions: `peek()`, `tail()`, `lenP()`, `subcall()`, `ask()`, etc.
3. Loop: call root LLM -> parse response -> execute code or finalize
4. LLM outputs Python code (executed in REPL) or `FINAL:`/`FINAL_VAR:` to end
5. REPL stdout/errors feed back to the LLM on next iteration

### 5. Policy Enforcement

`Policy` tracks and limits: steps, subcalls, recursion depth, total tokens, subcall tokens. Each has a corresponding exception (`MaxStepsExceeded`, etc.). Always respect policy checks when adding new execution paths.

## Coding Conventions

### Style
- **Formatter**: `ruff format` (line-length=100, double quotes, space indent)
- **Linter**: `ruff check`
- **Type checker**: `ty`
- **Python**: 3.12+ (use `X | Y` union syntax, not `Optional[X]`)

### Commits
- Use **conventional commits** via commitizen: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
- Run `uv run ruff format . && uv run ruff check .` before committing

### Testing
- Framework: `pytest` (config in `pyproject.toml`)
- Run: `uv run pytest`
- Use `FakeAdapter` from `adapters/fake.py` for deterministic tests (no API calls needed)
- Test files: `tests/test_*.py` matching source modules

### Dependencies
- **Only external dependency**: `httpx>=0.27`
- Dev deps: `pytest`, `ruff`, `ty`, `commitizen`
- Keep dependencies minimal by design

## How to Make Common Changes

### Adding a new REPL helper function

1. Define the function inside `RLM.run()` (it has access to `context`, `repl`, `policy`, `trace`)
2. Register it with `repl.set("function_name", function)`
3. Document it in the system prompt (`prompts.py`) so the LLM knows it exists
4. Add tests in `tests/test_rlm_loop.py` using `FakeAdapter`

### Adding a new model adapter

1. Create `src/pyrlm_runtime/adapters/your_adapter.py`
2. Implement the `ModelAdapter` protocol (just the `complete` method)
3. Return `ModelResponse(text=..., usage=Usage(prompt_tokens=..., completion_tokens=..., total_tokens=...))`
4. Export from `__init__.py` if it should be part of the public API

### Modifying the execution loop

1. Read `rlm.py` carefully - the loop has guards, fallbacks, and auto-finalization
2. Key flags: `repl_executed`, `subcall_made`, `invalid_responses`, `repl_errors`
3. New execution paths must call `policy.check_step()` / `policy.check_subcall()`
4. Record steps via `trace.add(TraceStep(...))` for observability
5. Test with `FakeAdapter` scripted responses to simulate multi-step conversations

### Adding a new system prompt variant

1. Add the prompt string to `prompts.py`
2. Follow existing patterns (available helpers, output format rules, examples)
3. The LLM must know about `FINAL:` / `FINAL_VAR:` syntax to terminate
4. Pass via `RLM(system_prompt=YOUR_PROMPT)`

### Modifying Context

1. `Context` is frozen - add new class methods or instance methods, don't change existing signatures
2. If adding document-level operations, handle both `context_type="string"` and `"document_list"`
3. Update `metadata()` if new fields are needed by prompts
4. Test in `tests/test_context.py`

## Public API (exported from `__init__.py`)

Core: `RLM`, `Context`, `PythonREPL`, `ExecResult`
Config: `Policy`, `PolicyError`, `MaxStepsExceeded`, `MaxSubcallsExceeded`, `MaxRecursionExceeded`, `MaxTokensExceeded`
Adapters: `ModelAdapter`, `ModelResponse`, `Usage`, `FileCache`
Routing: `SmartRouter`, `RouterConfig`, `RouterResult`, `ExecutionProfile`, `TraceFormatter`
Tracing: `Trace`, `TraceStep`

## Environment Variables

| Variable | Purpose | Example |
|---|---|---|
| `LLM_BASE_URL` | API endpoint | `http://localhost:11434/v1` |
| `LLM_MODEL` | Root model name | `qwen2.5-coder:14b-instruct` |
| `LLM_SUBCALL_MODEL` | Subcall model (can be smaller) | `qwen2.5-coder:7b-instruct` |
| `LLM_API_KEY` / `OPENAI_API_KEY` | API authentication | |
| `LLM_LOG_LEVEL` | Logging verbosity | `INFO`, `DEBUG` |
| `PARALLEL_SUBCALLS` | Enable parallel subcalls | `1` |
| `SHOW_TRAJECTORY` | Show execution trajectory | `1` |

## Running Commands

```bash
# Run all tests
uv run pytest

# Format code
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run ty check

# Build package
uv build

# Run an example
uv run python examples/minimal.py
```
