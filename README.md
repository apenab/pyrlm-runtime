# rlm-runtime

Minimal runtime for **Recursive Language Models (RLMs)** inspired by the MIT CSAIL paper
"Recursive Language Models". The core idea: the long prompt lives in a persistent environment
(Python REPL), and the LLM only sees metadata plus REPL outputs. The LLM writes code to
inspect the context and can make **subcalls** over small snippets.

## What this is
- A lightweight runtime loop: root LLM <-> REPL
- Deterministic `Context` helpers (slice/find/chunk)
- A safe-ish Python REPL with minimal builtins
- Provider-agnostic adapters + a FakeAdapter for tests
- Tracing + simple cache for replay

## What this is not
- A full agents framework
- A RAG system or vector database
- A production sandbox (this is an MVP)

## Quickstart
```bash
uv run python examples/minimal.py
```

## Example (FakeAdapter)
```python
from rlm_runtime import Context, RLM
from rlm_runtime.adapters import FakeAdapter

adapter = FakeAdapter(
    script=[
        "snippet = peek(80)\nsummary = llm_query(f'Summarize: {snippet}')\nanswer = summary",
        "FINAL_VAR: answer",
    ]
)
adapter.add_rule("You are a sub-LLM", "fake summary")

context = Context.from_text("RLMs treat long prompts as environment state.")
output, trace = RLM(adapter=adapter).run("Summarize.", context)
print(output)
```

## Design (paper-aligned)
- **Environment**: prompt lives as `P` in a REPL; helpers (`peek`, `tail`, `lenP`) are provided.
- **Context**: safe inspection (`slice`, `find`, `chunk`).
- **Policy**: step/subcall/token budgets.
- **Tracing**: structured steps + JSON export; subcalls record input/output hashes.

## Adapters
- `FakeAdapter` for tests/examples.
- `GenericChatAdapter` for schema-configurable chat endpoints.
- `OpenAICompatAdapter` for OpenAI-compatible endpoints (including Llama servers):
  - Uses `LLM_API_KEY` (preferred) or `OPENAI_API_KEY`.
  - Uses `LLM_BASE_URL` (preferred) or `OPENAI_BASE_URL`.
  - If no key is set, it sends no auth header (works for local servers).

### Llama notes
Local Llama models can be less compliant with the REPL protocol. Use the stricter
`LLAMA_SYSTEM_PROMPT` and set `require_repl_before_final=True` to force at least one REPL step.

## Roadmap
- Async subcalls
- Stronger sandboxing
- Optional tool calling

## Dev / Quality gates
```bash
uv run ruff check .
uv run ruff format .
uv run ty check
uv run pytest
```
