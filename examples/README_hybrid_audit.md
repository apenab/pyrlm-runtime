# Hybrid Audit Example

`examples/hybrid_audit.py` demonstrates a hybrid pipeline:

- Deterministic parsing and aggregation in the REPL.
- LLM subcalls for semantic classification (billing/login/bug/feature/other).
- Baselines (truncation + windowed, optional RAG-style line filtering).

## Run

```bash
LLM_MODEL=qwen2.5-coder:7b \
LLM_SUBCALL_MODEL=qwen2.5:3b \
N_ITEMS=400 \
BATCH_SIZE=20 \
BASELINE_MAX_CHARS=8000 \
uv run python examples/hybrid_audit.py
```

## Environment variables

- `LLM_MODEL` / `LLM_SUBCALL_MODEL`: root + subcall models.
- `LLM_BASE_URL`: OpenAI-compatible endpoint (default `http://localhost:11434/v1`).
- `LLM_TIMEOUT`: request timeout seconds.
- `LLM_API_KEY`: API key for hosted endpoints (fallback to `OPENAI_API_KEY`).
- `LLM_SUBCALL_API_KEY`: optional API key for subcall model (defaults to `LLM_API_KEY`).
- `N_ITEMS`: number of synthetic tickets (comma-separated list allowed).
- `BATCH_SIZE`: items per subcall batch.
- `BASELINE_MAX_CHARS`: truncation size for baseline.
- `PARALLEL_SUBCALLS`: set `1` to enable parallel subcalls.
- `MAX_WORKERS`: max workers for parallel subcalls.
- `RAG_BASELINE`: set `1` to enable the simple keyword RAG baseline.
- `RAG_LINES`: number of lines kept by the RAG baseline.
- `WRITE_DATA`: set `1` to write `examples/data/hybrid_audit_seeded.jsonl`.
- `WRITE_DATA_PATH`: override the output path for JSONL.
- `STRICT_LLM`: set `1` to disable the deterministic fallback when subcall outputs are malformed.

## Output

The script prints per-run metrics for:

- Baseline (truncation)
- Windowed baseline (head/mid/tail)
- Optional RAG baseline
- RLM hybrid pipeline

It also prints a summary table with tokens, latency, and winner.
