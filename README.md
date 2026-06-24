# pyrlm-runtime

Minimal Python runtime for **Recursive Language Models (RLMs)** — inspired by the [MIT CSAIL paper](https://arxiv.org/abs/2512.24601) _"Recursive Language Models"_.

RLMs solve the long-context problem: instead of sending huge contexts directly to an LLM (which truncates or degrades), the context lives as **environment state** in a Python REPL. The LLM writes code to inspect, search, and chunk the data, making **recursive subcalls** to smaller models when needed. Result: handle arbitrarily large contexts with constant token usage per step.

[![Become a Patron](docs/assets/become_a_patron_button.png)](https://www.patreon.com/u53467986)

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Live Rich Trace](#live-rich-trace)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
  - [RLM](#rlm)
  - [Context](#context)
  - [Adapters](#adapters)
  - [Policy](#policy)
  - [Trace](#trace)
  - [Cache](#cache)
  - [Router](#router)
  - [Reranking](#reranking)
  - [Multi-Query Retrieval](#multi-query-retrieval)
- [REPL Backends](#repl-backends)
- [REPL Functions Available to the LLM](#repl-functions-available-to-the-llm)
- [Retrieval Integration](#retrieval-integration)
- [Extending the REPL](#extending-the-repl)
- [Parallel Subcalls](#parallel-subcalls)
- [Multi-Turn Conversation History](#multi-turn-conversation-history)
- [Guard Mechanisms & Fallbacks](#guard-mechanisms--fallbacks)
- [Configuration](#configuration)
- [Examples](#examples)
- [When to Use RLMs](#when-to-use-rlms)
- [Benchmark: RLM vs Baseline](#benchmark-rlm-vs-baseline)
- [Development](#development)
- [References](#references)
- [License](#license)

## Installation

```bash
pip install pyrlm-runtime
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add pyrlm-runtime
```

For live terminal visualization of the REPL loop with `rich`:

```bash
pip install "pyrlm-runtime[rich]"
```

**Requirements:** Python 3.12+

**Optional:** For the secure Monty REPL backend (Rust sandbox):

```bash
pip install pydantic-monty
```

## Quickstart

### 1. Set your API key

```bash
export LLM_API_KEY="your-api-key-here"
# Optional: custom endpoint (Ollama, LM Studio, etc.)
# export LLM_BASE_URL="http://localhost:11434/v1"
```

### 2. Basic usage

```python
from pathlib import Path

from pyrlm_runtime import RLM, Context, FileCache
from pyrlm_runtime.adapters import OpenAICompatAdapter

# Load a whole folder of Markdown docs as context — this can be hundreds of
# files and millions of tokens. The data lives in the REPL, NOT in the prompt,
# so the size of this list is not bounded by the model's context window.
documents = [p.read_text(encoding="utf-8") for p in Path("docs/").rglob("*.md")]
context = Context.from_documents(documents)

# Initialize RLM with an adapter and a few useful options enabled
adapter = OpenAICompatAdapter(model="gpt-5.1")
rlm = RLM(
    adapter=adapter,
    # Route the many small sub-LLM calls to a cheaper model
    subcall_adapter=OpenAICompatAdapter(model="gpt-5.1-mini"),
    # Persist subcall results to disk — identical subcalls aren't paid twice
    cache=FileCache(root="./.rlm_cache"),
    # Let sub-LLMs run their own mini-RLM loop on large chunks (paper-aligned)
    recursive_subcalls=True,
    # Fan out independent subcalls concurrently (LLM calls are I/O-bound)
    parallel_subcalls=True,
)

# Ask questions over the entire corpus
answer, trace = rlm.run("What are the main themes across all documents?", context)
print(answer)
print(f"Solved in {len(trace.steps)} steps")  # the trace logs every step of the loop
```

> For unusually long trajectories you can also enable `compaction=True` with
> `compaction_threshold_pct=0.85` to summarize old turns instead of overflowing
> the window — see [Multi-Turn Conversation History](#multi-turn-conversation-history).

### 3. Run without external APIs (for testing)

```python
from pyrlm_runtime import RLM, Context
from pyrlm_runtime.adapters import FakeAdapter

adapter = FakeAdapter(script=[
    "snippet = peek(80)\nsummary = llm_query(f'Summarize: {snippet}')\nanswer = f'Summary -> {summary}'",
    "FINAL_VAR: answer",
])
adapter.add_rule("You are a sub-LLM", "[fake] short summary")

context = Context.from_text("RLMs treat long prompts as environment state.")
output, trace = RLM(adapter=adapter).run("Summarize this.", context)
print(output)  # Summary -> [fake] short summary
```

## Live Rich Trace

```python
from rich.console import Console

from pyrlm_runtime import Context, RLM
from pyrlm_runtime.adapters import FakeAdapter
from pyrlm_runtime.rich_trace import RichTraceListener

console = Console()
listener = RichTraceListener(console=console)

adapter = FakeAdapter(
    script=[
        "snippet = peek(40)\nsummary = llm_query(f'Summarize: {snippet}')\nprint(summary)\nanswer = summary",
        "FINAL_VAR: answer",
    ]
)
adapter.add_rule("You are a sub-LLM", "[fake] summary")

output, trace = RLM(adapter=adapter, event_listener=listener).run(
    "Summarize the first chunk.",
    Context.from_text("RLMs treat long prompts as environment state."),
)
```

With a real Azure OpenAI deployment:

```python
from dotenv import load_dotenv

from pyrlm_runtime import Context, RLM
from pyrlm_runtime.adapters import AzureOpenAIAdapter
from pyrlm_runtime.rich_trace import RichTraceListener

load_dotenv()

adapter = AzureOpenAIAdapter(model="gpt-5.1")
listener = RichTraceListener()

demo_text = "SpaceX Falcon 9 launched on Jan 6 with $50M revenue. ..."

output, trace = RLM(adapter=adapter, event_listener=listener).run(
    "Which launch had the largest revenue?",
    Context.from_text(demo_text),
)
```

Azure env contract for the live demo:

```bash
AZURE_OPENAI_API_KEY="..."
OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
# or: AZURE_ACCOUNT_NAME="your-resource"
AZURE_OPENAI_API_VERSION="2024-10-21"  # optional

uv run python examples/rich_repl_demo.py --model gpt-5.1
```

## Core Concepts

### How the RLM loop works

```
rlm.run(query, context)
  │
  ├── 1. Initialize REPL with context as variables `P` (text) and `ctx` (Context object)
  ├── 2. Build system prompt + user message with context metadata
  │
  └── 3. Loop (until FINAL or max_steps):
        │
        ├── LLM generates Python code (or FINAL answer)
        │
        ├── If code → execute in REPL sandbox
        │   ├── Code can call peek(), ctx.find(), ctx.chunk(), etc.
        │   ├── Code can call llm_query() / ask_chunks() for subcalls
        │   └── REPL output is sent back to LLM as next iteration
        │
        └── If FINAL → return answer
            ├── "FINAL: <answer>"        → inline answer
            ├── "FINAL_VAR: <varname>"   → return REPL variable value
            └── auto_finalize_var        → return when variable is set

Return: (output: str, trace: Trace)
```

### Finalization

The LLM signals completion in three ways:

| Method              | Example                                    | When to use                      |
| ------------------- | ------------------------------------------ | -------------------------------- |
| `FINAL: <text>`     | `FINAL: The answer is 42`                  | Short inline answers             |
| `FINAL_VAR: <name>` | `FINAL_VAR: result`                        | Return a REPL variable           |
| `auto_finalize_var` | `RLM(adapter, auto_finalize_var="answer")` | Auto-return when variable is set |

## API Reference

### RLM

The main entry point. Orchestrates the REPL loop, subcalls, and conversation history.

```python
from pyrlm_runtime import RLM

rlm = RLM(
    adapter,                            # Required: LLM adapter (see Adapters)
    policy=None,                        # Resource limits (see Policy)
    cache=None,                         # Subcall cache (see Cache)
    cache_dir=".rlm_cache",             # Directory used when a cache is enabled

    # Output limits
    max_output_tokens=4096,             # Max tokens the root LLM generates per call
                                        # (output cap, not a budget — see Policy.max_total_tokens)

    # System prompts
    system_prompt=BASE_SYSTEM_PROMPT,           # Root loop system prompt
    subcall_system_prompt=SUBCALL_SYSTEM_PROMPT,# System prompt for subcalls
    system_prompt_supplement="",                # Extra text appended after retrieval docs

    # REPL backend
    repl_backend="python",              # "python" (default) or "monty" (sandboxed)
    repl_exec_timeout=60.0,             # Per-exec CPU-time budget in seconds (python backend; None disables)

    # Conversation history & compaction
    conversation_history=True,          # Multi-turn mode (default: True)
    compaction=False,                   # Off by default; summarizes old turns when enabled
    compaction_threshold_tokens=80000,  # Token threshold that triggers compaction
    compaction_threshold_pct=0.0,       # If >0, threshold = pct * model context window (overrides above)
    compaction_model_context_limit=0,   # Model window in tokens (needed for *_pct)
    compaction_model_name="",           # Model name for accurate tiktoken counting (else len//4)
    compaction_summary_max_tokens=800,  # Max tokens for the running summary
    max_history_tokens=0,               # DEPRECATED: blunt history trim (0=disabled)

    # Retrieval
    retriever=None,                     # RetrieverProtocol impl (e.g. ElasticsearchRetriever)

    # Subcalls
    subcall_adapter=None,               # Separate (cheaper) adapter for subcalls
    subcall_max_output_tokens=1024,     # Output cap per subcall response
    parallel_subcalls=False,            # Run subcalls in parallel
    max_concurrent_subcalls=10,         # Max concurrent subcalls when parallel

    # Recursive subcalls (subcall runs a mini-RLM loop)
    recursive_subcalls=False,           # Enable recursion
    max_recursion_depth=2,              # Max recursion depth
    recursive_subcall_system_prompt=RECURSIVE_SUBCALL_SYSTEM_PROMPT,
    recursive_subcall_max_steps=3,      # Max steps for a recursive subcall RLM

    # Finalization control
    auto_finalize_var=None,             # REPL var name to auto-return as the answer
    auto_finalize_min_length=0,         # Min chars before auto_finalize_var triggers
    auto_finalize_reject_patterns=None, # Regexes that reject an auto-finalize answer
    auto_finalize_reject_limit=3,       # Max rejects before accepting anyway
    auto_finalize_reject_feedback=None, # Custom feedback template on reject
    min_steps=0,                        # Min steps before finalization is allowed

    # Guards & fallbacks
    require_repl_before_final=False,    # Enforce ≥1 REPL execution
    require_subcall_before_final=False, # Enforce ≥1 subcall
    invalid_response_limit=None,        # Max retries on non-code responses
    fallback_code=None,                 # Emergency code if LLM stalls
    repl_error_limit=None,              # Abort after N consecutive REPL errors
    loop_collapse_limit=5,              # Abort after N no-progress responses (None disables)
    subcall_guard_steps=None,           # Force a subcall if none made by step N
    fallback_guard_steps=None,          # Inject fallback_code if stalled by step N

    # Extensions / plugins (see "Extending the REPL")
    repl_extensions=None,               # Callable injecting domain-specific REPL functions
    doc_tools=None,                     # dict of on-demand PDF/doc tools to inject

    # Observability
    logger=None,                        # logging.Logger for the loop
    event_listener=None,                # RLMEventListener for streaming step events
    rlm_diagnostics=False,              # Emit extra diagnostic logging
    log_truncate_code=2000,             # Truncation limits (chars) for debug logs
    log_truncate_output=1000,
    log_truncate_prompt_summary=2000,
)

output, trace = rlm.run(query="Your question", context=context)
```

### Context

Wraps your data and provides safe inspection methods for the REPL.

```python
from pyrlm_runtime import Context

# From a single text
context = Context.from_text("Your long text here...")

# From multiple documents (joined by `separator`, default "\n\n---\n\n")
context = Context.from_documents([
    "Document 1 content...",
    "Document 2 content...",
    "Document 3 content...",
])  # override with separator="..." if you need a custom boundary

# Available methods (used by the LLM inside the REPL):
context.len_chars()                    # Total character count
context.num_documents()                # Number of documents
context.get_document(index)            # Get a specific document
context.document_lengths()             # List of document lengths
context.slice(start, end)             # Safe substring
context.find(pattern, regex=False)    # Search with optional regex
context.chunk(size, overlap=0)        # Split into chunks
context.chunk_documents(docs_per_chunk=10)  # Group documents into chunks
context.metadata()                    # Summary dict for system prompts
```

### Adapters

Adapters connect pyrlm-runtime to any LLM provider.

#### OpenAICompatAdapter

Works with OpenAI, Anthropic (via proxy), Ollama, LM Studio, vLLM, and any OpenAI-compatible API.

```python
from pyrlm_runtime.adapters import OpenAICompatAdapter

# OpenAI
adapter = OpenAICompatAdapter(model="gpt-5.1")

# Ollama (local)
adapter = OpenAICompatAdapter(
    model="llama3",
    base_url="http://localhost:11434/v1",
)

# Any OpenAI-compatible endpoint
adapter = OpenAICompatAdapter(
    model="my-model",
    base_url="https://my-endpoint.com/v1",
)
```

Uses environment variables: `LLM_API_KEY` (or `OPENAI_API_KEY`), `LLM_BASE_URL`.

#### GenericChatAdapter

For non-standard APIs with custom request/response formats.

```python
from pyrlm_runtime.adapters import GenericChatAdapter

adapter = GenericChatAdapter(
    base_url="https://custom-api.com",
    path="/chat/completions",
    model="custom-model",
    api_key="your-key",
    payload_builder=my_custom_builder,    # Custom request format
    response_parser=my_custom_parser,     # Custom response format
    timeout=60.0,
    max_retries=3,
)
```

Auto-retries on 429, 500, 502, 503, 504 with exponential backoff. Supports context manager (`with GenericChatAdapter(...) as adapter:`).

#### VertexAIAdapter

Google Cloud Vertex AI (Gemini). Requires `google-cloud-aiplatform` / `vertexai`
and GCP credentials (ADC or a service account).

```python
from pyrlm_runtime.adapters import VertexAIAdapter

adapter = VertexAIAdapter(
    project_id="my-gcp-project",
    location="us-central1",
    model="gemini-2.5-pro",
    api_transport="rest",   # default; use "grpc" to opt back into gRPC
)
```

`api_transport` defaults to **`"rest"`**: REST honors `HTTPS_PROXY` and the
system CA bundle (`REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE`) — required behind
corporate proxies with a self-signed TLS certificate — and is immune to the
gRPC pollset deadlock that long-running loops hit. Pass `api_transport="grpc"`
to restore the previous gRPC transport. The transport is configured via
`vertexai.init`, which is **process-global SDK state**: do not mix transports
across multiple adapters in the same process — the last `init` wins.

The adapter normalizes Gemini's finish reasons to the loop's vocabulary
(`MAX_TOKENS` → `"length"`, `STOP` → `"stop"`) in `ModelResponse.meta`, skips
thinking parts when extracting the answer text, and folds `thoughts_token_count`
into completion-token usage for Gemini 2.5 thinking models.

#### FakeAdapter

Deterministic adapter for testing. No external API needed.

```python
from pyrlm_runtime.adapters import FakeAdapter

adapter = FakeAdapter(
    script=["code step 1", "code step 2", "FINAL_VAR: result"]
)
# Pattern-based rules for subcall responses
adapter.add_rule(pattern="Summarize", response="This is a summary")
adapter.add_rule(pattern=r"find.*key", response="key_term", regex=True)
```

#### Custom adapters

Implement the `ModelAdapter` protocol:

```python
from pyrlm_runtime.adapters import ModelAdapter, ModelResponse

class MyAdapter:
    def complete(
        self,
        messages: list[dict[str, str]],
        *,                                  # max_tokens / temperature are keyword-only
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ModelResponse:
        # Call your LLM and return a ModelResponse
        ...
```

### Policy

Controls resource limits to prevent runaway execution.

```python
from pyrlm_runtime import Policy

policy = Policy(
    max_steps=40,              # Max RLM loop iterations
    max_subcalls=200,          # Max total subcalls
    max_recursion_depth=1,     # Max subcall nesting depth
    max_total_tokens=None,     # Token budget (root + subcalls); None = unlimited (default)
    max_subcall_tokens=None,   # Token budget for subcalls only; None = unlimited
)

rlm = RLM(adapter=adapter, policy=policy)
```

By default there is **no token budget** (`max_total_tokens=None`): a run is bounded by
`max_steps` / `max_subcalls` and terminates with a **graceful finalization** (the model is asked
for a best final answer from what it has gathered). Set `max_total_tokens` to an integer only if
you want a hard token ceiling — when it is hit the run also finalizes gracefully, just earlier
than `max_steps` would, so the answer reflects less gathered context. Prefer `max_steps` for
control.

Raises specific exceptions when the corresponding limits are set and exceeded: `MaxStepsExceeded`, `MaxSubcallsExceeded`, `MaxRecursionExceeded`, `MaxTokensExceeded` (the last only when `max_total_tokens` / `max_subcall_tokens` is set).

### Trace

Records every step of the RLM execution for debugging and analysis.

```python
output, trace = rlm.run(query, context)

# Inspect steps
for step in trace.steps:
    print(f"Step {step.step_id}: {step.kind}")
    if step.code:
        print(f"  Code: {step.code[:100]}")
    if step.stdout:
        print(f"  Output: {step.stdout[:100]}")
    if step.error:
        print(f"  Error: {step.error}")

# Serialize
json_str = trace.to_json()
trace_restored = Trace.from_json(json_str)
```

Step kinds: `root_call`, `repl_exec`, `subcall`, `recursive_subcall`, `sub_root_call`, `sub_repl_exec`, `sub_subcall`.

### Cache

File-based cache for subcall results. Avoids repeating identical LLM calls.

```python
from pyrlm_runtime import FileCache

cache = FileCache(root="./cache")
rlm = RLM(adapter=adapter, cache=cache)
```

The cache key includes the **effective subcall model** (the adapter that serves
the call, e.g. a cheaper `subcall_adapter`), so entries from different models
never collide in a shared cache directory. Writes are **atomic** and reads
tolerate a corrupt/half-written entry by degrading to a miss, so the cache is
safe to share with `parallel_subcalls=True`. (Changing the subcall model
invalidates prior entries by design — they simply become misses.)

> **Disclaimer:** the model identity is resolved from the adapter's `model_id` /
> `model_name` / `model` attribute (built-in adapters — `OpenAICompatAdapter`,
> `AzureOpenAIAdapter`, `VertexAIAdapter`, `GenericChatAdapter`, `FakeAdapter` —
> all expose one). Identity is by **model id only**, not by endpoint or adapter
> instance: two adapters pointing at the same model id but different `base_url` /
> region will share cache entries. Use a separate cache `root` per endpoint if
> that matters. Custom adapters exposing none of those attributes fall back to
> their class name.

### Router

Automatically selects between baseline (direct LLM call) and RLM based on context size.

```python
from pyrlm_runtime import SmartRouter, RouterConfig, ExecutionProfile

router = SmartRouter(
    adapter,
    config=RouterConfig(baseline_threshold=8000),  # chars
)

result = router.run(query, context, profile=ExecutionProfile.DETERMINISTIC_FIRST)
print(f"Method: {result.method}")   # "baseline" or "rlm"
print(f"Answer: {result.output}")
print(f"Tokens: {result.tokens_used}")
```

**Execution profiles:**

| Profile               | Strategy                                          |
| --------------------- | ------------------------------------------------- |
| `DETERMINISTIC_FIRST` | Try regex/`extract_after` first, minimal subcalls |
| `SEMANTIC_BATCHES`    | Parallel subcalls for classification tasks        |
| `HYBRID`              | Deterministic first, fall back to semantic        |
| `VERIFY`              | Double-check with recursive subcalls              |

### Reranking

pyrlm-runtime ships two rerankers that take a pool of retrieved documents and return a
reordered list prioritised for a given query. Both accept any `ModelAdapter`.

#### ListwiseReranker (sliding window)

Walks the candidate list bottom→top in overlapping windows, asking the LLM to permute
each window. Best for pools up to ~200 documents.

```python
from pyrlm_runtime import ListwiseReranker

reranker = ListwiseReranker(
    adapter,
    window_size=20,           # documents per LLM call
    step=10,                  # overlap between windows
    max_passage_chars=300,    # truncate each passage to this length
    cache=None,               # optional FileCache to skip repeated calls
)

results = reranker.rerank(query, candidates, top_k=10)
# candidates: list of dicts with at least {"doc_id": ..., "content": ...}
# returns: top_k dicts in reranked order
```

**Telemetry:** `reranker.llm_calls`, `reranker.cache_hits`

#### TournamentReranker

Shuffles the pool into batches, keeps the top-K survivors from each batch, and repeats
until a single batch remains. Designed for large pools (300–2,500 documents) where the
sliding window becomes expensive.

```python
from pyrlm_runtime import TournamentReranker

reranker = TournamentReranker(
    adapter,
    batch_size=20,            # documents per LLM call
    top_k_per_batch=4,        # survivors per batch
    shuffle_seed=42,          # reproducible shuffling
    max_passage_chars=300,
    cache=None,
)

results = reranker.rerank(query, candidates, top_k=10)
```

> **When to use which?**
> At pool sizes ≤ ~200 docs, `ListwiseReranker` wins because it preserves the BM25
> ordering and never permanently eliminates a document.
> `TournamentReranker` is the better choice at 300–2,500 docs where the sliding window
> becomes expensive and the initial ordering is less reliable.

#### Evaluation metrics

```python
from pyrlm_runtime import ndcg_at_k, recall_at_k

ndcg = ndcg_at_k(ranked_ids, qrels, k=10)   # qrels: {doc_id: relevance_score}
rec  = recall_at_k(ranked_ids, qrels, k=10)
```

---

### Multi-Query Retrieval

For **oblique queries** — where the relevant documents don't share surface vocabulary
with the query — a single BM25 pass misses most of the relevant corpus. The
multi-query pattern expands coverage by reformulating the query N times with diverse
vocabulary before retrieval, then merging and reranking the union.

```text
query → LLM rewriter (1 call) → N reformulations + original
                                       ↓
                              BM25 × (N+1) searches
                                       ↓
                              union_pool (deduplicated)
                                       ↓
                              ListwiseReranker (on ORIGINAL query)
                                       ↓
                                    top-10
```

#### QueryRewriter

Generates N vocabulary-diverse reformulations via a single LLM call.
The system prompt is caller-supplied so the class stays domain-agnostic.

```python
from pyrlm_runtime import QueryRewriter

REWRITE_PROMPT = """
You are a search-query reformulation expert. Given a query, produce exactly {n}
reformulations that attack the same underlying concept from different vocabulary angles.
Return JSON: {{"rewrites": ["...", ...]}}
""".format(n=5)

rewriter = QueryRewriter(
    adapter,
    n=5,
    system_prompt=REWRITE_PROMPT,
    max_tokens=400,
    cache=None,               # optional FileCache
)

rewrites = rewriter.rewrite("find proofs using induction on binary trees")
# → ["structural induction over recursive data", "tree depth recursion argument", ...]
```

#### union_pool

Merges multiple retrieval result lists into one deduplicated list. First occurrence
of each `doc_id` wins, preserving the highest-ranked result for each document.

```python
from pyrlm_runtime import union_pool

pool_a = bm25.search(query, top_n=25)
pool_b = bm25.search(rewrite_1, top_n=25)
pool_c = bm25.search(rewrite_2, top_n=25)

union = union_pool([pool_a, pool_b, pool_c])
# → deduplicated list, ~60 unique documents, first-seen order
```

#### Full pipeline example

```python
from pyrlm_runtime import QueryRewriter, union_pool, ListwiseReranker

rewriter = QueryRewriter(adapter, n=5, system_prompt=MY_PROMPT)
reranker = ListwiseReranker(adapter)

# Fan-out: reformulations + original query as anchor
searches = rewriter.rewrite(query) + [query]
pools = [bm25.search(q, top_n=25) for q in searches]
union = union_pool(pools)            # ~125 unique docs
top_10 = reranker.rerank(query, union, top_k=10)
```

> **Why include the original query?** The reformulations expand coverage into
> vocabulary-distant corners of the corpus. The original query guarantees you don't
> lose documents that BM25 already found — a critical anchor against regressions.

#### Measured results ([OBLIQ-Bench](https://arxiv.org/html/2605.06235) Math, N=151)

| System                                                           |   NDCG@10 |  vs BM25 |
| ---------------------------------------------------------------- | --------: | -------: |
| BM25 baseline                                                    |     0.028 |       1× |
| BM25 + `ListwiseReranker`                                        |     0.057 |     2.0× |
| `QueryRewriter` (5 rewrites) + `ListwiseReranker`                |     0.072 |     2.6× |
| `QueryRewriter` (5 rewrites + original) + `ListwiseReranker`     |     0.093 |     3.3× |
| **`QueryRewriter` (10 rewrites + original) + `ListwiseReranker`** | **0.103** | **3.7×** |

No index changes. No fine-tuning. Purely read-path composition.
See [`docs/obliq-bench/OBLIQ-PALANCA1-MULTIQUERY.md`](docs/obliq-bench/OBLIQ-PALANCA1-MULTIQUERY.md) for full
experimental details and [`examples/oblique_multiquery_bench.py`](examples/oblique_multiquery_bench.py)
to reproduce.

---

## REPL Backends

pyrlm-runtime ships with two interchangeable REPL backends:

### PythonREPL (default)

Uses `exec()` with a whitelist sandbox. Allowed modules: `re`, `math`, `json`, `textwrap`. Stdout capped at 4000 chars.

```python
rlm = RLM(adapter=adapter, repl_backend="python")
```

### MontyREPL (secure sandbox)

Uses [pydantic-monty](https://github.com/pydantic/pydantic-monty), a Rust-based Python interpreter with compile-time safety. Enforces resource limits: 5s duration, 128MB memory, 1M allocations.

```python
# Requires: pip install pydantic-monty
rlm = RLM(adapter=adapter, repl_backend="monty")
```

**How MontyREPL handles complex objects:** Python objects like `Context` can't run natively in the Rust sandbox. MontyREPL uses an **object proxy** system — methods are registered as external functions with `{name}__{method}` naming, and AST rewrites transform `ctx.method()` calls into `ctx__method()` calls transparently.

**Variable persistence:** MontyREPL uses AST-based detection of assignments, appending a capture dict to extract variable state from each execution.

Both backends implement the same `REPLProtocol` interface: `exec(code) -> ExecResult`, `get(name)`, `set(name, value)`.

## REPL Functions Available to the LLM

When the LLM generates code during the RLM loop, these functions are available in the REPL:

### Context inspection

```python
P                              # The full context text (str)
ctx                            # The Context object

peek(n=2000)                   # First n chars of context
tail(n=2000)                   # Last n chars of context
lenP()                         # Total character count

ctx.slice(start, end)          # Safe substring
ctx.find(pattern, regex=False) # Search (returns list of matches)
ctx.chunk(size, overlap=0)     # Split into char-based chunks
ctx.chunk_documents(docs_per_chunk=10)  # Group documents
ctx.num_documents()            # Document count
ctx.get_document(index)        # Get specific document
ctx.document_lengths()         # List of doc lengths
```

### Subcalls (call sub-LLMs)

```python
llm_query(text, model=None, max_tokens=None)
    # Single subcall to a sub-LLM
    # max_tokens defaults to subcall_max_output_tokens (1024) at runtime

llm_batch(prompts, model=None, max_tokens=None)
    # Process multiple prompts in parallel (always parallel, uses ThreadPoolExecutor)
    # max_tokens defaults to subcall_max_output_tokens (1024) at runtime
    # → Use this for independent batch operations
    # Example: llm_batch(["prompt1", "prompt2", "prompt3"])

llm_query_batch(chunks, model=None, max_tokens=None, parallel=None)
    # Batch subcall over multiple chunks
    # max_tokens defaults to subcall_max_output_tokens (1024) at runtime
    # → Parallel if parallel_subcalls=True or parallel=True (default: sequential)

ask(question, text, max_tokens=None)
    # Convenience: ask a question about a text snippet

ask_chunks(question, chunks, max_tokens=None, parallel=None)
    # Ask the same question over multiple chunks
    # → Parallel if parallel_subcalls=True or parallel=True (default: sequential)

ask_chunks_first(question, chunks, ...)
    # Return first valid (non-empty) answer from chunks (always sequential)

pick_first_answer(answers)
    # Filter and return first non-empty answer from a list
```

**Parallelization note:**

- `llm_batch()` always runs in parallel via ThreadPoolExecutor
- `ask_chunks()` and `llm_query_batch()` run:
  - **Sequential by default** (unless `RLM(parallel_subcalls=True)` or `ask_chunks(..., parallel=True)`)
  - **Parallel when enabled** (limited to `max_concurrent_subcalls`, default 10 workers)

### Retrieval (when retriever is configured)

```python
es_search(query, top_k=10, filters=None)
    # BM25 full-text search → list of {doc_id, preview, score, metadata}

es_vector_search(query, top_k=10, filters=None)
    # Semantic similarity search → list of {doc_id, preview, score, metadata}

es_hybrid_search(query, top_k=10, filters=None)
    # Combined BM25 + semantic (recommended) → list of {doc_id, preview, score, metadata}

es_get(doc_id)
    # Fetch full document → {doc_id, content, metadata}
```

### Deterministic extraction

```python
extract_after(marker, max_len=128)
    # Extract text after a marker without using a subcall (fast, 0 tokens)
```

## Retrieval Integration

For large corpora that don't fit in memory, the RLM can search external document indexes directly from the REPL loop. See the detailed architecture guide: **[docs/RETRIEVAL.md](docs/RETRIEVAL.md)**

### Quick Setup

First, install the optional Elasticsearch extra:

```bash
pip install "pyrlm-runtime[elasticsearch]"
```

```python
from pyrlm_runtime import RLM
from pyrlm_runtime.adapters import OpenAICompatAdapter
from pyrlm_runtime.retrieval import ElasticsearchRetriever

retriever = ElasticsearchRetriever(
    host="https://my-cluster.es.cloud.com",
    api_key="xxx",
    index="pdf_corpus",
    embedding_model="text-embedding-3-small",
)

rlm = RLM(adapter=OpenAICompatAdapter(model="gpt-5"), retriever=retriever)
answer, trace = rlm.run("Who signed document X?")  # No context needed
```

When a retriever is configured, four functions become available in the REPL:

```python
es_search(query, top_k=10, filters=None)        # BM25 keyword search
es_vector_search(query, top_k=10, filters=None)  # Semantic similarity
es_hybrid_search(query, top_k=10, filters=None)  # Combined (recommended)
es_get(doc_id)                                    # Fetch full document
```

The retrieval layer is **backend-agnostic**: any object implementing the `RetrieverProtocol` (with `search`, `vector_search`, `hybrid_search`, `get` methods) works as a drop-in backend.

## Extending the REPL

pyrlm-runtime is deliberately **generic** — domain logic lives in consumers, not in the
library. The supported way to add domain-specific behaviour is through three plugin
hooks on `RLM`, so you never have to fork the runtime.

### `repl_extensions` — inject custom REPL functions

A callable that receives the live loop state and returns a `{name: callable}` dict.
Each entry is registered into the REPL environment, so the LLM can call it like any
built-in function. This is the primary extension point for downstream projects
(e.g. `banking-rlm`).

```python
def my_extensions(rlm, repl, retriever, log_diag):
    """Called once per run, before the loop starts.

    rlm       -- the RLM instance (read its config, adapter, policy...)
    repl      -- the live REPL (use repl.get/set to share state)
    retriever -- the configured retriever, or None
    log_diag  -- callable for diagnostic logging
    Returns: dict of {function_name: callable} to expose in the REPL.
    """
    def normalize_amount(s: str) -> float:
        return float(s.replace(".", "").replace(",", "."))

    return {"normalize_amount": normalize_amount}

rlm = RLM(adapter=adapter, repl_extensions=my_extensions)
# Inside the REPL the LLM can now call normalize_amount("1.234,56")
```

### `doc_tools` — on-demand PDF / document access

A `dict` of `{name: callable}` injected into the REPL scaffold, following the same
plugin pattern as `repl_extensions`. Use it to give the RLM tools that read documents
on demand (e.g. the six tools in `pyrlm_runtime.doctools`: `list_pdfs`,
`get_pdf_info`, `read_pdf_pages`, `extract_table`, `search_in_pdf`, `search_corpus`).

```python
rlm = RLM(adapter=adapter, doc_tools={
    "read_pdf_pages": read_pdf_pages,
    "search_in_pdf": search_in_pdf,
})
```

### `system_prompt_supplement` — extra system-prompt text

Free-form text appended to the system prompt after the retrieval docs. Use it to
document the custom functions you injected via `repl_extensions`/`doc_tools` so the
model knows they exist and how to call them.

```python
rlm = RLM(
    adapter=adapter,
    repl_extensions=my_extensions,
    system_prompt_supplement=(
        "You also have normalize_amount(s) -> float for Spanish-formatted numbers."
    ),
)
```

All three hooks are **additive and optional** — the runtime stays domain-free, and the
injected names coexist with the built-in REPL functions and retrieval functions.

## Parallel Subcalls

See the detailed architecture guide: **[docs/PARALLEL_SUBCALLS.md](docs/PARALLEL_SUBCALLS.md)**

### Quick Summary

pyrlm-runtime supports three ways to parallelize LLM subcalls:

1. **`llm_batch(prompts)`** — Always parallel, best for independent prompts:

   ```python
   results = llm_batch(["Q1?", "Q2?", "Q3?"])  # All 3 run in parallel
   ```

2. **`ask_chunks(..., parallel=True)`** — Opt-in per-call:

   ```python
   answers = ask_chunks("Q?", chunks, parallel=True)  # Chunks processed in parallel
   ```

3. **`RLM(..., parallel_subcalls=True)`** — Global flag:
   ```python
   rlm = RLM(adapter, parallel_subcalls=True)  # All ask_chunks calls are parallel
   ```

**Why parallel?** LLM API calls are I/O-bound. Making 10 requests sequentially takes ~20s; in parallel, ~2s.

**Thread safety:** All parallel execution is protected by locks on `Policy`, `Trace`, and step ID counters.

**Limits:** Default 10 concurrent workers (`max_concurrent_subcalls`); adjust per your API's rate limits.

## Multi-Turn Conversation History

By default (`conversation_history=True`), the LLM sees its previous code attempts and REPL outputs across iterations. This enables self-correction.

```python
rlm = RLM(
    adapter=adapter,
    conversation_history=True,      # Default
)
```

**How it works:**

1. The initial message contains full query + context metadata
2. Each iteration appends a lightweight message with REPL results

### Keeping history within the context window

Most runs need none of this: in an RLM the large context lives in the REPL (the
model inspects it with code), not in the prompt, so the conversation history is
just code plus truncated REPL output and rarely approaches the context window.
Both mechanisms below default to **off** — turn one on only for unusually long
trajectories. When you do need to manage history, **compaction is preferred over
`max_history_tokens`**:

| Mechanism | What it does | Cost |
| --- | --- | --- |
| `compaction=True` | Summarizes old turns into a running summary; keeps a recoverable `history` REPL variable | One extra LLM call per compaction; preserves the gist |
| `max_history_tokens=N` (**deprecated**) | Blunt trim: drops the oldest middle turns outright | Free, no extra call; **discards information** |

Compaction triggers when the estimated history size crosses a threshold. Set the
threshold as a fraction of the model's context window with `compaction_threshold_pct`
(e.g. `0.85`) — the window is auto-resolved from the adapter's model id (or set
`compaction_model_name` / `compaction_model_context_limit` explicitly). Token counting
uses tiktoken when available, falling back to a `len // 4` estimate.

```python
rlm = RLM(
    adapter=adapter,
    compaction=True,
    compaction_threshold_pct=0.85,  # compact at 85% of the model's context window
)
```

Alternatively, set `compaction_threshold_tokens` for an absolute trigger. `max_history_tokens`
still works as a cheap, no-extra-LLM-call fallback but emits a `DeprecationWarning`; prefer
compaction, which summarizes rather than discards.

## Guard Mechanisms & Fallbacks

For robustness, RLM supports several guard mechanisms:

```python
rlm = RLM(
    adapter=adapter,

    # Require at least 1 REPL execution before accepting FINAL
    require_repl_before_final=True,

    # Require at least 1 subcall before accepting FINAL
    require_subcall_before_final=True,

    # Max non-code responses before giving up
    invalid_response_limit=5,

    # Emergency code to run if LLM stalls
    fallback_code="answer = pick_first_answer(ask_chunks('answer?', ctx))",
)
```

## Configuration

### Environment variables

```bash
# API key (checked in order)
LLM_API_KEY="your-key"        # Primary
OPENAI_API_KEY="your-key"     # Fallback

# Azure OpenAI
AZURE_OPENAI_API_KEY="your-key"
OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
# or: AZURE_ACCOUNT_NAME="your-resource"
AZURE_OPENAI_API_VERSION="2024-10-21"  # optional

# Custom endpoint (optional)
LLM_BASE_URL="https://..."

# For local models (no auth needed)
LLM_BASE_URL="http://localhost:11434/v1"  # Ollama
```

### Common configurations by use case

| Use case                       | Configuration                                                            |
| ------------------------------ | ------------------------------------------------------------------------ |
| Small context (<8K chars)      | Use `SmartRouter` — it will pick baseline automatically                  |
| Large corpus (10K+ docs)       | `RLM(adapter, retriever=ElasticsearchRetriever(...))` — search on demand |
| Large context (>100K chars)    | `RLM(adapter, conversation_history=True, parallel_subcalls=True)`        |
| Batch many independent prompts | Use `llm_batch(prompts)` — always parallel, no config needed             |
| Cost-sensitive                 | Use a cheaper `subcall_adapter` for subcalls                             |
| Safety-critical code execution | `repl_backend="monty"`                                                   |
| Deterministic extraction       | `SmartRouter` with `DETERMINISTIC_FIRST` profile                         |
| Complex multi-hop reasoning    | `recursive_subcalls=True, max_recursion_depth=2`                         |

### Supported providers

| Provider      | Setup                                                                       |
| ------------- | --------------------------------------------------------------------------- |
| **Azure**     | `AzureOpenAIAdapter(model="gpt-5.1")` + `AZURE_OPENAI_API_KEY` + endpoint   |
| **OpenAI**    | `OpenAICompatAdapter(model="gpt-5.1")` + `LLM_API_KEY`                      |
| **Anthropic** | Via OpenAI-compatible proxy                                                 |
| **Ollama**    | `OpenAICompatAdapter(model="llama3", base_url="http://localhost:11434/v1")` |
| **LM Studio** | `OpenAICompatAdapter(model="...", base_url="http://localhost:1234/v1")`     |
| **vLLM**      | `OpenAICompatAdapter(model="...", base_url="http://localhost:8000/v1")`     |
| **Custom**    | `GenericChatAdapter(...)` or implement `ModelAdapter`                       |

## Examples

| Example                                                                   | Description                                                   | Requires API? |
| ------------------------------------------------------------------------- | ------------------------------------------------------------- | ------------- |
| [`minimal.py`](examples/minimal.py)                                       | Basic RLM flow with FakeAdapter                               | No            |
| [`rlm_vs_baseline.py`](examples/rlm_vs_baseline.py)                       | Needle-in-haystack benchmark (MIT paper Figure 1)             | Yes           |
| [`smart_router_demo.py`](examples/smart_router_demo.py)                   | SmartRouter auto-selecting baseline vs RLM by context size    | Yes           |
| [`bench_repl_python_vs_monty.py`](examples/bench_repl_python_vs_monty.py) | Raw REPL performance: PythonREPL vs MontyREPL (no LLM calls)  | No            |
| [`bench_rlm_repl_backends.py`](examples/bench_rlm_repl_backends.py)       | Full RLM loop benchmark with both REPL backends (FakeAdapter) | No            |

Run any example:

```bash
uv run python examples/minimal.py
```

## When to Use RLMs

**Use RLM when:**

- Context size exceeds the model's window (>100K tokens)
- Information is scattered across the entire context
- The task requires examining most or all of the input
- Accuracy matters more than latency
- Context doesn't fit the RAG chunk paradigm

**Don't use RLM when:**

- Context always fits in the model window (<50K tokens)
- Simple keyword search would work
- Information is localized (RAG is faster)
- Real-time response is required (milliseconds)

## Benchmark: RLM vs Baseline

The [`rlm_vs_baseline.py`](examples/rlm_vs_baseline.py) example reproduces the key finding from the MIT paper (Figure 1): RLMs maintain accuracy as context grows, while baseline approaches degrade due to truncation.

![Figure 1 from MIT Paper](docs/figure1-mit-rlm.png)

_Figure 1: RLM accuracy remains high as distractor documents increase, while baseline accuracy drops._

### Running the benchmark

```bash
# Quick demo
RLM_CONTEXT_SIZES=5,30 uv run python examples/rlm_vs_baseline.py

# Full benchmark
RLM_CONTEXT_SIZES=5,20,50,120 uv run python examples/rlm_vs_baseline.py

# With detailed execution trajectory
SHOW_TRAJECTORY=1 RLM_CONTEXT_SIZES=5,30 uv run python examples/rlm_vs_baseline.py
```

### The crossover point

Around ~50 documents (~100K+ characters), the context exceeds the LLM's window and baseline accuracy drops to 0%. RLM maintains near-perfect accuracy by inspecting the context via code instead of sending it all as input.

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## References

- [MIT CSAIL Paper: Recursive Language Models](https://arxiv.org/abs/2512.24601) — Zhou, et al.
- This implementation is not affiliated with MIT.

## License

MIT License — see [LICENSE](LICENSE) for details.
