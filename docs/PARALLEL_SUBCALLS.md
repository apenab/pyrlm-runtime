# Parallel Subcalls Architecture

This document explains the architecture and implementation of parallel subcall execution in pyrlm-runtime, including the new `llm_batch()` function and thread safety improvements.

## Table of Contents

- [Overview](#overview)
- [The Problem: I/O-Bound Blocking](#the-problem-io-bound-blocking)
- [Architecture Decision: ThreadPoolExecutor](#architecture-decision-threadpoolexecutor)
- [Thread Safety Fixes](#thread-safety-fixes)
- [llm_batch(): The Parallel-First Interface](#llm_batch-the-parallel-first-interface)
- [Subcall Options at a Glance](#subcall-options-at-a-glance)
- [Usage Examples](#usage-examples)
- [Performance Considerations](#performance-considerations)

## Overview

pyrlm-runtime supports **three independent ways** to make LLM subcalls from the REPL:

1. **Sequential**: Default, no configuration needed
2. **Parallel via `ask_chunks(..., parallel=True)`**: Opt-in, controlled per-call
3. **Parallel via `llm_batch()`**: Always parallel, explicit batch control

All three options delegate to the same underlying `subcall()` mechanism, ensuring that caching, policy tracking, trace recording, and error handling work consistently.

## The Problem: I/O-Bound Blocking

LLM API calls are **I/O-bound** operations: the bottleneck is network latency (often 1-5 seconds per request), not CPU. When running multiple independent subcalls sequentially:

```python
# Sequential (slow): Total ~10 seconds for 5 calls
results = []
for chunk in chunks:  # 5 chunks
    response = llm_query(f"Summarize: {chunk}")  # 2s per call
    results.append(response)
```

With parallel execution, all 5 requests can happen **concurrently**:

```python
# Parallel (fast): Total ~2 seconds for 5 calls
results = llm_batch([
    f"Summarize: {chunk}"
    for chunk in chunks
])  # All 5 requests happen in parallel
```

**The key insight**: For I/O-bound work, parallelism via **threading** (not async/await) is simpler and sufficient. Python's Global Interpreter Lock (GIL) doesn't block on I/O, so multiple threads can make network requests concurrently.

## Architecture Decision: ThreadPoolExecutor

pyrlm-runtime uses **`concurrent.futures.ThreadPoolExecutor`** for parallelism:

### Why ThreadPoolExecutor?

1. **Simplicity**: No need for `async`/`await` rewrite — the entire codebase is synchronous
2. **Sufficient**: For I/O-bound LLM calls, threading provides real concurrency (GIL releases during I/O)
3. **Controlled**: Easy to limit concurrent workers (`max_concurrent_subcalls`, default 10)
4. **Existing**: Already used in `subcall_batch` before this refactor
5. **Resilient**: Errors in one worker don't crash others; exceptions propagate cleanly

### How It Works

```python
max_workers = min(self.max_concurrent_subcalls, len(prompts))  # e.g., 10
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all tasks
    futures = {
        executor.submit(worker_fn, prompt): idx
        for idx, prompt in enumerate(prompts)
    }
    # Collect results as they complete (order preserved via mapping)
    for future in as_completed(futures):
        result = future.result()  # Propagates exceptions
        results[idx] = result
```

## Thread Safety Fixes

The existing `subcall_batch` parallel path had **race conditions**:

### Issue 1: Unprotected `Policy` Counters

```python
# UNSAFE: Two threads could both check and increment simultaneously
if self.steps >= self.max_steps:  # Thread A reads: 39
    raise MaxStepsExceeded()
self.steps += 1                     # Thread B reads: 39 (before A increments!)
```

**Fix**: Add `threading.Lock` to `Policy`. All counter mutations (`check_step`, `check_subcall`, `add_tokens`, `add_subcall_tokens`) are now atomic:

```python
class Policy:
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def check_step(self) -> None:
        with self._lock:  # Atomic operation
            if self.steps >= self.max_steps:
                raise MaxStepsExceeded()
            self.steps += 1
```

### Issue 2: Unprotected `Trace.add()`

```python
# UNSAFE: Multiple threads appending to list simultaneously
with ThreadPoolExecutor(max_workers=10) as executor:
    # 10 threads calling: trace.add(step)
    # List append is not atomic!
    executor.submit(lambda: trace.add(step))
```

**Fix**: Add lock to `Trace.add()`:

```python
class Trace:
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, step: TraceStep) -> None:
        with self._lock:
            self.steps.append(step)  # Atomic append
```

### Issue 3: Unprotected `step_id` Counter

```python
# UNSAFE: Multiple threads incrementing step_id
step_id = 0
def next_step_id() -> int:
    nonlocal step_id
    step_id += 1  # Not atomic!
    return step_id
```

**Fix**: Add lock to `next_step_id()`:

```python
_step_lock = threading.Lock()
step_id = 0

def next_step_id() -> int:
    nonlocal step_id
    with _step_lock:  # Atomic increment
        step_id += 1
        return step_id
```

**Result**: Parallel execution is now safe. Multiple threads can call `subcall()`, `policy.add_tokens()`, and `trace.add()` without data corruption.

## llm_batch(): The Parallel-First Interface

`llm_batch()` is a new REPL function that makes parallel subcalls explicit and easy:

### Signature

```python
llm_batch(
    prompts: list[str],
    *,
    model: str | None = None,
    max_tokens: int | None = None
) -> list[str]
```

### Behavior

1. **Always parallel** — no `parallel=True` flag needed
2. **Deduplicates** identical prompts (avoids redundant API calls)
3. **Preserves order** — results in same order as input
4. **Fail-fast on worker failures** — adapter/subcall exceptions propagate to the caller
5. **Respects limits** — obeys `max_concurrent_subcalls` (default 10) and policy limits

### Example

```python
# In RLM REPL:
chunks = ctx.chunk(size=2000)
questions = [f"What is the main topic in chunk {i}?" for i in range(len(chunks))]

# Parallel: All questions asked simultaneously
results = llm_batch(questions)  # ~2s for 10 requests instead of ~20s sequential

# Results are in same order as input
for i, result in enumerate(results):
    print(f"Chunk {i}: {result}")
```

### Deduplication Example

```python
# If you ask the same question multiple times:
prompts = [
    "What is the capital of France?",
    "What is the capital of France?",  # Duplicate
    "What is the capital of Germany?",
]

results = llm_batch(prompts)
# Only 2 API calls made (not 3)
# results[0] == results[1] (cached)
# results = ["Paris", "Paris", "Berlin"]
```

### Single-Prompt Optimization

```python
# Single prompt: no thread overhead
results = llm_batch(["single prompt"])
# Just calls: subcall("single prompt")
# Returns: [response]
```

### Error Handling

```python
prompts = [
    "Valid prompt 1",
    None,  # This will cause an error
    "Valid prompt 2",
]

results = llm_batch(prompts)
# results[0] = "Normal response"
# results[1] = "[ERROR] llm_batch expects a list of prompt strings."
# results[2] = "Normal response"
# Invalid prompt items are reported per-item; valid prompts still run
```

Infrastructure failures from worker subcalls are different: if the adapter or
subcall machinery raises, `llm_batch()` propagates that exception instead of
returning a string that could be mistaken for model output.

## Subcall Options at a Glance

| Function                      | Parallelism | Config                                             | Use Case                         |
| ----------------------------- | ----------- | -------------------------------------------------- | -------------------------------- |
| `llm_query(text)`             | Never       | —                                                  | Single subjective prompt         |
| `ask(question, text)`         | Never       | —                                                  | Single Q&A on snippet            |
| `llm_batch(prompts)`          | Always      | —                                                  | Batch of independent prompts     |
| `llm_query_batch(chunks)`     | Opt-in      | `parallel=True` or global `parallel_subcalls=True` | Batch chunks                     |
| `ask_chunks(q, chunks)`       | Opt-in      | `parallel=True` or global `parallel_subcalls=True` | Ask same Q across chunks         |
| `ask_chunks_first(q, chunks)` | Never       | —                                                  | Return first valid answer (fast) |

## Usage Examples

### Example 1: Parallel Classification

Classify multiple documents without sequential bottleneck:

```python
documents = ctx.chunk_documents(docs_per_chunk=1)  # One doc per chunk
doc_snippets = [chunk[2] for chunk in documents]  # Extract text

# Create prompts
prompts = [
    f"Classify this document:\\n{snippet}\\n\\nCategory: business, legal, or technical?"
    for snippet in doc_snippets
]

# Parallel classification (2s instead of 20s for 10 docs)
classifications = llm_batch(prompts)

# Aggregate results
print(f"Business: {sum(1 for c in classifications if 'business' in c.lower())}")
print(f"Legal: {sum(1 for c in classifications if 'legal' in c.lower())}")
print(f"Technical: {sum(1 for c in classifications if 'technical' in c.lower())}")
```

### Example 2: Parallel Summarization

Summarize chunks in parallel:

```python
chunks = ctx.chunk(size=3000)  # (start, end, text) tuples
chunk_texts = [chunk[2] for chunk in chunks]

# Summarization prompts
prompts = [
    f"Summarize this text in one sentence:\\n{text}"
    for text in chunk_texts
]

summaries = llm_batch(prompts)
combined = " ".join(summaries)

# Now ask a follow-up question on the combined summary
final_answer = llm_query(f"Based on these summaries, answer the user's question: {P[:200]}")
```

### Example 3: Global Parallelization (Opt-in)

For batch operations that already use `ask_chunks`, enable global parallelization:

```python
# Option A: Global flag (all ask_chunks calls are parallel)
rlm = RLM(
    adapter=adapter,
    parallel_subcalls=True,  # Enable for all ask_chunks calls
)

# In REPL code:
chunks = ctx.chunk(size=2000)
answers = ask_chunks("What is the main topic?", chunks)  # Automatically parallel
```

Or per-call:

```python
# Option B: Per-call override
chunks = ctx.chunk(size=2000)
answers = ask_chunks("What is the main topic?", chunks, parallel=True)
```

### Example 4: Extracting Key Terms in Parallel

```python
# Find all paragraphs mentioning "revenue"
paragraphs = []
for start, end, text in ctx.chunk(size=500):
    if "revenue" in text:
        paragraphs.append(text)

# Extract exact figures in parallel
prompts = [
    f"Extract the revenue figure as a number (e.g., '$5M'):\\n{p}"
    for p in paragraphs
]

figures = llm_batch(prompts)

# Parse and aggregate
total = sum(
    float(fig.replace("$", "").replace("M", "").strip())
    for fig in figures
    if "$" in fig
)

print(f"Total revenue: ${total}M")
```

## Performance Considerations

### Concurrency Limits

```python
rlm = RLM(
    adapter=adapter,
    max_concurrent_subcalls=10,  # Default: 10 workers
)

# With 10 workers:
# - 10 prompts: 1 batch (all 10 run in parallel)
# - 50 prompts: 5 batches (first 10 run, then next 10, etc.)
```

Set conservatively:

- **Low** (3-5): Safe for rate-limited APIs
- **Medium** (10): Default; good balance
- **High** (20+): Use only if your API provider allows heavy concurrency

### Token Budget Strategy

```python
from pyrlm_runtime import Policy

policy = Policy(
    max_total_tokens=100_000,  # Total budget across all calls
    max_subcall_tokens=50_000,  # Subcall-only budget (optional)
)

rlm = RLM(adapter=adapter, policy=policy)

# With parallel subcalls, you can exhaust token budget faster
# Example: 10 parallel calls × 250 tokens each = 2,500 tokens at once
```

### Choosing Between Approaches

| Scenario                                | Recommendation                     |
| --------------------------------------- | ---------------------------------- |
| 2-3 questions on chunks                 | Sequential (`ask_chunks`)          |
| 5-10 independent questions              | `llm_batch()` (always parallel)    |
| Dynamic decision (based on chunk count) | `ask_chunks(..., parallel=True)`   |
| Deterministic extraction (no LLM calls) | `extract_after()` (0 tokens)       |
| Need first valid answer quickly         | `ask_chunks_first()` (stops early) |

### Caching Impact

Parallel execution greatly benefits from caching:

```python
# First run: all 10 requests are parallel
results1 = llm_batch(prompts)  # 2s (10 in parallel)

# Second run: all 10 results are cached
results2 = llm_batch(prompts)  # 0.01s (all cache hits, no threads)
```

This is why `llm_batch()` deduplicates: identical prompts within a batch only hit the API once.

### Error Handling

```python
try:
    results = llm_batch(prompts)
    # Check for errors in results
    errors = [r for r in results if r.startswith("[ERROR]")]
    if errors:
        print(f"Partial failures (could retry): {errors}")
except Exception as e:
    # Catastrophic error (e.g., adapter raises)
    print(f"Total failure: {e}")
```

## Thread Safety Guarantees

After the fixes in this implementation:

✅ **Policy counters are thread-safe**: `check_step()`, `check_subcall()`, `add_tokens()` are atomic
✅ **Trace mutations are thread-safe**: `Trace.add()` is synchronized
✅ **Step IDs are thread-safe**: `next_step_id()` increments atomically
✅ **Adapter calls are protected**: Policy limits prevent overspending
✅ **Cache is thread-safe**: FileCache uses its own locking

**However**, user code in the REPL should avoid mutable shared state:

```python
# UNSAFE: Don't do this
shared_list = []
for prompt in prompts:
    # If called in parallel from multiple threads, race condition!
    shared_list.append(llm_query(prompt))

# SAFE: Use llm_batch instead
results = llm_batch(prompts)
```

## Future Enhancements

Possible future improvements (not in scope):

1. **Adaptive concurrency**: Automatically adjust `max_concurrent_subcalls` based on API response times
2. **Priority queues**: Weight certain subcalls higher in the execution order
3. **Request batching**: Group identical requests into single API calls (beyond simple deduplication)
4. **Async support**: Migrate to `asyncio` for even more scalable parallelism if needed
5. **Metrics**: Expose parallelism stats (actual speedup, queue depth, retries)

## References

- [MIT CSAIL Paper: Recursive Language Models](https://arxiv.org/abs/2512.24601) — Inspired the overall RLM architecture
- Python docs: [concurrent.futures.ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor)
- Python docs: [threading.Lock](https://docs.python.org/3/library/threading.html#threading.Lock)
