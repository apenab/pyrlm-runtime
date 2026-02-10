---
name: rlm-debugging
description: Debug and diagnose RLM execution issues using the Trace system, REPL inspection, and policy analysis. Use when an RLM run produces incorrect results, hits policy limits, loops unexpectedly, or needs performance optimization.
license: MIT
metadata:
  author: apenab
  version: "1.0"
---

# RLM Debugging Guide

## Overview

RLM executions can fail or behave unexpectedly in several ways: wrong answers, infinite loops, policy limit exceptions, REPL errors, or excessive token usage. This skill helps you diagnose these issues using the built-in Trace system and execution analysis.

## The Trace System

Every `RLM.run()` call returns a `(result, trace)` tuple. The `Trace` object records every step of execution.

### TraceStep Fields

| Field | Type | Purpose |
|---|---|---|
| `step_id` | int | Sequential step number |
| `kind` | str | Step type (see below) |
| `depth` | int | Recursion depth (0 = root) |
| `prompt_summary` | str | Truncated prompt sent to LLM |
| `code` | str | Python code executed or LLM response |
| `stdout` | str | REPL output |
| `error` | str | REPL error or None |
| `usage` | Usage | Token counts (prompt, completion, total) |
| `cache_hit` | bool | Whether subcall was served from cache |
| `input_hash` | str | SHA256 of subcall input |
| `output_hash` | str | SHA256 of subcall output |
| `cache_key` | str | Full cache key for subcall |

### Step Kinds

| Kind | Meaning |
|---|---|
| `root_call` | Root LLM call (the controller) |
| `repl_exec` | Python code executed in REPL |
| `subcall` | Standard sub-LLM call |
| `recursive_subcall` | Subcall that ran its own mini-RLM loop |
| `sub_root_call` | Root call inside a recursive subcall |
| `sub_repl_exec` | REPL exec inside a recursive subcall |
| `sub_subcall` | Nested subcall inside a recursive subcall |

## Debugging Workflow

### Step 1: Capture the Trace

```python
from pyrlm_runtime import RLM, Context, Policy

rlm = RLM(adapter=my_adapter, policy=Policy(max_steps=20))
result, trace = rlm.run("your query", Context.from_text(your_text))
```

### Step 2: Inspect the Trace

```python
# Print all steps
for step in trace.steps:
    print(f"[{step.step_id}] {step.kind} depth={step.depth}")
    if step.code:
        print(f"  code: {step.code[:200]}")
    if step.stdout:
        print(f"  stdout: {step.stdout[:200]}")
    if step.error:
        print(f"  ERROR: {step.error}")
    if step.usage:
        print(f"  tokens: {step.usage.total_tokens}")
    if step.cache_hit:
        print(f"  (cache hit)")
```

### Step 3: Serialize for Later Analysis

```python
# Save trace to JSON
import json
with open("trace.json", "w") as f:
    f.write(trace.to_json())

# Load trace from JSON
with open("trace.json") as f:
    loaded_trace = Trace.from_json(f.read())
```

## Common Issues and Diagnosis

### 1. MaxStepsExceeded

**Symptom**: `MaxStepsExceeded` exception raised.

**Diagnosis**: The LLM is not converging to a `FINAL` answer.

**Check in trace**:
- Look at `root_call` steps: Is the LLM producing valid code?
- Look for `repl_exec` steps with errors: Is REPL code failing repeatedly?
- Check if the LLM keeps generating the same code (looping)

**Common causes**:
- LLM doesn't understand the `FINAL:` / `FINAL_VAR:` syntax -> check system prompt
- REPL errors prevent the LLM from getting useful output -> check error messages
- Context is too complex for the model -> try a more capable model or simplify the query
- `require_repl_before_final=True` but LLM keeps trying to answer directly

**Fix**: Increase `Policy(max_steps=...)`, improve the system prompt, or use a more capable model.

### 2. MaxSubcallsExceeded

**Symptom**: `MaxSubcallsExceeded` exception during subcall execution.

**Diagnosis**: Too many sub-LLM calls being made.

**Check in trace**:
- Count `subcall` and `recursive_subcall` steps
- Check chunk sizes: are chunks too small, creating too many subcalls?
- Look for `subcall_batch` calls with large chunk lists

**Common causes**:
- LLM is chunking too aggressively (e.g., 100-char chunks on a 1M-char context)
- LLM is calling `llm_query` in a loop instead of using `ask_chunks` batch
- Recursive subcalls creating nested subcalls

**Fix**: Increase `Policy(max_subcalls=...)`, adjust prompt to encourage larger chunks, or use `parallel_subcalls=True` for efficiency.

### 3. MaxTokensExceeded

**Symptom**: `MaxTokensExceeded` exception.

**Diagnosis**: Total token budget exhausted.

**Check in trace**:
```python
total = sum(s.usage.total_tokens for s in trace.steps if s.usage)
print(f"Total tokens used: {total}")

# Breakdown by kind
from collections import Counter
by_kind = Counter()
for s in trace.steps:
    if s.usage:
        by_kind[s.kind] += s.usage.total_tokens
print(by_kind)
```

**Common causes**:
- Too many subcalls (each costs tokens)
- Root LLM calls with very large prompts
- Recursive subcalls multiplying token usage

**Fix**: Increase `Policy(max_total_tokens=...)`, reduce subcall count, use caching.

### 4. Wrong Answer / NO_ANSWER

**Symptom**: RLM returns incorrect result or "NO_ANSWER".

**Diagnosis steps**:
1. **Check root_call responses**: Is the LLM writing good inspection code?
2. **Check repl_exec stdout**: Is the REPL returning useful data?
3. **Check subcall results**: Are sub-LLMs extracting correct information?
4. **Check the final step**: How was the answer resolved (FINAL vs FINAL_VAR)?

**Trace analysis**:
```python
# Find the final step
final_steps = [s for s in trace.steps if s.kind == "root_call"]
last_root = final_steps[-1]
print(f"Final LLM output: {last_root.code}")

# Check all subcall results
subcalls = [s for s in trace.steps if s.kind == "subcall"]
for s in subcalls:
    print(f"  input: {s.prompt_summary}")
    print(f"  output_hash: {s.output_hash}")
```

**Common causes**:
- The answer isn't in the context (verify with manual search)
- Sub-LLM prompt is too vague -> check `SUBCALL_SYSTEM_PROMPT`
- Chunking missed the relevant section -> check chunk overlap
- `extract_after` marker not found -> check deterministic extraction

### 5. REPL Errors

**Symptom**: `repl_exec` steps show errors repeatedly.

**Check in trace**:
```python
errors = [s for s in trace.steps if s.kind == "repl_exec" and s.error]
for s in errors:
    print(f"Step {s.step_id}: {s.error}")
    print(f"  Code: {s.code}")
```

**Common REPL errors**:
- `ImportError: import of 'X' is not allowed` -> LLM tried to import a blocked module. Only `re`, `math`, `json`, `textwrap` are allowed.
- `NameError` -> LLM referenced a function that isn't registered in the REPL
- `IndexError` / `KeyError` -> Bug in LLM-generated code, usually fixes on next iteration

### 6. Cache Issues

**Symptom**: Unexpected results when rerunning, or no caching benefit.

**Check**:
```python
cache_hits = [s for s in trace.steps if s.cache_hit]
cache_misses = [s for s in trace.steps if s.kind == "subcall" and not s.cache_hit]
print(f"Cache hits: {len(cache_hits)}, misses: {len(cache_misses)}")
```

**Fix**: Cache is stored in `.rlm_cache/`. Delete the directory to reset. Cache keys include model name, max_tokens, and recursive flag, so changing any of these causes cache misses.

## Performance Analysis

### Token Usage Summary

```python
def analyze_trace(trace):
    total_tokens = 0
    root_tokens = 0
    subcall_tokens = 0
    cache_hits = 0

    for step in trace.steps:
        if step.usage:
            total_tokens += step.usage.total_tokens
            if step.kind == "root_call":
                root_tokens += step.usage.total_tokens
            elif step.kind in ("subcall", "recursive_subcall"):
                subcall_tokens += step.usage.total_tokens
        if step.cache_hit:
            cache_hits += 1

    print(f"Total tokens: {total_tokens}")
    print(f"  Root LLM: {root_tokens}")
    print(f"  Subcalls: {subcall_tokens}")
    print(f"Root steps: {sum(1 for s in trace.steps if s.kind == 'root_call')}")
    print(f"Subcalls: {sum(1 for s in trace.steps if s.kind in ('subcall', 'recursive_subcall'))}")
    print(f"Cache hits: {cache_hits}")
    print(f"REPL errors: {sum(1 for s in trace.steps if s.error)}")
```

### Identifying Bottlenecks

1. **Too many root steps**: The LLM is not converging. Check if it's getting useful REPL feedback.
2. **Too many subcalls**: Chunking is too fine-grained. Increase chunk sizes.
3. **High token usage in subcalls**: Sub-LLM responses are verbose. Check `SUBCALL_SYSTEM_PROMPT`.
4. **No cache hits**: Either first run, or cache keys differ between runs (model/token params changed).

## Policy Tuning

| Scenario | Recommended Policy |
|---|---|
| Quick extraction (needle-in-haystack) | `Policy(max_steps=10, max_subcalls=20)` |
| Multi-document analysis | `Policy(max_steps=30, max_subcalls=200)` |
| Deep research (100+ docs) | `Policy(max_steps=40, max_subcalls=500, max_total_tokens=500_000)` |
| Development / testing | `Policy(max_steps=5, max_subcalls=10, max_total_tokens=50_000)` |

## Using SHOW_TRAJECTORY

Set `SHOW_TRAJECTORY=1` env var when running examples to see real-time execution flow. The `TraceFormatter` in `router.py` provides formatted trajectory output including ASCII visualization.

## Logging

Enable debug logging for detailed runtime output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("pyrlm_runtime")

rlm = RLM(adapter=my_adapter, logger=logger)
```

This shows: root_call steps, REPL exec results, subcall details, cache hits/misses, and policy state.
