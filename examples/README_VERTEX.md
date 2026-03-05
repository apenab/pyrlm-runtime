# Vertex AI Benchmark for pyrlm-runtime

This directory contains the Google Cloud Vertex AI implementation of the Oolong three-way benchmark, comparing three approaches to answering questions with large contexts:

1. **Baseline**: Single-shot direct prompt to LLM
2. **RLM-minimal**: Original rlm-minimal implementation with REPL loop
3. **PyRLM-runtime**: Full pyrlm_runtime framework

## Prerequisites

### 1. Google Cloud Setup

#### Authentication

Configure Google Cloud authentication using one of these methods:

**Option A: Application Default Credentials (Recommended)**

```bash
gcloud auth application-default login
gcloud config set project go-agl-poc-radax-p01-poc
```

**Option B: Service Account Key**

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

#### Required Permissions

Your GCP user/service account needs:

- Permission: `aiplatform.endpoints.predict`
- Role: `roles/aiplatform.user`

To grant this role:

```bash
gcloud projects add-iam-policy-binding go-agl-poc-radax-p01-poc \
  --member="user:your-email@domain.com" \
  --role="roles/aiplatform.user"
```

### 2. Installation

From the project root:

```bash
cd /Users/U01AB0B5/projects/pyrlm-runtime

# Install core dependencies
uv sync

# Install benchmark-specific dependencies
uv pip install -r examples/requirements_vertex.txt
```

This will install:

- `pyrlm-runtime` (from source)
- `rlm-minimal` (from source at `/Users/U01AB0B5/projects/rlm-minimal`)
- `google-cloud-aiplatform` and `vertexai`
- `datasets` and `python-dateutil` (for Oolong datasets)

### 3. Verify Setup

Test the adapters before running the full benchmark:

```bash
# Test both adapters
python examples/test_vertex_adapters.py

# Test only pyrlm_runtime adapter
python examples/test_vertex_adapters.py --adapter pyrlm

# Test only rlm-minimal adapter
python examples/test_vertex_adapters.py --adapter rlm-minimal
```

Expected output:

```
Testing VertexAIAdapter (pyrlm_runtime)
✓ Test 1 passed
✓ Test 2 passed
✓ Test 3 passed
✅ All VertexAIAdapter tests passed!

Testing VertexAIClientForRLMMinimal (rlm-minimal adapter)
✓ Test 1a passed (string input)
✓ Test 1b passed (message list input)
✓ Test 2 passed (instrumentation)
✅ All rlm-minimal adapter tests passed!
```

## Running the Benchmark

### Quick Test (1 example)

Test with a single example to verify everything works:

```bash
uv run python examples/oolong_three_way_benchmark_vertex.py \
  --max-examples 1 \
  --log-level DEBUG \
  --save-prompts \
  --latency-breakdown
```

### Standard Run (30 examples, stratified)

```bash
uv run python examples/oolong_three_way_benchmark_vertex.py \
  --model gemini-2.5-pro \
  --project-id go-agl-poc-radax-p01-poc \
  --location us-central1 \
  --dataset synth \
  --sample-strategy stratified \
  --seed 42 \
  --max-examples 30 \
  --baseline-max-tokens 1024 \
  --agent-max-tokens 2048 \
  --agent-subcall-max-tokens 2048 \
  --rlm-max-steps 35 \
  --rlm-max-subcalls 250 \
  --minimal-max-iterations 20 \
  --log-level INFO \
  --save-prompts \
  --latency-breakdown \
  --temperature 0.0
```

### Full Benchmark (100 examples)

```bash
uv run python examples/oolong_three_way_benchmark_vertex.py \
  --dataset synth \
  --max-examples 100 \
  --log-level INFO \
  --save-prompts
```

### Real-world Dataset (DND variant)

```bash
uv run python examples/oolong_three_way_benchmark_vertex.py \
  --dataset real \
  --max-examples 20 \
  --log-level DEBUG
```

## Command-Line Arguments

| Argument                     | Default                    | Description                              |
| ---------------------------- | -------------------------- | ---------------------------------------- |
| `--project-id`               | `go-agl-poc-radax-p01-poc` | GCP project ID                           |
| `--location`                 | `us-central1`              | GCP region for Vertex AI                 |
| `--model`                    | `gemini-2.5-pro`           | Gemini model to use                      |
| `--dataset`                  | `synth`                    | Dataset: `synth` or `real`               |
| `--max-examples`             | `100`                      | Number of examples to evaluate           |
| `--sample-strategy`          | `stratified`               | Sampling: `head`, `random`, `stratified` |
| `--seed`                     | `42`                       | Random seed for reproducibility          |
| `--min-context-len`          | `1024`                     | Minimum context length filter            |
| `--max-context-len`          | `131072`                   | Maximum context length filter            |
| `--baseline-max-tokens`      | `1024`                     | Max tokens for baseline                  |
| `--agent-max-tokens`         | `2048`                     | Max tokens for agent root calls          |
| `--agent-subcall-max-tokens` | `2048`                     | Max tokens for agent subcalls            |
| `--rlm-max-steps`            | `35`                       | Max steps for pyrlm_runtime              |
| `--rlm-max-subcalls`         | `250`                      | Max subcalls for pyrlm_runtime           |
| `--minimal-max-iterations`   | `20`                       | Max iterations for rlm-minimal           |
| `--temperature`              | `0.0`                      | Sampling temperature                     |
| `--log-level`                | `INFO`                     | Logging level: DEBUG/INFO/WARNING/ERROR  |
| `--save-prompts`             | Flag                       | Save all prompts to files                |
| `--latency-breakdown`        | Flag                       | Include latency metrics per phase        |

## Output Structure

Results are saved to `examples/exports/oolong_three_way/run_<timestamp>_<dataset>_<model>/`:

```
run_20260303_143052_synth_gemini-2.5-pro/
├── run_config.json                      # Complete configuration
├── summary.json                         # Aggregate metrics by engine
├── summary_by_context_bucket.json       # Breakdown by context size
├── per_example.json                     # Detailed results per example
├── flat_results.jsonl                   # Tabular format for analysis
├── summary.md                           # Human-readable table
├── execution.log                        # Full execution log
└── prompts/                             # If --save-prompts enabled
    ├── example_001_baseline_prompt.json
    ├── example_001_rlm_minimal_*.json
    └── example_001_pyrlm_*.json
```

### Summary Table Example

`summary.md`:

```markdown
| Engine        | Examples | Avg Score | Avg Tokens | Avg Time (s) | Errors |
| ------------- | -------: | --------: | ---------: | -----------: | -----: |
| baseline      |       30 |    0.8200 |      985.3 |         2.45 |      0 |
| rlm_minimal   |       30 |    0.8667 |     3421.7 |        12.84 |      2 |
| pyrlm_runtime |       30 |    0.9000 |     2856.2 |        10.12 |      1 |
```

## Analyzing Results

### Quick Summary

```bash
cat examples/exports/oolong_three_way/run_DIR/summary.md
```

### Inspect Specific Example

```bash
jq '.[] | select(.id == "example_123")' examples/exports/oolong_three_way/run_DIR/per_example.json
```

### Filter Errors

```bash
jq 'select(.error != null)' examples/exports/oolong_three_way/run_DIR/flat_results.jsonl
```

### Monitor Live Logs

```bash
tail -f examples/exports/oolong_three_way/run_DIR/execution.log | grep "DEBUG"
```

### Token Usage Analysis

```python
import json
from pathlib import Path

# Load results
results_dir = Path("examples/exports/oolong_three_way/run_DIR")
with open(results_dir / "per_example.json") as f:
    examples = json.load(f)

# Analyze token usage by context bucket
from collections import defaultdict
buckets = defaultdict(lambda: {"baseline": [], "rlm_minimal": [], "pyrlm_runtime": []})

for ex in examples:
    ctx_len = ex["context_len"]
    bucket = (
        "S(<8k)" if ctx_len < 8000 else
        "M(8k-32k)" if ctx_len < 32000 else
        "L(32k-128k)" if ctx_len < 128000 else
        "XL(>=128k)"
    )

    for engine in ["baseline", "rlm_minimal", "pyrlm_runtime"]:
        tokens = ex["results"][engine]["tokens"]
        buckets[bucket][engine].append(tokens)

# Print averages
for bucket in ["S(<8k)", "M(8k-32k)", "L(32k-128k)", "XL(>=128k)"]:
    print(f"\n{bucket}:")
    for engine in ["baseline", "rlm_minimal", "pyrlm_runtime"]:
        if buckets[bucket][engine]:
            avg = sum(buckets[bucket][engine]) / len(buckets[bucket][engine])
            print(f"  {engine:15s} {avg:.1f} tokens")
```

## Key Metrics to Compare

### 1. Accuracy (Score)

- **Baseline**: Strong on small contexts (<8K), degrades on large contexts
- **RLM-minimal**: Consistent across context sizes
- **PyRLM-runtime**: Should match or beat baseline+rlm-minimal

### 2. Efficiency (Tokens per correct answer)

Calculate: `total_tokens / score`

- **Baseline**: Most efficient on small contexts
- **RLM-minimal**: Higher token usage due to exploration
- **PyRLM-runtime**: Goal is <2x baseline average

### 3. Speed (Latency)

With `--latency-breakdown`, analyze:

- `message_prep`: Time to construct prompts
- `llm_call` / `rlm_execution`: Core LLM time
- `total`: End-to-end latency

### 4. Robustness (Error rate)

Common errors:

- **Timeouts**: Contexts >100K with baseline
- **Rate limits**: Too many subcalls (check `call_log`)
- **Parse failures**: Model didn't follow format

## Troubleshooting

### Authentication Errors

```
Error: Could not automatically determine credentials
```

**Solution**: Run `gcloud auth application-default login`

### Permission Denied

```
403 Permission Denied: aiplatform.endpoints.predict
```

**Solution**: Grant `roles/aiplatform.user` role (see Prerequisites)

### Model Not Found

```
404 Model gemini-2.5-pro not found
```

**Solution**: Verify model name is correct for your region. Try `gemini-1.5-pro` or check available models:

```bash
gcloud ai models list --region=us-central1
```

### Out of Memory / Timeout

**For large contexts (>100K)**:

- Increase `--agent-max-tokens` and `--agent-subcall-max-tokens`
- Reduce `--max-context-len` to filter out extreme examples
- Try smaller `--max-examples` first

### Rate Limiting

```
429 Resource exhausted: Quota exceeded
```

**Solutions**:

1. Add delays between examples (modify benchmark code)
2. Request quota increase in GCP console
3. Reduce `--max-examples` or `--rlm-max-subcalls`

## Next Steps

After running the benchmark:

1. **Compare results**: Which engine wins in each context bucket?
2. **Analyze logs**: Look for patterns in successful/failed examples
3. **Iterate on pyrlm-runtime**:
   - Tune prompts (`system_prompt`, `subcall_system_prompt`)
   - Adjust policy (`max_steps`, `max_subcalls`)
   - Optimize chunking strategy (`ctx.chunk()` parameters)
4. **Re-run benchmark**: Measure improvements

## Architecture Overview

### Adapters

- **`VertexAIAdapter`** (`src/pyrlm_runtime/adapters/vertex_ai.py`):
  - Implements `ModelAdapter` protocol for pyrlm_runtime
  - Returns `ModelResponse` with usage metrics
  - Handles retry logic and message conversion

- **`VertexAIClientForRLMMinimal`** (`examples/vertex_adapter_for_rlm_minimal.py`):
  - Compatible with rlm-minimal's `OpenAIClient` interface
  - Returns plain `str` (no metadata)
  - **`InstrumentedVertexAIClient`**: Subclass that tracks tokens/latency

### Benchmark Flow

```
oolong_three_way_benchmark_vertex.py
  │
  ├─ Load Oolong dataset (HuggingFace)
  ├─ Filter + stratify by context length
  │
  └─ For each example:
      │
      ├─ run_baseline(VertexAIAdapter)
      │   └─ Direct prompt to Gemini
      │
      ├─ run_rlm_minimal_external(InstrumentedVertexAIClient)
      │   └─ RLM_REPL.completion() with rlm-minimal library
      │
      └─ run_pyrlm(VertexAIAdapter)
          └─ pyrlm_runtime RLM.run() with Context + REPL
```

## References

- **Oolong Benchmark Paper**: [Link if available]
- **pyrlm-runtime Docs**: `/Users/U01AB0B5/projects/pyrlm-runtime/README.md`
- **rlm-minimal Docs**: `/Users/U01AB0B5/projects/rlm-minimal/README.md`
- **Vertex AI Gemini API**: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini
