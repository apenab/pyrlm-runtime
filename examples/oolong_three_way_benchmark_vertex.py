#!/usr/bin/env python3
"""
Three-way Oolong benchmark: baseline vs rlm-minimal vs pyrlm-runtime.
VERTEX AI VERSION - uses Google Cloud Vertex AI Gemini models.

This script reuses Oolong datasets ("oolongbench/oolong-synth" or "oolongbench/oolong-real")
and mirrors Oolong-style scoring while evaluating three execution strategies:
1) baseline single-shot prompt
2) rlm-minimal REPL loop (external library)
3) pyrlm-runtime agentic loop

Prerequisites:
  - GCP authentication: gcloud auth application-default login
  - Project access: go-agl-poc-radax-p01-poc
  - Permissions: roles/aiplatform.user

Usage:
  uv run python examples/oolong_three_way_benchmark_vertex.py \
    --model gemini-2.5-pro \
    --project-id go-agl-poc-radax-p01-poc \
    --dataset synth \
    --max-examples 30 \
    --log-level INFO \
    --save-prompts

Quick test:
  uv run python examples/oolong_three_way_benchmark_vertex.py \
    --max-examples 1 --log-level DEBUG
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Import Vertex AI adapters
from pyrlm_runtime import Context, Policy, RLM
from pyrlm_runtime.adapters.vertex_ai import VertexAIAdapter
from pyrlm_runtime.prompts import BASE_SYSTEM_PROMPT, SUBCALL_SYSTEM_PROMPT
from vertex_adapter_for_rlm_minimal import InstrumentedVertexAIClient

# Import external rlm-minimal library
from rlm.rlm_repl import RLM_REPL


def usage_total(response: Any) -> int:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    return int(getattr(usage, "total_tokens", 0) or 0)


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_model_name(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def context_bucket(context_len: int) -> str:
    if context_len < 8_000:
        return "S(<8k)"
    if context_len < 32_000:
        return "M(8k-32k)"
    if context_len < 128_000:
        return "L(32k-128k)"
    return "XL(>=128k)"


def synth_attempt_answer_parse(answer: str) -> tuple[str, str]:
    parse_confidence = "low"
    if ":" not in answer:
        if len(answer) < 20:
            return answer, parse_confidence
        return answer.split()[-1], parse_confidence

    candidate_answer = answer.split(":")[-1].strip()
    candidate_answer = candidate_answer.replace("*", "")
    candidate_answer = candidate_answer.replace("[", "").replace("]", "")
    parse_confidence = "med"

    if (
        "User:" in answer
        or "Answer:" in answer
        or "Date:" in answer
        or "Label" in answer
    ):
        parse_confidence = "high"
    if len(candidate_answer) < 20:
        parse_confidence = "vhigh"
    elif "more common" in candidate_answer:
        candidate_answer = "more common"
    elif "less common" in candidate_answer:
        candidate_answer = "less common"
    elif "same frequency" in candidate_answer:
        candidate_answer = "same frequency"

    return candidate_answer, parse_confidence


def synth_process_response(datapoint: dict[str, Any], output: str, model: str) -> dict[str, Any]:
    score = 0.0
    gold = (
        ast.literal_eval(datapoint["answer"])[0]
        if "datetime" not in datapoint["answer"]
        else datetime.strptime(datapoint["answer"], "[datetime.date(%Y, %m, %d)]")
    )

    trimmed_output, parse_confidence = synth_attempt_answer_parse(output)
    if str(trimmed_output) == str(gold):
        score = 1.0
    elif str(trimmed_output) in ["more common", "less common", "same frequency"]:
        if str(trimmed_output) in str(gold):
            score = 1.0
    elif datapoint["answer_type"] == "ANSWER_TYPE.NUMERIC":
        try:
            trimmed_output_int = int(trimmed_output)
            gold_int = int(gold)
            score = float(0.75 ** abs(gold_int - trimmed_output_int))
        except Exception:
            parse_confidence = "low"
    elif datapoint["answer_type"] == "ANSWER_TYPE.DATE":
        try:
            import dateutil.parser

            parsed = dateutil.parser.parse(trimmed_output)
            score = float(parsed == gold)
        except Exception:
            parse_confidence = "low"

    return {
        "id": datapoint["id"],
        "context_window_id": datapoint["context_window_id"],
        "dataset": datapoint["dataset"],
        "model": model,
        "attempted_parse": str(trimmed_output),
        "parse_confidence": parse_confidence,
        "full_answer": output,
        "score": score,
        "answer": str(gold),
    }


def dnd_parse_answer(answer: str) -> int | str | list[str]:
    try:
        return int(answer)
    except ValueError:
        pass
    if "," in answer:
        return [item.strip() for item in answer.split(",") if item.strip()]
    return answer


def dnd_parse_response(answer: str) -> tuple[Any, str]:
    match = re.search(r"\\boxed\{\\text\{([^}]*)\}\}", answer) or re.search(
        r"\\boxed[\{]+([^}]*)[\}]+", answer
    )
    if match:
        answer = match.group(1)
    else:
        return answer, "low"
    return dnd_parse_answer(answer), "high"


def dnd_process_response(datapoint: dict[str, Any], output: str, model: str) -> dict[str, Any]:
    gold = dnd_parse_answer(datapoint["answer"])
    trimmed_output, parse_confidence = dnd_parse_response(output)
    score = 0.0
    if isinstance(gold, int) and isinstance(trimmed_output, int):
        score = float(0.75 ** abs(gold - trimmed_output))
    elif isinstance(gold, str) and isinstance(trimmed_output, str):
        score = float(gold.strip().lower() == trimmed_output.strip().lower())
    elif isinstance(gold, list) and isinstance(trimmed_output, list):
        overlap = set(gold) & set(trimmed_output)
        score = float(len(overlap) / len(gold)) if gold else 0.0

    return {
        "id": datapoint["id"],
        "context_window_id": datapoint["context_window_id"],
        "model": model,
        "attempted_parse": trimmed_output,
        "parse_confidence": parse_confidence,
        "full_answer": output,
        "score": score,
        "answer": gold,
    }


def setup_logging(run_dir: Path, log_level: str) -> logging.Logger:
    """Configure comprehensive logging for the benchmark.

    Args:
        run_dir: Directory to store execution.log
        log_level: Desired console log level (DEBUG/INFO/WARNING/ERROR)

    Returns:
        Logger instance for the benchmark
    """
    log_file = run_dir / "execution.log"

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)

    # File handler - always DEBUG level
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    benchmark_logger = logging.getLogger("oolong_benchmark")
    benchmark_logger.info(f"Logging configured: console={log_level}, file=DEBUG")
    benchmark_logger.info(f"Log file: {log_file}")

    return benchmark_logger


def run_baseline(
    adapter: VertexAIAdapter,
    context_text: str,
    question: str,
    *,
    max_tokens: int,
    temperature: float,
    logger: logging.Logger,
    save_prompts: bool = False,
    prompts_dir: Path | None = None,
    example_id: str = "",
    latency_breakdown: bool = False,
) -> tuple[str, int, float, str | None, dict[str, Any]]:
    """Run baseline single-shot prompt.

    Args:
        adapter: VertexAI ada pter instance
        context_text: Full context text
        question: Question to answer
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        logger: Logger instance
        save_prompts: Whether to save prompts to files
        prompts_dir: Directory to save prompts
        example_id: ID for logging/filenames
        latency_breakdown: Whether to track detailed latencies

    Returns:
        Tuple of (output, tokens, elapsed, error, metadata)
    """
    timing = {"start": time.time()}

    # Build messages
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise solver. Return only the final answer. "
                "No explanations."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context_text}\n\nQuestion:\n{question}\n\n"
                "Return only the answer token/string."
            ),
        },
    ]
    timing["messages_built"] = time.time()

    # Save prompts if enabled
    if save_prompts and prompts_dir:
        prompt_file = prompts_dir / f"{example_id}_baseline_prompt.json"
        write_json(prompt_file, {"messages": messages, "max_tokens": max_tokens, "temperature": temperature})
        logger.debug(f"[BASELINE] Saved prompt to {prompt_file}")

    # Call LLM
    logger.info(f"[BASELINE] Calling LLM for {example_id} (context_len={len(context_text)})")
    timing["before_llm"] = time.time()

    try:
        resp = adapter.complete(messages, max_tokens=max_tokens, temperature=temperature)
        timing["after_llm"] = time.time()

        tokens = usage_total(resp)
        output = resp.text or ""

        # Calculate latencies
        metadata: dict[str, Any] = {}
        if latency_breakdown:
            metadata["latency_phases"] = {
                "message_prep": timing["messages_built"] - timing["start"],
                "llm_call": timing["after_llm"] - timing["before_llm"],
                "total": timing["after_llm"] - timing["start"],
            }

        if save_prompts:
            metadata["prompt_file"] = str(prompt_file) if prompts_dir else None

        elapsed = timing["after_llm"] - timing["start"]
        logger.info(f"[BASELINE] Completed {example_id}: {tokens} tokens, {elapsed:.2f}s")

        return output, tokens, elapsed, None, metadata

    except Exception as exc:
        elapsed = time.time() - timing["start"]
        logger.error(f"[BASELINE] Error for {example_id}: {exc}", exc_info=True)
        return "", 0, elapsed, str(exc), {}


def run_rlm_minimal_external(
    root_client: InstrumentedVertexAIClient,
    sub_client: InstrumentedVertexAIClient,
    context_text: str,
    question: str,
    *,
    max_iterations: int,
    logger: logging.Logger,
    save_prompts: bool = False,
    prompts_dir: Path | None = None,
    example_id: str = "",
    latency_breakdown: bool = False,
) -> tuple[str, int, float, str | None, dict[str, Any]]:
    """Run rlm-minimal using external library.

    Args:
        root_client: Instrumented Vertex AI client for root calls
        sub_client: Instrumented Vertex AI client for subcalls
        context_text: Full context text
        question: Question to answer
        max_iterations: Maximum REPL iterations
        logger: Logger instance
        save_prompts: Whether to save prompts (best effort)
        prompts_dir: Directory to save prompts
        example_id: ID for logging/filenames
        latency_breakdown: Whether to track detailed latencies

    Returns:
        Tuple of (output, tokens, elapsed, error, metadata)
    """
    timing = {"start": time.time()}

    # Reset client metrics
    root_client.reset_metrics()
    sub_client.reset_metrics()

    # Create RLM_REPL with injected clients
    logger.info(f"[RLM_MINIMAL] Starting for {example_id} (context_len={len(context_text)})")

    try:
        rlm = RLM_REPL(
            api_key=None,  # Not used with injected clients
            model="gemini-2.5-pro",  # Model name for metadata
            recursive_model="gemini-2.5-pro",
            max_iterations=max_iterations,
            enable_logging=False,  # Disable internal colorful logging
            client=root_client,  # Inject Vertex AI client
            recursive_client=sub_client,  # Inject Vertex AI subcall client
        )
        timing["rlm_init"] = time.time()

        # Call completion
        output = rlm.completion(context=context_text, query=question)
        timing["rlm_complete"] = time.time()

        # Collect metrics from instrumented clients
        total_tokens = root_client.total_tokens_used + sub_client.total_tokens_used
        elapsed = timing["rlm_complete"] - timing["start"]

        # Build metadata
        metadata: dict[str, Any] = {
            "root_calls": root_client.call_count,
            "subcalls": sub_client.call_count,
            "root_tokens": root_client.total_tokens_used,
            "sub_tokens": sub_client.total_tokens_used,
        }

        if latency_breakdown:
            metadata["latency_phases"] = {
                "rlm_init": timing["rlm_init"] - timing["start"],
                "rlm_execution": timing["rlm_complete"] - timing["rlm_init"],
                "total": elapsed,
            }
            metadata["root_call_log"] = root_client.call_log
            metadata["sub_call_log"] = sub_client.call_log

        logger.info(
            f"[RLM_MINIMAL] Completed {example_id}: {total_tokens} tokens "
            f"({root_client.call_count} root + {sub_client.call_count} sub), {elapsed:.2f}s"
        )

        return output, total_tokens, elapsed, None, metadata

    except Exception as exc:
        elapsed = time.time() - timing["start"]
        logger.error(f"[RLM_MINIMAL] Error for {example_id}: {exc}", exc_info=True)

        # Still collect partial metrics
        total_tokens = root_client.total_tokens_used + sub_client.total_tokens_used
        metadata = {
            "root_calls": root_client.call_count,
            "subcalls": sub_client.call_count,
            "partial": True,
        }

        return "", total_tokens, elapsed, str(exc), metadata


def summarize_trace_steps(trace: Any) -> dict[str, int]:
    if trace is None or not hasattr(trace, "steps"):
        return {}
    out: dict[str, int] = defaultdict(int)
    for step in trace.steps:
        out[step.kind] += 1
    return dict(out)


def run_pyrlm(
    root_adapter: VertexAIAdapter,
    sub_adapter: VertexAIAdapter,
    context_text: str,
    question: str,
    *,
    max_tokens: int,
    subcall_tokens: int,
    max_steps: int,
    max_subcalls: int,
    logger: logging.Logger,
    save_prompts: bool = False,
    prompts_dir: Path | None = None,
    example_id: str = "",
    latency_breakdown: bool = False,
) -> tuple[str, int, float, str | None, dict[str, Any]]:
    """Run pyrlm_runtime with logging.

    Args:
        root_adapter: VertexAI adapter for root calls
        sub_adapter: VertexAI adapter for subcalls
        context_text: Full context text
        question: Question to answer
        max_tokens: Max tokens for root calls
        subcall_tokens: Max tokens for subcalls
        max_steps: Max RLM steps
        max_subcalls: Max subcalls
        logger: Logger instance
        save_prompts: Whether to save prompts (best effort via trace)
        prompts_dir: Directory to save prompts
        example_id: ID for logging/filenames
        latency_breakdown: Whether to track detailed latencies

    Returns:
        Tuple of (output, tokens, elapsed, error, metadata)
    """
    timing = {"start": time.time()}

    logger.info(f"[PYRLM] Starting for {example_id} (context_len={len(context_text)})")

    context = Context.from_text(context_text)

    rlm = RLM(
        adapter=root_adapter,
        subcall_adapter=sub_adapter,
        policy=Policy(
            max_steps=max_steps,
            max_subcalls=max_subcalls,
            max_total_tokens=12_000_000,
        ),
        system_prompt=BASE_SYSTEM_PROMPT,
        subcall_system_prompt=SUBCALL_SYSTEM_PROMPT,
        max_tokens=max_tokens,
        subcall_max_tokens=subcall_tokens,
        require_repl_before_final=True,
        parallel_subcalls=True,
        max_concurrent_subcalls=20,
        conversation_history=True,
    )
    timing["rlm_init"] = time.time()

    trace = None
    try:
        output, trace = rlm.run(question, context)
        timing["rlm_complete"] = time.time()

        tokens = 0
        if trace is not None:
            tokens = sum((s.usage.total_tokens for s in trace.steps if s.usage), 0)

        elapsed = timing["rlm_complete"] - timing["start"]
        trace_steps_summary = summarize_trace_steps(trace)

        # Build metadata
        metadata: dict[str, Any] = {"trace_steps": trace_steps_summary}

        if latency_breakdown:
            metadata["latency_phases"] = {
                "rlm_init": timing["rlm_init"] - timing["start"],
                "rlm_execution": timing["rlm_complete"] - timing["rlm_init"],
                "total": elapsed,
            }

        # Save trace if requested
        if save_prompts and prompts_dir and trace:
            trace_file = prompts_dir / f"{example_id}_pyrlm_trace.json"
            try:
                trace_data = {
                    "steps": [
                        {
                            "kind": s.kind,
                            "tokens": s.usage.total_tokens if s.usage else 0,
                            "error": s.error,
                        }
                        for s in trace.steps
                    ]
                }
                write_json(trace_file, trace_data)
                metadata["trace_file"] = str(trace_file)
            except Exception as trace_exc:
                logger.warning(f"Could not save trace: {trace_exc}")

        logger.info(
            f"[PYRLM] Completed {example_id}: {tokens} tokens, "
            f"{trace_steps_summary}, {elapsed:.2f}s"
        )

        return output or "", tokens, elapsed, None, metadata

    except Exception as exc:
        elapsed = time.time() - timing["start"]
        logger.error(f"[PYRLM] Error for {example_id}: {exc}", exc_info=True)

        tokens = 0
        if trace is not None:
            tokens = sum((s.usage.total_tokens for s in trace.steps if s.usage), 0)

        return "", tokens, elapsed, str(exc), {"trace_steps": summarize_trace_steps(trace)}


def score_output(dataset_kind: str, datapoint: dict[str, Any], output: str, model: str) -> dict[str, Any]:
    if dataset_kind == "synth":
        return synth_process_response(datapoint, output, model)
    return dnd_process_response(datapoint, output, model)


def engine_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"examples": 0, "score_sum": 0.0, "tokens_sum": 0, "elapsed_sum": 0.0, "errors": 0}
    )
    for row in rows:
        e = row["engine"]
        grouped[e]["examples"] += 1
        grouped[e]["score_sum"] += float(row["score"])
        grouped[e]["tokens_sum"] += int(row["tokens"])
        grouped[e]["elapsed_sum"] += float(row["elapsed"])
        grouped[e]["errors"] += int(bool(row["error"]))

    out: dict[str, Any] = {}
    for engine, acc in grouped.items():
        n = max(1, acc["examples"])
        out[engine] = {
            "examples": acc["examples"],
            "avg_score": acc["score_sum"] / n,
            "avg_tokens": acc["tokens_sum"] / n,
            "avg_elapsed": acc["elapsed_sum"] / n,
            "errors": acc["errors"],
        }
    return out


def engine_summary_by_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "examples": 0,
                "score_sum": 0.0,
                "tokens_sum": 0,
                "elapsed_sum": 0.0,
                "errors": 0,
            }
        )
    )
    for row in rows:
        engine = row["engine"]
        bucket = row["context_bucket"]
        g = grouped[engine][bucket]
        g["examples"] += 1
        g["score_sum"] += float(row["score"])
        g["tokens_sum"] += int(row["tokens"])
        g["elapsed_sum"] += float(row["elapsed"])
        g["errors"] += int(bool(row["error"]))

    out: dict[str, Any] = {}
    for engine, by_bucket in grouped.items():
        out[engine] = {}
        for bucket, acc in by_bucket.items():
            n = max(1, acc["examples"])
            out[engine][bucket] = {
                "examples": acc["examples"],
                "avg_score": acc["score_sum"] / n,
                "avg_tokens": acc["tokens_sum"] / n,
                "avg_elapsed": acc["elapsed_sum"] / n,
                "errors": acc["errors"],
            }
    return out


def validate_rows_schema(rows: list[dict[str, Any]]) -> None:
    required = {
        "id",
        "engine",
        "score",
        "tokens",
        "elapsed",
        "error",
        "context_len",
        "context_bucket",
        "attempted_parse",
        "gold_answer",
        "parse_confidence",
    }
    for idx, row in enumerate(rows):
        missing = required - set(row.keys())
        if missing:
            raise ValueError(f"Row {idx} missing required keys: {sorted(missing)}")


def validate_result_consistency(
    rows: list[dict[str, Any]],
    per_example: list[dict[str, Any]],
    summary: dict[str, Any],
    run_config: dict[str, Any],
) -> None:
    validate_rows_schema(rows)

    expected_rows = len(per_example) * 3
    if len(rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows (3 engines x {len(per_example)} examples), got {len(rows)}"
        )

    if int(run_config.get("examples_evaluated", -1)) != len(per_example):
        raise ValueError(
            "run_config.examples_evaluated does not match number of evaluated examples"
        )

    recomputed = engine_summary(rows)
    if summary != recomputed:
        raise ValueError("summary payload is inconsistent with flat rows")


def select_rows(
    data: Any,
    *,
    strategy: str,
    max_examples: int,
    seed: int,
) -> Any:
    if max_examples <= 0 or len(data) <= max_examples:
        return data

    if strategy == "head":
        return data.select(range(max_examples))

    if strategy == "random":
        rng = random.Random(seed)
        idxs = list(range(len(data)))
        rng.shuffle(idxs)
        return data.select(sorted(idxs[:max_examples]))

    # Stratified by log2(context_len) bins to avoid overfitting to one length regime.
    bins: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(data):
        ctx_len = int(row["context_len"])
        key = int(math.log2(max(2, ctx_len)))
        bins[key].append(i)

    rng = random.Random(seed)
    keys = sorted(bins.keys())
    selected: list[int] = []
    per_bin = max(1, max_examples // max(1, len(keys)))
    for key in keys:
        choices = bins[key][:]
        rng.shuffle(choices)
        selected.extend(choices[:per_bin])

    # Fill remaining slots from leftovers globally.
    if len(selected) < max_examples:
        remaining = [i for i in range(len(data)) if i not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_examples - len(selected)])

    selected = sorted(selected[:max_examples])
    return data.select(selected)


def write_markdown_summary(path: Path, summary: dict[str, Any], by_bucket: dict[str, Any]) -> None:
    lines = []
    lines.append("# Oolong Three-Way Benchmark Summary")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Engine | Examples | Avg Score | Avg Tokens | Avg Time (s) | Errors |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for engine in ("baseline", "rlm_minimal", "pyrlm_runtime"):
        if engine not in summary:
            continue
        s = summary[engine]
        lines.append(
            f"| {engine} | {s['examples']} | {s['avg_score']:.4f} | {s['avg_tokens']:.1f} | {s['avg_elapsed']:.2f} | {s['errors']} |"
        )

    lines.append("")
    lines.append("## By Context Bucket")
    lines.append("")
    lines.append("| Engine | Bucket | Examples | Avg Score | Avg Tokens | Avg Time (s) | Errors |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    bucket_order = ["S(<8k)", "M(8k-32k)", "L(32k-128k)", "XL(>=128k)"]
    for engine in ("baseline", "rlm_minimal", "pyrlm_runtime"):
        if engine not in by_bucket:
            continue
        for bucket in bucket_order:
            if bucket not in by_bucket[engine]:
                continue
            s = by_bucket[engine][bucket]
            lines.append(
                f"| {engine} | {bucket} | {s['examples']} | {s['avg_score']:.4f} | {s['avg_tokens']:.1f} | {s['avg_elapsed']:.2f} | {s['errors']} |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synth", "real"], default="synth")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--project-id", default="go-agl-poc-radax-p01-poc")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument(
        "--sample-strategy",
        choices=["head", "random", "stratified"],
        default="stratified",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-context-len", type=int, default=1024)
    parser.add_argument("--max-context-len", type=int, default=131072)
    parser.add_argument("--labels", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--baseline-max-tokens", type=int, default=1024)
    parser.add_argument("--agent-max-tokens", type=int, default=2048)
    parser.add_argument("--agent-subcall-max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--rlm-max-steps", type=int, default=35)
    parser.add_argument("--rlm-max-subcalls", type=int, default=250)
    parser.add_argument("--minimal-max-iterations", type=int, default=20)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (DEBUG for verbose output)",
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Save all prompts and traces to files",
    )
    parser.add_argument(
        "--latency-breakdown",
        action="store_true",
        help="Include detailed latency metrics per phase",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with: uv pip install datasets"
        ) from exc

    split_to_use = "context_window_text_with_labels" if args.labels else "context_window_text"
    if args.dataset == "synth":
        data = load_dataset("oolongbench/oolong-synth")["test"]
    else:
        data = load_dataset("oolongbench/oolong-real", "dnd")["test"]

    data = data.filter(lambda x: x["context_len"] <= args.max_context_len)
    data = data.filter(lambda x: x["context_len"] > args.min_context_len)
    data = select_rows(
        data,
        strategy=args.sample_strategy,
        max_examples=args.max_examples,
        seed=args.seed,
    )

    baseline_max_tokens = args.baseline_max_tokens
    agent_max_tokens = args.agent_max_tokens
    agent_subcall_max_tokens = args.agent_subcall_max_tokens
    if args.max_tokens is not None:
        baseline_max_tokens = args.max_tokens
        agent_max_tokens = args.max_tokens
        agent_subcall_max_tokens = args.max_tokens

    run_dir = (
        Path("examples/exports/oolong_three_way")
        / f"run_{now_tag()}_{args.dataset}_{safe_model_name(args.model)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(run_dir, args.log_level)
    logger.info(f"Benchmark started: {args.dataset} dataset, model={args.model}")
    logger.info(f"Project: {args.project_id}, Location: {args.location}")
    logger.info(f"Results will be saved to: {run_dir}")

    # Create prompts directory if needed
    prompts_dir = None
    if args.save_prompts:
        prompts_dir = run_dir / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        logger.info(f"Prompts will be saved to: {prompts_dir}")

    # Create Vertex AI adapters
    logger.info("Initializing Vertex AI adapters...")

    # Baseline adapter
    base_adapter = VertexAIAdapter(
        project_id=args.project_id,
        location=args.location,
        model=args.model,
        timeout=900,
        logger_instance=logger,
    )

    # RLM-minimal adapters (instrumented for metrics)
    min_root = InstrumentedVertexAIClient(
        project_id=args.project_id,
        location=args.location,
        model=args.model,
        logger_instance=logger,
    )
    min_sub = InstrumentedVertexAIClient(
        project_id=args.project_id,
        location=args.location,
        model=args.model,
        logger_instance=logger,
    )

    # PyRLM adapters
    py_root = VertexAIAdapter(
        project_id=args.project_id,
        location=args.location,
        model=args.model,
        timeout=900,
        logger_instance=logger,
    )
    py_sub = VertexAIAdapter(
        project_id=args.project_id,
        location=args.location,
        model=args.model,
        timeout=900,
        logger_instance=logger,
    )

    rows: list[dict[str, Any]] = []
    per_example: list[dict[str, Any]] = []
    print(
        f"Running {len(data)} examples from Oolong-{args.dataset} with model={args.model} "
        f"(sampling={args.sample_strategy}, seed={args.seed})"
    )
    for i, dp in enumerate(data):
        context_text = dp[split_to_use]
        question = dp["question"]
        example_id = dp["id"]
        print(f"[{i + 1}/{len(data)}] id={example_id} ctx_len={dp['context_len']}")

        b_out, b_tok, b_s, b_err, b_meta = run_baseline(
            base_adapter,
            context_text,
            question,
            max_tokens=baseline_max_tokens,
            temperature=args.temperature,
            logger=logger,
            save_prompts=args.save_prompts,
            prompts_dir=prompts_dir,
            example_id=example_id,
            latency_breakdown=args.latency_breakdown,
        )
        m_out, m_tok, m_s, m_err, m_meta = run_rlm_minimal_external(
            min_root,
            min_sub,
            context_text,
            question,
            max_iterations=args.minimal_max_iterations,
            logger=logger,
            save_prompts=args.save_prompts,
            prompts_dir=prompts_dir,
            example_id=example_id,
            latency_breakdown=args.latency_breakdown,
        )
        p_out, p_tok, p_s, p_err, p_meta = run_pyrlm(
            py_root,
            py_sub,
            context_text,
            question,
            max_tokens=agent_max_tokens,
            subcall_tokens=agent_subcall_max_tokens,
            max_steps=args.rlm_max_steps,
            max_subcalls=args.rlm_max_subcalls,
            logger=logger,
            save_prompts=args.save_prompts,
            prompts_dir=prompts_dir,
            example_id=example_id,
            latency_breakdown=args.latency_breakdown,
        )

        engines = [
            ("baseline", b_out, b_tok, b_s, b_err, b_meta),
            ("rlm_minimal", m_out, m_tok, m_s, m_err, m_meta),
            ("pyrlm_runtime", p_out, p_tok, p_s, p_err, p_meta),
        ]
        example_payload = {"id": example_id, "question": question, "results": {}}

        for engine, out, tok, elapsed, err, extra in engines:
            eval_out = score_output(args.dataset, dict(dp), out, args.model)
            row = {
                "id": example_id,
                "engine": engine,
                "score": float(eval_out["score"]),
                "tokens": int(tok),
                "elapsed": float(elapsed),
                "error": err,
                "context_len": int(dp["context_len"]),
                "context_bucket": context_bucket(int(dp["context_len"])),
                "attempted_parse": eval_out["attempted_parse"],
                "gold_answer": eval_out["answer"],
                "parse_confidence": eval_out["parse_confidence"],
            }
            rows.append(row)
            example_payload["results"][engine] = {
                "output": out,
                "tokens": tok,
                "elapsed": elapsed,
                "error": err,
                "score": eval_out["score"],
                "attempted_parse": eval_out["attempted_parse"],
                "gold_answer": eval_out["answer"],
                "parse_confidence": eval_out["parse_confidence"],
                "extra": extra,
            }
        per_example.append(example_payload)

    summary = engine_summary(rows)
    summary_by_bucket = engine_summary_by_bucket(rows)
    config = {
        "dataset": args.dataset,
        "model": args.model,
        "project_id": args.project_id,
        "location": args.location,
        "max_examples": args.max_examples,
        "sample_strategy": args.sample_strategy,
        "seed": args.seed,
        "min_context_len": args.min_context_len,
        "max_context_len": args.max_context_len,
        "labels": args.labels,
        "max_tokens_legacy_override": args.max_tokens,
        "baseline_max_tokens": baseline_max_tokens,
        "agent_max_tokens": agent_max_tokens,
        "agent_subcall_max_tokens": agent_subcall_max_tokens,
        "temperature": args.temperature,
        "rlm_max_steps": args.rlm_max_steps,
        "rlm_max_subcalls": args.rlm_max_subcalls,
        "minimal_max_iterations": args.minimal_max_iterations,
        "log_level": args.log_level,
        "save_prompts": args.save_prompts,
        "latency_breakdown": args.latency_breakdown,
        "examples_evaluated": len(data),
    }
    validate_result_consistency(rows, per_example, summary, config)

    write_json(run_dir / "run_config.json", config)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "summary_by_context_bucket.json", summary_by_bucket)
    write_json(run_dir / "per_example.json", per_example)
    write_jsonl(run_dir / "flat_results.jsonl", rows)
    write_markdown_summary(run_dir / "summary.md", summary, summary_by_bucket)

    print("\nSummary")
    print("=" * 72)
    for engine in ("baseline", "rlm_minimal", "pyrlm_runtime"):
        if engine not in summary:
            continue
        s = summary[engine]
        print(
            f"{engine:14s} score={s['avg_score']:.4f} "
            f"avg_tokens={s['avg_tokens']:.1f} avg_time={s['avg_elapsed']:.2f}s "
            f"errors={s['errors']}/{s['examples']}"
        )
    print("\nBy Context Bucket")
    print("=" * 72)
    bucket_order = ["S(<8k)", "M(8k-32k)", "L(32k-128k)", "XL(>=128k)"]
    for engine in ("baseline", "rlm_minimal", "pyrlm_runtime"):
        if engine not in summary_by_bucket:
            continue
        for bucket in bucket_order:
            if bucket not in summary_by_bucket[engine]:
                continue
            s = summary_by_bucket[engine][bucket]
            print(
                f"{engine:14s} {bucket:12s} score={s['avg_score']:.4f} "
                f"avg_tokens={s['avg_tokens']:.1f} avg_time={s['avg_elapsed']:.2f}s "
                f"n={s['examples']} err={s['errors']}"
            )
    print(f"\nArtifacts: {run_dir}")


if __name__ == "__main__":
    main()
