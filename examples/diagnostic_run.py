#!/usr/bin/env python3
"""
Quick diagnostic: run 3 small examples through pyrlm_runtime only,
comparing the OLD config (4-line stub prompt + auto_finalize_var) against
the FIXED config (BASE_SYSTEM_PROMPT + require_repl_before_final).

Usage:
  uv run python examples/diagnostic_run.py \
    --project-id go-agl-poc-radax-p01-poc \
    --model gemini-2.5-pro
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from pyrlm_runtime import Context, Policy, RLM
from pyrlm_runtime.adapters.vertex_ai import VertexAIAdapter
from pyrlm_runtime.prompts import BASE_SYSTEM_PROMPT, SUBCALL_SYSTEM_PROMPT

# Example IDs chosen from the S(<8k) bucket where pyrlm_runtime failed badly
TARGET_IDS = {411040047, 412040051, 712070053}


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


def score_synth(datapoint: dict[str, Any], output: str) -> tuple[float, str, str]:
    gold = (
        ast.literal_eval(datapoint["answer"])[0]
        if "datetime" not in datapoint["answer"]
        else datetime.strptime(datapoint["answer"], "[datetime.date(%Y, %m, %d)]")
    )
    trimmed, confidence = synth_attempt_answer_parse(output)
    score = 0.0
    if str(trimmed) == str(gold):
        score = 1.0
    elif str(trimmed) in ["more common", "less common", "same frequency"]:
        if str(trimmed) in str(gold):
            score = 1.0
    elif datapoint["answer_type"] == "ANSWER_TYPE.NUMERIC":
        try:
            score = float(0.75 ** abs(int(trimmed) - int(gold)))
        except Exception:
            pass
    return score, str(trimmed), str(gold)


def run_with_config(
    adapter: VertexAIAdapter,
    sub_adapter: VertexAIAdapter,
    context_text: str,
    question: str,
    config_name: str,
    *,
    max_tokens: int = 2048,
    subcall_tokens: int = 2048,
    max_steps: int = 35,
    max_subcalls: int = 250,
) -> dict[str, Any]:
    context = Context.from_text(context_text)

    if config_name == "old_stub":
        # Reproduce the old buggy configuration
        system_prompt = (
            "You have Python REPL access with P and ctx.\n"
            "Answer exactly the question and store final result in final_answer.\n"
            "Use deterministic string search first if possible.\n"
            "Return only the final answer."
        )
        subcall_prompt = "Answer the user question from provided text. Output only answer."
        rlm = RLM(
            adapter=adapter,
            subcall_adapter=sub_adapter,
            policy=Policy(max_steps=max_steps, max_subcalls=max_subcalls, max_total_tokens=12_000_000),
            system_prompt=system_prompt,
            subcall_system_prompt=subcall_prompt,
            max_tokens=max_tokens,
            subcall_max_tokens=subcall_tokens,
            auto_finalize_var="final_answer",
            parallel_subcalls=True,
            max_concurrent_subcalls=20,
            conversation_history=True,
        )
    elif config_name == "fixed":
        # The fixed configuration: proper prompts, no auto-finalize, require REPL
        rlm = RLM(
            adapter=adapter,
            subcall_adapter=sub_adapter,
            policy=Policy(max_steps=max_steps, max_subcalls=max_subcalls, max_total_tokens=12_000_000),
            system_prompt=BASE_SYSTEM_PROMPT,
            subcall_system_prompt=SUBCALL_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            subcall_max_tokens=subcall_tokens,
            require_repl_before_final=True,
            parallel_subcalls=True,
            max_concurrent_subcalls=20,
            conversation_history=True,
        )
    else:
        raise ValueError(f"Unknown config: {config_name}")

    start = time.time()
    trace = None
    try:
        output, trace = rlm.run(question, context)
        elapsed = time.time() - start
        tokens = 0
        if trace is not None:
            tokens = sum((s.usage.total_tokens for s in trace.steps if s.usage), 0)
        steps_summary = {}
        if trace:
            from collections import defaultdict
            out = defaultdict(int)
            for step in trace.steps:
                out[step.kind] += 1
            steps_summary = dict(out)
        return {
            "config": config_name,
            "output": output or "",
            "tokens": tokens,
            "elapsed": elapsed,
            "error": None,
            "steps": steps_summary,
        }
    except Exception as exc:
        elapsed = time.time() - start
        return {
            "config": config_name,
            "output": "",
            "tokens": 0,
            "elapsed": elapsed,
            "error": str(exc),
            "steps": {},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic run for pyrlm_runtime fixes")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--project-id", default="go-agl-poc-radax-p01-poc")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["old_stub", "fixed"],
        choices=["old_stub", "fixed"],
        help="Which configurations to test (default: both)",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Specific example IDs to test (default: 411040047 412040051 712070053)",
    )
    args = parser.parse_args()

    target_ids = set(int(x) for x in args.ids) if args.ids else TARGET_IDS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("diagnostic")

    logger.info("Loading oolong-synth dataset...")
    from datasets import load_dataset
    data = load_dataset("oolongbench/oolong-synth")["test"]

    # Filter to target IDs
    examples = [dp for dp in data if dp["id"] in target_ids]
    logger.info(f"Found {len(examples)} matching examples out of {len(target_ids)} requested")

    if not examples:
        logger.error("No matching examples found. Check IDs.")
        return

    # Create adapters
    adapter = VertexAIAdapter(
        project_id=args.project_id,
        location=args.location,
        model=args.model,
        timeout=900,
    )
    sub_adapter = VertexAIAdapter(
        project_id=args.project_id,
        location=args.location,
        model=args.model,
        timeout=900,
    )

    results: list[dict[str, Any]] = []

    for dp in examples:
        example_id = dp["id"]
        question = dp["question"]
        context_text = dp["context_window_text"]
        ctx_len = dp["context_len"]
        logger.info(f"\n{'='*72}")
        logger.info(f"Example {example_id} (ctx_len={ctx_len})")
        logger.info(f"Question: {question}")
        logger.info(f"Gold answer: {ast.literal_eval(dp['answer'])[0]}")

        for config_name in args.configs:
            logger.info(f"\n  --- Config: {config_name} ---")
            result = run_with_config(
                adapter, sub_adapter, context_text, question, config_name
            )
            score, parsed, gold = score_synth(dict(dp), result["output"])
            result["score"] = score
            result["parsed"] = parsed
            result["gold"] = gold
            result["example_id"] = example_id
            result["ctx_len"] = ctx_len
            results.append(result)

            logger.info(
                f"  [{config_name}] score={score:.4f} parsed={parsed!r} gold={gold!r} "
                f"tokens={result['tokens']} time={result['elapsed']:.1f}s "
                f"steps={result['steps']}"
            )
            if result["error"]:
                logger.warning(f"  [{config_name}] ERROR: {result['error']}")

    # Print summary table
    print(f"\n{'='*72}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*72}")
    print(f"{'Example':<14} {'Config':<12} {'Score':>6} {'Parsed':<20} {'Gold':<20} {'Tokens':>8} {'Time':>7}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['example_id']:<14} {r['config']:<12} {r['score']:>6.4f} "
            f"{r['parsed']:<20} {r['gold']:<20} {r['tokens']:>8} {r['elapsed']:>6.1f}s"
        )

    # Save results
    out_path = Path("examples/exports/diagnostic_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
