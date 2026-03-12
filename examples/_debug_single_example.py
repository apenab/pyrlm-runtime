#!/usr/bin/env python3
"""Debug a single Oolong example by ID — dumps full RLM trace."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from _azure_check import check_azure_connection
from pyrlm_runtime import Context, Policy, RLM
from pyrlm_runtime.adapters import AzureOpenAIAdapter
from pyrlm_runtime.prompts import BASE_SYSTEM_PROMPT, SUBCALL_SYSTEM_PROMPT

from oolong_rlm_vs_baseline import (
    OOLONG_ENV_TIPS,
    OOLONG_SUBCALL_TIPS,
    _serialize_trace,
    _trace_diagnostics,
    _trace_total_tokens,
    score_output,
    write_json,
)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Debug single Oolong example")
    parser.add_argument("example_id", type=int, help="Oolong example ID")
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--env-tips", action="store_true", default=True)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--max-subcalls", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--subcall-max-tokens", type=int, default=1024)
    parser.add_argument("--text-limit", type=int, default=4000,
                        help="Trace text truncation limit (default: 4000 chars)")
    parser.add_argument("--no-labels", action="store_true", default=False,
                        help="Use context_window_text instead of context_window_text_with_labels")
    args = parser.parse_args()

    check_azure_connection(args.model)

    from datasets import load_dataset

    data = load_dataset("oolongbench/oolong-synth")["test"]
    row = None
    for r in data:
        if r.get("id") == args.example_id:
            row = r
            break
    if row is None:
        print(f"Example {args.example_id} not found")
        return

    context_col = "context_window_text" if args.no_labels else "context_window_text_with_labels"
    context_text = row[context_col]
    question = row["question"]
    print(f"context_col: {context_col}")
    print(f"ID:          {row['id']}")
    print(f"context_len: {row['context_len']}")
    print(f"question:    {question}")
    print(f"gold answer: {row['answer']}")
    print(f"answer_type: {row.get('answer_type', '?')}")
    print()

    system_prompt = BASE_SYSTEM_PROMPT
    if args.env_tips:
        system_prompt += OOLONG_ENV_TIPS
    subcall_system_prompt = SUBCALL_SYSTEM_PROMPT + OOLONG_SUBCALL_TIPS

    adapter = AzureOpenAIAdapter(model=args.model, timeout=900.0)
    sub_adapter = AzureOpenAIAdapter(model=args.model, timeout=900.0)

    context = Context.from_text(context_text)
    rlm = RLM(
        adapter=adapter,
        subcall_adapter=sub_adapter,
        policy=Policy(
            max_steps=args.max_steps,
            max_subcalls=args.max_subcalls,
            max_total_tokens=12_000_000,
        ),
        system_prompt=system_prompt,
        subcall_system_prompt=subcall_system_prompt,
        max_tokens=args.max_tokens,
        subcall_max_tokens=args.subcall_max_tokens,
        require_repl_before_final=True,
        parallel_subcalls=True,
        max_concurrent_subcalls=20,
        conversation_history=True,
    )

    import time

    start = time.time()
    output, trace = rlm.run(question, context)
    elapsed = time.time() - start

    tokens = _trace_total_tokens(trace)
    eval_out = score_output("synth", dict(row), output or "", args.model)

    print(f"\n{'=' * 72}")
    print(f"RLM output:  {output}")
    print(f"Score:       {eval_out['score']}")
    print(f"Parse:       {eval_out['attempted_parse']} (confidence={eval_out['parse_confidence']})")
    print(f"Gold:        {eval_out['answer']}")
    print(f"Tokens:      {tokens}")
    print(f"Elapsed:     {elapsed:.1f}s")
    print(f"{'=' * 72}")

    # Dump full trace
    out_path = Path(f"examples/exports/debug_{args.example_id}.json")
    write_json(
        out_path,
        {
            "id": args.example_id,
            "question": question,
            "gold_answer": row["answer"],
            "rlm_output": output,
            "score": eval_out["score"],
            "tokens": tokens,
            "elapsed": elapsed,
            "diagnostics": _trace_diagnostics(trace),
            "steps": _serialize_trace(trace, text_limit=args.text_limit),
        },
    )
    print(f"\nFull trace: {out_path}")


if __name__ == "__main__":
    main()
