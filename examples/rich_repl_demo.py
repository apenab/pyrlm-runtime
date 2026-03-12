"""
Azure GPT-5.1 Live Rich trace demo for pyrlm-runtime.

This example is meant to be visually clear in a terminal recording or screenshot:
it loads Azure credentials from `.env`, runs a short RLM task, and prints the
full REPL trajectory live with Rich panels.

Required env vars:
    AZURE_OPENAI_API_KEY
    OPENAI_ENDPOINT or AZURE_ACCOUNT_NAME

Optional env vars:
    AZURE_OPENAI_API_VERSION (default: 2024-10-21)

Run with:
    uv run python examples/rich_repl_demo.py
    uv run python examples/rich_repl_demo.py --model gpt-5.1
    uv run python examples/rich_repl_demo.py --query "Which product sold the most units?"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _azure_check import check_azure_connection
from pyrlm_runtime import Context, RLM
from pyrlm_runtime.adapters import AzureOpenAIAdapter
from pyrlm_runtime.rich_trace import RichTraceListener

DEFAULT_QUERY = "Which product sold the most units, and how many units was that?"

DEMO_TEXT = """
Sales report:

- notebook: units=12
- mug: units=7
- sticker: units=20

Answer using only this report.
""".strip()


def _load_demo_env() -> None:
    loaded = False

    discovered = find_dotenv(usecwd=True)
    if discovered:
        load_dotenv(discovered, override=False)
        loaded = True

    if os.getenv("AZURE_OPENAI_API_KEY"):
        return

    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / ".env",
        here.parents[2] / ".env",
        here.parents[2] / "ocr-documents" / ".env",
    ]
    for candidate in candidates:
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            loaded = True
            if os.getenv("AZURE_OPENAI_API_KEY"):
                return

    if not loaded:
        load_dotenv(override=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Azure GPT-5.1 Live Rich trace demo")
    parser.add_argument(
        "--model",
        default="gpt-5.1",
        help="Azure deployment name to use for the demo",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Question to ask over the demo context",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_demo_env()
    check_azure_connection(args.model)

    adapter = AzureOpenAIAdapter(model=args.model)
    listener = RichTraceListener()
    runtime = RLM(adapter=adapter, event_listener=listener)

    output, trace = runtime.run(args.query, Context.from_text(DEMO_TEXT))
    total_tokens = sum(step.usage.total_tokens for step in trace.steps if step.usage is not None)

    print("\nFinal answer:")
    print(output)
    print(f"\nTrace summary: {len(trace.steps)} steps, {total_tokens} tokens")


if __name__ == "__main__":
    main()
