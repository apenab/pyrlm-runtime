"""
RLM Example with Local Ollama
=============================

PURPOSE:
    Demonstrate RLM running with a real LLM through Ollama.
    This is the functional "Hello World" that connects to a local server.

WHAT IT SHOWS:
    1. GenericChatAdapter: connects to any OpenAI-compatible API
    2. Policy: execution limits (steps, tokens, subcalls)
    3. require_repl_before_final: forces the model to execute code before answering
    4. LLAMA_SYSTEM_PROMPT: system prompt tuned for Llama/Qwen models

RLM ARCHITECTURE (MIT CSAIL paper):
    ┌─────────────────────────────────────────────────────────┐
    │  RLM treats the long prompt as "environment state"      │
    │  instead of sending it fully to the model.              │
    │                                                         │
    │  Context (P) ─┬─► Python REPL ◄── Generated code        │
    │              │       │                                  │
    │              │       ▼                                  │
    │              └─► peek(), ctx.find(), llm_query()        │
    │                                                         │
    │  The model INSPECTS the context programmatically        │
    │  instead of seeing it all at once.                      │
    └─────────────────────────────────────────────────────────┘

PREREQUISITES:
    1. Install Ollama: https://ollama.ai/download
    2. Download a model: ollama pull llama3.2:latest
    3. Start the server: ollama serve (or it may already run as a service)

ENVIRONMENT VARIABLES:
    LLM_BASE_URL    Server URL (default: http://localhost:11434/v1)
    LLM_MODEL       Model to use (default: llama3.2:latest)

HOW TO RUN:
    # With defaults
    uv run python examples/ollama_example.py

    # With a specific model
    LLM_MODEL=qwen2.5-coder:7b uv run python examples/ollama_example.py

EXPECTED OUTPUT:
    oolong

RECOMMENDED MODELS (by quality):
    - qwen2.5-coder:14b   <- Best instruction following
    - qwen2.5-coder:7b    <- Good quality/speed balance
    - deepseek-coder:6.7b <- Solid alternative
    - llama3.2:latest     <- Default, works but less precise
"""

import os

from rlm_runtime import Context, Policy, RLM
from rlm_runtime.adapters import GenericChatAdapter
from rlm_runtime.prompts import LLAMA_SYSTEM_PROMPT


def main() -> None:
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv("LLM_MODEL", "llama3.2:latest")

    adapter = GenericChatAdapter(base_url=base_url, model=model)
    policy = Policy(max_steps=10, max_subcalls=8, max_total_tokens=12000)
    context = Context.from_text(
        "RLMs treat long prompts as environment state. The key term is: oolong."
    )

    rlm = RLM(
        adapter=adapter,
        policy=policy,
        system_prompt=LLAMA_SYSTEM_PROMPT,
        require_repl_before_final=True,
    )

    output, _trace = rlm.run(
        "What is the key term? Use the REPL to inspect P. Reply as: FINAL: <answer>.",
        context,
    )
    print(output)


if __name__ == "__main__":
    main()
