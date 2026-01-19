"""
RLM Example with a Cloud API (OpenAI-compatible)
================================================

PURPOSE:
    Show RLM connecting to cloud APIs that require authentication.
    Useful for stronger models like GPT-4, Claude, or enterprise APIs.

WHAT IT SHOWS:
    1. API key authentication (Bearer token)
    2. auto_finalize_var: if the model never emits FINAL, RLM uses the variable
    3. extract_after(): deterministic helper to extract text without subcalls
    4. Extended timeout for slower cloud APIs

DIFFERENCES VS OLLAMA_EXAMPLE:
    ┌────────────────────┬─────────────────────┬─────────────────────┐
    │                    │   ollama_example    │   cloud_example     │
    ├────────────────────┼─────────────────────┼─────────────────────┤
    │ Authentication     │ None                │ API key required    │
    │ Latency            │ Low (~100ms)        │ Higher (~1-3s)       │
    │ Default timeout    │ 60s                 │ 180s                │
    │ Cost               │ Free                │ Per token           │
    │ auto_finalize_var  │ No                  │ Yes ("key")         │
    └────────────────────┴─────────────────────┴─────────────────────┘

WHAT auto_finalize_var MEANS:
    If the model does not emit "FINAL:" or "FINAL_VAR:", RLM automatically
    returns the value of the specified variable (here: "key").

    Example flow:
    1. Model generates: key = extract_after('The key term is:')
    2. REPL executes: key = "oolong"
    3. Model generates more code or text without FINAL
    4. max_steps reached -> RLM returns value of `key`

REQUIRED ENVIRONMENT VARIABLES:
    LLM_BASE_URL    Endpoint URL (e.g., https://api.openai.com/v1)
    LLM_API_KEY     Your API key

OPTIONAL ENVIRONMENT VARIABLES:
    LLM_MODEL       Model to use (default: nemotron-3-nano:30b-cloud)
    LLM_TIMEOUT     Timeout in seconds (default: 180)

HOW TO RUN:
    # With OpenAI
    LLM_BASE_URL=https://api.openai.com/v1 \
    LLM_API_KEY=sk-... \
    LLM_MODEL=gpt-4o-mini \
    uv run python examples/cloud_example.py

    # With Together.ai
    LLM_BASE_URL=https://api.together.xyz/v1 \
    LLM_API_KEY=... \
    LLM_MODEL=meta-llama/Llama-3-70b-chat-hf \
    uv run python examples/cloud_example.py

EXPECTED OUTPUT:
    oolong

SECURITY:
    - Never hardcode your API key in code
    - Use environment variables or .env files
    - The script validates required variables are present
"""

import os

from rlm_runtime import Context, Policy, RLM
from rlm_runtime.adapters import GenericChatAdapter
from rlm_runtime.prompts import LLAMA_SYSTEM_PROMPT


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for cloud execution")
    return value


def main() -> None:
    base_url = require_env("LLM_BASE_URL")
    api_key = require_env("LLM_API_KEY")
    model = os.getenv("LLM_MODEL", "nemotron-3-nano:30b-cloud")
    timeout = float(os.getenv("LLM_TIMEOUT", "180"))

    adapter = GenericChatAdapter(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
    )
    policy = Policy(max_steps=10, max_subcalls=6, max_total_tokens=12000)
    context = Context.from_text(
        "RLMs treat long prompts as environment state. The key term is: oolong."
    )

    rlm = RLM(
        adapter=adapter,
        policy=policy,
        system_prompt=LLAMA_SYSTEM_PROMPT,
        require_repl_before_final=True,
        auto_finalize_var="key",
    )

    query = (
        "Find the key term defined by 'The key term is:'. "
        "Set key = extract_after('The key term is:') and reply with FINAL_VAR: key."
    )
    output, _trace = rlm.run(query, context)
    print(output)


if __name__ == "__main__":
    main()
