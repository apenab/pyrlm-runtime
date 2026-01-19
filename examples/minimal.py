"""
Minimal RLM Example (Recursive Language Model)
==============================================

PURPOSE:
    Demonstrate the basic RLM flow without a real LLM server.
    Ideal for understanding the architecture and running unit tests.

WHAT IT SHOWS:
    1. The REPL loop: the model generates Python code that is executed
    2. Subcalls: llm_query() lets you call sub-LLMs
    3. Finalization: FINAL_VAR returns a variable value as the answer
    4. FakeAdapter: simulates LLM responses with predefined scripts

EXECUTION FLOW:
    Step 1: The "model" (FakeAdapter) generates code:
            snippet = peek(80)
            summary = llm_query(f'Summarize: {snippet}')
            answer = f'Summary -> {summary}'

    Step 2: The REPL runs the code:
            - peek(80) reads the first 80 chars of the context
            - llm_query() performs a subcall returning "[fake] short summary"
            - answer is assigned with the result

    Step 3: The model emits "FINAL_VAR: answer"

    Step 4: RLM returns the value of `answer` as the final response

HOW TO RUN:
    uv run python examples/minimal.py

EXPECTED OUTPUT:
    Summary -> [fake] short summary
    Trace steps: 2

WHY IT IS USEFUL:
    - No Ollama or external API required
    - Runs instantly (<1s)
    - Perfect for tests and CI/CD
    - Demonstrates core RLM components
"""

from rlm_runtime import Context, RLM
from rlm_runtime.adapters import FakeAdapter


def main() -> None:
    adapter = FakeAdapter(
        script=[
            "\n".join(
                [
                    "snippet = peek(80)",
                    "summary = llm_query(f'Summarize: {snippet}')",
                    "answer = f'Summary -> {summary}'",
                ]
            ),
            "FINAL_VAR: answer",
        ]
    )
    adapter.add_rule("You are a sub-LLM", "[fake] short summary")

    context = Context.from_text(
        "RLMs treat long prompts as environment state and inspect them via code."
    )
    runtime = RLM(adapter=adapter)
    output, trace = runtime.run("Give a short summary.", context)

    print(output)
    print("Trace steps:", len(trace.steps))


if __name__ == "__main__":
    main()
