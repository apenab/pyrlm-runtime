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
