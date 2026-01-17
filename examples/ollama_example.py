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
