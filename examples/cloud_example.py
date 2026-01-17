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
