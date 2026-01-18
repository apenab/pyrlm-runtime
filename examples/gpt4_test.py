"""Quick test of RLM with GPT-4o."""
import os
from rlm_runtime import Context, RLM
from rlm_runtime.adapters import GenericChatAdapter
from rlm_runtime.prompts import LLAMA_SYSTEM_PROMPT

# Simple test with just one RLM call
base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
api_key = os.getenv("LLM_API_KEY")
model = os.getenv("LLM_MODEL", "gpt-4o-mini")

adapter = GenericChatAdapter(base_url=base_url, model=model, api_key=api_key, timeout=180)

context = Context.from_text(
    "Here is some information about beverages. "
    "Coffee is popular worldwide. "
    "Tea comes in many varieties. "
    "The key term is: oolong. "
    "Water is essential for life."
)

rlm = RLM(
    adapter=adapter,
    system_prompt=LLAMA_SYSTEM_PROMPT,
    require_repl_before_final=True,
)

output, trace = rlm.run(
    "Find the key term defined by 'The key term is:'. Use extract_after() helper.",
    context
)

print(f"Answer: {output}")
print(f"Steps: {len(trace.steps)}")
print(f"Total tokens: {sum(s.usage.total_tokens for s in trace.steps if s.usage)}")
