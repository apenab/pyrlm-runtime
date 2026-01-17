"""
Ejemplo de RLM con API Cloud (OpenAI-compatible)
=================================================

PROPÓSITO:
    Demostrar RLM conectándose a APIs cloud que requieren autenticación.
    Útil para usar modelos más potentes como GPT-4, Claude, o APIs empresariales.

QUÉ DEMUESTRA:
    1. Autenticación con API key (Bearer token)
    2. auto_finalize_var: si el modelo no responde FINAL, RLM usa la variable indicada
    3. extract_after(): helper determinista para extraer texto sin subcalls
    4. Timeout extendido para APIs cloud más lentas

DIFERENCIAS CON OLLAMA_EXAMPLE:
    ┌────────────────────┬─────────────────────┬─────────────────────┐
    │                    │   ollama_example    │   cloud_example     │
    ├────────────────────┼─────────────────────┼─────────────────────┤
    │ Autenticación      │ Ninguna             │ API key requerida   │
    │ Latencia           │ Baja (~100ms)       │ Alta (~1-3s)        │
    │ Timeout default    │ 60s                 │ 180s                │
    │ Costo              │ Gratis              │ Por token           │
    │ auto_finalize_var  │ No                  │ Sí ("key")          │
    └────────────────────┴─────────────────────┴─────────────────────┘

QUÉ ES auto_finalize_var:
    Si el modelo no genera "FINAL:" o "FINAL_VAR:", RLM automáticamente
    retorna el valor de la variable especificada (en este caso "key").

    Ejemplo de flujo:
    1. Modelo genera: key = extract_after('The key term is:')
    2. REPL ejecuta: key = "oolong"
    3. Modelo genera más código o texto sin FINAL
    4. Se alcanza max_steps → RLM retorna valor de `key`

VARIABLES DE ENTORNO (requeridas):
    LLM_BASE_URL    URL del endpoint (ej: https://api.openai.com/v1)
    LLM_API_KEY     Tu API key

VARIABLES DE ENTORNO (opcionales):
    LLM_MODEL       Modelo a usar (default: nemotron-3-nano:30b-cloud)
    LLM_TIMEOUT     Timeout en segundos (default: 180)

CÓMO EJECUTAR:
    # Con OpenAI
    LLM_BASE_URL=https://api.openai.com/v1 \
    LLM_API_KEY=sk-... \
    LLM_MODEL=gpt-4o-mini \
    uv run python examples/cloud_example.py

    # Con Together.ai
    LLM_BASE_URL=https://api.together.xyz/v1 \
    LLM_API_KEY=... \
    LLM_MODEL=meta-llama/Llama-3-70b-chat-hf \
    uv run python examples/cloud_example.py

OUTPUT ESPERADO:
    oolong

SEGURIDAD:
    - Nunca hardcodees tu API key en el código
    - Usa variables de entorno o .env files
    - El código valida que las variables requeridas estén presentes
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
