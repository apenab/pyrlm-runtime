"""
Ejemplo de RLM con Ollama Local
================================

PROPÓSITO:
    Demostrar RLM ejecutándose con un modelo LLM real a través de Ollama.
    Es el "Hello World" funcional que conecta con un servidor local.

QUÉ DEMUESTRA:
    1. GenericChatAdapter: conecta con cualquier API compatible OpenAI
    2. Policy: límites de ejecución (pasos, tokens, subcalls)
    3. require_repl_before_final: obliga al modelo a ejecutar código antes de responder
    4. LLAMA_SYSTEM_PROMPT: prompt de sistema optimizado para modelos Llama/Qwen

ARQUITECTURA RLM (del paper MIT CSAIL):
    ┌─────────────────────────────────────────────────────────┐
    │  RLM trata el prompt largo como "estado del entorno"    │
    │  en lugar de pasarlo completo al modelo.                │
    │                                                         │
    │  Contexto (P) ─┬─► REPL Python ◄── Código generado     │
    │                │       │                                │
    │                │       ▼                                │
    │                └─► peek(), ctx.find(), llm_query()     │
    │                                                         │
    │  El modelo INSPECCIONA el contexto programáticamente    │
    │  en lugar de verlo todo de una vez.                     │
    └─────────────────────────────────────────────────────────┘

REQUISITOS PREVIOS:
    1. Instalar Ollama: https://ollama.ai/download
    2. Descargar un modelo: ollama pull llama3.2:latest
    3. Iniciar el servidor: ollama serve (o ya corre como servicio)

VARIABLES DE ENTORNO:
    LLM_BASE_URL    URL del servidor (default: http://localhost:11434/v1)
    LLM_MODEL       Modelo a usar (default: llama3.2:latest)

CÓMO EJECUTAR:
    # Con defaults
    uv run python examples/ollama_example.py

    # Con modelo específico
    LLM_MODEL=qwen2.5-coder:7b uv run python examples/ollama_example.py

OUTPUT ESPERADO:
    oolong

MODELOS RECOMENDADOS (en orden de calidad):
    - qwen2.5-coder:14b   ← Mejor seguimiento de instrucciones
    - qwen2.5-coder:7b    ← Buen balance calidad/velocidad
    - deepseek-coder:6.7b ← Alternativa sólida
    - llama3.2:latest     ← Default, funciona pero menos preciso
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
