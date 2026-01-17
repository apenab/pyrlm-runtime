"""
Ejemplo Mínimo de RLM (Recursive Language Model)
================================================

PROPÓSITO:
    Demostrar el flujo básico de RLM sin necesidad de un servidor LLM real.
    Ideal para entender la arquitectura y ejecutar tests unitarios.

QUÉ DEMUESTRA:
    1. El ciclo REPL: el modelo genera código Python que se ejecuta
    2. Sub-llamadas (subcalls): llm_query() permite consultar sub-LLMs
    3. Finalización: FINAL_VAR devuelve el valor de una variable como respuesta
    4. FakeAdapter: simula respuestas LLM con scripts predefinidos

FLUJO DE EJECUCIÓN:
    Paso 1: El "modelo" (FakeAdapter) genera código:
            snippet = peek(80)
            summary = llm_query(f'Summarize: {snippet}')
            answer = f'Summary -> {summary}'

    Paso 2: El REPL ejecuta el código:
            - peek(80) lee los primeros 80 chars del contexto
            - llm_query() hace una subcall que retorna "[fake] short summary"
            - answer se asigna con el resultado

    Paso 3: El modelo genera "FINAL_VAR: answer"

    Paso 4: RLM retorna el valor de `answer` como respuesta final

CÓMO EJECUTAR:
    uv run python examples/minimal.py

OUTPUT ESPERADO:
    Summary -> [fake] short summary
    Trace steps: 2

POR QUÉ ES ÚTIL:
    - No requiere Ollama ni API externa
    - Ejecuta instantáneamente (<1s)
    - Perfecto para tests y CI/CD
    - Demuestra los componentes core de RLM
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
