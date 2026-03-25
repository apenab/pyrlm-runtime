"""
Demo interactivo de RLM (Recursive Language Model) con trazas Rich.

=== ¿Qué es RLM? ===

Un LLM normal recibe un prompt y devuelve texto directamente.
Un RLM es diferente: el modelo genera CÓDIGO PYTHON que se ejecuta
en un REPL, y puede llamar a sub-LLMs para analizar partes del texto.
Es como darle al modelo una calculadora y acceso a asistentes.

=== ¿Qué muestra este ejemplo? ===

  1. Se le da al RLM un reporte de ventas como contexto
  2. Se le hace una pregunta sobre ese reporte
  3. El RLM genera código Python para inspeccionar los datos
  4. El REPL ejecuta ese código y devuelve resultados
  5. El RLM sintetiza la respuesta final

=== Variables de entorno necesarias ===

    AZURE_OPENAI_API_KEY
    OPENAI_ENDPOINT  (o AZURE_ACCOUNT_NAME)

=== Cómo ejecutar ===

    uv run python examples/rich_repl_demo.py
    uv run python examples/rich_repl_demo.py --modelo gpt-5.1
    uv run python examples/rich_repl_demo.py --pregunta "¿Cuál región vendió más?"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from _azure_check import check_azure_connection
from pyrlm_runtime import Context, RLM
from pyrlm_runtime.adapters import AzureOpenAIAdapter
from pyrlm_runtime.rich_trace import RichTraceListener

# ---------------------------------------------------------------------------
# Datos de ejemplo: un reporte de ventas trimestral en español.
# Este texto es el "contexto" que el RLM va a explorar con código Python.
# ---------------------------------------------------------------------------
REPORTE_VENTAS = """
Reporte de ventas - Q1 2025

Región Norte:
  - Laptops: 150 unidades, $225,000
  - Monitores: 80 unidades, $64,000
  - Teclados: 200 unidades, $10,000

Región Sur:
  - Laptops: 90 unidades, $135,000
  - Monitores: 120 unidades, $96,000
  - Teclados: 310 unidades, $15,500

Región Centro:
  - Laptops: 200 unidades, $300,000
  - Monitores: 95 unidades, $76,000
  - Teclados: 180 unidades, $9,000
""".strip()

PREGUNTA_DEFAULT = "¿Qué región tuvo el mayor ingreso total y cuánto fue?"


# ---------------------------------------------------------------------------
# Carga de credenciales Azure (busca .env en varias ubicaciones)
# ---------------------------------------------------------------------------
def _cargar_env() -> None:
    descubierto = find_dotenv(usecwd=True)
    if descubierto:
        load_dotenv(descubierto, override=False)

    if os.getenv("AZURE_OPENAI_API_KEY"):
        return

    aqui = Path(__file__).resolve()
    candidatos = [
        aqui.parents[1] / ".env",
        aqui.parents[2] / ".env",
    ]
    for candidato in candidatos:
        if candidato.is_file():
            load_dotenv(candidato, override=False)
            if os.getenv("AZURE_OPENAI_API_KEY"):
                return

    load_dotenv(override=False)


def parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo de RLM con trazas Rich en terminal"
    )
    parser.add_argument(
        "--modelo",
        default="gpt-5.1",
        help="Nombre del deployment en Azure (default: gpt-5.1)",
    )
    parser.add_argument(
        "--pregunta",
        default=PREGUNTA_DEFAULT,
        help="Pregunta a hacer sobre el reporte de ventas",
    )
    return parser.parse_args()


def main() -> None:
    args = parsear_args()

    # --- Paso 1: Cargar credenciales y verificar conexión ---
    _cargar_env()
    check_azure_connection(args.modelo)

    # --- Paso 2: Crear los componentes del RLM ---
    #
    #   adapter  = conexión al LLM (Azure OpenAI en este caso)
    #   listener = imprime cada paso del RLM con paneles Rich
    #   runtime  = el motor RLM que orquesta todo
    #
    adapter = AzureOpenAIAdapter(model=args.modelo)
    listener = RichTraceListener()
    runtime = RLM(adapter=adapter, event_listener=listener)

    # --- Paso 3: Ejecutar el RLM ---
    #
    #   El RLM recibe:
    #     - Una pregunta (query)
    #     - Un contexto (los datos que el modelo puede explorar con código)
    #
    #   Internamente el RLM hace un loop:
    #     1. El LLM genera código Python
    #     2. El REPL lo ejecuta (puede usar peek(), ctx.find(), llm_query()...)
    #     3. El LLM ve el resultado y decide: ¿genero más código o ya tengo la respuesta?
    #     4. Cuando está listo, emite FINAL_VAR con la variable que contiene la respuesta
    #
    contexto = Context.from_text(REPORTE_VENTAS)
    respuesta, traza = runtime.run(args.pregunta, contexto)

    # --- Paso 4: Mostrar resultados ---
    tokens_total = sum(
        paso.usage.total_tokens
        for paso in traza.steps
        if paso.usage is not None
    )

    print("\n" + "=" * 50)
    print("RESPUESTA FINAL:")
    print(respuesta)
    print(f"\nResumen: {len(traza.steps)} pasos, {tokens_total} tokens")
    print("=" * 50)


if __name__ == "__main__":
    main()
