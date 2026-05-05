"""
Demo: contar las 'r' en 50 nombres de frutas generados via subcall.

=== ¿Qué hace este ejemplo? ===

  1. El RLM recibe el problema: generar 50 nombres de frutas y contar
     cuántas letras 'r' tiene cada nombre.
  2. Para generar los nombres, el modelo llama a llm_query() — eso crea
     un subcall visible en la traza como nodo hijo.
  3. El modelo ejecuta un assert para verificar que hay exactamente 50
     frutas antes de construir el resultado.
  4. Devuelve un diccionario {nombre_fruta: cantidad_de_r}.
  5. Al terminar, abre el trace viewer interactivo para explorar cada paso.

=== Variables de entorno necesarias ===

    AZURE_OPENAI_API_KEY
    OPENAI_ENDPOINT  (o AZURE_ACCOUNT_NAME)

=== Cómo ejecutar ===

    uv run python examples/fruits_r_count_demo.py
    uv run python examples/fruits_r_count_demo.py --modelo gpt-4o
    uv run python examples/fruits_r_count_demo.py --no-viewer   # sólo imprime
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
from pyrlm_runtime.events import RLMEvent, RLMEventListener
from pyrlm_runtime.rich_trace import RichTraceListener

# ---------------------------------------------------------------------------
# La tarea: sin contexto externo — el problema es autocontenido.
# El campo clave es "using a sub-call" para que el modelo use llm_query().
# ---------------------------------------------------------------------------
QUERY = (
    "Generate the names of exactly 50 fruits by calling llm_query() once to get a "
    "newline-separated list of 50 unique fruit names. "
    "Parse the result into a Python list, then assert len(fruit_list) == 50. "
    "Finally, build and return a dictionary mapping each fruit name to the count of "
    "the letter 'r' (case-insensitive) in that name."
)


def _cargar_env() -> None:
    descubierto = find_dotenv(usecwd=True)
    if descubierto:
        load_dotenv(descubierto, override=False)

    if os.getenv("AZURE_OPENAI_API_KEY"):
        return

    aqui = Path(__file__).resolve()
    for candidato in [aqui.parents[1] / ".env", aqui.parents[2] / ".env"]:
        if candidato.is_file():
            load_dotenv(candidato, override=False)
            if os.getenv("AZURE_OPENAI_API_KEY"):
                return

    load_dotenv(override=False)


def parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cuenta las 'r' en 50 frutas generadas por subcall, luego abre el trace viewer"
    )
    parser.add_argument(
        "--modelo",
        default="gpt-5.1",
        help="Modelo principal para el RLM (default: gpt-5.1)",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="No abrir el trace viewer al terminar (sólo imprime el resultado)",
    )
    return parser.parse_args()


class _SimpleProgressListener(RLMEventListener):
    """Minimal listener for viewer mode: prints a single line per step."""

    def handle(self, event: RLMEvent) -> None:
        if event.kind == "run_started":
            print("Running...", flush=True)
        elif event.kind == "step_completed" and event.step is not None:
            s = event.step
            tokens = s.usage.total_tokens if s.usage else 0
            print(f"  step {s.step_id}: {s.kind}  ({tokens} tokens)", flush=True)
        elif event.kind == "run_finished":
            print(f"Done — {event.total_steps} steps, {event.tokens_used} tokens\n", flush=True)


def main() -> None:
    args = parsear_args()

    _cargar_env()
    check_azure_connection(args.modelo)

    adapter = AzureOpenAIAdapter(model=args.modelo)
    # When opening the TUI viewer, use a plain text progress listener so Rich
    # panels don't bleed through into the TUI's alternate screen.
    listener: RLMEventListener = RichTraceListener() if args.no_viewer else _SimpleProgressListener()
    runtime = RLM(adapter=adapter, event_listener=listener, parallel_subcalls=False)

    respuesta, traza = runtime.run(QUERY, Context.from_text(""))

    tokens_total = sum(
        paso.usage.total_tokens for paso in traza.steps if paso.usage is not None
    )
    subcalls = sum(1 for paso in traza.steps if paso.kind in {"subcall", "sub_subcall"})

    print("\n" + "=" * 60)
    print("RESULTADO FINAL:")
    print(respuesta)
    print(f"\nResumen: {len(traza.steps)} pasos | {subcalls} subcall(s) | {tokens_total:,} tokens")
    print("=" * 60)

    if not args.no_viewer:
        try:
            from pyrlm_runtime.trace_viewer import TraceViewerApp
        except ImportError:
            print(
                "\nPara abrir el trace viewer instala: "
                'pip install "pyrlm-runtime[tui]"'
            )
            return

        print("\nAbriendo trace viewer... (q para salir)\n")
        app = TraceViewerApp.from_trace(traza, source_name="fruits_r_count", answer=respuesta)
        app.run()


if __name__ == "__main__":
    main()
