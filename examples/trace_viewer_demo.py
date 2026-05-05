"""Interactive TUI trace viewer demo.

Usage:
    uv run python examples/trace_viewer_demo.py examples/exports/debug_316030026.json

Or with a trace saved by Trace.to_json():
    uv run python examples/trace_viewer_demo.py my_trace.json

Keyboard shortcuts inside the viewer:
    ↑ / k   Move up
    ↓ / j   Move down
    q       Quit
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive TUI viewer for pyrlm-runtime trace JSON files."
    )
    parser.add_argument("trace_file", type=Path, help="Path to the trace JSON file")
    args = parser.parse_args()

    if not args.trace_file.exists():
        print(f"Error: file not found: {args.trace_file}")
        raise SystemExit(1)

    try:
        from pyrlm_runtime.trace_viewer import TraceViewerApp
    except ImportError:
        print(
            "textual is required to run the trace viewer.\n"
            'Install it with: pip install "pyrlm-runtime[tui]"'
        )
        raise SystemExit(1)

    app = TraceViewerApp.from_json_file(args.trace_file)
    app.run()


if __name__ == "__main__":
    main()
