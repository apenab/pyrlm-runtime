from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)


def test_importing_pyrlm_runtime_does_not_import_mcp():
    result = _run(
        "import sys, pyrlm_runtime; "
        "assert 'mcp' not in sys.modules, 'mcp imported'; "
        "assert 'pyrlm_runtime.mcp_server' not in sys.modules, 'mcp_server imported'"
    )

    assert result.returncode == 0, result.stderr


def test_missing_mcp_extra_gives_the_install_command():
    result = _run(
        "import sys; sys.modules['mcp'] = None\n"
        "try:\n"
        "    import pyrlm_runtime.mcp_server\n"
        "except ImportError as exc:\n"
        "    print(exc)\n"
    )

    assert result.returncode == 0, result.stderr
    assert 'pip install "pyrlm-runtime[mcp]"' in result.stdout


def test_service_module_does_not_import_mcp():
    package = Path(importlib.util.find_spec("pyrlm_runtime").origin).parent
    tree = ast.parse((package / "mcp_server" / "service.py").read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported |= {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }

    assert not {name for name in imported if name == "mcp" or name.startswith("mcp.")}
