from __future__ import annotations

from dataclasses import dataclass
import io
import json
import math
import re
import textwrap
from contextlib import redirect_stdout
from typing import Any, Dict


@dataclass(frozen=True)
class ExecResult:
    stdout: str
    error: str | None


class PythonREPL:
    """Persisted REPL with a minimal, controlled global scope."""

    def __init__(
        self,
        *,
        stdout_limit: int = 4000,
        allowed_modules: Dict[str, Any] | None = None,
        allowed_builtins: Dict[str, Any] | None = None,
    ) -> None:
        self._stdout_limit = stdout_limit
        self._globals: Dict[str, Any] = {}

        modules = allowed_modules or {
            "re": re,
            "math": math,
            "json": json,
            "textwrap": textwrap,
        }
        builtins = allowed_builtins or self._default_builtins()

        self._globals.update(modules)
        self._globals["__builtins__"] = builtins

    def _default_builtins(self) -> Dict[str, Any]:
        return {
            "print": print,
            "len": len,
            "range": range,
            "min": min,
            "max": max,
            "sum": sum,
            "sorted": sorted,
            "enumerate": enumerate,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "abs": abs,
            "any": any,
            "all": all,
            "Exception": Exception,
            "ValueError": ValueError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "RuntimeError": RuntimeError,
        }

    def exec(self, code: str) -> ExecResult:
        buffer = io.StringIO()
        error: str | None = None
        if not code.strip():
            return ExecResult(stdout="", error=None)
        try:
            try:
                compiled = compile(code, "<repl>", "eval")
            except SyntaxError:
                with redirect_stdout(buffer):
                    exec(code, self._globals, None)
            else:
                with redirect_stdout(buffer):
                    result = eval(compiled, self._globals, None)
                    if result is not None:
                        print(result)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        stdout = self._truncate(buffer.getvalue())
        return ExecResult(stdout=stdout, error=error)

    def get(self, name: str) -> Any:
        return self._globals.get(name)

    def set(self, name: str, value: Any) -> None:
        self._globals[name] = value

    def _truncate(self, text: str) -> str:
        if len(text) <= self._stdout_limit:
            return text
        return text[: self._stdout_limit] + "...<truncated>"
