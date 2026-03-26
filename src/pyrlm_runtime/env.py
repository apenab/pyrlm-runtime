from __future__ import annotations

import collections
from dataclasses import dataclass
import datetime
import io
import json
import math
import pprint
import re
import textwrap
import time
from contextlib import redirect_stdout
from typing import Any, Dict, Protocol, runtime_checkable

from .context import Context


@dataclass(frozen=True)
class ExecResult:
    stdout: str
    error: str | None


@runtime_checkable
class REPLProtocol(Protocol):
    """Common interface for REPL backends (PythonREPL, MontyREPL)."""

    def exec(self, code: str) -> ExecResult: ...
    def get(self, name: str) -> Any: ...
    def set(self, name: str, value: Any) -> None: ...
    def snapshot_state(self) -> Dict[str, str]: ...
    def describe_state(self, max_items: int = 20) -> str: ...


def _truncate_repr(value: Any, max_chars: int = 80) -> str:
    text = repr(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 14] + "...<truncated>"


def _summarize_value(value: Any) -> str:
    if isinstance(value, Context):
        meta = value.metadata()
        return f"Context[len={meta['total_length']}, docs={meta['num_documents']}]"
    if isinstance(value, str):
        preview = _truncate_repr(value, max_chars=60)
        return f"str[len={len(value)}] {preview}"
    if isinstance(value, list):
        return f"list[len={len(value)}]"
    if isinstance(value, tuple):
        return f"tuple[len={len(value)}]"
    if isinstance(value, set):
        return f"set[len={len(value)}]"
    if isinstance(value, dict):
        keys = list(value.keys())
        safe_keys = [
            k if isinstance(k, (str, int, float, bool)) else type(k).__name__
            for k in keys[:3]
        ]
        suffix = ", ..." if len(keys) > 3 else ""
        preview = _truncate_repr(safe_keys, max_chars=50)
        return f"dict[len={len(value)}] keys={preview}{suffix}"
    if isinstance(value, (int, float, bool)):
        return f"{type(value).__name__}({_truncate_repr(value, max_chars=32)})"
    return type(value).__name__


def _snapshot_user_state(values: Dict[str, Any], scaffold: set[str] | None = None) -> Dict[str, str]:
    skip = {"__builtins__", "__name__"} | (scaffold or set())
    snapshot: Dict[str, str] = {}
    for name, value in values.items():
        if name.startswith("_") or name in skip:
            continue
        snapshot[name] = _summarize_value(value)
    return dict(sorted(snapshot.items()))


def _format_syntax_error(exc: SyntaxError, code: str) -> str:
    line_text = ""
    if exc.text:
        line_text = exc.text.rstrip("\n")
    elif exc.lineno and 1 <= exc.lineno <= len(code.splitlines()):
        line_text = code.splitlines()[exc.lineno - 1]

    detail = exc.msg or "invalid syntax"
    if exc.lineno:
        detail += f" at line {exc.lineno}"
    if exc.offset and line_text:
        pointer = " " * max(exc.offset - 1, 0) + "^"
        detail += f":\n{line_text}\n{pointer}"
    elif line_text:
        detail += f":\n{line_text}"
    if len(code.splitlines()) > 80:
        detail += "\nHint: split the task into smaller Python blocks."
    return f"SyntaxError: {detail}"


class PythonREPL:
    """Persisted REPL with a minimal, controlled global scope."""

    def __init__(
        self,
        *,
        stdout_limit: int = 16000,
        allowed_modules: Dict[str, Any] | None = None,
        allowed_builtins: Dict[str, Any] | None = None,
    ) -> None:
        self._stdout_limit = stdout_limit
        self._globals: Dict[str, Any] = {}

        modules = allowed_modules or {
            "re": _RegexProxy(re),
            "math": math,
            "json": json,
            "textwrap": textwrap,
            "collections": collections,
            "pprint": pprint,
            "datetime": datetime,
            "time": time,
        }
        builtins = allowed_builtins or self._default_builtins()
        builtins["__import__"] = self._safe_import(modules)

        self._globals.update(modules)
        self._globals["__builtins__"] = builtins
        # Track names injected at init-time so show_vars() excludes them
        # by default.  rlm.py overwrites this with the full scaffold set.
        self._scaffold_names: set[str] = set(modules.keys())

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
            "zip": zip,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "abs": abs,
            "round": round,
            "any": any,
            "all": all,
            "hasattr": hasattr,
            "getattr": getattr,
            "type": type,
            "isinstance": isinstance,
            "dir": dir,
            "map": map,
            "filter": filter,
            "reversed": reversed,
            "chr": chr,
            "ord": ord,
            "repr": repr,
            "Exception": Exception,
            "ValueError": ValueError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "RuntimeError": RuntimeError,
            "NameError": NameError,
            "globals": globals,
            "locals": locals,
        }

    def _safe_import(self, modules: Dict[str, Any]) -> Any:
        allowed = dict(modules)

        def _import(
            name: str,
            globals: Dict[str, Any] | None = None,
            locals: Dict[str, Any] | None = None,
            fromlist: tuple | list = (),
            level: int = 0,
        ) -> Any:  # noqa: A002
            if name in allowed:
                return allowed[name]
            raise ImportError(f"import of '{name}' is not allowed")

        return _import

    def exec(self, code: str) -> ExecResult:
        buffer = io.StringIO()
        error: str | None = None
        if not code.strip():
            return ExecResult(stdout="", error=None)
        try:
            try:
                compiled = compile(code, "<repl>", "eval")
            except SyntaxError:
                compiled_exec = compile(code, "<repl>", "exec")
                with redirect_stdout(buffer):
                    exec(compiled_exec, self._globals, None)
            else:
                with redirect_stdout(buffer):
                    result = eval(compiled, self._globals, None)
                    if result is not None:
                        print(result)
        except SyntaxError as exc:
            error = _format_syntax_error(exc, code)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        stdout = self._truncate(buffer.getvalue())
        # Expose last error as _stderr so the model can inspect and recover programmatically
        self._globals["_stderr"] = error or ""
        return ExecResult(stdout=stdout, error=error)

    def get(self, name: str) -> Any:
        return self._globals.get(name)

    def set(self, name: str, value: Any) -> None:
        self._globals[name] = value

    def show_vars(self) -> str:
        """Return user-defined variables, excluding scaffold and built-ins.

        Mirrors original alexzhang13/rlm's SHOW_VARS() so the model can
        inspect what it has created before calling FINAL_VAR:.
        """
        scaffold: set[str] = getattr(self, "_scaffold_names", set())
        skip = {"__builtins__", "__name__"} | scaffold
        user_vars = {
            k: type(v).__name__
            for k, v in self._globals.items()
            if not k.startswith("_") and k not in skip
        }
        if not user_vars:
            return "No variables created yet. Write code to create variables first."
        return "Available variables: " + ", ".join(
            f"{k} ({t})" for k, t in sorted(user_vars.items())
        )

    def snapshot_state(self) -> Dict[str, str]:
        scaffold: set[str] = getattr(self, "_scaffold_names", set())
        return _snapshot_user_state(self._globals, scaffold)

    def describe_state(self, max_items: int = 20) -> str:
        snapshot = self.snapshot_state()
        if not snapshot:
            return "No user variables available."
        items = list(snapshot.items())
        lines = [f"{name} = {summary}" for name, summary in items[:max_items]]
        if len(items) > max_items:
            lines.append(f"... and {len(items) - max_items} more variable(s)")
        return "REPL state:\n" + "\n".join(lines)

    def restore_names(self, names: dict[str, Any]) -> None:
        """Restore scaffold names in globals after each exec.

        Prevents the model from accidentally overwriting helpers like
        llm_query, ask_chunks, peek, etc.  Mirrors original's
        _restore_scaffold() in local_repl.py.
        """
        self._globals.update(names)

    def _truncate(self, text: str) -> str:
        if len(text) <= self._stdout_limit:
            return text
        return text[: self._stdout_limit] + "...<truncated>"


def _coerce_text(value: Any) -> str:
    if isinstance(value, Context):
        return value.text
    if isinstance(value, list):
        if value and isinstance(value[0], tuple) and len(value[0]) >= 3:
            return "\n".join(str(item[2]) for item in value)
        return "\n".join(str(item) for item in value)
    if isinstance(value, tuple):
        if value and isinstance(value[0], str):
            return "\n".join(value)
        return str(value)
    return str(value)


class _RegexProxy:
    def __init__(self, module: Any) -> None:
        self._module = module

    def search(self, pattern: str, string: Any, flags: int = 0) -> Any:
        return self._module.search(pattern, _coerce_text(string), flags)

    def match(self, pattern: str, string: Any, flags: int = 0) -> Any:
        return self._module.match(pattern, _coerce_text(string), flags)

    def findall(self, pattern: str, string: Any, flags: int = 0) -> Any:
        return self._module.findall(pattern, _coerce_text(string), flags)

    def finditer(self, pattern: str, string: Any, flags: int = 0) -> Any:
        return self._module.finditer(pattern, _coerce_text(string), flags)

    def sub(self, pattern: str, repl: Any, string: Any, count: int = 0, flags: int = 0) -> Any:
        return self._module.sub(pattern, repl, _coerce_text(string), count, flags)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)
