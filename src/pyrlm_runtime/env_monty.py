from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable, Dict

from .env import ExecResult

# Primitive types that Monty can handle directly as inputs.
_MONTY_PRIMITIVES = (str, int, float, bool, type(None))


def _is_monty_serializable(value: Any) -> bool:
    """Check if a value can be passed directly as a Monty input."""
    if isinstance(value, _MONTY_PRIMITIVES):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_monty_serializable(v) for v in value)
    if isinstance(value, dict):
        return all(
            isinstance(k, str) and _is_monty_serializable(v)
            for k, v in value.items()
        )
    return False

try:
    from pydantic_monty import (
        Monty,
        MontyError,
        MontyRuntimeError,
        MontySyntaxError,
    )

    MONTY_AVAILABLE = True
except ImportError:
    MONTY_AVAILABLE = False


@dataclass(frozen=True)
class MontyLimits:
    """Resource limits for Monty REPL execution."""

    max_duration_secs: float = 5.0
    max_memory: int = 128 * 1024 * 1024  # 128 MB
    max_allocations: int = 1_000_000
    max_recursion_depth: int = 100

    def to_dict(self) -> dict:
        return {
            "max_duration_secs": self.max_duration_secs,
            "max_memory": self.max_memory,
            "max_allocations": self.max_allocations,
            "max_recursion_depth": self.max_recursion_depth,
        }


def _find_assigned_names(code: str) -> list[str]:
    """Extract top-level variable names assigned in the code using AST."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.iter_child_nodes(tree):
        names.extend(_extract_targets(node))
    return names


def _names_from_target(target: ast.AST) -> list[str]:
    """Recursively extract variable names from an assignment target node."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        result: list[str] = []
        for elt in target.elts:
            result.extend(_names_from_target(elt))
        return result
    if isinstance(target, ast.Starred):
        return _names_from_target(target.value)
    return []


def _extract_targets(node: ast.AST) -> list[str]:
    """Return all assigned variable names from an AST node."""
    if isinstance(node, ast.Assign):
        names: list[str] = []
        for target in node.targets:
            names.extend(_names_from_target(target))
        return names
    if isinstance(node, ast.AnnAssign):
        return _names_from_target(node.target)
    if isinstance(node, (ast.AugAssign, ast.For)):
        return _names_from_target(node.target)
    return []


class _MethodCallRewriter(ast.NodeTransformer):
    """Rewrite ``name.method(args)`` calls to ``name__method(args)`` in the AST.

    This allows LLM code like ``ctx.find("x")`` to be transparently mapped to
    the external function ``ctx__find("x")`` that Monty can invoke.
    Attribute accesses that are *not* method calls (e.g. ``ctx.text``) are left
    untouched so Monty can resolve them on the dataclass input.
    """

    def __init__(self, name: str, methods: set[str]) -> None:
        self._name = name
        self._methods = methods

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        if not isinstance(node.func, ast.Attribute):
            return node
        attr: ast.Attribute = node.func
        if not isinstance(attr.value, ast.Name):
            return node
        if attr.value.id != self._name or attr.attr not in self._methods:
            return node
        # Replace ctx.method(...) -> ctx__method(...)
        new_func = ast.Name(id=f"{self._name}__{attr.attr}", ctx=ast.Load())
        ast.copy_location(new_func, node.func)
        node.func = new_func
        return node


def _rewrite_method_calls(code: str, name: str, methods: set[str]) -> str:
    """Rewrite ``name.method(...)`` calls to ``name__method(...)`` in source code."""
    if not methods:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    rewriter = _MethodCallRewriter(name, methods)
    tree = rewriter.visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


class MontyREPL:
    """Sandboxed REPL backed by pydantic-monty (Rust interpreter).

    Exposes the same interface as PythonREPL so they are interchangeable.
    Each exec() call creates a fresh Monty instance with the current
    variables as inputs and registered callables as external functions.

    Variable persistence across exec() calls is achieved by:
    1. Passing current variables as Monty inputs
    2. Detecting assignments via AST and appending a capture dict
    3. Extracting assigned values from the capture dict result
    """

    def __init__(
        self,
        *,
        stdout_limit: int = 4000,
        limits: MontyLimits | None = None,
    ) -> None:
        if not MONTY_AVAILABLE:
            raise ImportError(
                "pydantic-monty is not installed. "
                "Install it with: pip install pydantic-monty"
            )
        self._stdout_limit = stdout_limit
        self._limits = limits or MontyLimits()
        self._variables: Dict[str, Any] = {}
        self._external_fns: Dict[str, Callable] = {}
        self._object_methods: Dict[str, set[str]] = {}

    def set(self, name: str, value: Any) -> None:
        if callable(value) and not isinstance(value, (str, int, float, bool, list, dict, tuple)):
            self._external_fns[name] = value
        elif _is_monty_serializable(value):
            self._variables[name] = value
        else:
            self._register_object(name, value)

    def get(self, name: str) -> Any:
        return self._variables.get(name)

    def _register_object(self, name: str, obj: Any) -> None:
        """Decompose a complex object into external functions for Monty.

        Passes the object itself as a Monty input (for attribute access like
        ``ctx.text``) and registers each public method as an external function
        named ``{name}__{method}``.  At exec time, an AST rewrite transforms
        ``ctx.method(args)`` into ``ctx__method(args)`` so Monty invokes the
        external function while plain attribute access remains untouched.
        """
        self._variables[name] = obj

        methods: set[str] = set()
        for attr_name in dir(obj):
            if attr_name.startswith("_"):
                continue
            attr = getattr(obj, attr_name, None)
            if not callable(attr):
                continue
            self._external_fns[f"{name}__{attr_name}"] = attr
            methods.add(attr_name)
        self._object_methods[name] = methods

    def exec(self, code: str) -> ExecResult:
        if not code.strip():
            return ExecResult(stdout="", error=None)

        assigned = _find_assigned_names(code)
        augmented_code = self._augment_code(code, assigned)

        stdout_parts: list[str] = []

        def print_callback(_stream: str, text: str) -> None:
            stdout_parts.append(text)

        try:
            result_value = self._run_monty(augmented_code, print_callback)
            self._capture_variables(assigned, result_value)
        except MontySyntaxError as exc:
            return ExecResult(stdout="", error=f"SyntaxError: {exc}")
        except MontyRuntimeError as exc:
            return ExecResult(
                stdout=self._truncate("".join(stdout_parts)),
                error=exc.display("type-msg"),
            )
        except MontyError as exc:
            return ExecResult(stdout="", error=str(exc))
        except Exception as exc:  # noqa: BLE001
            return ExecResult(stdout="", error=f"{type(exc).__name__}: {exc}")

        return ExecResult(stdout=self._truncate("".join(stdout_parts)), error=None)

    def _augment_code(self, code: str, assigned: list[str]) -> str:
        """Append a capture dict expression to extract assigned variable values."""
        if not assigned:
            return code
        capture_expr = "{" + ", ".join(f'"{n}": {n}' for n in assigned) + "}"
        return code.rstrip() + "\n" + capture_expr

    def _run_monty(self, code: str, print_callback: Callable) -> Any:
        """Create a Monty instance and execute the code."""
        for name, methods in self._object_methods.items():
            code = _rewrite_method_calls(code, name, methods)

        input_names = list(self._variables.keys())
        ext_fn_names = list(self._external_fns.keys())

        monty = Monty(
            code,
            inputs=input_names if input_names else None,
            external_functions=ext_fn_names if ext_fn_names else None,
        )

        run_kwargs: dict[str, Any] = {
            "inputs": dict(self._variables) if input_names else None,
            "limits": self._limits.to_dict(),
            "print_callback": print_callback,
        }
        if ext_fn_names:
            run_kwargs["external_functions"] = dict(self._external_fns)

        return monty.run(**run_kwargs)

    def _capture_variables(self, assigned: list[str], result_value: Any) -> None:
        """Extract assigned variable values from the Monty result dict."""
        if not assigned or not isinstance(result_value, dict):
            return
        for var_name in assigned:
            if var_name in result_value:
                self._variables[var_name] = result_value[var_name]

    def _truncate(self, text: str) -> str:
        if len(text) <= self._stdout_limit:
            return text
        return text[: self._stdout_limit] + "...<truncated>"
