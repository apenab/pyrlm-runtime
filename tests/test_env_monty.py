"""Tests for MontyREPL and AST variable extraction helpers."""

import pytest

from pyrlm_runtime.env_monty import (
    MONTY_AVAILABLE,
    MontyLimits,
    MontyREPL,
    _find_assigned_names,
)

pytestmark = pytest.mark.skipif(not MONTY_AVAILABLE, reason="pydantic-monty not installed")


# ---------------------------------------------------------------------------
# Unit tests for _find_assigned_names / _extract_targets / _names_from_target
# ---------------------------------------------------------------------------


class TestFindAssignedNames:
    def test_simple_assign(self):
        assert _find_assigned_names("x = 1") == ["x"]

    def test_multiple_assigns(self):
        assert _find_assigned_names("x = 1\ny = 2") == ["x", "y"]

    def test_tuple_unpack(self):
        names = _find_assigned_names("a, b = 1, 2")
        assert names == ["a", "b"]

    def test_nested_tuple_unpack(self):
        names = _find_assigned_names("(a, (b, c)) = (1, (2, 3))")
        assert names == ["a", "b", "c"]

    def test_list_unpack(self):
        names = _find_assigned_names("[a, b] = [1, 2]")
        assert names == ["a", "b"]

    def test_starred_unpack(self):
        names = _find_assigned_names("a, *b, c = [1, 2, 3, 4]")
        assert names == ["a", "b", "c"]

    def test_annotated_assign(self):
        names = _find_assigned_names("x: int = 5")
        assert names == ["x"]

    def test_augmented_assign(self):
        names = _find_assigned_names("x += 1")
        assert names == ["x"]

    def test_for_loop(self):
        names = _find_assigned_names("for i in range(10):\n    pass")
        assert names == ["i"]

    def test_chained_assign(self):
        # a = b = 1 has two targets in a single Assign node
        names = _find_assigned_names("a = b = 1")
        assert names == ["a", "b"]

    def test_syntax_error_returns_empty(self):
        assert _find_assigned_names("def (broken") == []

    def test_no_assignments(self):
        assert _find_assigned_names("print('hello')") == []

    def test_empty_code(self):
        assert _find_assigned_names("") == []

    def test_attribute_assign_ignored(self):
        # obj.attr = 1 -> target is ast.Attribute, not ast.Name
        assert _find_assigned_names("obj.attr = 1") == []

    def test_subscript_assign_ignored(self):
        # d['key'] = 1 -> target is ast.Subscript, not ast.Name
        assert _find_assigned_names("d['key'] = 1") == []

    def test_mixed_assignments(self):
        code = "x = 1\na, b = 2, 3\ny: str = 'hi'\nz += 10"
        names = _find_assigned_names(code)
        assert names == ["x", "a", "b", "y", "z"]


# ---------------------------------------------------------------------------
# Integration tests for MontyREPL
# ---------------------------------------------------------------------------


class TestMontyREPLBasic:
    def test_simple_exec(self):
        repl = MontyREPL()
        result = repl.exec("x = 1 + 2")
        assert result.error is None
        assert repl.get("x") == 3

    def test_print_output(self):
        repl = MontyREPL()
        result = repl.exec("print('hello')")
        assert result.error is None
        assert "hello" in result.stdout

    def test_empty_code(self):
        repl = MontyREPL()
        result = repl.exec("")
        assert result.error is None
        assert result.stdout == ""

    def test_variable_persistence(self):
        repl = MontyREPL()
        repl.exec("x = 10")
        repl.exec("y = x + 5")
        assert repl.get("y") == 15

    def test_set_and_get(self):
        repl = MontyREPL()
        repl.set("name", "test")
        result = repl.exec("greeting = 'hello ' + name")
        assert result.error is None
        assert repl.get("greeting") == "hello test"

    def test_syntax_error(self):
        repl = MontyREPL()
        result = repl.exec("def (broken")
        assert result.error is not None
        assert "SyntaxError" in result.error


class TestMontyREPLVariableCapture:
    """Tests for variable capture across exec() calls with various patterns."""

    def test_tuple_unpack_capture(self):
        repl = MontyREPL()
        result = repl.exec("a, b = 10, 20")
        assert result.error is None
        assert repl.get("a") == 10
        assert repl.get("b") == 20

    def test_starred_unpack_capture(self):
        repl = MontyREPL()
        result = repl.exec("first, *rest = [1, 2, 3, 4]")
        assert result.error is None
        assert repl.get("first") == 1
        assert repl.get("rest") == [2, 3, 4]

    def test_annotated_assign_capture(self):
        repl = MontyREPL()
        result = repl.exec("x: int = 42")
        assert result.error is None
        assert repl.get("x") == 42

    def test_augmented_assign_capture(self):
        repl = MontyREPL()
        repl.set("count", 5)
        result = repl.exec("count += 3")
        assert result.error is None
        assert repl.get("count") == 8

    def test_chained_assign_not_supported_by_monty(self):
        # Monty does not support chained assignment (a = b = 99)
        repl = MontyREPL()
        result = repl.exec("a = b = 99")
        assert result.error is not None
        assert "SyntaxError" in result.error

    def test_multi_step_capture(self):
        repl = MontyREPL()
        repl.exec("x = 1")
        repl.exec("y = x + 1")
        repl.exec("z = x + y")
        assert repl.get("z") == 3

    def test_list_unpack_capture(self):
        repl = MontyREPL()
        result = repl.exec("[a, b] = [100, 200]")
        assert result.error is None
        assert repl.get("a") == 100
        assert repl.get("b") == 200


class TestMontyREPLExternalFunctions:
    def test_callable_as_external_fn(self):
        repl = MontyREPL()
        repl.set("double", lambda x: x * 2)
        result = repl.exec("result = double(5)")
        assert result.error is None
        assert repl.get("result") == 10

    def test_non_callable_as_input(self):
        repl = MontyREPL()
        repl.set("data", [1, 2, 3])
        result = repl.exec("total = len(data)")
        assert result.error is None
        assert repl.get("total") == 3


class TestMontyREPLObjectProxy:
    """Tests for complex object registration via AST method call rewriting."""

    def test_context_attribute_access(self):
        from pyrlm_runtime import Context

        repl = MontyREPL()
        ctx = Context.from_text("hello world")
        repl.set("ctx", ctx)
        result = repl.exec("t = ctx.text")
        assert result.error is None
        assert repl.get("t") == "hello world"

    def test_context_method_find(self):
        from pyrlm_runtime import Context

        repl = MontyREPL()
        ctx = Context.from_text("The key term is: oolong.")
        repl.set("ctx", ctx)
        result = repl.exec('matches = ctx.find("key term")')
        assert result.error is None
        assert repl.get("matches") == [(4, 12, "key term")]

    def test_context_method_slice(self):
        from pyrlm_runtime import Context

        repl = MontyREPL()
        ctx = Context.from_text("abcdefghij")
        repl.set("ctx", ctx)
        result = repl.exec("s = ctx.slice(2, 5)")
        assert result.error is None
        assert repl.get("s") == "cde"

    def test_context_method_len_chars(self):
        from pyrlm_runtime import Context

        repl = MontyREPL()
        ctx = Context.from_text("hello")
        repl.set("ctx", ctx)
        result = repl.exec("n = ctx.len_chars()")
        assert result.error is None
        assert repl.get("n") == 5

    def test_context_method_chunk(self):
        from pyrlm_runtime import Context

        repl = MontyREPL()
        ctx = Context.from_text("abcdef")
        repl.set("ctx", ctx)
        result = repl.exec("chunks = ctx.chunk(3)")
        assert result.error is None
        assert repl.get("chunks") == [(0, 3, "abc"), (3, 6, "def")]

    def test_non_method_attribute_untouched(self):
        from pyrlm_runtime import Context

        repl = MontyREPL()
        ctx = Context.from_text("test")
        repl.set("ctx", ctx)
        # context_type is a plain attribute, not a method call
        result = repl.exec("ct = ctx.context_type")
        assert result.error is None
        assert repl.get("ct") == "string"


class TestMontyREPLSecurity:
    def test_infinite_loop_timeout(self):
        repl = MontyREPL(limits=MontyLimits(max_duration_secs=0.5))
        result = repl.exec("while True:\n    pass")
        assert result.error is not None

    def test_stdout_truncation(self):
        repl = MontyREPL(stdout_limit=50)
        result = repl.exec("print('A' * 200)")
        assert result.error is None
        assert len(result.stdout) <= 70  # 50 + "...<truncated>"
        assert result.stdout.endswith("...<truncated>")
