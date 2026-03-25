from pyrlm_runtime import PythonREPL


def test_repl_persistence_and_restrictions() -> None:
    repl = PythonREPL(stdout_limit=1000)
    repl.exec("x = 5")
    result = repl.exec("print(x + 1)")
    assert result.stdout.strip() == "6"

    blocked = repl.exec("import os")
    assert blocked.error is not None

    blocked_open = repl.exec("open('tmp.txt', 'w')")
    assert blocked_open.error is not None


def test_syntax_error_includes_line_detail_and_hint_for_large_blocks() -> None:
    repl = PythonREPL()
    lines = [f"x{i} = {i}" for i in range(85)]
    lines.append("if True print('boom')")
    result = repl.exec("\n".join(lines))

    assert result.error is not None
    assert "SyntaxError: invalid syntax at line 86" in result.error
    assert "if True print('boom')" in result.error
    assert "Hint: split the task into smaller Python blocks." in result.error


# ---------------------------------------------------------------------------
# show_vars and restore_names — mirrors original alexzhang13/rlm design
# ---------------------------------------------------------------------------


def test_show_vars_empty() -> None:
    """show_vars() returns a helpful message when no user variables exist."""
    # PythonREPL initialises _scaffold_names with the default module names, so
    # a freshly-created REPL with no user code should report "No variables".
    repl = PythonREPL()
    msg = repl.show_vars()
    assert "No variables" in msg


def test_show_vars_lists_user_variables() -> None:
    """show_vars() shows user-created variables, not scaffold names."""
    repl = PythonREPL()
    scaffold = {"peek": lambda: None, "ask": lambda: None}
    repl.restore_names(scaffold)
    # Union with existing scaffold names so the default modules stay hidden too
    repl._scaffold_names = repl._scaffold_names | set(scaffold.keys())  # type: ignore[attr-defined]
    repl.exec("my_result = 'hello world'")
    repl.exec("counter = 42")
    msg = repl.show_vars()
    assert "my_result" in msg
    assert "counter" in msg
    # Scaffold names must NOT appear
    assert "peek" not in msg
    assert "ask" not in msg


def test_restore_names_prevents_scaffold_overwrite() -> None:
    """restore_names() resets scaffold names after an accidental overwrite.

    Regression: without scaffold restoration, `llm_query = None` would
    permanently break all subsequent subcalls in the RLM loop.
    """
    repl = PythonREPL()
    sentinel = object()
    scaffold = {"llm_query": sentinel}
    repl.restore_names(scaffold)

    # Model accidentally overwrites llm_query
    repl.exec("llm_query = None")
    assert repl.get("llm_query") is None  # confirm overwrite happened

    # After restore_names, scaffold is back
    repl.restore_names(scaffold)
    assert repl.get("llm_query") is sentinel


def test_show_vars_integration_with_rlm() -> None:
    """SHOW_VARS() is accessible inside the RLM REPL and returns variable info."""
    from pyrlm_runtime import Context, RLM
    from pyrlm_runtime.adapters import FakeAdapter

    adapter = FakeAdapter(
        script=[
            # Step 1: create a variable and call SHOW_VARS()
            "my_answer = 'hello'\nprint(SHOW_VARS())",
            "FINAL_VAR: my_answer",
        ]
    )
    context = Context.from_text("test context")
    runtime = RLM(adapter=adapter)
    output, trace = runtime.run("Test SHOW_VARS.", context)

    assert output == "hello"
    # SHOW_VARS stdout should have appeared in the first REPL step
    repl_steps = [s for s in trace.steps if s.kind == "repl_exec"]
    assert any("my_answer" in (s.stdout or "") for s in repl_steps)
