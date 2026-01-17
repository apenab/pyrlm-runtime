from rlm_runtime import PythonREPL


def test_repl_persistence_and_restrictions() -> None:
    repl = PythonREPL(stdout_limit=1000)
    repl.exec("x = 5")
    result = repl.exec("print(x + 1)")
    assert result.stdout.strip() == "6"

    blocked = repl.exec("import os")
    assert blocked.error is not None

    blocked_open = repl.exec("open('tmp.txt', 'w')")
    assert blocked_open.error is not None
