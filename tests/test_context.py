import pytest

from rlm_runtime import Context


def test_len_and_slice() -> None:
    ctx = Context.from_text("hello world")
    assert ctx.len_chars() == 11
    assert ctx.slice(0, 5) == "hello"
    assert ctx.slice(-5, 5) == "hello"
    assert ctx.slice(6, 50) == "world"
    assert ctx.slice(10, 5) == ""


def test_find_literal_and_regex() -> None:
    ctx = Context.from_text("aba abb aab")
    assert ctx.find("ab")[:2] == [(0, 2, "ab"), (4, 6, "ab")]
    matches = ctx.find(r"a.b", regex=True, max_matches=10)
    assert matches[0][2] == "abb"


def test_chunking() -> None:
    ctx = Context.from_text("abcdefghij")
    chunks = ctx.chunk(4, overlap=1)
    assert chunks[0] == (0, 4, "abcd")
    assert chunks[1] == (3, 7, "defg")
    assert chunks[-1][2] == "ghij"

    with pytest.raises(ValueError):
        ctx.chunk(0)
    with pytest.raises(ValueError):
        ctx.chunk(4, overlap=4)
