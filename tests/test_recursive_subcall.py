"""Differential tests for recursive subcalls: the new child-RLM path
(``recursion_impl="child"``) vs the legacy fork (``recursion_impl="fork"``).

The child path re-enters the real ``RLM.run()`` loop as a child RLM, so a
recursive subcall has the full toolset (``llm_batch``/parallel, deeper
recursion) and shares the budget/cache with the parent. The fork path runs a
trimmed mini-loop that cannot do any of that — these tests pin the difference.
"""
from __future__ import annotations

import pytest

from pyrlm_runtime import RLM, Context, Policy
from pyrlm_runtime.adapters.fake import FakeAdapter, FakeRule
from pyrlm_runtime.policy import MaxTokensExceeded
from pyrlm_runtime.prompts import (
    BASE_SYSTEM_PROMPT,
    RECURSIVE_SUBCALL_SYSTEM_PROMPT,
    SUBCALL_SYSTEM_PROMPT,
)

# Distinctive, mutually-exclusive opening slices of each system prompt — used to
# tell which agent (root / recursive child / single-shot leaf) is calling.
ROOT = BASE_SYSTEM_PROMPT[:50]
CHILD = RECURSIVE_SUBCALL_SYSTEM_PROMPT[:50]
LEAF = SUBCALL_SYSTEM_PROMPT[:50]
# Phase markers within an agent's own loop.
INIT = "have not interacted"   # build_root_user_message (step 1)
ITER = "[REPL Result]"          # build_iteration_message (step >= 2)


def _is_root(p: str) -> bool:
    return ROOT in p and CHILD not in p and LEAF not in p


# With conversation_history on, the initial user message (containing INIT) stays
# in the prompt on every turn, so "first turn" must mean INIT present AND no
# iteration result yet (ITER absent). "later turn" means ITER present.
def _first(agent) -> object:
    return lambda p: agent(p) and INIT in p and ITER not in p


def _later(agent) -> object:
    return lambda p: agent(p) and ITER in p


def _is_child(p: str) -> bool:
    return CHILD in p


def _rule(pred, response: str) -> FakeRule:
    return FakeRule(matcher=pred, response=response)


def _run(adapter: FakeAdapter, *, tmp_path, **rlm_kwargs):
    rlm = RLM(adapter=adapter, cache_dir=tmp_path / "cache", **rlm_kwargs)
    return rlm.run("ROOTQ", Context.from_text("ROOT_CTX"))


# Root always: emit a recursive subcall, store its result, then finalize.
ROOT_RULES = [
    _rule(_first(_is_root), "res = ask('CHILDQ', 'SNIP')\nanswer = res"),
    _rule(_later(_is_root), "FINAL_VAR: answer"),
]


def test_child_recursion_runs_full_loop_with_batch(tmp_path) -> None:
    """child path: a recursive subcall can itself run ``llm_batch`` (parallel
    fan-out) and a real multi-step loop — impossible in the fork."""
    rules = ROOT_RULES + [
        # Child runs a real loop: batch two grandchild calls, then finalize.
        _rule(_first(_is_child),
              "parts = llm_batch(['GQ1', 'GQ2'])\nanswer = parts[0]"),
        _rule(_later(_is_child), "FINAL_VAR: answer"),
        # Leaf grandchildren (single-shot at max depth).
        _rule(lambda p: LEAF in p and "GQ1" in p, "LEAF1"),
        _rule(lambda p: LEAF in p and "GQ2" in p, "LEAF2"),
    ]
    output, trace = _run(
        FakeAdapter(rules=rules), tmp_path=tmp_path,
        recursive_subcalls=True, recursion_impl="child", max_recursion_depth=2,
    )
    kinds = {s.kind for s in trace.steps}
    assert output == "LEAF1"
    assert "recursive_subcall" in kinds          # root spawned a child RLM
    assert "sub_repl_exec" in kinds              # child ran real REPL code
    assert "sub_subcall" in kinds                # child fanned out to grandchildren
    assert max((s.depth or 0) for s in trace.steps) >= 2


def test_child_subtree_respects_global_budget(tmp_path) -> None:
    """Anti-leak: subcalls made *inside* a recursive subtree count against the
    global ``max_subcalls`` (rolled up into the parent policy)."""
    policy = Policy(max_subcalls=2, max_steps=40)
    rules = ROOT_RULES + [
        _rule(_first(_is_child),
              "a = llm_query('GQ1')\nb = llm_query('GQ2')\nanswer = a"),
        _rule(_later(_is_child), "FINAL_VAR: answer"),
        _rule(lambda p: LEAF in p and "GQ1" in p, "LEAF1"),
        _rule(lambda p: LEAF in p and "GQ2" in p, "LEAF2"),
    ]
    _run(
        FakeAdapter(rules=rules), tmp_path=tmp_path, policy=policy,
        recursive_subcalls=True, recursion_impl="child", max_recursion_depth=2,
    )
    # Root recursive call (1) + one grandchild that fit the budget, rolled up (→2).
    # The second grandchild is blocked by the shared budget.
    assert policy.subcalls == 2


def test_fork_subtree_leaks_global_budget(tmp_path) -> None:
    """Legacy behavior (documented): the fork's internal subcalls bypass the
    policy, so the global counter under-counts the subtree. This is the leak the
    child path fixes — kept as an explicit guard against accidental reuse."""
    policy = Policy(max_subcalls=2, max_steps=40)
    rules = ROOT_RULES + [
        _rule(_first(_is_child),
              "a = llm_query('GQ1')\nb = llm_query('GQ2')\nanswer = a"),
        _rule(_later(_is_child), "FINAL_VAR: answer"),
        _rule(lambda p: LEAF in p and "GQ1" in p, "LEAF1"),
        _rule(lambda p: LEAF in p and "GQ2" in p, "LEAF2"),
    ]
    _run(
        FakeAdapter(rules=rules), tmp_path=tmp_path, policy=policy,
        recursive_subcalls=True, recursion_impl="fork", max_recursion_depth=2,
    )
    # Only the root recursive call is counted; the two internal calls leak.
    assert policy.subcalls == 1


def test_child_recursion_shares_cache(tmp_path) -> None:
    """Two identical recursive subcalls: the second is served from the shared
    cache (cache_hit) instead of re-running the child loop."""
    rules = [
        # Root issues the SAME recursive subcall twice, then finalizes.
        _rule(_first(_is_root),
              "x = ask('CHILDQ', 'SNIP')\ny = ask('CHILDQ', 'SNIP')\nanswer = x"),
        _rule(_later(_is_root), "FINAL_VAR: answer"),
        _rule(_first(_is_child), "answer = 'CVAL'"),
        _rule(_later(_is_child), "FINAL_VAR: answer"),
    ]
    output, trace = _run(
        FakeAdapter(rules=rules), tmp_path=tmp_path,
        recursive_subcalls=True, recursion_impl="child", max_recursion_depth=2,
    )
    assert output == "CVAL"
    assert sum(1 for s in trace.steps if s.kind == "recursive_subcall") == 1
    assert any(s.cache_hit for s in trace.steps)


def test_depth_bound_falls_back_to_single_shot(tmp_path) -> None:
    """At ``max_recursion_depth`` the subcall is single-shot (the leaf): no child
    RLM is spawned and recursion terminates."""
    rules = ROOT_RULES + [
        _rule(lambda p: LEAF in p and "CHILDQ" in p, "LEAFANS"),
    ]
    output, trace = _run(
        FakeAdapter(rules=rules), tmp_path=tmp_path,
        recursive_subcalls=True, recursion_impl="child", max_recursion_depth=1,
    )
    kinds = {s.kind for s in trace.steps}
    assert output == "LEAFANS"
    assert "recursive_subcall" not in kinds
    assert "subcall" in kinds
    assert max((s.depth or 0) for s in trace.steps) == 1


def test_default_no_recursion_keeps_single_shot(tmp_path) -> None:
    """``recursive_subcalls=False`` (default): ``ask``/``llm_query`` are plain
    single-shot subcalls regardless of depth budget."""
    rules = ROOT_RULES + [
        _rule(lambda p: LEAF in p and "CHILDQ" in p, "LEAFANS"),
    ]
    output, trace = _run(
        FakeAdapter(rules=rules), tmp_path=tmp_path,
        recursive_subcalls=False, max_recursion_depth=2,
    )
    kinds = {s.kind for s in trace.steps}
    assert output == "LEAFANS"
    assert "recursive_subcall" not in kinds
    assert "subcall" in kinds


@pytest.mark.parametrize("impl", ["child", "fork"])
def test_both_impls_terminate(tmp_path, impl) -> None:
    """Sanity: both implementations bound recursion and return the leaf answer."""
    rules = ROOT_RULES + [
        _rule(_first(_is_child), "answer = llm_query('GQ1')"),
        _rule(_later(_is_child), "FINAL_VAR: answer"),
        _rule(lambda p: LEAF in p and "GQ1" in p, "LEAF1"),
    ]
    output, _ = _run(
        FakeAdapter(rules=rules), tmp_path=tmp_path,
        recursive_subcalls=True, recursion_impl=impl, max_recursion_depth=2,
    )
    assert output == "LEAF1"


class _ChildBudgetRaisingAdapter(FakeAdapter):
    """Raises MaxTokensExceeded on the recursive child's LLM call, simulating a
    child that exhausts the shared budget mid-run (the exception then escapes
    ``child.run()``)."""

    def complete(self, messages, *, max_tokens=512, temperature=0.0):
        prompt = "\n".join(m.get("content", "") for m in messages)
        if _is_child(prompt):
            raise MaxTokensExceeded("max_total_tokens exceeded")
        return super().complete(messages, max_tokens=max_tokens, temperature=temperature)


def test_recursive_child_budget_exhaustion_does_not_abort_parent_cell(tmp_path) -> None:
    """A recursive child that runs out of budget must NOT propagate the exception
    into the parent's REPL exec (which would abort the whole cell and discard the
    parent's own work). The parent's ``ask()`` should receive a [SUBCALL_LIMIT]
    sentinel so the statement after it still runs.
    """
    # Root: ask() (will recurse + blow up), then a SECOND statement that must run
    # iff ask() returned instead of raising. Finalize from that second statement.
    rules = [
        _rule(_first(_is_root),
              "res = ask('CHILDQ', 'SNIP')\nsurvived = 'OK:' + res[:15]"),
        _rule(_later(_is_root), "FINAL_VAR: survived"),
        # Child rules are never reached — the adapter raises on the child call.
    ]
    output, trace = _run(
        _ChildBudgetRaisingAdapter(rules=rules), tmp_path=tmp_path,
        recursive_subcalls=True, recursion_impl="child", max_recursion_depth=2,
    )
    # The cell survived: `survived` was assigned from the sentinel, so the answer
    # carries the OK prefix and the SUBCALL_LIMIT marker (not a NO_ANSWER / crash).
    assert output.startswith("OK:")
    assert "SUBCALL_LIMIT" in output
    # The truncation is visible in the trace as a recursive_subcall with an error.
    rec = [s for s in trace.steps if s.kind == "recursive_subcall"]
    assert rec and rec[0].error is not None


def test_child_does_not_inherit_parent_system_prompt_supplement(tmp_path) -> None:
    """The recursive child has retriever=None / doc_tools=None, so a parent's
    retrieval-oriented ``system_prompt_supplement`` must NOT leak into it — else
    the child is told to call es_* tools that do not exist (observed in
    autodoc-rlm: NameError in the child's REPL). The child reasons over P only.
    """
    LEAK = "ZZZ_RETRIEVAL_SUPPLEMENT_LEAK_MARKER_call_es_find_pages_text"
    rules = ROOT_RULES + [
        _rule(_first(_is_child), "answer = llm_query('GQ1')"),
        _rule(_later(_is_child), "FINAL_VAR: answer"),
        _rule(lambda p: LEAF in p and "GQ1" in p, "LEAF1"),
    ]
    adapter = FakeAdapter(rules=rules)
    output, _ = _run(
        adapter, tmp_path=tmp_path,
        recursive_subcalls=True, recursion_impl="child", max_recursion_depth=2,
        system_prompt_supplement="\n\n## RETRIEVAL\n" + LEAK,
    )
    assert output == "LEAF1"

    # call_log holds every messages list (root + child + leaf share the adapter).
    system_prompts = [
        msgs[0]["content"] for msgs in adapter.call_log
        if msgs and msgs[0].get("role") == "system"
    ]
    child_prompts = [p for p in system_prompts if _is_child(p)]
    root_prompts = [p for p in system_prompts if _is_root(p)]

    assert child_prompts, "expected at least one recursive-child LLM call"
    # Sanity: the supplement DOES reach the root (otherwise the test is vacuous).
    assert any(LEAK in p for p in root_prompts)
    # The fix: it must NOT reach any child.
    assert all(LEAK not in p for p in child_prompts)
