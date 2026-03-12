import pytest

from pyrlm_runtime import Policy
from pyrlm_runtime.policy import (
    MaxRecursionExceeded,
    MaxStepsExceeded,
    MaxSubcallsExceeded,
    MaxTokensExceeded,
)


def test_policy_limits() -> None:
    policy = Policy(
        max_steps=1,
        max_subcalls=1,
        max_recursion_depth=1,
        max_total_tokens=5,
        max_subcall_tokens=3,
    )

    policy.check_step()
    with pytest.raises(MaxStepsExceeded):
        policy.check_step()

    policy.check_subcall(depth=1)
    with pytest.raises(MaxSubcallsExceeded):
        policy.check_subcall(depth=1)

    with pytest.raises(MaxRecursionExceeded):
        policy.check_subcall(depth=2)

    policy.add_tokens(3)
    with pytest.raises(MaxTokensExceeded):
        policy.add_tokens(3)

    policy = Policy(max_total_tokens=10, max_subcall_tokens=3)
    policy.add_subcall_tokens(2)
    with pytest.raises(MaxTokensExceeded):
        policy.add_subcall_tokens(2)


def test_add_subcall_tokens_does_not_partially_mutate_on_total_limit() -> None:
    policy = Policy(max_total_tokens=3, max_subcall_tokens=10)
    policy.add_subcall_tokens(2)

    with pytest.raises(MaxTokensExceeded, match="max_total_tokens exceeded"):
        policy.add_subcall_tokens(2)

    assert policy.subcall_tokens == 2
    assert policy.total_tokens == 2


def test_subcall_token_reservation_and_finalize() -> None:
    policy = Policy(max_total_tokens=20, max_subcall_tokens=10)

    policy.reserve_subcall_tokens(6)
    with pytest.raises(MaxTokensExceeded, match="max_subcall_tokens exceeded"):
        policy.reserve_subcall_tokens(5)

    policy.finalize_subcall_tokens(reserved_tokens=6, actual_tokens=4)

    assert policy.subcall_tokens == 4
    assert policy.total_tokens == 4


def test_subcall_token_release_clears_reservation_after_failure() -> None:
    policy = Policy(max_total_tokens=20, max_subcall_tokens=10)

    policy.reserve_subcall_tokens(6)
    policy.release_subcall_tokens(6)
    policy.reserve_subcall_tokens(10)

    assert policy.subcall_tokens == 0
    assert policy.total_tokens == 0
