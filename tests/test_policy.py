import pytest

from rlm_runtime import Policy
from rlm_runtime.policy import (
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
