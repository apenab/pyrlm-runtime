from __future__ import annotations

from dataclasses import dataclass


class PolicyError(RuntimeError):
    pass


class MaxStepsExceeded(PolicyError):
    pass


class MaxSubcallsExceeded(PolicyError):
    pass


class MaxRecursionExceeded(PolicyError):
    pass


class MaxTokensExceeded(PolicyError):
    pass


@dataclass
class Policy:
    max_steps: int = 40
    max_subcalls: int = 200
    max_recursion_depth: int = 1
    max_total_tokens: int = 200_000
    max_subcall_tokens: int | None = None
    steps: int = 0
    subcalls: int = 0
    total_tokens: int = 0
    subcall_tokens: int = 0

    def check_step(self) -> None:
        if self.steps >= self.max_steps:
            raise MaxStepsExceeded("max_steps exceeded")
        self.steps += 1

    def check_subcall(self, depth: int) -> None:
        if depth > self.max_recursion_depth:
            raise MaxRecursionExceeded("max_recursion_depth exceeded")
        if self.subcalls >= self.max_subcalls:
            raise MaxSubcallsExceeded("max_subcalls exceeded")
        self.subcalls += 1

    def add_tokens(self, tokens: int) -> None:
        if tokens <= 0:
            return
        if self.total_tokens + tokens > self.max_total_tokens:
            raise MaxTokensExceeded("max_total_tokens exceeded")
        self.total_tokens += tokens

    def add_subcall_tokens(self, tokens: int) -> None:
        if tokens <= 0:
            return
        if self.max_subcall_tokens is not None:
            if self.subcall_tokens + tokens > self.max_subcall_tokens:
                raise MaxTokensExceeded("max_subcall_tokens exceeded")
        self.subcall_tokens += tokens
        self.add_tokens(tokens)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)
