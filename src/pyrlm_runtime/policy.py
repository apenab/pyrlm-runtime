from __future__ import annotations

import threading
from dataclasses import dataclass, field


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
    _reserved_total_tokens: int = field(default=0, repr=False, compare=False)
    _reserved_subcall_tokens: int = field(default=0, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def check_step(self) -> None:
        with self._lock:
            if self.steps >= self.max_steps:
                raise MaxStepsExceeded("max_steps exceeded")
            self.steps += 1

    def check_subcall(self, depth: int) -> None:
        with self._lock:
            if depth > self.max_recursion_depth:
                raise MaxRecursionExceeded("max_recursion_depth exceeded")
            if self.subcalls >= self.max_subcalls:
                raise MaxSubcallsExceeded("max_subcalls exceeded")
            self.subcalls += 1

    def add_tokens(self, tokens: int) -> None:
        with self._lock:
            if tokens <= 0:
                return
            if self.total_tokens + tokens > self.max_total_tokens:
                raise MaxTokensExceeded("max_total_tokens exceeded")
            self.total_tokens += tokens

    def add_subcall_tokens(self, tokens: int) -> None:
        with self._lock:
            if tokens <= 0:
                return
            new_subcall_tokens = self.subcall_tokens + tokens
            new_total_tokens = self.total_tokens + tokens
            if self.max_subcall_tokens is not None:
                if new_subcall_tokens > self.max_subcall_tokens:
                    raise MaxTokensExceeded("max_subcall_tokens exceeded")
            if new_total_tokens > self.max_total_tokens:
                raise MaxTokensExceeded("max_total_tokens exceeded")
            self.subcall_tokens = new_subcall_tokens
            self.total_tokens = new_total_tokens

    def reserve_subcall_tokens(self, tokens: int) -> None:
        with self._lock:
            if tokens <= 0:
                return
            new_reserved_subcall_tokens = self._reserved_subcall_tokens + tokens
            new_reserved_total_tokens = self._reserved_total_tokens + tokens
            if self.max_subcall_tokens is not None:
                if self.subcall_tokens + new_reserved_subcall_tokens > self.max_subcall_tokens:
                    raise MaxTokensExceeded("max_subcall_tokens exceeded")
            if self.total_tokens + new_reserved_total_tokens > self.max_total_tokens:
                raise MaxTokensExceeded("max_total_tokens exceeded")
            self._reserved_subcall_tokens = new_reserved_subcall_tokens
            self._reserved_total_tokens = new_reserved_total_tokens

    def release_subcall_tokens(self, tokens: int) -> None:
        with self._lock:
            if tokens <= 0:
                return
            if tokens > self._reserved_subcall_tokens or tokens > self._reserved_total_tokens:
                raise ValueError("cannot release more reserved subcall tokens than reserved")
            self._reserved_subcall_tokens -= tokens
            self._reserved_total_tokens -= tokens

    def finalize_subcall_tokens(self, reserved_tokens: int, actual_tokens: int) -> None:
        with self._lock:
            if reserved_tokens < 0 or actual_tokens < 0:
                raise ValueError("token counts must be non-negative")
            if reserved_tokens > self._reserved_subcall_tokens:
                raise ValueError("reserved subcall token budget underflow")
            if reserved_tokens > self._reserved_total_tokens:
                raise ValueError("reserved total token budget underflow")

            remaining_reserved_subcall_tokens = self._reserved_subcall_tokens - reserved_tokens
            remaining_reserved_total_tokens = self._reserved_total_tokens - reserved_tokens
            new_subcall_tokens = self.subcall_tokens + actual_tokens
            new_total_tokens = self.total_tokens + actual_tokens

            if self.max_subcall_tokens is not None:
                if new_subcall_tokens + remaining_reserved_subcall_tokens > self.max_subcall_tokens:
                    self._reserved_subcall_tokens = remaining_reserved_subcall_tokens
                    self._reserved_total_tokens = remaining_reserved_total_tokens
                    raise MaxTokensExceeded("max_subcall_tokens exceeded")
            if new_total_tokens + remaining_reserved_total_tokens > self.max_total_tokens:
                self._reserved_subcall_tokens = remaining_reserved_subcall_tokens
                self._reserved_total_tokens = remaining_reserved_total_tokens
                raise MaxTokensExceeded("max_total_tokens exceeded")

            self._reserved_subcall_tokens = remaining_reserved_subcall_tokens
            self._reserved_total_tokens = remaining_reserved_total_tokens
            self.subcall_tokens = new_subcall_tokens
            self.total_tokens = new_total_tokens


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)
