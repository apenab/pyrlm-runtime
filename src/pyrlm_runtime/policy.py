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
    # Total token budget (root + subcalls). None = unlimited. When None, the run is bounded
    # by max_steps / max_subcalls and terminates with a graceful finalization rather than a
    # MaxTokensExceeded abort.
    max_total_tokens: int | None = None
    max_subcall_tokens: int | None = None
    steps: int = 0
    subcalls: int = 0
    total_tokens: int = 0
    subcall_tokens: int = 0
    _reserved_total_tokens: int = field(default=0, init=False, repr=False, compare=False)
    _reserved_subcall_tokens: int = field(default=0, init=False, repr=False, compare=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

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

    def account_subtree(self, *, subcalls: int) -> None:
        """Roll a recursive subtree's consumed subcall count up into this policy.

        A recursive subcall runs a child ``RLM`` with its **own** ``Policy`` seeded
        from this policy's remaining budget. The child's internal subcalls are
        counted on the child policy, so after it returns we add them here to keep
        the global ``max_subcalls`` budget honest across sibling subtrees (otherwise
        two siblings would each read the same ``remaining`` and double-spend). Token
        accounting already rolls up via ``add_subcall_tokens``/``finalize_subcall_tokens``.
        """
        with self._lock:
            if subcalls <= 0:
                return
            self.subcalls += subcalls

    def add_tokens(self, tokens: int) -> None:
        with self._lock:
            if tokens <= 0:
                return
            if (
                self.max_total_tokens is not None
                and self.total_tokens + tokens > self.max_total_tokens
            ):
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
            if self.max_total_tokens is not None and new_total_tokens > self.max_total_tokens:
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
            if (
                self.max_total_tokens is not None
                and self.total_tokens + new_reserved_total_tokens > self.max_total_tokens
            ):
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
            if (
                self.max_total_tokens is not None
                and new_total_tokens + remaining_reserved_total_tokens > self.max_total_tokens
            ):
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


# ---------------------------------------------------------------------------
# Accurate token counting (tiktoken-based, with len//4 fallback)
# ---------------------------------------------------------------------------

_DEFAULT_CONTEXT_LIMIT = 128_000

# Longest-key-wins substring match on model name.
_MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "gpt-5.5": 272_000,
    "gpt-5.4-mini": 272_000,
    "gpt-5.4": 272_000,
    "gpt-5.2": 272_000,
    "gpt-5-nano": 272_000,
    "gpt-5": 272_000,
    "gpt-4o-mini": 128_000,
    "gpt-4o-2024": 128_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4-32k": 32_768,
    "gpt-4": 8_192,
    "gpt-3.5-turbo-16k": 16_385,
    "gpt-3.5-turbo": 16_385,
    "o1-mini": 128_000,
    "o1-preview": 128_000,
    "o1": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "gemini-2.5-flash": 1_000_000,
    "gemini-2.5-pro": 1_000_000,
    "gemini-2.0-flash": 1_000_000,
    "gemini-1.5-pro": 1_000_000,
    "gemini-1.5-flash": 1_000_000,
}


def get_context_limit(model_name: str) -> int:
    """Return context window size in tokens for ``model_name``.

    Uses longest-key substring match.  Falls back to 128K for unknown models.
    """
    if not model_name:
        return _DEFAULT_CONTEXT_LIMIT
    exact = _MODEL_CONTEXT_LIMITS.get(model_name)
    if exact is not None:
        return exact
    best_len, best_limit = 0, _DEFAULT_CONTEXT_LIMIT
    for key, limit in _MODEL_CONTEXT_LIMITS.items():
        if key in model_name and len(key) > best_len:
            best_len, best_limit = len(key), limit
    return best_limit


def count_tokens(messages: list[dict], model_name: str = "") -> int:
    """Count tokens in a list of ``{role, content}`` message dicts.

    Uses tiktoken when the package is installed and ``model_name`` is
    recognisable; otherwise falls back to ``len(content) // 4`` per message.
    This is a significant improvement over calling ``estimate_tokens`` per
    message for dense content (math, code, JSON) where tiktoken ratios deviate
    far from the 4-chars-per-token heuristic.

    Note: tiktoken only ships exact encodings for OpenAI models. For non-OpenAI
    models (Claude, Gemini) there is no native encoding, so the count uses the
    ``cl100k_base`` encoding as an approximation — close enough for budgeting
    decisions but not an exact provider-side token count.
    """
    if not messages:
        return 0
    if model_name:
        n = _count_tokens_tiktoken(messages, model_name)
        if n is not None:
            return n
    # Fallback: sum of per-message char estimates.
    total = 0
    for m in messages:
        raw = m.get("content") or ""
        total += estimate_tokens(raw if isinstance(raw, str) else str(raw))
    return total


def _count_tokens_tiktoken(messages: list[dict], model_name: str) -> int | None:
    """Return tiktoken count for ``messages`` or ``None`` if unavailable."""
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None
    # OpenAI message-format overhead (3 tokens/message + reply primer).
    total = 3  # reply primer
    for m in messages:
        total += 3  # per-message overhead
        content = m.get("content")
        if isinstance(content, str):
            total += len(enc.encode(content))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(enc.encode(part.get("text", "") or ""))
        elif content is not None:
            total += len(enc.encode(str(content)))
        if m.get("name"):
            total += 1
    return total
