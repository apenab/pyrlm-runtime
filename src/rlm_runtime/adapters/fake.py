from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Iterable

from .base import ModelAdapter, ModelResponse, Usage, estimate_usage


@dataclass
class FakeRule:
    matcher: Callable[[str], bool]
    response: str
    once: bool = False
    usage: Usage | None = None


class FakeAdapter(ModelAdapter):
    """Deterministic adapter for tests and examples."""

    def __init__(
        self,
        *,
        rules: Iterable[FakeRule] | None = None,
        script: Iterable[str] | None = None,
    ) -> None:
        self._rules = list(rules or [])
        self._script = list(script or [])

    def add_rule(
        self,
        pattern: str,
        response: str,
        *,
        regex: bool = False,
        once: bool = False,
        usage: Usage | None = None,
    ) -> None:
        if regex:

            def matcher(prompt: str) -> bool:
                return re.search(pattern, prompt) is not None
        else:

            def matcher(prompt: str) -> bool:
                return pattern in prompt

        self._rules.append(FakeRule(matcher=matcher, response=response, once=once, usage=usage))

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ModelResponse:
        prompt = "\n".join(msg.get("content", "") for msg in messages)

        for rule in list(self._rules):
            if rule.matcher(prompt):
                if rule.once:
                    self._rules.remove(rule)
                usage = rule.usage or estimate_usage(prompt, rule.response)
                return ModelResponse(text=rule.response, usage=usage)

        if self._script:
            response = self._script.pop(0)
            usage = estimate_usage(prompt, response)
            return ModelResponse(text=response, usage=usage)

        raise RuntimeError("FakeAdapter has no matching rule or remaining script")
