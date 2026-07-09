"""PEEK: Context Map as an Orientation Cache for Long-Context LLM Agents."""

from peek.core import (
    SECTIONS,
    CachePolicy,
    Cartographer,
    CartographerOutput,
    ContextMap,
    Distiller,
    DistillerOutput,
    Item,
    ItemTag,
    Operation,
    UpdateResult,
    Usage,
    evict,
    update_scores,
)
from peek.llm import LMClient

__version__ = "0.1.0"

__all__ = [
    "CachePolicy",
    "Cartographer",
    "CartographerOutput",
    "ContextMap",
    "Distiller",
    "DistillerOutput",
    "Item",
    "ItemTag",
    "LMClient",
    "Operation",
    "SECTIONS",
    "UpdateResult",
    "Usage",
    "__version__",
    "evict",
    "update_scores",
]


def __getattr__(name: str):
    if name == "OpenAIClient":
        from peek.llm.openai_client import OpenAIClient

        return OpenAIClient
    if name == "AnthropicClient":
        from peek.llm.anthropic_client import AnthropicClient

        return AnthropicClient
    if name == "GeminiClient":
        from peek.llm.gemini_client import GeminiClient

        return GeminiClient
    raise AttributeError(name)
