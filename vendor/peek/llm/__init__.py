from peek.llm.base import LMClient


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


__all__ = ["LMClient", "OpenAIClient", "AnthropicClient", "GeminiClient"]
