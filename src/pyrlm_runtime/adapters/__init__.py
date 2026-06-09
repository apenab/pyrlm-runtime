from __future__ import annotations

from typing import Any

from .base import ModelAdapter, ModelResponse, Usage
from .azure_openai import AzureOpenAIAdapter
from .fake import FakeAdapter, FakeRule
from .generic_chat import GenericChatAdapter
from .openai_compat import OpenAICompatAdapter

__all__ = [
    "ModelAdapter",
    "ModelResponse",
    "Usage",
    "AzureOpenAIAdapter",
    "FakeAdapter",
    "FakeRule",
    "GenericChatAdapter",
    "OpenAICompatAdapter",
    "VertexAIAdapter",
]


def __getattr__(name: str) -> Any:
    # Lazy export: VertexAIAdapter pulls in google-cloud-aiplatform / vertexai,
    # which are optional. Import it on demand so `import pyrlm_runtime.adapters`
    # does not require the Google SDKs for consumers that don't use Vertex.
    if name == "VertexAIAdapter":
        from .vertex_ai import VertexAIAdapter

        return VertexAIAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
