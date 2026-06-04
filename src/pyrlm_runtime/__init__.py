"""Minimal runtime for Recursive Language Models (RLMs)."""

__version__ = "1.0.0"

from .adapters.base import ModelAdapter, ModelResponse, Usage
from .cache import FileCache
from .context import Context
from .doctools import (
    DocIndexStoreProtocol,
    DocInfo,
    DocumentCache,
    DocumentPolicy,
    DocumentPolicyError,
    MaxPDFsExceeded,
    MaxPagesExceeded,
    MaxTablesExceeded,
    PageInfo,
    PageReaderProtocol,
    create_doc_tools,
)
from .env import ExecResult, PythonREPL
from .policy import (
    MaxRecursionExceeded,
    MaxStepsExceeded,
    MaxSubcallsExceeded,
    MaxTokensExceeded,
    Policy,
    PolicyError,
)
from .multiquery import QueryRewriter, union_pool
from .rerank import (
    ListwiseReranker,
    RerankerProtocol,
    TournamentReranker,
    ndcg_at_k,
    recall_at_k,
)
from .retrieval import (
    AsyncElasticsearchRetriever,
    AsyncRetrieverProtocol,
    ElasticsearchRetriever,
    RetrieverProtocol,
)
from .rlm import RLM
from .router import (
    ExecutionProfile,
    RouterConfig,
    RouterResult,
    SmartRouter,
    TraceFormatter,
)
from .trace import Trace, TraceStep

__all__ = [
    "Context",
    "DocInfo",
    "PageInfo",
    "PageReaderProtocol",
    "DocIndexStoreProtocol",
    "DocumentPolicy",
    "DocumentPolicyError",
    "MaxPDFsExceeded",
    "MaxPagesExceeded",
    "MaxTablesExceeded",
    "DocumentCache",
    "create_doc_tools",
    "AsyncElasticsearchRetriever",
    "AsyncRetrieverProtocol",
    "ElasticsearchRetriever",
    "RetrieverProtocol",
    "QueryRewriter",
    "union_pool",
    "ListwiseReranker",
    "TournamentReranker",
    "RerankerProtocol",
    "ndcg_at_k",
    "recall_at_k",
    "PythonREPL",
    "ExecResult",
    "Policy",
    "PolicyError",
    "MaxStepsExceeded",
    "MaxSubcallsExceeded",
    "MaxRecursionExceeded",
    "MaxTokensExceeded",
    "Trace",
    "TraceStep",
    "ModelAdapter",
    "ModelResponse",
    "Usage",
    "FileCache",
    "RLM",
    "SmartRouter",
    "RouterConfig",
    "RouterResult",
    "ExecutionProfile",
    "TraceFormatter",
]
