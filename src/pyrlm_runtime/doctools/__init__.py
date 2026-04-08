"""Document intelligence tools for on-demand PDF access in the RLM REPL."""

from .cache import DocumentCache
from .policy import (
    DocumentPolicy,
    DocumentPolicyError,
    MaxPDFsExceeded,
    MaxPagesExceeded,
    MaxTablesExceeded,
)
from .protocols import DocIndexStoreProtocol, PageReaderProtocol
from .schema import DocInfo, PageInfo
from .tools import DEFAULT_LABELS, create_doc_tools

__all__ = [
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
    "DEFAULT_LABELS",
    "create_doc_tools",
]
