"""Configuration of the MCP server.

:class:`ServerConfig` holds every non-secret setting and serialises to JSON.
Secrets (the bearer token and the Elasticsearch API key) live in
:class:`Secrets`, which never serialises. :meth:`ServerConfig.from_env` and
:meth:`Secrets.from_env` are the only functions that read the environment.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from pyrlm_runtime import ElasticsearchRetriever, RetrieverProtocol

ENV_PREFIX = "PYRLM_MCP_"

Transport = Literal["stdio", "streamable-http"]
SearchMode = Literal["hybrid", "lexical", "semantic"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

ALL_SEARCH_MODES: tuple[SearchMode, ...] = ("hybrid", "lexical", "semantic")


class ConfigError(ValueError):
    """The server configuration is invalid or has nothing to publish."""


class ElasticsearchConfig(BaseModel):
    """Non-secret settings for the built-in :class:`ElasticsearchRetriever`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str
    index: str
    content_field: str = "content"
    vector_field: str = "embedding"
    embedding_provider: str = "openai"
    embedding_model: str | None = None
    embedding_base_url: str = "https://api.openai.com/v1"


class ServerConfig(BaseModel):
    """Every non-secret setting of the MCP server.

    The first entry of ``search_modes`` is the mode ``search`` uses when the
    client does not pass one. ``filterable_fields=None`` accepts any filter key
    except those in ``metadata_denylist``. ``base_filters`` are merged over the
    client's filters on every search, with the server's value winning; they
    narrow relevance and are not an access boundary, because reads by id skip
    them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "pyrlm-runtime"
    transport: Transport = "stdio"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=0, le=65535)
    allowed_hosts: tuple[str, ...] = ()
    allowed_origins: tuple[str, ...] = ()
    allow_unauthenticated: bool = False
    log_level: LogLevel = "INFO"

    elasticsearch: ElasticsearchConfig | None = None
    retriever_factory: str | None = Field(
        default=None, description='Dotted path "pkg.module:callable" returning a retriever.'
    )
    search_modes: tuple[SearchMode, ...] = ALL_SEARCH_MODES

    max_top_k: int = Field(default=50, ge=1)
    max_content_chars: int = Field(default=4_000, ge=1)
    max_document_chars: int = Field(default=20_000, ge=1)
    max_response_chars: int = Field(default=60_000, ge=1)
    max_pages: int = Field(default=20, ge=1)
    metadata_denylist: tuple[str, ...] = ()
    filterable_fields: tuple[str, ...] | None = None
    base_filters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> ServerConfig:
        if not self.search_modes:
            raise ValueError("search_modes must name at least one mode")
        if len(set(self.search_modes)) != len(self.search_modes):
            raise ValueError("search_modes must not repeat a mode")
        if self.elasticsearch is not None and self.retriever_factory is not None:
            raise ValueError("set either elasticsearch or retriever_factory, not both")
        return self

    @property
    def default_search_mode(self) -> SearchMode:
        return self.search_modes[0]

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ServerConfig:
        """Build the configuration from ``PYRLM_MCP_*`` variables.

        Lists are comma-separated; ``BASE_FILTERS`` is a JSON object. Without
        ``SEARCH_MODES``, an Elasticsearch retriever with no
        ``EMBEDDING_MODEL`` publishes only the ``lexical`` mode.

        Raises:
            ConfigError: a variable has an invalid value.
        """
        env = _Env(os.environ if environ is None else environ)
        es_host = env.text("ES_HOST")
        elasticsearch = None
        if es_host is not None:
            index = env.text("ES_INDEX")
            if index is None:
                raise ConfigError(f"{ENV_PREFIX}ES_INDEX is required with {ENV_PREFIX}ES_HOST")
            elasticsearch = ElasticsearchConfig(
                host=es_host,
                index=index,
                **env.present(
                    content_field="ES_CONTENT_FIELD",
                    vector_field="ES_VECTOR_FIELD",
                    embedding_provider="EMBEDDING_PROVIDER",
                    embedding_model="EMBEDDING_MODEL",
                    embedding_base_url="EMBEDDING_BASE_URL",
                ),
            )
        values: dict[str, Any] = env.present(
            name="NAME",
            transport="TRANSPORT",
            host="HOST",
            port="PORT",
            log_level="LOG_LEVEL",
            retriever_factory="RETRIEVER_FACTORY",
            max_top_k="MAX_TOP_K",
            max_content_chars="MAX_CONTENT_CHARS",
            max_document_chars="MAX_DOCUMENT_CHARS",
            max_response_chars="MAX_RESPONSE_CHARS",
            max_pages="MAX_PAGES",
        )
        for key, var in (
            ("allowed_hosts", "ALLOWED_HOSTS"),
            ("allowed_origins", "ALLOWED_ORIGINS"),
            ("search_modes", "SEARCH_MODES"),
            ("metadata_denylist", "METADATA_DENYLIST"),
            ("filterable_fields", "FILTERABLE_FIELDS"),
        ):
            items = env.items(var)
            if items is not None:
                values[key] = items
        if elasticsearch is not None:
            values["elasticsearch"] = elasticsearch
            if "search_modes" not in values and elasticsearch.embedding_model is None:
                values["search_modes"] = ("lexical",)
        allow_unauthenticated = env.flag("ALLOW_UNAUTHENTICATED")
        if allow_unauthenticated is not None:
            values["allow_unauthenticated"] = allow_unauthenticated
        base_filters = env.json_object("BASE_FILTERS")
        if base_filters is not None:
            values["base_filters"] = base_filters
        try:
            return cls(**values)
        except ValidationError as exc:
            raise ConfigError(str(exc)) from exc


@dataclass(frozen=True)
class Secrets:
    """Credentials kept out of :class:`ServerConfig` so it can be logged and shared."""

    bearer_token: str | None = field(default=None, repr=False)
    es_api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Secrets:
        """Read ``PYRLM_MCP_BEARER_TOKEN`` and ``PYRLM_MCP_ES_API_KEY``."""
        env = _Env(os.environ if environ is None else environ)
        return cls(bearer_token=env.text("BEARER_TOKEN"), es_api_key=env.text("ES_API_KEY"))


def load_object(dotted_path: str) -> Any:
    """Import ``"pkg.module:attribute"`` and return the attribute.

    Raises:
        ConfigError: the path is malformed or does not resolve.
    """
    module_name, sep, attribute = dotted_path.partition(":")
    if not sep or not module_name or not attribute:
        raise ConfigError(f'expected "pkg.module:attribute", got {dotted_path!r}')
    try:
        target: Any = importlib.import_module(module_name)
        for part in attribute.split("."):
            target = getattr(target, part)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(f"cannot load {dotted_path!r}: {exc}") from exc
    return target


def build_retriever(
    config: ServerConfig, secrets: Secrets | None = None
) -> RetrieverProtocol | None:
    """Build the retriever that ``config`` describes, or return ``None``.

    ``retriever_factory`` is called with no arguments. The Elasticsearch
    retriever checks that the ``elasticsearch`` package is installed here, at
    start-up, instead of on the first request.

    Raises:
        ConfigError: the factory cannot be loaded, returns something that is not
            a retriever, or the ``elasticsearch`` extra is missing.
    """
    if config.retriever_factory is not None:
        factory: Callable[[], Any] = load_object(config.retriever_factory)
        retriever = factory()
        if not isinstance(retriever, RetrieverProtocol):
            raise ConfigError(
                f"{config.retriever_factory} returned {type(retriever).__name__}, "
                "which does not implement RetrieverProtocol"
            )
        return retriever
    es = config.elasticsearch
    if es is None:
        return None
    try:
        importlib.import_module("elasticsearch")
    except ImportError as exc:
        raise ConfigError(
            'the Elasticsearch retriever needs: pip install "pyrlm-runtime[mcp,elasticsearch]"'
        ) from exc
    return ElasticsearchRetriever(
        host=es.host,
        api_key=(secrets.es_api_key if secrets else None) or "",
        index=es.index,
        content_field=es.content_field,
        vector_field=es.vector_field,
        embedding_provider=es.embedding_provider,
        embedding_model=es.embedding_model,
        embedding_base_url=es.embedding_base_url,
    )


_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})


class _Env:
    def __init__(self, environ: Mapping[str, str]) -> None:
        self._environ = environ

    def text(self, name: str) -> str | None:
        value = self._environ.get(ENV_PREFIX + name)
        if value is None or not value.strip():
            return None
        return value.strip()

    def present(self, **names: str) -> dict[str, str]:
        found = {key: self.text(var) for key, var in names.items()}
        return {key: value for key, value in found.items() if value is not None}

    def items(self, name: str) -> tuple[str, ...] | None:
        value = self.text(name)
        if value is None:
            return None
        return tuple(item.strip() for item in value.split(",") if item.strip())

    def flag(self, name: str) -> bool | None:
        value = self._environ.get(ENV_PREFIX + name)
        if value is None:
            return None
        lowered = value.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise ConfigError(f"{ENV_PREFIX}{name} must be a boolean, got {value!r}")

    def json_object(self, name: str) -> dict[str, Any] | None:
        value = self.text(name)
        if value is None:
            return None
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{ENV_PREFIX}{name} is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ConfigError(f"{ENV_PREFIX}{name} must be a JSON object")
        return parsed
