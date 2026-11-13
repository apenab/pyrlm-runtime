from __future__ import annotations

import importlib.util
import json

import pytest

pytest.importorskip("mcp")

from _mcp_fakes import PagedRetriever  # noqa: E402

from pyrlm_runtime.mcp_server.config import (  # noqa: E402
    ConfigError,
    Secrets,
    ServerConfig,
    build_retriever,
)

P = "PYRLM_MCP_"


class TestFromEnv:
    def test_defaults_with_empty_environment(self):
        config = ServerConfig.from_env({})

        assert config == ServerConfig()
        assert config.transport == "stdio"
        assert config.host == "127.0.0.1"
        assert config.default_search_mode == "hybrid"

    def test_parses_lists_booleans_numbers_and_json(self):
        config = ServerConfig.from_env(
            {
                P + "TRANSPORT": "streamable-http",
                P + "PORT": "9001",
                P + "ALLOWED_HOSTS": "mcp.example.com, mcp.example.com:*",
                P + "ALLOW_UNAUTHENTICATED": "1",
                P + "SEARCH_MODES": "lexical,hybrid",
                P + "MAX_TOP_K": "7",
                P + "METADATA_DENYLIST": "ssn",
                P + "FILTERABLE_FIELDS": "year,doc_id",
                P + "BASE_FILTERS": '{"tenant": "acme"}',
            }
        )

        assert config.transport == "streamable-http"
        assert config.port == 9001
        assert config.allowed_hosts == ("mcp.example.com", "mcp.example.com:*")
        assert config.allow_unauthenticated is True
        assert config.search_modes == ("lexical", "hybrid")
        assert config.default_search_mode == "lexical"
        assert config.max_top_k == 7
        assert config.metadata_denylist == ("ssn",)
        assert config.filterable_fields == ("year", "doc_id")
        assert config.base_filters == {"tenant": "acme"}

    @pytest.mark.parametrize(
        "name, value",
        [
            ("TRANSPORT", "sse"),
            ("PORT", "eighty"),
            ("SEARCH_MODES", "fuzzy"),
            ("ALLOW_UNAUTHENTICATED", "maybe"),
            ("BASE_FILTERS", "[1, 2]"),
            ("BASE_FILTERS", "{not json"),
            ("MAX_TOP_K", "0"),
        ],
    )
    def test_invalid_values_raise_config_error(self, name, value):
        with pytest.raises(ConfigError):
            ServerConfig.from_env({P + name: value})

    def test_empty_filterable_fields_allows_no_filter_key(self):
        assert ServerConfig.from_env({P + "FILTERABLE_FIELDS": ""}).filterable_fields == ()

    def test_elasticsearch_without_embedding_model_publishes_lexical_only(self):
        config = ServerConfig.from_env({P + "ES_HOST": "http://es:9200", P + "ES_INDEX": "docs"})

        assert config.elasticsearch is not None
        assert config.elasticsearch.index == "docs"
        assert config.search_modes == ("lexical",)

    def test_elasticsearch_with_embedding_model_keeps_every_mode(self):
        config = ServerConfig.from_env(
            {
                P + "ES_HOST": "http://es:9200",
                P + "ES_INDEX": "docs",
                P + "EMBEDDING_MODEL": "text-embedding-3-small",
            }
        )

        assert config.search_modes == ("hybrid", "lexical", "semantic")

    def test_es_host_without_index_is_rejected(self):
        with pytest.raises(ConfigError, match="ES_INDEX"):
            ServerConfig.from_env({P + "ES_HOST": "http://es:9200"})

    def test_elasticsearch_and_factory_are_exclusive(self):
        with pytest.raises(ConfigError):
            ServerConfig.from_env(
                {
                    P + "ES_HOST": "http://es:9200",
                    P + "ES_INDEX": "docs",
                    P + "RETRIEVER_FACTORY": "_mcp_fakes:make_retriever",
                }
            )


class TestSecrets:
    ENV = {
        P + "BEARER_TOKEN": "tok-SECRET",
        P + "ES_API_KEY": "es-SECRET",
        P + "ES_HOST": "http://es:9200",
        P + "ES_INDEX": "docs",
    }

    def test_config_round_trips_through_json_without_secrets(self):
        config = ServerConfig.from_env(self.ENV)

        dumped = config.model_dump_json()

        assert "SECRET" not in dumped
        assert ServerConfig.model_validate(json.loads(dumped)) == config

    def test_secrets_are_read_but_not_printed(self):
        secrets = Secrets.from_env(self.ENV)

        assert secrets.bearer_token == "tok-SECRET"
        assert secrets.es_api_key == "es-SECRET"
        assert "SECRET" not in repr(secrets)


class TestBuildRetriever:
    def test_nothing_configured_returns_none(self):
        assert build_retriever(ServerConfig()) is None

    def test_factory_builds_the_retriever(self):
        retriever = build_retriever(ServerConfig(retriever_factory="_mcp_fakes:make_retriever"))

        assert isinstance(retriever, PagedRetriever)

    @pytest.mark.parametrize(
        "path", ["no_colon", "_mcp_fakes:missing", "missing_module_xyz:thing", ":x"]
    )
    def test_bad_factory_path_raises_config_error(self, path):
        with pytest.raises(ConfigError):
            build_retriever(ServerConfig(retriever_factory=path))

    def test_factory_returning_a_non_retriever_is_rejected(self):
        with pytest.raises(ConfigError, match="RetrieverProtocol"):
            build_retriever(ServerConfig(retriever_factory="_mcp_fakes:make_not_a_retriever"))

    @pytest.mark.skipif(
        importlib.util.find_spec("elasticsearch") is not None,
        reason="elasticsearch is installed",
    )
    def test_elasticsearch_without_the_extra_fails_at_startup(self):
        config = ServerConfig.from_env({P + "ES_HOST": "http://es:9200", P + "ES_INDEX": "docs"})

        with pytest.raises(ConfigError, match=r"pyrlm-runtime\[mcp,elasticsearch\]"):
            build_retriever(config)


def test_cli_missing_env_file_exits_with_an_error(tmp_path, capsys):
    from pyrlm_runtime.mcp_server.__main__ import main

    assert main(["--env-file", str(tmp_path / "missing.env")]) == 2
    assert "env file not found" in capsys.readouterr().err
