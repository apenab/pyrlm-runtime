"""Unit tests for the adapter layer."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from pyrlm_runtime.adapters import (
    AzureOpenAIAdapter,
    FakeAdapter,
    FakeRule,
    GenericChatAdapter,
    ModelResponse,
    OpenAICompatAdapter,
    Usage,
)
from pyrlm_runtime.adapters.base import estimate_usage
from pyrlm_runtime.adapters.generic_chat import (
    default_payload_builder,
    default_response_parser,
)

MESSAGES = [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# Usage & ModelResponse
# ---------------------------------------------------------------------------


class TestUsage:
    def test_from_dict_complete(self) -> None:
        usage = Usage.from_dict({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15

    def test_from_dict_partial(self) -> None:
        usage = Usage.from_dict({"prompt_tokens": 10})
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 10

    def test_from_dict_empty(self) -> None:
        usage = Usage.from_dict({})
        assert usage.total_tokens == 0

    def test_to_dict_round_trip(self) -> None:
        original = Usage(prompt_tokens=3, completion_tokens=7, total_tokens=10)
        restored = Usage.from_dict(original.to_dict())
        assert restored == original

    def test_estimate_usage(self) -> None:
        usage = estimate_usage("hello world", "ok")
        assert usage.prompt_tokens >= 1
        assert usage.completion_tokens >= 1
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


class TestModelResponse:
    def test_default_model_id_is_none(self) -> None:
        resp = ModelResponse(text="hi", usage=Usage(1, 1, 2))
        assert resp.model_id is None

    def test_model_id_preserved(self) -> None:
        resp = ModelResponse(text="hi", usage=Usage(1, 1, 2), model_id="gpt-4")
        assert resp.model_id == "gpt-4"


# ---------------------------------------------------------------------------
# Default builders / parsers
# ---------------------------------------------------------------------------


class TestDefaultPayloadBuilder:
    def test_includes_model_when_set(self) -> None:
        payload = default_payload_builder(MESSAGES, 100, 0.5, "gpt-4")
        assert payload["model"] == "gpt-4"
        assert payload["max_tokens"] == 100
        assert payload["temperature"] == 0.5

    def test_excludes_model_when_none(self) -> None:
        payload = default_payload_builder(MESSAGES, 100, 0.0, None)
        assert "model" not in payload


class TestDefaultResponseParser:
    def test_parses_openai_format(self) -> None:
        data = {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        content, usage = default_response_parser(data)
        assert content == "answer"
        assert usage is not None
        assert usage.total_tokens == 8

    def test_parses_without_usage(self) -> None:
        data = {"choices": [{"message": {"content": "answer"}}]}
        content, usage = default_response_parser(data)
        assert content == "answer"
        assert usage is None


# ---------------------------------------------------------------------------
# GenericChatAdapter
# ---------------------------------------------------------------------------


def _ok_response(content: str = "hello") -> httpx.Response:
    """Build a fake successful OpenAI-style response."""
    body = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    return httpx.Response(200, json=body)


def _error_response(status: int) -> httpx.Response:
    return httpx.Response(status, json={"error": "fail"})


class TestGenericChatAdapterInit:
    def test_requires_endpoint_or_base_url(self) -> None:
        with pytest.raises(ValueError, match="endpoint or base_url is required"):
            GenericChatAdapter()

    def test_builds_endpoint_from_base_url(self) -> None:
        adapter = GenericChatAdapter(base_url="http://localhost:8000/v1")
        assert adapter.endpoint == "http://localhost:8000/v1/chat/completions"
        adapter.close()

    def test_custom_path(self) -> None:
        adapter = GenericChatAdapter(base_url="http://localhost", path="/api/chat")
        assert adapter.endpoint == "http://localhost/api/chat"
        adapter.close()

    def test_direct_endpoint(self) -> None:
        adapter = GenericChatAdapter(endpoint="http://my-api/chat")
        assert adapter.endpoint == "http://my-api/chat"
        adapter.close()

    def test_api_key_sets_auth_header(self) -> None:
        adapter = GenericChatAdapter(endpoint="http://x", api_key="sk-test")
        assert adapter.headers["Authorization"] == "Bearer sk-test"
        adapter.close()

    def test_context_manager(self) -> None:
        with GenericChatAdapter(endpoint="http://x") as adapter:
            assert adapter.endpoint == "http://x"


class TestGenericChatAdapterComplete:
    def test_successful_request(self) -> None:
        adapter = GenericChatAdapter(endpoint="http://x", max_retries=0)
        adapter._client = httpx.Client(transport=httpx.MockTransport(lambda _: _ok_response()))
        result = adapter.complete(MESSAGES)
        assert result.text == "hello"
        assert result.usage.total_tokens == 7
        adapter.close()

    def test_model_id_in_response(self) -> None:
        adapter = GenericChatAdapter(endpoint="http://x", model="llama3", max_retries=0)
        adapter._client = httpx.Client(transport=httpx.MockTransport(lambda _: _ok_response()))
        result = adapter.complete(MESSAGES)
        assert result.model_id == "llama3"
        adapter.close()

    def test_estimates_usage_when_missing(self) -> None:
        body = {"choices": [{"message": {"content": "hi"}}]}
        resp = httpx.Response(200, json=body)
        adapter = GenericChatAdapter(endpoint="http://x", max_retries=0)
        adapter._client = httpx.Client(transport=httpx.MockTransport(lambda _: resp))
        result = adapter.complete(MESSAGES)
        assert result.usage.total_tokens > 0
        adapter.close()

    def test_retries_on_429(self) -> None:
        call_count = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return _error_response(429)
            return _ok_response()

        adapter = GenericChatAdapter(endpoint="http://x", max_retries=3)
        adapter._client = httpx.Client(transport=httpx.MockTransport(handler))
        adapter._wait = lambda _: None  # type: ignore[assignment]
        result = adapter.complete(MESSAGES)
        assert result.text == "hello"
        assert call_count == 3
        adapter.close()

    def test_retries_on_500(self) -> None:
        call_count = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _error_response(500)
            return _ok_response()

        adapter = GenericChatAdapter(endpoint="http://x", max_retries=2)
        adapter._client = httpx.Client(transport=httpx.MockTransport(handler))
        adapter._wait = lambda _: None  # type: ignore[assignment]
        result = adapter.complete(MESSAGES)
        assert result.text == "hello"
        assert call_count == 2
        adapter.close()

    def test_no_retry_on_400(self) -> None:
        adapter = GenericChatAdapter(endpoint="http://x", max_retries=3)
        adapter._client = httpx.Client(
            transport=httpx.MockTransport(lambda _: _error_response(400))
        )
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            adapter.complete(MESSAGES)
        assert exc_info.value.response.status_code == 400
        adapter.close()

    def test_no_retry_on_401(self) -> None:
        adapter = GenericChatAdapter(endpoint="http://x", max_retries=3)
        adapter._client = httpx.Client(
            transport=httpx.MockTransport(lambda _: _error_response(401))
        )
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            adapter.complete(MESSAGES)
        assert exc_info.value.response.status_code == 401
        adapter.close()

    def test_retries_exhausted_raises(self) -> None:
        adapter = GenericChatAdapter(endpoint="http://x", max_retries=2)
        adapter._client = httpx.Client(
            transport=httpx.MockTransport(lambda _: _error_response(502))
        )
        adapter._wait = lambda _: None  # type: ignore[assignment]
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            adapter.complete(MESSAGES)
        assert exc_info.value.response.status_code == 502
        adapter.close()

    def test_retries_on_timeout(self) -> None:
        call_count = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ReadTimeout("timed out")
            return _ok_response()

        adapter = GenericChatAdapter(endpoint="http://x", max_retries=2)
        adapter._client = httpx.Client(transport=httpx.MockTransport(handler))
        adapter._wait = lambda _: None  # type: ignore[assignment]
        result = adapter.complete(MESSAGES)
        assert result.text == "hello"
        assert call_count == 2
        adapter.close()

    def test_timeout_exhausted_raises(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out")

        adapter = GenericChatAdapter(endpoint="http://x", max_retries=1)
        adapter._client = httpx.Client(transport=httpx.MockTransport(handler))
        adapter._wait = lambda _: None  # type: ignore[assignment]
        with pytest.raises(httpx.ReadTimeout):
            adapter.complete(MESSAGES)
        adapter.close()

    def test_retries_on_connect_error(self) -> None:
        call_count = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("connection refused")
            return _ok_response()

        adapter = GenericChatAdapter(endpoint="http://x", max_retries=2)
        adapter._client = httpx.Client(transport=httpx.MockTransport(handler))
        adapter._wait = lambda _: None  # type: ignore[assignment]
        result = adapter.complete(MESSAGES)
        assert result.text == "hello"
        adapter.close()

    def test_malformed_json_raises_value_error(self) -> None:
        resp = httpx.Response(
            200, content=b"not json", headers={"content-type": "application/json"}
        )
        adapter = GenericChatAdapter(endpoint="http://x", max_retries=0)
        adapter._client = httpx.Client(transport=httpx.MockTransport(lambda _: resp))
        with pytest.raises(ValueError, match="Malformed JSON"):
            adapter.complete(MESSAGES)
        adapter.close()

    def test_unexpected_structure_raises_value_error(self) -> None:
        resp = httpx.Response(200, json={"unexpected": True})
        adapter = GenericChatAdapter(endpoint="http://x", max_retries=0)
        adapter._client = httpx.Client(transport=httpx.MockTransport(lambda _: resp))
        with pytest.raises(ValueError, match="Unexpected response structure"):
            adapter.complete(MESSAGES)
        adapter.close()

    def test_custom_payload_builder(self) -> None:
        captured: list[dict] = []

        def my_builder(messages, max_tokens, temperature, model):
            payload = {"prompt": messages[0]["content"], "n_predict": max_tokens}
            captured.append(payload)
            return payload

        adapter = GenericChatAdapter(endpoint="http://x", payload_builder=my_builder, max_retries=0)
        adapter._client = httpx.Client(transport=httpx.MockTransport(lambda _: _ok_response()))
        adapter.complete(MESSAGES)
        assert captured[0]["prompt"] == "hello"
        assert captured[0]["n_predict"] == 512
        adapter.close()

    def test_custom_response_parser(self) -> None:
        def my_parser(data):
            return data["result"], Usage(1, 1, 2)

        resp = httpx.Response(200, json={"result": "custom"})
        adapter = GenericChatAdapter(endpoint="http://x", response_parser=my_parser, max_retries=0)
        adapter._client = httpx.Client(transport=httpx.MockTransport(lambda _: resp))
        result = adapter.complete(MESSAGES)
        assert result.text == "custom"
        assert result.usage.total_tokens == 2
        adapter.close()


# ---------------------------------------------------------------------------
# OpenAICompatAdapter
# ---------------------------------------------------------------------------


class TestOpenAICompatAdapter:
    def test_default_base_url(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            adapter = OpenAICompatAdapter(model="gpt-4")
            assert "api.openai.com" in adapter._adapter.endpoint
            adapter._adapter.close()

    def test_env_base_url(self) -> None:
        with patch.dict("os.environ", {"LLM_BASE_URL": "http://custom:8080/v1"}):
            adapter = OpenAICompatAdapter(model="test")
            assert "custom:8080" in adapter._adapter.endpoint
            adapter._adapter.close()

    def test_env_api_key(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test123"}):
            adapter = OpenAICompatAdapter(model="gpt-4")
            assert adapter._adapter.headers["Authorization"] == "Bearer sk-test123"
            adapter._adapter.close()

    def test_explicit_base_url_overrides_env(self) -> None:
        with patch.dict("os.environ", {"LLM_BASE_URL": "http://env"}):
            adapter = OpenAICompatAdapter(model="m", base_url="http://explicit/v1")
            assert "explicit" in adapter._adapter.endpoint
            adapter._adapter.close()


# ---------------------------------------------------------------------------
# AzureOpenAIAdapter
# ---------------------------------------------------------------------------


class TestAzureOpenAIAdapter:
    def test_uses_openai_endpoint_when_present(self) -> None:
        env = {
            "AZURE_OPENAI_API_KEY": " azure-key ",
            "OPENAI_ENDPOINT": " https://demo.openai.azure.com/openai/v1 ",
        }
        with patch.dict("os.environ", env, clear=True):
            adapter = AzureOpenAIAdapter(model=" gpt-5.1 ")
            assert (
                adapter.endpoint
                == "https://demo.openai.azure.com/openai/deployments/gpt-5.1/chat/completions"
                "?api-version=2024-10-21"
            )
            assert adapter._adapter.headers["api-key"] == "azure-key"
            assert "Authorization" not in adapter._adapter.headers
            adapter.close()

    def test_builds_endpoint_from_account_name(self) -> None:
        env = {
            "AZURE_OPENAI_API_KEY": "azure-key",
            "AZURE_ACCOUNT_NAME": "my-account",
            "AZURE_OPENAI_API_VERSION": "2025-01-01-preview",
        }
        with patch.dict("os.environ", env, clear=True):
            adapter = AzureOpenAIAdapter(model="gpt-5.1")
            assert (
                adapter.endpoint
                == "https://my-account.openai.azure.com/openai/deployments/gpt-5.1/chat/completions"
                "?api-version=2025-01-01-preview"
            )
            adapter.close()

    def test_explicit_api_version_overrides_env(self) -> None:
        env = {
            "AZURE_OPENAI_API_KEY": "azure-key",
            "AZURE_ACCOUNT_NAME": "my-account",
            "AZURE_OPENAI_API_VERSION": "2025-01-01-preview",
        }
        with patch.dict("os.environ", env, clear=True):
            adapter = AzureOpenAIAdapter(model="gpt-5.1", api_version="2024-10-21")
            assert adapter.api_version == "2024-10-21"
            adapter.close()

    def test_requires_api_key(self) -> None:
        with patch.dict("os.environ", {"AZURE_ACCOUNT_NAME": "my-account"}, clear=True):
            with pytest.raises(EnvironmentError, match="Missing AZURE_OPENAI_API_KEY"):
                AzureOpenAIAdapter(model="gpt-5.1")

    def test_requires_endpoint_or_account_name(self) -> None:
        with patch.dict("os.environ", {"AZURE_OPENAI_API_KEY": "azure-key"}, clear=True):
            with pytest.raises(EnvironmentError, match="Set OPENAI_ENDPOINT or AZURE_ACCOUNT_NAME"):
                AzureOpenAIAdapter(model="gpt-5.1")

    def test_sends_azure_payload_and_parses_usage(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = dict(request.headers)
            captured["body"] = request.content.decode("utf-8")
            return _ok_response("azure hello")

        env = {
            "AZURE_OPENAI_API_KEY": "azure-key",
            "OPENAI_ENDPOINT": "https://demo.openai.azure.com",
        }
        with patch.dict("os.environ", env, clear=True):
            adapter = AzureOpenAIAdapter(model="gpt-5.1")
            adapter._adapter._client = httpx.Client(transport=httpx.MockTransport(handler))
            result = adapter.complete(MESSAGES, max_tokens=321, temperature=0.2)

            assert result.text == "azure hello"
            assert result.usage.total_tokens == 7
            assert result.model_id == "gpt-5.1"
            assert "api-key" in captured["headers"]
            assert "authorization" not in captured["headers"]
            payload = json.loads(str(captured["body"]))
            assert payload["max_completion_tokens"] == 321
            assert payload["temperature"] == 0.2
            assert "model" not in payload
            adapter.close()


# ---------------------------------------------------------------------------
# FakeAdapter
# ---------------------------------------------------------------------------


class TestFakeAdapter:
    def test_script_returns_in_order(self) -> None:
        adapter = FakeAdapter(script=["first", "second", "third"])
        assert adapter.complete(MESSAGES).text == "first"
        assert adapter.complete(MESSAGES).text == "second"
        assert adapter.complete(MESSAGES).text == "third"

    def test_empty_raises(self) -> None:
        adapter = FakeAdapter()
        with pytest.raises(RuntimeError, match="no matching rule or remaining script"):
            adapter.complete(MESSAGES)

    def test_rule_substring_match(self) -> None:
        adapter = FakeAdapter()
        adapter.add_rule("hello", "matched!")
        result = adapter.complete(MESSAGES)
        assert result.text == "matched!"

    def test_rule_no_match_falls_to_script(self) -> None:
        adapter = FakeAdapter(script=["fallback"])
        adapter.add_rule("xyz", "nope")
        result = adapter.complete(MESSAGES)
        assert result.text == "fallback"

    def test_rule_regex_match(self) -> None:
        adapter = FakeAdapter()
        adapter.add_rule(r"hel+o", "regex!", regex=True)
        result = adapter.complete(MESSAGES)
        assert result.text == "regex!"

    def test_rule_once_removed_after_use(self) -> None:
        adapter = FakeAdapter(script=["fallback"])
        adapter.add_rule("hello", "once!", once=True)
        assert adapter.complete(MESSAGES).text == "once!"
        assert adapter.complete(MESSAGES).text == "fallback"

    def test_rule_persistent_reused(self) -> None:
        adapter = FakeAdapter()
        adapter.add_rule("hello", "persistent")
        assert adapter.complete(MESSAGES).text == "persistent"
        assert adapter.complete(MESSAGES).text == "persistent"

    def test_custom_usage_in_rule(self) -> None:
        custom = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        adapter = FakeAdapter()
        adapter.add_rule("hello", "resp", usage=custom)
        result = adapter.complete(MESSAGES)
        assert result.usage == custom

    def test_estimated_usage_when_none(self) -> None:
        adapter = FakeAdapter(script=["response"])
        result = adapter.complete(MESSAGES)
        assert result.usage.total_tokens > 0

    def test_rules_from_constructor(self) -> None:
        rule = FakeRule(matcher=lambda p: "hello" in p, response="from_init")
        adapter = FakeAdapter(rules=[rule])
        assert adapter.complete(MESSAGES).text == "from_init"


# ---------------------------------------------------------------------------
# GenericChatAdapter backoff
# ---------------------------------------------------------------------------


class TestCalculateDelay:
    def test_exponential_growth(self) -> None:
        adapter = GenericChatAdapter(endpoint="http://x", retry_base_delay=1.0)
        d0 = adapter._calculate_delay(0)
        d1 = adapter._calculate_delay(1)
        d2 = adapter._calculate_delay(2)
        # With jitter, d1 should generally be larger than d0 base
        # base: 1, 2, 4 — jittered: [0.5, 1.5], [1.0, 3.0], [2.0, 6.0]
        assert 0.5 <= d0 <= 1.5
        assert 1.0 <= d1 <= 3.0
        assert 2.0 <= d2 <= 6.0
        adapter.close()

    def test_respects_max_delay(self) -> None:
        adapter = GenericChatAdapter(
            endpoint="http://x", retry_base_delay=10.0, retry_max_delay=5.0
        )
        delay = adapter._calculate_delay(5)
        # max_delay=5, jitter max 1.5x → max 7.5
        assert delay <= 7.5
        adapter.close()


class TestShouldRetry:
    def test_retryable_codes(self) -> None:
        adapter = GenericChatAdapter(endpoint="http://x")
        for code in (429, 500, 502, 503, 504):
            assert adapter._should_retry(code) is True
        adapter.close()

    def test_non_retryable_codes(self) -> None:
        adapter = GenericChatAdapter(endpoint="http://x")
        for code in (400, 401, 403, 404, 422):
            assert adapter._should_retry(code) is False
        adapter.close()
