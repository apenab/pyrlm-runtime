"""Regression tests for VertexAIAdapter transport selection.

The adapter must configure the SDK transport via ``vertexai.init`` and default
to REST (HTTP), which honors corporate proxy / system-CA settings and avoids
the gRPC pollset deadlock. ``api_transport="grpc"`` must restore gRPC. These
tests stub ``vertexai.init`` and ``GenerativeModel`` so no network or GCP
credentials are touched.
"""

from __future__ import annotations

import pytest

try:
    from pyrlm_runtime.adapters import vertex_ai

    VERTEX_AVAILABLE = True
except Exception:  # pragma: no cover - vertex deps missing
    VERTEX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not VERTEX_AVAILABLE, reason="vertexai not installed")


def _patch(monkeypatch) -> dict:
    captured: dict = {}

    def fake_init(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(vertex_ai.vertexai, "init", fake_init)
    monkeypatch.setattr(vertex_ai, "GenerativeModel", lambda name: object())
    return captured


def test_default_transport_is_rest(monkeypatch) -> None:
    captured = _patch(monkeypatch)
    vertex_ai.VertexAIAdapter(project_id="p", location="us-central1")
    assert captured["api_transport"] == "rest"
    assert captured["project"] == "p"
    assert captured["location"] == "us-central1"


def test_grpc_transport_opt_in(monkeypatch) -> None:
    captured = _patch(monkeypatch)
    vertex_ai.VertexAIAdapter(project_id="p", api_transport="grpc")
    assert captured["api_transport"] == "grpc"


def test_lazy_export_from_adapters_package() -> None:
    """VertexAIAdapter is exported lazily from pyrlm_runtime.adapters."""
    from pyrlm_runtime.adapters import VertexAIAdapter

    assert VertexAIAdapter is vertex_ai.VertexAIAdapter
