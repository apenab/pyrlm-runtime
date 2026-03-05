"""Test suite for Vertex AI adapters.

This script provides basic smoke tests for the Vertex AI adapters to ensure
they work correctly before running the full benchmark.

Prerequisites:
    - GCP authentication configured (gcloud auth application-default login)
    - Project set to go-agl-poc-radax-p01-poc
    - Required dependencies installed (google-cloud-aiplatform, vertexai)

Usage:
    # Test pyrlm_runtime adapter only
    python examples/test_vertex_adapters.py --adapter pyrlm

    # Test rlm-minimal adapter only
    python examples/test_vertex_adapters.py --adapter rlm-minimal

    # Test both (default)
    python examples/test_vertex_adapters.py
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_pyrlm_adapter(project_id: str, model: str):
    """Test VertexAIAdapter for pyrlm_runtime.

    Args:
        project_id: GCP project ID
        model: Gemini model name

    Returns:
        True if test passes, False otherwise
    """
    logger.info("=" * 70)
    logger.info("Testing VertexAIAdapter (pyrlm_runtime)")
    logger.info("=" * 70)

    try:
        from pyrlm_runtime.adapters.vertex_ai import VertexAIAdapter

        # Initialize adapter
        logger.info(f"Initializing adapter: project={project_id}, model={model}")
        adapter = VertexAIAdapter(
            project_id=project_id,
            model=model,
            logger_instance=logger,
        )

        # Test 1: Simple completion
        logger.info("\nTest 1: Simple user message")
        messages = [{"role": "user", "content": "Say hello to me!"}]
        response = adapter.complete(messages, max_tokens=50, temperature=0.0)

        logger.info(f"Response text: {response.text}")
        logger.info(f"Tokens: {response.usage.total_tokens}")
        logger.info(f"Model: {response.model_id}")

        assert response.text is not None, "Response text is None"
        assert "hello" in response.text.lower(), f"Expected 'hello' in response: {response.text}"
        logger.info("✓ Test 1 passed")

        # Test 2: System prompt + user message
        logger.info("\nTest 2: System + user messages")
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        response = adapter.complete(messages, max_tokens=50, temperature=0.0)

        logger.info(f"Response text: {response.text}")
        logger.info(f"Tokens: {response.usage.total_tokens}")

        assert response.text is not None, "Response text is None"
        assert "4" in response.text, f"Expected '4' in response: {response.text}"
        logger.info("✓ Test 2 passed")

        # Test 3: Multi-turn conversation
        logger.info("\nTest 3: Multi-turn conversation")
        messages = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
            {"role": "user", "content": "What is my name?"},
        ]
        response = adapter.complete(messages, max_tokens=50, temperature=0.0)

        logger.info(f"Response text: {response.text}")
        logger.info(f"Tokens: {response.usage.total_tokens}")

        assert response.text is not None, "Response text is None"
        assert "alice" in response.text.lower(), f"Expected 'Alice' in response: {response.text}"
        logger.info("✓ Test 3 passed")

        logger.info("\n" + "=" * 70)
        logger.info("✅ All VertexAIAdapter tests passed!")
        logger.info("=" * 70)
        return True

    except Exception as exc:
        logger.error(f"\n❌ VertexAIAdapter test failed: {exc}", exc_info=True)
        return False


def test_rlm_minimal_adapter(project_id: str, model: str):
    """Test VertexAIClientForRLMMinimal and InstrumentedVertexAIClient.

    Args:
        project_id: GCP project ID
        model: Gemini model name

    Returns:
        True if test passes, False otherwise
    """
    logger.info("=" * 70)
    logger.info("Testing VertexAIClientForRLMMinimal (rlm-minimal adapter)")
    logger.info("=" * 70)

    try:
        from vertex_adapter_for_rlm_minimal import (
            InstrumentedVertexAIClient,
            VertexAIClientForRLMMinimal,
        )

        # Test 1: Basic client (non-instrumented)
        logger.info(f"\nTest 1: Basic VertexAIClientForRLMMinimal")
        client = VertexAIClientForRLMMinimal(
            project_id=project_id,
            model=model,
            logger_instance=logger,
        )

        # Test string input
        response = client.completion("What is machine learning?", max_tokens=50)
        logger.info(f"Response: {response}")
        assert response is not None, "Response is None"
        assert isinstance(response, str), f"Expected str, got {type(response)}"
        assert len(response) > 3, f"Response too short: {response}"
        logger.info("✓ Test 1a passed (string input)")

        # Test message list input
        messages = [{"role": "user", "content": "What is 3+3?"}]
        response = client.completion(messages, max_tokens=50, temperature=0.0)
        logger.info(f"Response: {response}")
        assert "6" in response, f"Expected '6' in response: {response}"
        logger.info("✓ Test 1b passed (message list input)")

        # Test 2: Instrumented client
        logger.info("\nTest 2: InstrumentedVertexAIClient")
        instrumented = InstrumentedVertexAIClient(
            project_id=project_id,
            model=model,
            logger_instance=logger,
        )

        # Make multiple calls
        response1 = instrumented.completion("Count to three.", max_tokens=60)
        response2 = instrumented.completion("What is the capital of France?", max_tokens=60)

        logger.info(f"Response 1: {response1}")
        logger.info(f"Response 2: {response2}")
        logger.info(f"Total tokens used: {instrumented.total_tokens_used}")
        logger.info(f"Call count: {instrumented.call_count}")
        logger.info(f"Call log entries: {len(instrumented.call_log)}")

        assert instrumented.call_count == 2, f"Expected 2 calls, got {instrumented.call_count}"
        assert instrumented.total_tokens_used > 0, "Total tokens should be > 0"
        assert len(instrumented.call_log) == 2, f"Expected 2 log entries, got {len(instrumented.call_log)}"

        # Verify log structure
        for i, log_entry in enumerate(instrumented.call_log, 1):
            logger.info(f"Call {i} log: {log_entry}")
            assert "call_index" in log_entry
            assert "prompt_tokens" in log_entry
            assert "completion_tokens" in log_entry
            assert "total_tokens" in log_entry
            assert "elapsed" in log_entry

        logger.info("✓ Test 2 passed (instrumentation)")

        logger.info("\n" + "=" * 70)
        logger.info("✅ All rlm-minimal adapter tests passed!")
        logger.info("=" * 70)
        return True

    except Exception as exc:
        logger.error(f"\n❌ RLM-minimal adapter test failed: {exc}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Vertex AI adapters")
    parser.add_argument(
        "--project-id",
        default="go-agl-poc-radax-p01-poc",
        help="GCP project ID",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="Gemini model name",
    )
    parser.add_argument(
        "--adapter",
        choices=["pyrlm", "rlm-minimal", "both"],
        default="both",
        help="Which adapter(s) to test",
    )
    args = parser.parse_args()

    results = {}

    logger.info(f"\nTesting with project_id={args.project_id}, model={args.model}\n")

    if args.adapter in ("pyrlm", "both"):
        results["pyrlm"] = test_pyrlm_adapter(args.project_id, args.model)

    if args.adapter in ("rlm-minimal", "both"):
        results["rlm-minimal"] = test_rlm_minimal_adapter(args.project_id, args.model)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    for adapter_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{adapter_name:15s} {status}")

    all_passed = all(results.values())
    if all_passed:
        logger.info("\n🎉 All tests passed! Adapters are ready for benchmarking.")
        sys.exit(0)
    else:
        logger.error("\n⚠️  Some tests failed. Check logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
