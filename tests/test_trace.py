from pyrlm_runtime import Trace, TraceStep
from pyrlm_runtime.adapters.base import Usage


def test_trace_roundtrip() -> None:
    trace = Trace(steps=[])
    trace.add(
        TraceStep(
            step_id=1,
            kind="root_call",
            depth=0,
            prompt_summary="hello",
            code="print('hi')",
            output="print('hi')",
            usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            elapsed=0.123,
        )
    )
    trace.add(
        TraceStep(
            step_id=2,
            kind="subcall",
            depth=1,
            prompt_summary="chunk",
            cache_hit=True,
            input_hash="in",
            output_hash="out",
            cache_key="key",
            parallel_group_id="parallel-1",
            parallel_index=0,
            parallel_total=2,
        )
    )

    raw = trace.to_json()
    restored = Trace.from_json(raw)
    assert len(restored.steps) == 2
    assert restored.steps[0].usage is not None
    assert restored.steps[0].output == "print('hi')"
    assert restored.steps[0].elapsed == 0.123
    assert restored.steps[1].cache_hit is True
    assert restored.steps[1].parallel_group_id == "parallel-1"
    assert restored.steps[1].parallel_index == 0
    assert restored.steps[1].parallel_total == 2


def test_trace_from_json_backward_compatible_without_new_fields() -> None:
    raw = """
    [
      {
        "step_id": 1,
        "kind": "root_call",
        "depth": 0,
        "prompt_summary": "hello",
        "code": "print('hi')",
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
      }
    ]
    """

    restored = Trace.from_json(raw)

    assert len(restored.steps) == 1
    assert restored.steps[0].output is None
    assert restored.steps[0].elapsed is None
    assert restored.steps[0].parallel_group_id is None
