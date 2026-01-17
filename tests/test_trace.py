from rlm_runtime import Trace, TraceStep
from rlm_runtime.adapters.base import Usage


def test_trace_roundtrip() -> None:
    trace = Trace(steps=[])
    trace.add(
        TraceStep(
            step_id=1,
            kind="root_call",
            depth=0,
            prompt_summary="hello",
            code="print('hi')",
            usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
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
        )
    )

    raw = trace.to_json()
    restored = Trace.from_json(raw)
    assert len(restored.steps) == 2
    assert restored.steps[0].usage is not None
    assert restored.steps[1].cache_hit is True
