from typing import TYPE_CHECKING, Dict, Sequence

if TYPE_CHECKING:
    from .model_client.base_client import GenerationAttempt


def summarize_request_timing(
    attempts: Sequence["GenerationAttempt"], *, total_seconds: float, queue_seconds: float,
) -> Dict[str, float]:
    """Provider 往返耗时不是纯推理时间；单列本地排队和失败调用，保留逻辑请求总耗时。"""
    provider_attempts = [attempt for attempt in attempts if not attempt.operation.startswith("model.queue")]
    failed = sum(max(0.0, attempt.duration_ms) / 1000 for attempt in provider_attempts if attempt.status != "succeeded")
    final = (
        max(0.0, provider_attempts[-1].duration_ms) / 1000
        if provider_attempts and provider_attempts[-1].status == "succeeded" else 0.0
    )
    total = max(0.0, total_seconds)
    queue = max(0.0, queue_seconds)
    return {
        "queue_seconds": queue,
        "failed_provider_seconds": failed,
        "final_provider_seconds": final,
        "other_seconds": max(0.0, total - queue - failed - final),
        "total_seconds": total,
    }
