from types import SimpleNamespace

import pytest

from src.llm_models.request_timing import summarize_request_timing


def attempt(seconds, status, operation="response"):
    return SimpleNamespace(duration_ms=seconds * 1000, status=status, operation=operation)


def test_failed_candidates_are_not_attributed_to_final_provider():
    timing = summarize_request_timing(
        [attempt(60, "failed"), attempt(60, "failed"), attempt(41, "succeeded")],
        total_seconds=177.5, queue_seconds=10,
    )
    assert timing == {
        "queue_seconds": 10, "failed_provider_seconds": 120, "final_provider_seconds": 41,
        "other_seconds": 6.5, "total_seconds": 177.5,
    }


def test_failed_queue_does_not_count_as_provider_call():
    timing = summarize_request_timing(
        [attempt(10, "failed", "model.queue_timeout"), attempt(2, "succeeded")],
        total_seconds=12.5, queue_seconds=10,
    )
    assert timing["failed_provider_seconds"] == 0
    assert timing["final_provider_seconds"] == 2
    assert timing["other_seconds"] == pytest.approx(0.5)


def test_all_failed_has_no_final_success_latency():
    timing = summarize_request_timing([attempt(5, "failed")], total_seconds=6, queue_seconds=1)
    assert timing["final_provider_seconds"] == 0
    assert timing["failed_provider_seconds"] == 5
