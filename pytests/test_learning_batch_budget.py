from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.learners.pending_learning_buffer import PendingLearningBuffer


def buffer(*items):
    value = PendingLearningBuffer(lambda item: item)
    value.extend(items)
    return value


def test_learning_batch_preserves_fifo_and_unconsumed_sources():
    pending = buffer(*(f"message-{index}" for index in range(45)))
    first = pending.bounded_snapshot(str)
    assert first == [f"message-{index}" for index in range(20)]
    assert len(pending) == 45
    pending.discard(first)
    assert pending.bounded_snapshot(str)[0] == "message-20"
    assert len(pending) == 25


def test_learning_batch_limits_chars_without_slicing_source_text():
    pending = buffer("abc", "def", "ghi")
    assert pending.bounded_snapshot(str, max_chars=6) == ["abc", "def"]
    assert pending.snapshot() == ["abc", "def", "ghi"]


def test_single_oversized_source_is_not_lost_or_stuck():
    pending = buffer("a" * 7000, "short")
    first = pending.bounded_snapshot(str)
    assert first == ["a" * 7000]
    pending.discard(first)
    assert pending.bounded_snapshot(str) == ["short"]


@pytest.mark.parametrize("limits", [{"max_chars": 0}, {"max_messages": 0}])
def test_invalid_batch_limits_are_explicit(limits):
    with pytest.raises(ValueError):
        buffer("abc").bounded_snapshot(str, **limits)


def test_compact_expression_review_uses_fast_task():
    from src.learners.expression_utils import judge_llm

    assert judge_llm.task_name == "fast_model"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["expression", "jargon"])
@pytest.mark.parametrize("outcome", ["completed", "no_results", "failed", "raised"])
async def test_learning_only_consumes_terminal_nonfailed_batches(monkeypatch, kind, outcome):
    from src.learners import expression_learner, jargon_learner

    module = expression_learner if kind == "expression" else jargon_learner
    learner_type = module.ExpressionLearner if kind == "expression" else module.JargonLearner
    learner = learner_type.__new__(learner_type)
    learner.session_id = "chat-1"
    learner.learning_status = "running"
    learner.pending_messages = buffer("first", "second")
    learner._resolve_learning_session_id = lambda _: "chat-1"

    async def run(*args, **kwargs):
        if outcome == "raised":
            raise RuntimeError("provider failed")
        learner.learning_status = outcome
        return outcome == "completed"

    learner._run_learning_batch = run
    gate = SimpleNamespace(acquire=AsyncMock(return_value=SimpleNamespace(acquired=True)), release=AsyncMock())
    monkeypatch.setattr(module, f"{kind}_learning_batch_gate", gate)
    if kind == "expression":
        operation = learner._learn_from_session_messages(["first"])
    else:
        operation = learner._learn_from_sources(["first"], jargon_miner=object())
    if outcome == "raised":
        with pytest.raises(RuntimeError, match="provider failed"):
            await operation
    else:
        await operation
    assert learner.pending_messages.snapshot() == (
        ["first", "second"] if outcome in ("failed", "raised") else ["second"]
    )
    gate.release.assert_awaited_once_with("chat-1")
