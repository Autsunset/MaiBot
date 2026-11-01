from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import __future__
import ast
import asyncio
import time

import pytest

from src.llm_models.exceptions import LLMTaskTimeoutError
from src.llm_models.request_admission import (
    ModelQueueFullError,
    ModelQueueTimeoutError,
    ModelRequestAdmission,
)


def slot(pool, model="vision", **overrides):
    options = {"max_concurrency": 1, "max_queue_size": 64, "queue_timeout": 2.0}
    options.update(overrides)
    return pool.slot(("provider", model), **options)


@pytest.mark.asyncio
async def test_fifo_limits_concurrency_and_reclaims_empty_buckets() -> None:
    pool = ModelRequestAdmission()
    active = 0
    peak = 0
    order = []

    async def run(index):
        nonlocal active, peak
        async with slot(pool):
            active += 1
            peak = max(peak, active)
            order.append(index)
            await asyncio.sleep(0)
            active -= 1

    await asyncio.gather(*(run(i) for i in range(18)))
    assert peak == 1
    assert order == list(range(18))
    assert not pool._buckets


@pytest.mark.asyncio
async def test_unlimited_default_and_independent_models_do_not_block() -> None:
    pool = ModelRequestAdmission()
    async with slot(pool):
        async with slot(pool, "spark", max_queue_size=0):
            async with slot(pool, max_concurrency=0):
                pass
    assert not pool._buckets


@pytest.mark.asyncio
async def test_queue_capacity_rejects_before_inference() -> None:
    pool = ModelRequestAdmission()
    async with slot(pool, max_queue_size=0):
        with pytest.raises(ModelQueueFullError):
            async with slot(pool, max_queue_size=0):
                pytest.fail("A saturated zero-capacity queue must not admit a request")
    assert not pool._buckets


@pytest.mark.asyncio
async def test_queue_timeout_and_cancellation_remove_waiters() -> None:
    pool = ModelRequestAdmission()
    async with slot(pool):
        with pytest.raises(ModelQueueTimeoutError):
            async with slot(pool, queue_timeout=0.01):
                pytest.fail("A timed-out waiter must not run")

        async def queued():
            async with slot(pool):
                pytest.fail("A cancelled waiter must not run")

        task = asyncio.create_task(queued())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not pool._buckets[("provider", "vision")].waiting
    async with slot(pool, max_queue_size=0):
        pass
    assert not pool._buckets


@pytest.mark.asyncio
async def test_cancellation_after_grant_does_not_leak_the_slot() -> None:
    pool = ModelRequestAdmission()
    holder = slot(pool)
    await holder.__aenter__()

    async def queued():
        async with slot(pool):
            pytest.fail("Cancellation must win before the waiter resumes")

    task = asyncio.create_task(queued())
    await asyncio.sleep(0)
    await holder.__aexit__(None, None, None)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with slot(pool, max_queue_size=0):
        pass
    assert not pool._buckets


@pytest.mark.asyncio
async def test_provider_error_and_running_cancellation_release_slots() -> None:
    pool = ModelRequestAdmission()
    with pytest.raises(ValueError, match="provider failed"):
        async with slot(pool):
            raise ValueError("provider failed")

    entered = asyncio.Event()

    async def running():
        async with slot(pool):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(running())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with slot(pool, max_queue_size=0):
        pass
    assert not pool._buckets


@pytest.mark.asyncio
async def test_hot_reloaded_limit_wakes_existing_waiters() -> None:
    pool = ModelRequestAdmission()
    entered = asyncio.Event()

    async def queued():
        async with slot(pool):
            entered.set()

    async with slot(pool):
        task = asyncio.create_task(queued())
        await asyncio.sleep(0)
        async with slot(pool, max_concurrency=2):
            assert entered.is_set()
        await task
    assert not pool._buckets


def test_bot_and_webui_loops_share_a_single_concurrency_budget() -> None:
    pool = ModelRequestAdmission()
    barrier = Barrier(2)
    lock = Lock()
    active = 0
    peak = 0
    completed = 0

    async def request():
        nonlocal active, peak, completed
        async with slot(pool):
            with lock:
                active += 1
                peak = max(peak, active)
            await asyncio.sleep(0.005)
            with lock:
                active -= 1
                completed += 1

    def event_loop():
        barrier.wait(timeout=3)

        async def batch():
            await asyncio.gather(*(request() for _ in range(5)))

        asyncio.run(batch())

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(event_loop) for _ in range(2)]
        for future in futures:
            future.result(timeout=5)
    assert peak == 1
    assert completed == 10
    assert not pool._buckets


def timeout_wrapper(pool, *, hard_timeout=0.05, queue_timeout=1.0, max_queue_size=64):
    """Exercise the actual orchestration method with only I/O replaced."""
    tree = ast.parse(Path("src/llm_models/utils_model.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LLMOrchestrator")
    method = next(
        node for node in cls.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_attempt_request_on_model_with_timeout"
    )
    snapshot = Mock(return_value=Path("/tmp/synthetic-request-snapshot.json"))
    scope = {
        "asyncio": asyncio,
        "model_request_admission": pool,
        "ModelQueueFullError": ModelQueueFullError,
        "ModelQueueTimeoutError": ModelQueueTimeoutError,
        "LLMTaskTimeoutError": LLMTaskTimeoutError,
        "save_failed_request_snapshot": snapshot,
        "serialize_client_request_snapshot": lambda request: {},
        "attach_request_snapshot": Mock(),
        "logger": Mock(),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "<production-timeout-wrapper>", "exec",
                flags=__future__.annotations.compiler_flag),
        scope,
    )
    orchestrator = SimpleNamespace(
        task_name="vlm", request_type="image",
        model_for_task=SimpleNamespace(hard_timeout=hard_timeout),
        _attempt_request_on_model=AsyncMock(return_value="response"),
    )
    provider = SimpleNamespace(name="provider", client_type="fake")
    request = SimpleNamespace(
        model_info=SimpleNamespace(
            model_identifier="vision", max_concurrency=1,
            max_queue_size=max_queue_size, queue_timeout=queue_timeout,
        ),
        trace_context=None,
    )

    async def execute():
        return await scope["_attempt_request_on_model_with_timeout"](
            orchestrator, provider, object(), request, "vision",
        )

    return execute, orchestrator, snapshot


@pytest.mark.asyncio
async def test_queue_wait_does_not_consume_inference_deadline() -> None:
    pool = ModelRequestAdmission()
    execute, orchestrator, snapshot = timeout_wrapper(pool, hard_timeout=0.02)
    holder = slot(pool)
    await holder.__aenter__()
    started = time.monotonic()
    task = asyncio.create_task(execute())
    await asyncio.sleep(0.04)
    orchestrator._attempt_request_on_model.assert_not_called()
    await holder.__aexit__(None, None, None)
    assert await task == "response"
    assert time.monotonic() - started > 0.02
    snapshot.assert_not_called()
    assert not pool._buckets


@pytest.mark.asyncio
async def test_inference_still_times_out_and_releases_its_slot() -> None:
    pool = ModelRequestAdmission()
    execute, orchestrator, snapshot = timeout_wrapper(pool, hard_timeout=0.01)

    async def slow(*args, **kwargs):
        await asyncio.Event().wait()

    orchestrator._attempt_request_on_model.side_effect = slow
    with pytest.raises(LLMTaskTimeoutError):
        await execute()
    assert snapshot.call_args.kwargs["operation"] == "task.hard_timeout"
    assert not pool._buckets


@pytest.mark.asyncio
@pytest.mark.parametrize("full", [False, True])
async def test_queue_failures_are_distinct_and_do_not_send_provider_requests(full: bool) -> None:
    pool = ModelRequestAdmission()
    execute, orchestrator, snapshot = timeout_wrapper(
        pool, queue_timeout=0.01, max_queue_size=0 if full else 64,
    )
    async with slot(pool):
        with pytest.raises(ModelQueueFullError if full else ModelQueueTimeoutError):
            await execute()
    orchestrator._attempt_request_on_model.assert_not_called()
    assert snapshot.call_args.kwargs["operation"] == ("model.queue_full" if full else "model.queue_timeout")
    assert not pool._buckets
