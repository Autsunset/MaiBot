from collections import deque
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import AsyncIterator, Deque, Dict, Tuple

import asyncio
import time

from .exceptions import ModelAttemptFailed


class ModelQueueFullError(ModelAttemptFailed):
    """A request was rejected before contacting a saturated model."""


class ModelQueueTimeoutError(ModelAttemptFailed):
    """A request exceeded its admission deadline, not its inference deadline."""


@dataclass(eq=False)
class _Lease:
    ready: Future[None] = field(default_factory=Future)
    granted: bool = False


@dataclass
class _Bucket:
    limit: int
    active: int = 0
    waiting: Deque[_Lease] = field(default_factory=deque)


class ModelRequestAdmission:
    """Bound model traffic across both the Bot and WebUI event loops."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._buckets: Dict[Tuple[str, str], _Bucket] = {}

    @staticmethod
    def _grant_waiters(bucket: _Bucket) -> None:
        while bucket.waiting and bucket.active < bucket.limit:
            lease = bucket.waiting.popleft()
            lease.granted = True
            bucket.active += 1
            lease.ready.set_result(None)

    def _finish(self, key: Tuple[str, str], lease: _Lease) -> None:
        with self._lock:
            bucket = self._buckets[key]
            if lease.granted:
                bucket.active -= 1
            else:
                bucket.waiting.remove(lease)
                lease.ready.cancel()
            self._grant_waiters(bucket)
            if not bucket.active and not bucket.waiting:
                del self._buckets[key]

    @asynccontextmanager
    async def slot(
        self,
        key: Tuple[str, str],
        *,
        max_concurrency: int,
        max_queue_size: int,
        queue_timeout: float,
    ) -> AsyncIterator[float]:
        """Yield queue wait time after admission; always return the slot."""
        if max_concurrency == 0:
            yield 0.0
            return

        started = time.monotonic()
        lease = _Lease()
        with self._lock:
            bucket = self._buckets.setdefault(key, _Bucket(limit=max_concurrency))
            bucket.limit = max_concurrency
            self._grant_waiters(bucket)
            if bucket.active >= bucket.limit and len(bucket.waiting) >= max_queue_size:
                raise ModelQueueFullError(
                    f"Model '{key[1]}' admission queue is full "
                    f"(concurrency={max_concurrency}, max_queue_size={max_queue_size})"
                )
            bucket.waiting.append(lease)
            self._grant_waiters(bucket)

        # A concurrent Future can be notified safely by either event-loop thread.
        # Shield it so cancellation cannot race with a cross-thread slot grant.
        ready = asyncio.wrap_future(lease.ready)
        try:
            try:
                await asyncio.wait_for(asyncio.shield(ready), timeout=queue_timeout)
            except asyncio.TimeoutError as exc:
                raise ModelQueueTimeoutError(
                    f"Model '{key[1]}' admission queue timed out after {queue_timeout}s "
                    f"(concurrency={max_concurrency})"
                ) from exc
            yield time.monotonic() - started
        finally:
            # A grant may already have happened when the waiter is cancelled.
            self._finish(key, lease)
            if not ready.done():
                ready.cancel()


model_request_admission = ModelRequestAdmission()
