from pathlib import Path

import pytest

from src.services.image_writeback_journal import ImageWritebackJournal


def test_unfinished_job_survives_reopen_and_completed_job_is_not_replayed(tmp_path: Path):
    path = tmp_path / 'jobs.sqlite3'
    journal = ImageWritebackJournal(path)
    journal.enqueue('real-chat', 'message-1')
    assert journal.next_job()['message_id'] == 'message-1'
    journal.close()
    journal = ImageWritebackJournal(path)
    try:
        assert journal.next_job()['session_id'] == 'real-chat'
        journal.complete('real-chat', 'message-1')
        journal.enqueue('real-chat', 'message-1')
        assert journal.next_job() is None
        assert journal.list_jobs('done', 25, 0)['total'] == 1
    finally:
        journal.close()


def test_failure_keeps_error_caps_retries_and_can_be_requeued(tmp_path: Path, monkeypatch):
    monkeypatch.setattr('src.services.image_writeback_journal.time.time', lambda: 100)
    journal = ImageWritebackJournal(tmp_path / 'jobs.sqlite3')
    try:
        journal.enqueue('real-chat', 'message-1')
        journal.fail(journal.next_job(), '图片文件缺失', 2)
        assert journal.next_job() is None
        monkeypatch.setattr('src.services.image_writeback_journal.time.time', lambda: 103)
        job = journal.next_job()
        assert job['attempts'] == 1
        journal.fail(job, '模型不可用', 2)
        assert journal.next_job() is None
        failed = journal.list_jobs('failed', 25, 0)['items'][0]
        assert failed['last_error'] == '模型不可用'
        journal.close()
        journal = ImageWritebackJournal(tmp_path / 'jobs.sqlite3')
        assert journal.retry_failed() == 1
        assert journal.next_job()['attempts'] == 0
    finally:
        journal.close()


def test_message_identity_includes_real_session(tmp_path: Path):
    journal = ImageWritebackJournal(tmp_path / 'jobs.sqlite3')
    try:
        journal.enqueue('chat-a', 'same-message')
        journal.enqueue('chat-b', 'same-message')
        assert journal.list_jobs('', 1, 0)['total'] == 2
        assert len(journal.list_jobs('', 1, 1)['items']) == 1
        with pytest.raises(ValueError):
            journal.enqueue('', 'same-message')
    finally:
        journal.close()


@pytest.mark.asyncio
async def test_worker_reloads_pending_message_after_restart(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import asyncio

    from src.services import memory_flow_service as flow

    path = tmp_path / 'jobs.sqlite3'
    journal = ImageWritebackJournal(path)
    journal.enqueue('real-chat', 'persisted-message')
    journal.close()
    service = flow.ImageMemoryWritebackService(path)
    monkeypatch.setattr(service, "_enabled", lambda: True)
    service._journal = ImageWritebackJournal(path)
    message = SimpleNamespace(session_id='real-chat', message_id='persisted-message')
    loaded = []

    def find(**kwargs):
        loaded.append(kwargs)
        return [message]

    async def handle(value):
        assert value is message
        service._stopping = True
        return True

    monkeypatch.setattr(flow, 'find_messages', find)
    monkeypatch.setattr(service, '_handle_message', AsyncMock(side_effect=handle))
    try:
        await asyncio.wait_for(service._worker_loop(), 2)
        assert loaded == [{'session_id': 'real-chat', 'message_id': 'persisted-message', 'limit': 1}]
        assert service._journal.list_jobs('done', 25, 0)['total'] == 1
    finally:
        service._journal.close()


def test_permanent_components_do_not_requeue_and_preserve_history(tmp_path: Path):
    journal = ImageWritebackJournal(tmp_path / "jobs.sqlite3")
    try:
        journal.enqueue("chat", "message")
        journal.record_component(
            "chat", "message", "0", "permanent_failure",
            {"type": "ImageAssetValidationError", "code": "unsupported_format", "message": "GIF"},
        )
        journal.fail(journal.next_job(), "unsupported GIF", 6, retryable=False)
        assert journal.next_job() is None
        assert journal.retry_failed() == 0
        failed = journal.list_jobs("failed", 10, 0)["items"][0]
        assert failed["attempts"] == 1
        assert failed["components"][0]["error"]["code"] == "unsupported_format"
        assert journal.connection.execute("SELECT COUNT(*) FROM image_writeback_component_history").fetchone()[0] == 1
    finally:
        journal.close()


def test_group_error_keeps_original_children_and_notes():
    from src.services.image_writeback_journal import image_error_details

    error = ValueError("original child")
    error.add_note("component_path=2")
    details = image_error_details(ExceptionGroup("images", [error]))
    assert details["children"][0]["message"] == "original child"
    assert "component_path=2" in details["children"][0]["traceback"]


@pytest.mark.asyncio
async def test_concurrent_start_does_not_create_duplicate_workers(tmp_path, monkeypatch):
    from unittest.mock import Mock

    import asyncio

    from src.services import memory_flow_service as flow

    service = flow.ImageMemoryWritebackService(tmp_path / "jobs.sqlite3")
    monkeypatch.setattr(service, "_enabled", lambda: True)
    factory = Mock(wraps=ImageWritebackJournal)
    monkeypatch.setattr(flow, "ImageWritebackJournal", factory)

    async def wait():
        await asyncio.Event().wait()

    monkeypatch.setattr(service, "_worker_loop", wait)
    monkeypatch.setattr(service, "_compensation_loop", wait)
    await asyncio.gather(*(service.start() for _ in range(8)))
    assert factory.call_count == 1
    await service.shutdown()
    assert service._journal is None


@pytest.mark.asyncio
async def test_mixed_components_only_retry_transient_failure(tmp_path: Path, monkeypatch):
    from datetime import datetime
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import io

    from PIL import Image

    from src.A_memorix.core.image.asset_store import ImageAssetValidationError
    from src.chat.image_system.image_manager import image_manager
    from src.services import memory_flow_service as flow

    def encoded(fmt):
        output = io.BytesIO()
        Image.new("RGB", (2, 2)).save(output, format=fmt)
        return output.getvalue()

    originals = [encoded("PNG"), encoded("GIF"), encoded("PNG")]
    components = []
    for index, payload in enumerate(originals):
        component = SimpleNamespace(binary_data=payload, binary_hash=f"hash-{index}", content="")

        async def load(target=component, data=payload):
            target.binary_data = data

        component.load_image_binary = AsyncMock(side_effect=load)
        components.append(component)
    message = SimpleNamespace(
        session_id="chat", message_id="message", timestamp=datetime.now(),
        raw_message=SimpleNamespace(components=components),
    )
    monkeypatch.setattr(flow, "iter_message_image_components", lambda _: [(str(i), c) for i, c in enumerate(components)])
    monkeypatch.setattr(image_manager, "get_cached_image_description", lambda _: None)
    calls = []

    async def ingest(**kwargs):
        path = kwargs["component_path"]
        calls.append(path)
        if path == "2" and calls.count("2") == 1:
            raise RuntimeError("provider unavailable")
        return {"success": True, "content_hash": f"hash-{path}"}

    monkeypatch.setattr(flow.memory_service, "image_memory", ingest)
    service = flow.ImageMemoryWritebackService(tmp_path / "jobs.sqlite3")
    service._journal = ImageWritebackJournal(service._journal_path)
    service._journal.enqueue("chat", "message")
    try:
        with pytest.raises(ExceptionGroup) as first:
            await service._handle_message(message)
        assert flow._image_error_retryable(first.value)
        assert len(first.value.exceptions) == 2
        states = service._journal.components("chat", "message")
        assert [states[str(i)]["status"] for i in range(3)] == [
            "done", "permanent_failure", "retryable_failure",
        ]
        with pytest.raises(ExceptionGroup) as second:
            await service._handle_message(message)
        assert not flow._image_error_retryable(second.value)
        assert isinstance(second.value.exceptions[0], ImageAssetValidationError)
        assert calls == ["0", "2", "2"]
        states = service._journal.components("chat", "message")
        assert [states[str(i)]["attempts"] for i in range(3)] == [1, 1, 2]
        assert components[0].load_image_binary.await_count == 0
        assert components[1].load_image_binary.await_count == 0
        assert originals[1] == encoded("GIF")
    finally:
        service._journal.close()
