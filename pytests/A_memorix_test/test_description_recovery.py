from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncio
import json
import sqlite3

import pytest
import pytest_asyncio

from pytests.A_memorix_test.test_image_memory_runtime import _Embedder, _png, _runtime
from src.services import memory_flow_service as flow
from src.services.memory_service import MemoryWriteResult
from src.A_memorix.core.runtime.services.image_service import MemoryImageService


@pytest_asyncio.fixture
async def failed_jobs(tmp_path):
    metadata, runtime = _runtime(tmp_path, _Embedder())
    targets = []
    for index in (1, 2):
        image = await runtime.ingest(
            image_bytes=_png((index, 90, 100)), source_kind="chat", external_ref=f"chat:c:m{index}:0",
            scope_type="chat", chat_id="c", message_id=f"m{index}", component_path="0", occurred_at=10.0,
        )
        await runtime.apply_model_description(content_hash=image["content_hash"], text=f"description {index}")
        item = metadata.claim_image_description_compensations(
            lease_token=f"original-{index}", lease_seconds=30, max_attempts=6, limit=1,
        )[0]
        target = {"occurrence_id": item["occurrence_id"], "description_hash": item["description_hash"]}
        metadata.complete_image_description_compensation(
            **target, lease_token=f"original-{index}", success=False, error=f"original failure {index}",
        )
        with metadata.transaction() as database:
            database.execute(
                "UPDATE image_description_compensations SET attempt_count=6 WHERE occurrence_id=?",
                (target["occurrence_id"],),
            )
        targets.append(target)
    yield metadata, targets
    metadata.close()


def row(metadata, target):
    return dict(metadata.query(
        "SELECT * FROM image_description_compensations WHERE occurrence_id=? AND description_hash=?",
        (target["occurrence_id"], target["description_hash"]),
    )[0])


@pytest.mark.asyncio
async def test_selective_claim_preserves_counts_errors_and_other_jobs(failed_jobs):
    metadata, targets = failed_jobs
    first, other = targets
    original, unrelated = row(metadata, first), row(metadata, other)
    preview = metadata.prepare_image_description_recovery(**first, lease_token="unused")
    assert preview["eligible"] and row(metadata, first) == original
    claim = metadata.prepare_image_description_recovery(**first, lease_token="manual", dry_run=False)
    current = row(metadata, first)
    assert current["attempt_count"] == 7
    assert current["last_error"] == "original failure 1"
    assert claim["item"]["attempt_count"] == 6
    assert row(metadata, other) == unrelated
    assert metadata.prepare_image_description_recovery(
        **first, lease_token="competing", dry_run=False,
    )["reason"] == "lease_active"
    assert metadata.complete_image_description_compensation(**first, lease_token="manual", success=True)
    assert metadata.prepare_image_description_recovery(**first, lease_token="unused")["reason"] == "already_done"
    assert row(metadata, first)["attempt_count"] == 7


@pytest.mark.asyncio
async def test_recovery_does_not_revive_deleted_source(failed_jobs):
    metadata, targets = failed_jobs
    with metadata.transaction() as database:
        database.execute(
            "UPDATE image_occurrences SET status='deleted' WHERE occurrence_id=?", (targets[0]["occurrence_id"],),
        )
    assert metadata.prepare_image_description_recovery(
        **targets[0], lease_token="unused", dry_run=False,
    )["reason"] == "source_unavailable"
    assert row(metadata, targets[0])["attempt_count"] == 6


@pytest.mark.asyncio
async def test_runtime_dispatcher_defaults_to_nonmutating_preview(failed_jobs):
    metadata, targets = failed_jobs
    service = MemoryImageService(SimpleNamespace(image_memory_runtime=SimpleNamespace(metadata_store=metadata)))
    preview = await service.execute(action="prepare_description_recovery", **targets[0])
    assert preview["eligible"]
    assert row(metadata, targets[0])["attempt_count"] == 6


@pytest.mark.asyncio
async def test_expired_manual_lease_can_resume_without_counter_reset(failed_jobs, monkeypatch):
    from src.A_memorix.core.storage import metadata_image

    metadata, targets = failed_jobs
    clock = metadata_image._now()
    monkeypatch.setattr(metadata_image, "_now", lambda: clock)
    metadata.prepare_image_description_recovery(**targets[0], lease_token="expired", dry_run=False, lease_seconds=1)
    monkeypatch.setattr(metadata_image, "_now", lambda: clock + 2)
    resumed = metadata.prepare_image_description_recovery(**targets[0], lease_token="resumed", dry_run=False)
    assert resumed["eligible"]
    assert row(metadata, targets[0])["attempt_count"] == 8
    assert not metadata.complete_image_description_compensation(**targets[0], lease_token="expired", success=True)


def wire_live_service(monkeypatch, metadata, service, *, chat_exists=True):
    from src.chat.message_receive.chat_manager import chat_manager

    monkeypatch.setattr(chat_manager, "get_existing_session_by_session_id", lambda _: object() if chat_exists else None)
    monkeypatch.setattr(flow, "find_messages", lambda **_: [object()])

    async def image_memory(*, action, **kwargs):
        if action == "prepare_description_recovery":
            return metadata.prepare_image_description_recovery(**kwargs, lease_token="manual")
        if action == "complete_description_compensation":
            return {"success": metadata.complete_image_description_compensation(**kwargs)}
        raise AssertionError(f"unexpected action: {action}")

    monkeypatch.setattr(flow.memory_service, "image_memory", image_memory)


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_exists", [True, False])
async def test_preview_never_claims_or_calls_models(failed_jobs, tmp_path, monkeypatch, chat_exists):
    metadata, targets = failed_jobs
    service = flow.ImageMemoryWritebackService(tmp_path / "recovery.sqlite3")
    wire_live_service(monkeypatch, metadata, service, chat_exists=chat_exists)
    generate = AsyncMock(side_effect=AssertionError("preview must not invoke models"))
    monkeypatch.setattr(flow.memory_service, "ingest_summary", generate)
    preview = await service.recover_description_compensation(**targets[0])
    assert preview["eligible"] is chat_exists
    assert row(metadata, targets[0])["attempt_count"] == 6
    assert not service._journal_path.exists()
    generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_anchor_cannot_summarize_a_neighboring_window(failed_jobs, tmp_path, monkeypatch):
    metadata, targets = failed_jobs
    service = flow.ImageMemoryWritebackService(tmp_path / "recovery.sqlite3")
    wire_live_service(monkeypatch, metadata, service)
    monkeypatch.setattr(flow, "find_messages", lambda **_: [])
    result = await service.recover_description_compensation(**targets[0], dry_run=False)
    assert result["reason"] == "source_message_unavailable" and not result["eligible"]
    assert row(metadata, targets[0])["attempt_count"] == 6
    assert not service._journal_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("succeeded", [True, False])
async def test_recovery_preserves_failure_receipt_and_exact_idempotency_key(
    failed_jobs, tmp_path, monkeypatch, succeeded,
):
    metadata, targets = failed_jobs
    service = flow.ImageMemoryWritebackService(tmp_path / "recovery.sqlite3")
    wire_live_service(monkeypatch, metadata, service)
    generate = AsyncMock(return_value=MemoryWriteResult(
        success=succeeded, stored_ids=["summary"] if succeeded else [],
        detail="generated" if succeeded else "invalid summary",
    ))
    monkeypatch.setattr(flow.memory_service, "ingest_summary", generate)
    result = await service.recover_description_compensation(**targets[0], dry_run=False)
    assert result["success"] is succeeded
    assert result["attempt_count"] == 7
    expected = f"image_description_compensation:{targets[0]['occurrence_id']}:{targets[0]['description_hash']}"
    assert generate.call_args.kwargs["external_id"] == expected
    assert row(metadata, targets[0])["attempt_count"] == 7
    assert row(metadata, targets[1])["attempt_count"] == 6
    with sqlite3.connect(service._journal_path) as database:
        before, after, status = database.execute(
            "SELECT before_json,result_json,status FROM image_description_recovery_history",
        ).fetchone()
    assert json.loads(before)["last_error"] == "original failure 1"
    assert json.loads(before)["attempt_count"] == 6
    assert json.loads(after)["success"] is succeeded
    assert status == ("done" if succeeded else "failed")
    if succeeded:
        replay = await service.recover_description_compensation(**targets[0], dry_run=False)
        assert replay["reason"] == "already_done"
        generate.assert_awaited_once()


def test_recovery_api_defaults_to_preview():
    from src.webui.routers.memory import DescriptionRecoveryRequest

    assert DescriptionRecoveryRequest(occurrence_id="occurrence", description_hash="description").dry_run


@pytest.mark.asyncio
async def test_cancelled_recovery_keeps_audit_and_releases_lease(failed_jobs, tmp_path, monkeypatch):
    metadata, targets = failed_jobs
    service = flow.ImageMemoryWritebackService(tmp_path / "recovery.sqlite3")
    wire_live_service(monkeypatch, metadata, service)
    monkeypatch.setattr(service, "_run_description_compensation", AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await service.recover_description_compensation(**targets[0], dry_run=False)
    current = row(metadata, targets[0])
    assert current["status"] == "failed" and current["attempt_count"] == 7
    assert current["lease_token"] == ""
    with sqlite3.connect(service._journal_path) as database:
        assert database.execute("SELECT status FROM image_description_recovery_history").fetchone()[0] == "interrupted"
