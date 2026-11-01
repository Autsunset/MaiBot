from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.A_memorix.core.runtime import sdk_memory_kernel
from src.A_memorix.core.runtime.services import ingest_service
from src.A_memorix.core.runtime.services.ingest_service import MemoryIngestService
from src.A_memorix.core.runtime.services.runtime_dependency_service import MemoryRuntimeDependencyService


@pytest.mark.parametrize("dual_ready", [False, True])
def test_summary_importer_receives_active_paragraph_pool(monkeypatch, dual_ready):
    single, paragraph, graph = object(), object(), object()
    active = paragraph if dual_ready else single
    runtime_config = {"graph_vector_store": graph}
    importer = Mock()
    for name in (
        "AggregateQueryService",
        "PersonProfileService",
        "EpisodeSegmentationService",
        "EpisodeService",
    ):
        monkeypatch.setattr(sdk_memory_kernel, name, Mock())
    monkeypatch.setattr(sdk_memory_kernel, "SummaryImporter", importer)
    kernel = SimpleNamespace(
        metadata_store=object(),
        graph_store=object(),
        vector_store=single,
        paragraph_vector_store=paragraph,
        graph_vector_store=graph,
        embedding_manager=object(),
        sparse_index=None,
        retriever=None,
        _paragraph_store=lambda: active,
        _build_runtime_config=lambda: runtime_config,
    )

    MemoryRuntimeDependencyService(kernel)._refresh_runtime_dependents()

    assert importer.call_args.kwargs["vector_store"] is active
    assert importer.call_args.kwargs["plugin_config"]["graph_vector_store"] is graph
    assert kernel.vector_store is single


@pytest.mark.asyncio
@pytest.mark.parametrize("dual_ready", [False, True])
async def test_person_fact_dedup_receives_active_paragraph_pool(monkeypatch, dual_ready):
    single, paragraph = object(), object()
    active = paragraph if dual_ready else single
    stop = RuntimeError("stop before writes")
    deduplicator = Mock(return_value=SimpleNamespace(plan=AsyncMock(side_effect=stop)))
    monkeypatch.setattr(ingest_service, "FactDeduplicator", deduplicator)
    kernel = SimpleNamespace(
        metadata_store=SimpleNamespace(get_external_memory_ref=Mock(return_value=None)),
        embedding_manager=object(),
        vector_store=single,
        _paragraph_store=lambda: active,
    )
    with pytest.raises(RuntimeError, match="stop before writes"):
        await MemoryIngestService(kernel)._ingest_person_fact(
            {"external_id": "observation-1", "text": "A likes tea.", "chat_id": "chat-1"},
            {"platform": "test", "evidence_message_ids": ["message-1"]},
            "person-1",
        )

    assert deduplicator.call_args.args[2] is active
    assert kernel.vector_store is single
