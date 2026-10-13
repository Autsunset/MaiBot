from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import asyncio

import pytest

from src.config.model_configs import APIProvider, ModelInfo, TaskConfig
from src.llm_models import utils_model
from src.llm_models.exceptions import EmptyResponseException, ModelAttemptFailed, ReqAbortException
from src.llm_models.utils_model import LLMOrchestrator, RequestType


@pytest.fixture
def vlm(monkeypatch):
    names = ["vision-a", "vision-b", "vision-c", "vision-d"]
    task = TaskConfig(model_list=names, selection_strategy="sequential", hard_timeout=42)
    models = [
        ModelInfo(name=name, model_identifier=name, api_provider="test-provider", model_type="llm", visual=True)
        for name in names
    ]
    provider = APIProvider(name="test-provider", base_url="https://example.invalid/v1", auth_type="none")
    config = SimpleNamespace(
        model_task_config=SimpleNamespace(vlm=task, utils=task),
        models=models,
        api_providers=[provider],
    )
    monkeypatch.setattr(utils_model.config_manager, "get_model_config", lambda: config)
    monkeypatch.setattr(utils_model, "ensure_configured_clients_loaded", lambda: None)
    monkeypatch.setattr(utils_model, "has_request_snapshot", lambda exc: True)
    monkeypatch.setattr(utils_model, "update_failed_request_attempt", lambda *args, **kwargs: None)
    monkeypatch.setattr(utils_model, "mark_request_final_failure", lambda *args, **kwargs: None)
    monkeypatch.setattr(LLMOrchestrator, "_schedule_llm_error_event", lambda *args, **kwargs: None)
    factory = Mock(return_value=object())
    monkeypatch.setattr(utils_model.client_registry, "get_client_class_instance", factory)
    orchestrator = LLMOrchestrator("vlm", "image")
    return orchestrator, factory


@pytest.mark.asyncio
async def test_client_initialization_failure_tries_next_model(vlm, monkeypatch):
    orchestrator, factory = vlm
    factory.side_effect = [AssertionError(), object()]
    request = AsyncMock(return_value=SimpleNamespace(content="a visible object", usage=None))
    monkeypatch.setattr(orchestrator, "_attempt_request_on_model_with_timeout", request)

    result = await orchestrator._execute_request(RequestType.RESPONSE)

    assert result.model_info.name == "vision-b"
    assert factory.call_count == 2
    request.assert_awaited_once()
    assert orchestrator.model_usage["vision-a"] == (0, 1, 0)
    assert orchestrator.model_usage["vision-b"] == (0, 0, 0)


@pytest.mark.asyncio
async def test_initialization_failure_tries_the_whole_configured_pool(vlm, monkeypatch):
    orchestrator, factory = vlm
    factory.side_effect = [AssertionError(), AssertionError(), AssertionError(), object()]
    request = AsyncMock(return_value=SimpleNamespace(content="a visible object", usage=None))
    monkeypatch.setattr(orchestrator, "_attempt_request_on_model_with_timeout", request)

    result = await orchestrator._execute_request(RequestType.RESPONSE)

    assert result.model_info.name == "vision-d"
    assert factory.call_count == 4
    assert orchestrator.model_for_task.hard_timeout == 42


@pytest.mark.asyncio
async def test_exhausted_client_initialization_preserves_error_and_penalties(vlm, monkeypatch):
    orchestrator, factory = vlm
    original_error = AssertionError("synthetic initialization error")
    factory.side_effect = original_error
    request = AsyncMock()
    monkeypatch.setattr(orchestrator, "_attempt_request_on_model_with_timeout", request)

    with pytest.raises(AssertionError) as caught:
        await orchestrator._execute_request(RequestType.RESPONSE)

    assert caught.value is original_error
    assert factory.call_count == len(orchestrator.model_for_task.model_list)
    assert all(usage == (0, 1, 0) for usage in orchestrator.model_usage.values())
    request.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [None, "", " \n"])
async def test_empty_image_description_tries_next_model(vlm, monkeypatch, content):
    orchestrator, _ = vlm
    request = AsyncMock(
        side_effect=[
            SimpleNamespace(content=content, usage=None),
            SimpleNamespace(content="a visible object", usage=None),
        ]
    )
    monkeypatch.setattr(orchestrator, "_attempt_request_on_model_with_timeout", request)

    result = await orchestrator._execute_request(RequestType.RESPONSE)

    assert result.model_info.name == "vision-b"
    assert request.await_count == 2
    assert orchestrator.model_usage["vision-a"] == (0, 1, 0)


@pytest.mark.asyncio
async def test_all_empty_results_propagate_failure(vlm, monkeypatch):
    orchestrator, _ = vlm
    request = AsyncMock(return_value=SimpleNamespace(content="", usage=None))
    monkeypatch.setattr(orchestrator, "_attempt_request_on_model_with_timeout", request)

    with pytest.raises(EmptyResponseException):
        await orchestrator._execute_request(RequestType.RESPONSE)

    assert request.await_count == len(orchestrator.model_for_task.model_list)
    assert all(usage[2] == 0 for usage in orchestrator.model_usage.values())


@pytest.mark.asyncio
async def test_existing_request_failure_still_tries_next_model(vlm, monkeypatch):
    orchestrator, _ = vlm
    request = AsyncMock(
        side_effect=[
            ModelAttemptFailed("synthetic request failure", ValueError("synthetic provider error")),
            SimpleNamespace(content="a visible object", usage=None),
        ]
    )
    monkeypatch.setattr(orchestrator, "_attempt_request_on_model_with_timeout", request)

    result = await orchestrator._execute_request(RequestType.RESPONSE)

    assert result.model_info.name == "vision-b"
    assert request.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [asyncio.CancelledError(), ReqAbortException()])
async def test_cancellation_does_not_try_another_model(vlm, monkeypatch, error):
    orchestrator, _ = vlm
    request = AsyncMock(side_effect=error)
    monkeypatch.setattr(orchestrator, "_attempt_request_on_model_with_timeout", request)

    with pytest.raises(type(error)):
        await orchestrator._execute_request(RequestType.RESPONSE)

    request.assert_awaited_once()
    assert all(usage == (0, 0, 0) for usage in orchestrator.model_usage.values())


@pytest.mark.asyncio
async def test_explicit_model_does_not_fall_back_to_another_model(vlm, monkeypatch):
    orchestrator, factory = vlm
    factory.side_effect = AssertionError()
    request = AsyncMock()
    monkeypatch.setattr(orchestrator, "_attempt_request_on_model_with_timeout", request)

    with pytest.raises(AssertionError):
        await orchestrator._execute_request(RequestType.RESPONSE, model_name="vision-b")

    factory.assert_called_once()
    request.assert_not_called()


@pytest.mark.asyncio
async def test_non_vlm_initialization_behavior_is_unchanged(vlm):
    _, factory = vlm
    factory.side_effect = AssertionError()
    orchestrator = LLMOrchestrator("utils", "text")

    with pytest.raises(AssertionError):
        await orchestrator._execute_request(RequestType.RESPONSE)

    factory.assert_called_once()


@pytest.mark.asyncio
async def test_initialization_log_excludes_provider_exception_payload(vlm, monkeypatch):
    orchestrator, factory = vlm
    secret = "sensitive-provider-payload"
    factory.side_effect = [ValueError(secret), object()]
    request = AsyncMock(return_value=SimpleNamespace(content="a visible object", usage=None))
    monkeypatch.setattr(orchestrator, "_attempt_request_on_model_with_timeout", request)
    errors = []
    monkeypatch.setattr(utils_model.logger, "error", errors.append)

    await orchestrator._execute_request(RequestType.RESPONSE)

    assert "ValueError" in errors[0]
    assert "_select_model" in errors[0]
    assert secret not in errors[0]
