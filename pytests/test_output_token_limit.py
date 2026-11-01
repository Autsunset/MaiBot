from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import __future__
import ast
import asyncio

import pytest

from src.llm_models.exceptions import (
    EmptyResponseException,
    ModelAttemptFailed,
    NetworkConnectionError,
    OutputTokenLimitError,
    ReqAbortException,
    RespNotOkException,
    RespParseException,
)


def parser_scope():
    tree = ast.parse(Path("src/llm_models/model_client/openai_responses_client.py").read_text())
    names = {"_get_value", "_parse_completed_response", "_consume_response_stream"}
    methods = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    scope = {
        "asyncio": asyncio, "cast": cast,
        "OutputTokenLimitError": OutputTokenLimitError,
        "RespParseException": RespParseException,
        "ReqAbortException": ReqAbortException,
    }
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), "<production-responses-parser>", "exec",
                flags=__future__.annotations.compiler_flag),
        scope,
    )
    return scope


@pytest.mark.parametrize("object_response", [False, True])
def test_incomplete_token_limit_remains_a_failure_with_original_response(object_response: bool) -> None:
    scope = parser_scope()
    response = {
        "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{"type": "message", "content": '{"unfinished":'}],
    }
    if object_response:
        response = SimpleNamespace(
            **{**response, "incomplete_details": SimpleNamespace(reason="max_output_tokens")},
        )
    with pytest.raises(OutputTokenLimitError) as caught:
        scope["_parse_completed_response"](response, None, "provider", "endpoint", "strict")
    assert caught.value.ext_info is response
    assert isinstance(caught.value, RespParseException)


@pytest.mark.parametrize("status,reason", [("incomplete", "content_filter"), ("failed", None), ("cancelled", None)])
def test_other_incomplete_or_failed_statuses_keep_existing_error_type(status, reason) -> None:
    scope = parser_scope()
    with pytest.raises(RespParseException) as caught:
        scope["_parse_completed_response"](
            {"status": status, "incomplete_details": {"reason": reason}},
            None, "provider", "endpoint", "strict",
        )
    assert not isinstance(caught.value, OutputTokenLimitError)


@pytest.mark.asyncio
async def test_stream_token_limit_closes_stream_and_does_not_accept_partial_output() -> None:
    scope = parser_scope()
    response = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}

    class Stream:
        close = AsyncMock()

        async def __aiter__(self):
            yield {"type": "response.incomplete", "response": response}

    stream = Stream()
    with pytest.raises(OutputTokenLimitError) as caught:
        await scope["_consume_response_stream"](stream, None)
    assert caught.value.ext_info is response
    stream.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("limited", [False, True])
async def test_token_limit_stops_same_budget_retries_but_parse_errors_still_retry(limited: bool) -> None:
    tree = ast.parse(Path("src/llm_models/utils_model.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LLMOrchestrator")
    method = next(
        node for node in cls.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_attempt_request_on_model"
    )

    class ResponseRequest(SimpleNamespace):
        pass

    class EmbeddingRequest:
        pass

    @asynccontextmanager
    async def track(*args):
        yield

    scope = {
        "asyncio": asyncio, "ResponseRequest": ResponseRequest,
        "EmbeddingRequest": EmbeddingRequest, "track_model_task_usage": track,
        "has_request_snapshot": lambda error: True,
        "EmptyResponseException": EmptyResponseException,
        "NetworkConnectionError": NetworkConnectionError,
        "OutputTokenLimitError": OutputTokenLimitError,
        "ReqAbortException": ReqAbortException, "RespNotOkException": RespNotOkException,
        "RespParseException": RespParseException, "ModelAttemptFailed": ModelAttemptFailed,
        "update_failed_request_attempt": Mock(), "logger": Mock(),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "<production-provider-retries>", "exec",
                flags=__future__.annotations.compiler_flag),
        scope,
    )
    error = OutputTokenLimitError(message="max_output_tokens") if limited else RespParseException(message="invalid JSON")
    client = SimpleNamespace(get_response=AsyncMock(side_effect=error))
    orchestrator = SimpleNamespace(
        task_name="memory", request_type="classify",
        _get_original_error_info=lambda error: "",
        _schedule_llm_retry_event=Mock(),
    )
    provider = SimpleNamespace(max_retry=3, retry_interval=0, name="provider")
    request = ResponseRequest(model_info=SimpleNamespace(name="model", model_identifier="model"), trace_context=None)
    with pytest.raises(ModelAttemptFailed) as caught:
        await scope["_attempt_request_on_model"](orchestrator, provider, client, request)
    assert caught.value.original_exception is error
    assert client.get_response.await_count == (1 if limited else 3)
    assert orchestrator._schedule_llm_retry_event.call_count == (0 if limited else 2)
