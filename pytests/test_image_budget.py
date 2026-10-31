from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import List
from unittest.mock import Mock

import __future__
import ast
import asyncio
import base64
import inspect
import io
import time

import pytest

from src.llm_models.image_budget import (
    get_request_image_limit,
    limit_context_images,
    limit_latest_images_in_messages,
)
from src.llm_models.payload_content.context_item import (
    AssistantMessageItem,
    ContextImagePart,
    ContextItem,
    ContextItemMeta,
    ContextTextPart,
    ProviderReplayFragment,
    ProviderScope,
    SystemMessageItem,
    UserMessageItem,
)
from src.maisaka.visual.message_limiter import limit_latest_images_in_messages as compatibility_limiter


def make_message(count: int, item_id: str = "images") -> UserMessageItem:
    parts = (ContextTextPart("Existing image descriptions"),) + tuple(
        ContextImagePart("png", f"image-{index}") for index in range(count)
    )
    return UserMessageItem(meta=ContextItemMeta.create(item_id=item_id), parts=parts)


def images(items: List[ContextItem]) -> List[str]:
    return [
        part.image_base64
        for item in items
        if isinstance(item, (SystemMessageItem, UserMessageItem, AssistantMessageItem))
        for part in item.parts
        if isinstance(part, ContextImagePart)
    ]


@pytest.mark.parametrize("model", ["Agents-A1", "intern-s2", "qwen3.8-27b"])
def test_discovery_route_is_limited_to_eight(model: str) -> None:
    assert get_request_image_limit("https://discovery-api.intern-ai.org.cn/v1/", model) == 8


@pytest.mark.parametrize(
    "model",
    [
        "intern-latest", "group/auto-hy3", "group/glm-5.3-flash-all",
        "group/deepseek-v4.1-flash-all", "auto/minimax-m3.1-flash",
        "loomy/qwen3.8-flash", "deepseek-v4.1-flash-uncensored",
    ],
)
def test_verified_model_budgets_without_private_deployment_addresses(model: str) -> None:
    assert get_request_image_limit("https://provider.example/v1", model) == 32


def test_mini_and_unknown_policies() -> None:
    assert get_request_image_limit("http://local.example/v1", "MiniCPM-V-4.7-1B") == 1
    assert get_request_image_limit("https://provider.example/v1", "untested-model") is None
    assert get_request_image_limit(
        "https://discovery-api.intern-ai.org.cn/v1", "group/auto-hy3"
    ) == 8


def test_total_budget_keeps_latest_without_mutating_history() -> None:
    original = make_message(40)
    limited, omitted = limit_context_images([original], 8)
    assert omitted == 32
    assert images(limited) == [f"image-{index}" for index in range(32, 40)]
    assert limited[0].meta is original.meta
    assert limited[0].parts[0] is original.parts[0]
    assert len(images([original])) == 40
    assert len([part for part in limited[0].parts if isinstance(part, ContextTextPart)]) == 33


def test_existing_exemptions_stay_compatible_but_final_budget_counts_collages() -> None:
    history = [make_message(8), make_message(1, "candidate-collage")]
    assert compatibility_limiter is limit_latest_images_in_messages
    ordinary = compatibility_limiter(
        history, max_image_num=8, preserved_item_ids={"candidate-collage"}
    )
    assert len(images(ordinary)) == 9
    final, omitted = limit_context_images(ordinary, 8)
    assert omitted == 1
    assert len(images(final)) == 8
    assert final[-1] is history[-1]
    assert len(images(history)) == 9


def test_ordinary_zero_limit_preserves_old_collage_exemption() -> None:
    history = [make_message(2), make_message(1, "candidate-collage")]
    limited = compatibility_limiter(
        history, max_image_num=0, preserved_item_ids={"candidate-collage"}
    )
    assert images(limited) == ["image-0"]
    total, _ = limit_context_images(history, 0)
    assert images(total) == []
    with pytest.raises(ValueError):
        limit_context_images(history, -1)


def test_system_user_assistant_images_share_one_budget_and_stale_replay_is_cleared() -> None:
    scope = ProviderScope(1, "openai", "example", "endpoint", "model")
    replay = ProviderReplayFragment.from_payload(scope, {"role": "assistant", "content": "native"})
    old_assistant = AssistantMessageItem(
        meta=ContextItemMeta.create(), parts=make_message(2).parts, replay=replay,
    )
    system = SystemMessageItem(meta=ContextItemMeta.create(), parts=make_message(1).parts)
    recent = make_message(3)
    result, omitted = limit_context_images([old_assistant, system, recent], 3)
    assert omitted == 3
    assert images(result) == images([recent])
    assert result[0].replay is None
    assert old_assistant.replay is replay


def test_unchanged_items_keep_identity_and_replay() -> None:
    item = make_message(2)
    output, omitted = limit_context_images([item], 2)
    assert omitted == 0 and output[0] is item


def test_each_candidate_uses_its_own_copy_of_the_same_history() -> None:
    history = [make_message(40)]
    narrow, _ = limit_context_images(history, 8)
    broad, _ = limit_context_images(history, 32)
    assert len(images(narrow)) == 8
    assert len(images(broad)) == 32
    assert len(images(history)) == 40


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["ordinary", "split", "embedding"])
async def test_real_scheduler_rebuilds_each_attempt_and_rechecks_after_splitting(case: str) -> None:
    """Run the production retry method with network/DB boundaries replaced."""
    tree = ast.parse(Path("src/llm_models/utils_model.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LLMOrchestrator")
    method = next(
        node for node in cls.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_execute_request"
    )

    class RequestType:
        RESPONSE = "response"
        EMBEDDING = "embedding"

    class AttemptFailed(Exception):
        original_exception = RuntimeError("synthetic first-route failure")

    class Abort(Exception):
        pass

    class HttpError(Exception):
        status_code = 0

    original = [make_message(1 if case == "split" else 40)]
    normalized = Mock(side_effect=lambda items, **kwargs: [make_message(40)] if case == "split" else items)
    scope = {
        "asyncio": asyncio, "datetime": datetime, "inspect": inspect,
        "get_request_image_limit": get_request_image_limit,
        "limit_context_images": limit_context_images,
        "normalize_context_images": normalized,
        "RequestType": RequestType,
        "RequestTraceContext": lambda **kwargs: SimpleNamespace(
            **kwargs, model_attempt=0, current_attempt_started_at=time.time(),
        ),
        "LLMExecutionResult": lambda **kwargs: SimpleNamespace(**kwargs),
        "ModelAttemptFailed": AttemptFailed,
        "ReqAbortException": Abort,
        "RespNotOkException": HttpError,
        "has_request_snapshot": lambda error: True,
        "update_failed_request_attempt": lambda *args, **kwargs: None,
        "logger": Mock(),
    }
    compiled = compile(
        ast.Module(body=[method], type_ignores=[]), "<production-request-loop>", "exec",
        flags=__future__.annotations.compiler_flag,
    )
    exec(compiled, scope)
    candidates = iter([
        (
            SimpleNamespace(name="narrow", model_identifier="intern-s2"),
            SimpleNamespace(name="discovery", base_url="https://discovery-api.intern-ai.org.cn/v1"),
            object(),
        ),
        (
            SimpleNamespace(name="broad", model_identifier="group/glm-5.3-flash-all"),
            SimpleNamespace(name="other", base_url="https://provider.example/v1"),
            object(),
        ),
    ])
    sent = []

    async def select(**kwargs):
        return next(candidates)

    async def send(provider, client, request, model_name):
        sent.append(len(images(request.context_items)))
        if model_name == "narrow":
            raise AttemptFailed()
        return SimpleNamespace(usage=None)

    fake = SimpleNamespace(
        model_for_task=SimpleNamespace(model_list=["narrow", "broad"]),
        task_name="test", request_type="test",
        _resolve_effective_session_id=lambda session_id: session_id,
        _select_model=select,
        _build_client_request=lambda **kwargs: SimpleNamespace(**kwargs),
        _attempt_request_on_model_with_timeout=send,
        _adjust_model_usage=Mock(),
    )
    await scope["_execute_request"](
        fake,
        RequestType.EMBEDDING if case == "embedding" else RequestType.RESPONSE,
        context_factory=lambda client: original,
    )
    assert sent == ([40, 40] if case == "embedding" else [8, 32])
    assert len(images(original)) == (1 if case == "split" else 40)
    assert normalized.call_count == (0 if case == "embedding" else 2)


def test_single_image_budget_keeps_both_ends_of_a_long_screenshot() -> None:
    from PIL import Image, ImageDraw
    from src.llm_models.image_normalizer import normalize_context_images

    image = Image.new("RGB", (100, 9000), "green")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 99, 999), fill="red")
    draw.rectangle((0, 8000, 99, 8999), fill="blue")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    part = ContextImagePart("png", base64.b64encode(stream.getvalue()).decode())
    item = UserMessageItem(meta=ContextItemMeta.create(), parts=(part,))
    result = normalize_context_images([item], max_images=1)
    payloads = images(result)
    assert len(payloads) == 1
    normalized = Image.open(io.BytesIO(base64.b64decode(payloads[0])))
    assert max(normalized.size) <= 8000
    assert normalized.getpixel((normalized.width // 2, 0)) == (255, 0, 0)
    assert normalized.getpixel((normalized.width // 2, normalized.height - 1)) == (0, 0, 255)


def test_runtime_admin_result_keeps_command_specific_values() -> None:
    source = Path("src/A_memorix/host_service.py").read_text()
    tree = ast.parse(source)
    payload = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        and len(node.keys) == 2
        and all(key is None for key in node.keys)
        and any(isinstance(value, ast.Name) and value.id == "result" for value in node.values)
        and any(
            isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
            and value.func.attr == "_startup_status_payload"
            for value in node.values
        )
    )

    class Host:
        def _startup_status_payload(self):
            return {"status": "starting", "shared": "old"}

    result = {"shared": "fresh", "vector_spaces": ["current"]}
    merged = eval(compile(ast.Expression(payload), "<admin-payload>", "eval"), {"self": Host(), "result": result})
    assert merged == {"status": "starting", **result}
