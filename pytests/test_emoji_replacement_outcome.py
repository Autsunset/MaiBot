from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.emoji_system import emoji_manager


@pytest.fixture
def manager(monkeypatch):
    value = emoji_manager.EmojiManager.__new__(emoji_manager.EmojiManager)
    value.emojis = [SimpleNamespace(description="old", query_count=0, register_time=None)]
    value._emoji_num = 1
    value.unregister_emoji = Mock(return_value=True)
    value.register_emoji_to_db = Mock(return_value="registered")
    prompt = SimpleNamespace(add_context=Mock())
    monkeypatch.setattr(emoji_manager.prompt_manager, "get_prompt", lambda _: prompt)
    monkeypatch.setattr(emoji_manager.prompt_manager, "render_prompt", AsyncMock(return_value="prompt"))
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "expected"),
    [("不取消注册", "skipped"), ("取消注册编号1", "registered"), ("取消注册编号999", "failed"), ("???", "failed")],
)
async def test_replacement_distinguishes_policy_skip_from_error(monkeypatch, manager, decision, expected):
    client = SimpleNamespace(generate_response=AsyncMock(return_value=SimpleNamespace(response=decision)))
    monkeypatch.setattr(emoji_manager, "LLMServiceClient", lambda **_: client)
    result = await manager._replace_an_emoji_outcome(SimpleNamespace(description="new"))
    assert result == expected
    assert manager.unregister_emoji.call_count == (1 if expected == "registered" else 0)
    assert manager.register_emoji_to_db.call_count == (1 if expected == "registered" else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["registered", "skipped", "failed"])
async def test_legacy_boolean_api_remains_compatible(manager, monkeypatch, outcome):
    monkeypatch.setattr(manager, "_replace_an_emoji_outcome", AsyncMock(return_value=outcome))
    assert await manager.replace_an_emoji_by_llm(SimpleNamespace()) is (outcome == "registered")


@pytest.mark.asyncio
@pytest.mark.parametrize("register_status", ["skipped", "failed"])
async def test_partial_replacement_is_not_a_policy_skip(manager, monkeypatch, register_status):
    client = SimpleNamespace(generate_response=AsyncMock(return_value=SimpleNamespace(response="取消注册编号1")))
    monkeypatch.setattr(emoji_manager, "LLMServiceClient", lambda **_: client)
    manager.register_emoji_to_db.return_value = register_status
    assert await manager._replace_an_emoji_outcome(SimpleNamespace(description="new")) == "failed"
    manager.unregister_emoji.assert_called_once()
