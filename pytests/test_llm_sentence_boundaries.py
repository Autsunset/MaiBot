from types import SimpleNamespace
from unittest.mock import AsyncMock

import json

import pytest

from src.chat.utils import llm_sentence_splitter as splitter


TEXT = "A sufficiently long original sentence, with its original spacing.\nSecond original sentence, unchanged!"


@pytest.mark.asyncio
async def test_short_reply_does_not_call_model(monkeypatch):
    def unexpected(**kwargs):
        raise AssertionError("short reply must not call a model")

    monkeypatch.setattr(splitter, "LLMServiceClient", unexpected)
    assert await splitter.split_text_with_llm("short, ") == [("short", ", ")]


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", [[], [1], [1, 2]])
async def test_selected_boundaries_preserve_all_original_characters(monkeypatch, selection):
    generate = AsyncMock(return_value=SimpleNamespace(response=json.dumps({"break_after": selection})))
    monkeypatch.setattr(splitter, "LLMServiceClient", lambda **_: SimpleNamespace(generate_response=generate))
    result = await splitter.split_text_with_llm(TEXT)
    assert "".join(content + separator for content, separator in result) == TEXT
    assert len(result) == len(selection) + 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    ['["rewritten sentence"]', '{"break_after":[true]}', '{"break_after":[999]}',
     '{"break_after":[2,1]}', '{"break_after":[1,1]}', '{"break_after":"1"}',
     '{"break_after":[],"text":"rewritten"}', "not json"],
)
async def test_rewrites_and_invalid_boundaries_are_rejected(monkeypatch, response):
    generate = AsyncMock(return_value=SimpleNamespace(response=response))
    monkeypatch.setattr(splitter, "LLMServiceClient", lambda **_: SimpleNamespace(generate_response=generate))
    with pytest.raises(ValueError):
        await splitter.split_text_with_llm(TEXT)
