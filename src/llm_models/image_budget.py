"""Observed image budgets for this deployment's verified model routes."""

from dataclasses import replace
from typing import AbstractSet, List, Sequence, Set, Tuple
from urllib.parse import urlsplit

from src.llm_models.payload_content.context_item import (
    AssistantMessageItem,
    ContextImagePart,
    ContextItem,
    ContextTextPart,
    SystemMessageItem,
    UserMessageItem,
)


# These are tested route policies, not advertised maximum model capacities.
_VERIFIED_32_IMAGE_MODELS: Set[str] = {
    "intern-latest",
    "group/auto-hy3",
    "group/glm-5.3-flash-all",
    "group/deepseek-v4.1-flash-all",
    "auto/minimax-m3.1-flash",
    "loomy/qwen3.8-flash",
    "deepseek-v4.1-flash-uncensored",
}
IMAGE_LIMIT_PLACEHOLDER = "[\u56fe\u7247]"


def get_request_image_limit(base_url: str, model_identifier: str) -> int | None:
    """Return this fork's observed route/model policy, or no new limit."""
    parsed = urlsplit(base_url)
    host = (parsed.hostname or "").lower()
    if host == "discovery-api.intern-ai.org.cn":
        return 8
    if model_identifier.casefold() == "minicpm-v-4.7-1b":
        return 1
    if model_identifier in _VERIFIED_32_IMAGE_MODELS:
        return 32
    return None


def limit_context_images(
    items: Sequence[ContextItem], max_images: int,
) -> Tuple[List[ContextItem], int]:
    """Keep the latest images across all messages without changing history.

    Candidate collages use ordinary user image parts, so they consume this
    final request budget too. Existing surrounding text/descriptions remain.
    """
    if max_images < 0:
        raise ValueError("max_images must not be negative")
    return _limit_images(items, max_images, IMAGE_LIMIT_PLACEHOLDER, frozenset())


def limit_latest_images_in_messages(
    messages: Sequence[ContextItem],
    *,
    max_image_num: int,
    placeholder: str = IMAGE_LIMIT_PLACEHOLDER,
    preserved_item_ids: AbstractSet[str] = frozenset(),
) -> List[ContextItem]:
    """Preserve the existing ordinary-image limiter's API and exemptions."""
    limited, _ = _limit_images(messages, max(0, int(max_image_num)), placeholder, preserved_item_ids)
    return limited


def _limit_images(
    items: Sequence[ContextItem],
    max_images: int,
    placeholder: str,
    preserved_item_ids: AbstractSet[str],
) -> Tuple[List[ContextItem], int]:
    positions = [
        (item_index, part_index)
        for item_index, item in enumerate(items)
        if isinstance(item, (SystemMessageItem, UserMessageItem, AssistantMessageItem))
        and item.meta.item_id not in preserved_item_ids
        for part_index, part in enumerate(item.parts)
        if isinstance(part, ContextImagePart)
    ]
    removed = max(0, len(positions) - max_images)
    if not removed:
        return list(items), 0

    keep = set(positions[-max_images:]) if max_images else set()
    limited: List[ContextItem] = []
    for item_index, item in enumerate(items):
        if (
            not isinstance(item, (SystemMessageItem, UserMessageItem, AssistantMessageItem))
            or item.meta.item_id in preserved_item_ids
        ):
            limited.append(item)
            continue
        parts = tuple(
            ContextTextPart(placeholder)
            if isinstance(part, ContextImagePart) and (item_index, part_index) not in keep
            else part
            for part_index, part in enumerate(item.parts)
        )
        if parts == item.parts:
            limited.append(item)
        elif isinstance(item, AssistantMessageItem):
            # Native replay must not restore image parts removed above.
            limited.append(replace(item, parts=parts, replay=None))
        else:
            limited.append(replace(item, parts=parts))
    return limited, removed
