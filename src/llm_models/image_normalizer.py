"""模型请求图片的尺寸限制与长截图分段。"""

from dataclasses import replace
from io import BytesIO
from typing import List
import base64
import binascii
import math

from PIL import Image, UnidentifiedImageError

from src.llm_models.payload_content.context_item import (
    AssistantMessageItem,
    ContextContentPart,
    ContextImagePart,
    ContextItem,
    SystemMessageItem,
    UserMessageItem,
)

MODEL_IMAGE_MAX_SIDE = 8000
MODEL_IMAGE_SPLIT_RATIO = 3
MODEL_IMAGE_OVERLAP = 24
MODEL_IMAGE_MAX_SEGMENTS = 16


def _resize_image(image: Image.Image, scale: float) -> Image.Image:
    """保持比例缩小图片，并保证极窄图片仍有至少一个像素。"""
    return image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.Resampling.LANCZOS,
    )


def normalize_image_part(
    part: ContextImagePart, max_segments: int = MODEL_IMAGE_MAX_SEGMENTS,
) -> List[ContextImagePart]:
    """仅处理单边超过 8000 的图片；长图按顺序分段，其他图片等比例缩小。

    超限动图使用首帧，PNG 等图片保留透明通道。尺寸合规的图片保持原始数据。
    """
    if max_segments < 1:
        raise ValueError("max_segments must be positive")
    segment_limit = min(max_segments, MODEL_IMAGE_MAX_SEGMENTS)
    try:
        image_bytes = base64.b64decode(part.image_base64, validate=True)
        image = Image.open(BytesIO(image_bytes))
    except (binascii.Error, ValueError, UnidentifiedImageError, OSError):
        # 无效图片仍交给原有客户端校验逻辑处理，尺寸处理不改变其既有行为。
        return [part]

    with image:
        long_side, short_side = max(image.size), min(image.size)
        if long_side <= MODEL_IMAGE_MAX_SIDE:
            return [part]

        image.seek(0)
        frame = image.copy()
        if long_side / short_side <= MODEL_IMAGE_SPLIT_RATIO:
            frame = _resize_image(frame, MODEL_IMAGE_MAX_SIDE / long_side)
            segments = [frame]
        else:
            # 先约束短边和最大总长度，再裁剪，避免极长图片生成无界数量的图片。
            max_length = (
                MODEL_IMAGE_MAX_SIDE * segment_limit
                - MODEL_IMAGE_OVERLAP * (segment_limit - 1)
            )
            scale = min(1.0, MODEL_IMAGE_MAX_SIDE / short_side, max_length / long_side)
            if scale < 1.0:
                frame = _resize_image(frame, scale)
            long_side, short_side = max(frame.size), min(frame.size)
            segment_length = min(
                MODEL_IMAGE_MAX_SIDE,
                max(
                    short_side * MODEL_IMAGE_SPLIT_RATIO,
                    math.ceil(
                        (long_side + MODEL_IMAGE_OVERLAP * (segment_limit - 1))
                        / segment_limit
                    ),
                ),
            )
            segments: List[Image.Image] = []
            start = 0
            while start < long_side:
                end = min(start + segment_length, long_side)
                box = (0, start, frame.width, end) if frame.height >= frame.width else (start, 0, end, frame.height)
                segments.append(frame.crop(box))
                if end == long_side:
                    break
                start = end - MODEL_IMAGE_OVERLAP

        normalized_parts: List[ContextImagePart] = []
        image_format = "jpeg" if image.format == "JPEG" else "png"
        for segment in segments:
            if image_format == "jpeg":
                segment = segment.convert("RGB")
            elif segment.mode not in {"RGB", "RGBA", "L", "LA", "P", "I;16"}:
                segment = segment.convert("RGBA")
            output = BytesIO()
            if image_format == "jpeg":
                segment.save(output, format="JPEG", quality=95)
            else:
                segment.save(output, format="PNG")
            normalized_parts.append(
                ContextImagePart(image_format, base64.b64encode(output.getvalue()).decode("ascii"))
            )
        return normalized_parts


def normalize_context_images(
    items: List[ContextItem], max_images: int | None = None,
) -> List[ContextItem]:
    """规范化请求中的全部图片，保留消息顺序、元信息以及未修改的历史对象。"""
    image_count = sum(
        isinstance(part, ContextImagePart)
        for item in items
        if isinstance(item, (SystemMessageItem, UserMessageItem, AssistantMessageItem))
        for part in item.parts
    )
    if max_images is not None and max_images < image_count:
        raise ValueError("Apply the image budget before normalization")
    extra_segments = None if max_images is None else max_images - image_count
    normalized_items: List[ContextItem] = []
    for item in items:
        if not isinstance(item, (SystemMessageItem, UserMessageItem, AssistantMessageItem)):
            normalized_items.append(item)
            continue
        parts: List[ContextContentPart] = []
        changed = False
        for part in item.parts:
            if isinstance(part, ContextImagePart):
                # Resize the whole screenshot when its split would exceed the
                # remaining budget, rather than silently dropping its beginning.
                segment_limit = MODEL_IMAGE_MAX_SEGMENTS if extra_segments is None else extra_segments + 1
                images = normalize_image_part(part, max_segments=segment_limit)
                if extra_segments is not None:
                    extra_segments -= len(images) - 1
                parts.extend(images)
                changed = changed or len(images) != 1 or images[0] is not part
            else:
                parts.append(part)
        if not changed:
            normalized_items.append(item)
        elif isinstance(item, AssistantMessageItem):
            # 修改正文后，原始 Provider 回放片段已失效，不能绕过尺寸限制重发原图。
            normalized_items.append(replace(item, parts=tuple(parts), replay=None))
        else:
            normalized_items.append(replace(item, parts=tuple(parts)))
    return normalized_items
