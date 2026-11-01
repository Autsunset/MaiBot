"""由模型选择原文边界，代码负责切片的回复断句。"""

from typing import List, Tuple

import json
import re

from src.common.logger import get_logger
from src.common.prompt_i18n import load_prompt
from src.services.llm_service import LLMServiceClient

logger = get_logger("llm_sentence_splitter")

_BOUNDARY_PATTERN = re.compile(r'(?:[。！？!?；;，,]|\.(?=\s|$))[”’"」』）)\]]*[ \t]*|\n+')
_SHORT_TEXT_LIMIT = 40
_MAX_BOUNDARIES = 64


def _normalize_json_payload(payload: str) -> str:
    """去除模型偶尔添加的 Markdown JSON 代码块包装。"""
    normalized = payload.strip()
    if normalized.startswith("```json") and normalized.endswith("```"):
        return normalized[7:-3].strip()
    if normalized.startswith("```") and normalized.endswith("```"):
        return normalized[3:-3].strip()
    return normalized


def _extract_trailing_comma(sentence: str) -> Tuple[str, str]:
    """将段尾逗号移入分隔符，逐条发送时省略，压缩合并时恢复。"""
    content = sentence.rstrip("，, \t\r\n")
    separator = sentence[len(content) :]
    if content.strip() and ("，" in separator or "," in separator):
        return content, separator
    return sentence, ""


def _parse_boundary_selection(payload: str, candidate_count: int) -> List[int]:
    try:
        selection = json.loads(_normalize_json_payload(payload))
    except json.JSONDecodeError as exc:
        raise ValueError("LLM 断句结果不是合法的边界选择 JSON") from exc
    if not isinstance(selection, dict) or set(selection) != {"break_after"}:
        raise ValueError("LLM 断句结果必须只包含 break_after 字段")
    selected = selection["break_after"]
    if (
        not isinstance(selected, list)
        or any(type(index) is not int or not 1 <= index <= candidate_count for index in selected)
        or selected != sorted(set(selected))
    ):
        raise ValueError("LLM 断句边界必须是递增、不重复且在候选范围内的整数编号")
    return selected


async def split_text_with_llm(text: str) -> List[Tuple[str, str]]:
    """只接受候选边界编号，不接收模型重写的句子或自行计算的字符偏移。"""
    if not text:
        return []
    if len(text.strip()) <= _SHORT_TEXT_LIMIT and "\n" not in text:
        logger.debug("[business-outcome] task=response.splitter outcome=policy_skip reason=short_text")
        return [_extract_trailing_comma(text)]

    boundaries = [match.end() for match in _BOUNDARY_PATTERN.finditer(text) if match.end() < len(text)]
    if not boundaries or len(boundaries) > _MAX_BOUNDARIES:
        return [_extract_trailing_comma(text)]
    candidates = [
        {"id": index, "before": text[max(0, end - 20) : end], "after": text[end : end + 20]}
        for index, end in enumerate(boundaries, 1)
    ]
    payload = json.dumps({"text": text, "candidates": candidates}, ensure_ascii=False)
    client = LLMServiceClient(task_name="fast_model", request_type="response.splitter")
    result = await client.generate_response(load_prompt("response_splitter", payload=payload))
    try:
        selected = _parse_boundary_selection(result.response, len(boundaries))
    except ValueError:
        logger.warning("[business-outcome] task=response.splitter outcome=validation_failed api_status=succeeded")
        raise

    # 直接取原文切片，模型不再复述正文；保留原来的完整性校验和逗号分隔符语义。
    ends = [boundaries[index - 1] for index in selected] + [len(text)]
    starts = [0] + ends[:-1]
    sentences = [text[start:end] for start, end in zip(starts, ends, strict=True)]
    if "".join(sentences) != text:
        raise ValueError("LLM 断句未完整保留原文")
    return [_extract_trailing_comma(sentence) for sentence in sentences]
