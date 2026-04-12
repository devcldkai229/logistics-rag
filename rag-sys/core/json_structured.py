"""
Gọi LLM (ChatOpenAI / DeepSeek) rồi parse JSON → Pydantic.

DeepSeek và một số endpoint không hỗ trợ `response_format` JSON Schema của
`with_structured_output`; dùng prompt + `json.loads` thay thế.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _ai_text(msg: BaseMessage) -> str:
    raw = msg.content
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts: list[str] = []
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts).strip()
    return str(raw).strip()


def extract_json_dict(text: str) -> dict:
    """Lấy object JSON đầu tiên từ chuỗi (có thể bọc ```json ... ```)."""
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    i = s.find("{")
    if i < 0:
        raise ValueError("Không tìm thấy '{' trong phản hồi.")
    decoder = json.JSONDecoder()
    obj, _end = decoder.raw_decode(s[i:])
    if not isinstance(obj, dict):
        raise ValueError("JSON gốc không phải object.")
    return obj


def _compact_json_instruction(schema: type[BaseModel]) -> str:
    lines: list[str] = []
    for name, field in schema.model_fields.items():
        ann = getattr(field.annotation, "__name__", None) or str(field.annotation)
        lines.append(f'  "{name}": ({ann})')
    return (
        "You MUST output a single JSON object only. No markdown fences, no text before or after.\n"
        "Fields:\n"
        + "\n".join(lines)
    )


def _merge_system(messages: list[BaseMessage], addition: str) -> list[BaseMessage]:
    out = list(messages)
    if out and isinstance(out[0], SystemMessage):
        c = out[0].content
        if isinstance(c, str):
            out[0] = SystemMessage(content=c.strip() + "\n\n" + addition)
        else:
            out.insert(0, SystemMessage(content=addition))
    else:
        out.insert(0, SystemMessage(content=addition))
    return out


async def ainvoke_json_model(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    schema: type[T],
) -> T | None:
    """
    Gọi `llm.ainvoke` với hướng dẫn JSON; validate bằng Pydantic.
    Trả None nếu parse/validate thất bại.
    """
    instr = _compact_json_instruction(schema)
    merged = _merge_system(messages, instr)
    try:
        resp = await llm.ainvoke(merged)
    except Exception:
        logger.exception("json_structured: LLM ainvoke failed")
        return None

    if not isinstance(resp, AIMessage):
        logger.warning("json_structured: unexpected message type %s", type(resp))

    raw = _ai_text(resp)
    if not raw:
        logger.warning("json_structured: empty model content")
        return None

    try:
        data = extract_json_dict(raw)
        return schema.model_validate(data)
    except (json.JSONDecodeError, ValueError, ValidationError) as e:
        logger.warning("json_structured: parse/validate failed: %s | preview=%r", e, raw[:400])
        return None
