"""
Bộ nhớ hội thoại trong RAM (theo session_id) cho pipeline MACA.

- Chỉ dùng khi client gửi `session_id` (API chat).
- Mặc định: mỗi request gửi **một** tin nhắn user; server nối với lịch sử đã lưu.
- Nếu request gửi **nhiều** tin → coi là full transcript, thay thế lịch sử phiên.
- Giới hạn số tin: MACA_MEMORY_MAX_MESSAGES (mặc định 80).

Lưu ý: chỉ tồn tại trong một process; nhiều worker uvicorn không chia sẻ RAM.
"""

from __future__ import annotations

import asyncio
import os
from copy import deepcopy

from langchain_core.messages import AIMessage, BaseMessage

_lock = asyncio.Lock()
_store: dict[str, list[BaseMessage]] = {}

_MAX = max(8, min(500, int(os.getenv("MACA_MEMORY_MAX_MESSAGES", "80"))))


def _trim(msgs: list[BaseMessage]) -> list[BaseMessage]:
    if len(msgs) <= _MAX:
        return msgs
    return msgs[-_MAX:]


async def get_messages(session_id: str) -> list[BaseMessage]:
    async with _lock:
        return deepcopy(_store.get(session_id, []))


async def set_messages(session_id: str, messages: list[BaseMessage]) -> None:
    async with _lock:
        _store[session_id] = _trim(list(messages))


async def clear_session(session_id: str) -> None:
    async with _lock:
        _store.pop(session_id, None)


def session_count() -> int:
    return len(_store)
