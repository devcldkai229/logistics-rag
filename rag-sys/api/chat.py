"""
Chat HTTP: chạy graph MACA (LangGraph) và trả bản nháp + metadata.
Hỗ trợ SSE `/message/stream` và bộ nhớ phiên RAM (`session_id`).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from core.conversation_memory import clear_session, get_messages as memory_get_messages, set_messages
from core.state import MacaState
from core.workflow import app as maca_graph

logger = logging.getLogger(__name__)

router = APIRouter()

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class ChatMessagePart(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessagePart] = Field(min_length=1)
    retrieval_limit: int | None = Field(default=5, ge=1, le=30)
    session_id: str | None = Field(
        default=None,
        description=(
            "Nếu có: lịch sử lưu RAM trên server. "
            "Nên gửi mỗi lượt **một** tin role=user; server nối với các lượt trước. "
            "Nếu gửi nhiều tin → coi là full transcript và ghi đè phiên."
        ),
    )


class ChatResponse(BaseModel):
    reply: str
    session_id: str | None = None
    current_route: str | None = None
    is_valid: bool | None = None
    judge_feedback: str | None = None
    retry_count: int | None = None
    retrieved_count: int | None = None
    retrieval_error: str | None = None


def _to_lc_messages(parts: list[ChatMessagePart]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for p in parts:
        if p.role == "user":
            out.append(HumanMessage(content=p.content.strip()))
        else:
            out.append(AIMessage(content=p.content.strip()))
    return out


async def _resolve_lc_messages(body: ChatRequest) -> tuple[list[BaseMessage], str | None]:
    incoming = _to_lc_messages(body.messages)
    if not incoming:
        raise HTTPException(status_code=400, detail="Không có tin nhắn hợp lệ.")

    sid = (body.session_id or "").strip() or None
    if not sid:
        return incoming, None

    prior = await memory_get_messages(sid)
    if len(body.messages) == 1 and body.messages[0].role == "user":
        combined = prior + incoming
        logger.debug(
            "Chat memory: session=%s append 1 user turn (prior_len=%d)",
            sid[:8],
            len(prior),
        )
    else:
        combined = incoming
        logger.debug(
            "Chat memory: session=%s full sync (%d messages from client)",
            sid[:8],
            len(incoming),
        )
    return combined, sid


async def _invoke_graph_with_messages(
    messages: list[BaseMessage],
    retrieval_limit: int | None,
) -> dict[str, Any]:
    state_in: MacaState = {"messages": messages}
    if retrieval_limit is not None:
        state_in["retrieval_limit"] = retrieval_limit
    return await maca_graph.ainvoke(state_in)


async def _run_chat_pipeline(body: ChatRequest) -> tuple[dict[str, Any], str | None, list[BaseMessage]]:
    combined, sid = await _resolve_lc_messages(body)
    # Checkpoint sau tin user mới để lỗi graph không làm lệch phiên so với UI.
    if sid:
        await set_messages(sid, list(combined))
    result = await _invoke_graph_with_messages(combined, body.retrieval_limit)
    return result, sid, combined


def _final_reply_from_result(result: dict[str, Any]) -> str:
    reply = (result.get("draft_response") or "").strip()
    if not reply:
        reply = "(Không có nội dung trả lời — kiểm tra log server và cấu hình LLM.)"
    return reply


def _meta_payload(result: dict[str, Any], *, session_id: str | None) -> dict[str, Any]:
    chunks = result.get("retrieved_chunks") or []
    re = result.get("retrieval_error")
    retrieval_error = re if isinstance(re, str) and re.strip() else None
    return {
        "type": "meta",
        "session_id": session_id,
        "current_route": result.get("current_route"),
        "is_valid": result.get("is_valid"),
        "judge_feedback": result.get("judge_feedback"),
        "retry_count": result.get("retry_count"),
        "retrieved_count": len(chunks) if isinstance(chunks, list) else None,
        "retrieval_error": retrieval_error,
    }


async def _persist_reply(session_id: str | None, combined: list[BaseMessage], reply: str) -> None:
    if not session_id:
        return
    await set_messages(session_id, list(combined) + [AIMessage(content=reply)])


def _sse_data(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@router.post("/message", response_model=ChatResponse)
async def chat_message(body: ChatRequest) -> ChatResponse:
    try:
        result, sid, combined = await _run_chat_pipeline(body)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("MACA graph invoke failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    reply = _final_reply_from_result(result)
    await _persist_reply(sid, combined, reply)

    m = _meta_payload(result, session_id=sid)
    return ChatResponse(
        reply=reply,
        session_id=sid,
        current_route=m.get("current_route"),
        is_valid=m.get("is_valid"),
        judge_feedback=m.get("judge_feedback"),
        retry_count=m.get("retry_count"),
        retrieved_count=m.get("retrieved_count"),
        retrieval_error=m.get("retrieval_error"),
    )


@router.post("/message/stream")
async def chat_message_stream(body: ChatRequest) -> StreamingResponse:
    """
    Server-Sent Events: chạy full graph, stream nội dung trả lời; meta có `session_id` nếu dùng RAM.
    """

    async def event_gen() -> AsyncIterator[str]:
        try:
            result, sid, combined = await _run_chat_pipeline(body)
        except HTTPException as he:
            detail: str
            if isinstance(he.detail, str):
                detail = he.detail
            else:
                detail = json.dumps(he.detail, ensure_ascii=False)
            yield _sse_data({"type": "error", "detail": detail})
            return
        except Exception as e:
            logger.exception("MACA graph stream invoke failed")
            yield _sse_data({"type": "error", "detail": str(e)})
            return

        reply = _final_reply_from_result(result)
        await _persist_reply(sid, combined, reply)

        yield _sse_data(_meta_payload(result, session_id=sid))
        step = 24
        for i in range(0, len(reply), step):
            yield _sse_data({"type": "delta", "text": reply[i : i + step]})
            await asyncio.sleep(0)
        yield _sse_data({"type": "done"})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.delete("/session/{session_id}")
async def clear_chat_session(session_id: str) -> dict[str, str]:
    """Xóa lịch sử RAM của một phiên (vd. nút 'Chat mới' trên UI)."""
    await clear_session(session_id.strip())
    return {"status": "ok", "session_id": session_id.strip()}
