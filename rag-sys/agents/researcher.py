"""
Researcher node: truy hồi vector + rerank theo state đã chuẩn bị bởi query_formulator.
"""

from __future__ import annotations

import logging

from core.demo_log import log_agent_flow
from core.state import MacaState
from data.database import AsyncSessionLocal
from tools.hybird_retriever import retrieve_from_maca_state

logger = logging.getLogger(__name__)


async def run_researcher(state: MacaState) -> dict:
    """
    Gọi hybrid retriever (dense + rerank) với session DB mới.

    - `general_chat`: không truy hồi.
    - Lỗi kết nối / truy vấn DB: `retrieved_chunks=[]`, `retrieval_error` báo lỗi.
    """
    route = (state.get("current_route") or "").strip().lower()
    if route == "general_chat":
        logger.info("Researcher: route=general_chat — skip retrieval")
        log_agent_flow("researcher", "bỏ qua truy hồi", reason="general_chat")
        return {}

    sq = (state.get("standalone_query") or "").strip()
    hy = (state.get("hypothetical_answer") or "").strip()
    tt = state.get("target_table")
    lim = state.get("retrieval_limit")
    log_agent_flow(
        "researcher",
        "bắt đầu truy hồi (đầu vào từ Query formulator)",
        current_route=route,
        standalone_query=sq or "(rỗng)",
        hypothetical_answer=hy or "(rỗng)",
        target_table=tt,
        retrieval_limit=lim,
        extracted_metadata_preview=state.get("extracted_metadata") or {},
    )

    try:
        async with AsyncSessionLocal() as db:
            chunks = await retrieve_from_maca_state(db, state)
    except Exception:
        logger.exception("Researcher: database or retrieval failed")
        err_msg = (
            "Database Error: không kết nối hoặc truy vấn PostgreSQL / pgvector thất bại."
        )
        log_agent_flow(
            "researcher",
            "THẤT BẠI — lỗi DB / retrieval",
            retrieval_error=err_msg,
            chunk_count=0,
        )
        return {
            "retrieved_chunks": [],
            "retrieval_error": err_msg,
        }

    logger.info("Researcher: retrieved %d chunks successfully", len(chunks))
    log_agent_flow(
        "researcher",
        "truy hồi thành công",
        chunk_count=len(chunks),
        retrieval_error=None,
    )
    return {"retrieved_chunks": chunks, "retrieval_error": None}


async def researcher_node(state: MacaState) -> dict:
    return await run_researcher(state)
