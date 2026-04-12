"""
Truy hồi MACA: dense (pgvector) + Cross-Encoder rerank.

Khớp với `core.state.MacaState` (`target_table`, `extracted_metadata`, `retrieval_limit`,
`standalone_query`, `hypothetical_answer`) và chunk Markdown từ `pipeline.semantic_chunker`.
Hybrid BM25/FTS + RRF có thể bổ sung sau; hiện tại chỉ vector + rerank như pipeline cũ.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Mapping

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.state import MacaState
from data.models import FeedbackChunk, PolicyChunk
from pipeline.embedder import generate_embedding, get_reranker

# Số ứng viên vector trước rerank.
RECALL_TOP_K = max(
    3,
    min(
        50,
        int(os.getenv("MACA_RECALL_TOP_K", os.getenv("LIGA_RECALL_TOP_K", "10"))),
    ),
)

# Cột FeedbackChunk dùng để lọc cứng (ưu tiên hơn JSONB khi trùng key trong extracted_metadata).
_FEEDBACK_FILTER_COLUMNS = frozenset(
    {
        "feedback_id",
        "order_code",
        "customer_code",
        "customer_tier",
        "route_name",
        "sentiment",
        "rating",
    }
)


def _norm_meta_key(raw: str) -> str:
    """Chuẩn hóa tên key từ LLM/CSV (Customer Code → customer_code)."""
    return str(raw).strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_target_table(raw: str | None) -> str:
    t = (raw or "policy").strip().lower()
    if t in ("feedback", "csv", "voc"):
        return "feedback"
    if t in ("policy", "docx", "document", "markdown"):
        return "policy"
    raise ValueError(f"target_table không hợp lệ: {raw!r} (chờ 'feedback' hoặc 'policy')")


def _feedback_citation(chunk: FeedbackChunk) -> str:
    head = []
    if chunk.feedback_id:
        head.append(f"Feedback {chunk.feedback_id}")
    if chunk.order_code:
        head.append(f"Đơn {chunk.order_code}")
    tail = f"KH {chunk.customer_code} | {chunk.route_name} | tier {chunk.customer_tier} | {chunk.sentiment}"
    return " · ".join(head + [tail]) if head else tail


def _policy_citation(chunk: PolicyChunk) -> str:
    if chunk.section_title:
        return f"{chunk.document_name} — {chunk.section_title}"
    return chunk.document_name


def _apply_feedback_metadata_filters(
    stmt: Select[tuple[FeedbackChunk]],
    metadata: Mapping[str, Any],
) -> Select[tuple[FeedbackChunk]]:
    col_values: dict[str, Any] = {}
    extra_json: dict[str, Any] = {}
    for raw_key, value in metadata.items():
        if value is None:
            continue
        nk = _norm_meta_key(str(raw_key))
        if nk in _FEEDBACK_FILTER_COLUMNS:
            col_values[nk] = value
        else:
            # Giữ key gốc cho JSONB @> (trùng cấu trúc lúc ingest từ CSV).
            extra_json[str(raw_key)] = value

    for nk, value in col_values.items():
        col = getattr(FeedbackChunk, nk, None)
        if col is None:
            continue
        stmt = stmt.where(col == value)

    if extra_json:
        stmt = stmt.where(FeedbackChunk.metadata_json.contains(extra_json))
    return stmt


def _apply_policy_metadata_filters(
    stmt: Select[tuple[PolicyChunk]],
    metadata: Mapping[str, Any],
) -> Select[tuple[PolicyChunk]]:
    if not metadata:
        return stmt
    document_name: Any = None
    section_title: Any = None
    rest: dict[str, Any] = {}
    for raw_key, v in metadata.items():
        if v is None:
            continue
        nk = _norm_meta_key(str(raw_key))
        if nk == "document_name":
            document_name = v
        elif nk == "section_title":
            section_title = v
        else:
            rest[str(raw_key)] = v

    if document_name is not None:
        stmt = stmt.where(PolicyChunk.document_name == document_name)
    if section_title is not None:
        stmt = stmt.where(PolicyChunk.section_title == section_title)
    if rest:
        stmt = stmt.where(PolicyChunk.metadata_json.contains(rest))
    return stmt


def _chunk_to_payload(
    chunk: FeedbackChunk | PolicyChunk,
    *,
    source: str,
) -> dict[str, Any]:
    if isinstance(chunk, FeedbackChunk):
        citation = _feedback_citation(chunk)
    else:
        citation = _policy_citation(chunk)
    return {
        "citation": citation,
        "content": chunk.content,
        "chunk_id": str(chunk.id),
        "source": source,
    }


async def retrieve_maca_chunks(
    db: AsyncSession,
    *,
    embedding_text: str,
    rerank_query: str,
    target_table: str,
    extracted_metadata: Mapping[str, Any] | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """
    Truy hồi theo bảng `feedback_chunks` hoặc `policy_chunks`.

    - `embedding_text`: sinh vector (có thể gộp standalone + HyDE).
    - `rerank_query`: câu ngắn cho bge-reranker (thường `standalone_query`).
    - `extracted_metadata`: lọc cột + JSONB containment (PostgreSQL).
    """
    meta = dict(extracted_metadata or {})
    target = _normalize_target_table(target_table)
    emb_preview = embedding_text[:120] + ("..." if len(embedding_text) > 120 else "")
    rq_preview = rerank_query[:120] + ("..." if len(rerank_query) > 120 else "")
    print(
        f"[MACA Retriever] target={target} | embedding (preview): {emb_preview!r} | "
        f"rerank (preview): {rq_preview!r} | recall_k={RECALL_TOP_K} | final_k={limit} | "
        f"filters={meta or {}}"
    )

    if not embedding_text.strip():
        print("[MACA Retriever] Bỏ qua: embedding_text rỗng.")
        return []

    if not rerank_query.strip():
        print("[MACA Retriever] Cảnh báo: rerank_query rỗng — dùng embedding_text làm fallback rerank.")
        rerank_query = embedding_text

    query_vector = await asyncio.to_thread(generate_embedding, embedding_text)

    if target == "feedback":
        stmt = (
            select(FeedbackChunk)
            .where(FeedbackChunk.embedding.isnot(None))
            .order_by(FeedbackChunk.embedding.cosine_distance(query_vector))
            .limit(RECALL_TOP_K)
        )
        stmt = _apply_feedback_metadata_filters(stmt, meta)
        source_label = "feedback"
    else:
        stmt = (
            select(PolicyChunk)
            .where(PolicyChunk.embedding.isnot(None))
            .order_by(PolicyChunk.embedding.cosine_distance(query_vector))
            .limit(RECALL_TOP_K)
        )
        stmt = _apply_policy_metadata_filters(stmt, meta)
        source_label = "policy"

    result = await db.execute(stmt)
    chunks = list(result.scalars().all())
    print(f"[MACA Retriever] Bước 1: {len(chunks)} ứng viên từ DB (<= {RECALL_TOP_K}).")

    candidates: list[dict[str, Any]] = []
    for ch in chunks:
        candidates.append(_chunk_to_payload(ch, source=source_label))

    if not candidates:
        print("[MACA Retriever] Không có ứng viên sau retrieval.")
        return []

    reranker = get_reranker()
    inputs: list[tuple[str, str]] = [(rerank_query, d["content"]) for d in candidates]
    print(f"[MACA Retriever] Bước 2: rerank {len(inputs)} cặp (query, passage).")

    scores = await asyncio.to_thread(
        lambda: reranker.predict(inputs, show_progress_bar=False)
    )
    score_list = [float(s) for s in scores]
    for doc, score in zip(candidates, score_list, strict=True):
        doc["rerank_score"] = score

    candidates.sort(key=lambda d: d["rerank_score"], reverse=True)

    top = candidates[:limit]
    out: list[dict[str, Any]] = []
    for d in top:
        out.append(
            {
                "citation": d["citation"],
                "content": d["content"],
                "chunk_id": d["chunk_id"],
                "source": d["source"],
            }
        )
    print(f"[MACA Retriever] Bước 3: trả về {len(out)} chunk (top-{limit}).")
    return out


def build_embedding_text_from_state(state: Mapping[str, Any]) -> str:
    """Gộp standalone + hypothetical (HyDE) cho bước encode vector."""
    sq = (state.get("standalone_query") or "").strip()
    ha = (state.get("hypothetical_answer") or "").strip()
    if ha:
        return f"{sq}\n\n[Giả định truy hồi]\n{ha}".strip()
    return sq


def build_rerank_query_from_state(state: Mapping[str, Any]) -> str:
    """Reranker cần query tương đối ngắn — ưu tiên standalone_query."""
    return (state.get("standalone_query") or "").strip()


async def retrieve_from_maca_state(
    db: AsyncSession,
    state: MacaState,
) -> list[dict[str, Any]]:
    """
    Tiện ích cho node LangGraph: đọc trường state và gọi `retrieve_maca_chunks`.

    Cần ít nhất `standalone_query` hoặc `hypothetical_answer` để có embedding_text.
    """
    embedding_text = build_embedding_text_from_state(state)
    rerank_query = build_rerank_query_from_state(state) or embedding_text
    target = state.get("target_table")
    limit = int(state.get("retrieval_limit") or 3)
    meta = state.get("extracted_metadata") or {}

    return await retrieve_maca_chunks(
        db,
        embedding_text=embedding_text,
        rerank_query=rerank_query,
        target_table=target or "policy",
        extracted_metadata=meta,
        limit=limit,
    )
