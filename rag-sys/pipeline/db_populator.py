"""
Nhúng chunk từ UniversalIngestor và ghi vào PostgreSQL (FeedbackChunk / PolicyChunk).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from data.models import FeedbackChunk, PolicyChunk
from data.database import AsyncSessionLocal
from pipeline.embedder import generate_embedding
from pipeline.universal_ingestor import UniversalIngestor

logger = logging.getLogger(__name__)

_FEEDBACK_COLUMN_KEYS = frozenset(
    {
        "feedback_id",
        "order_code",
        "customer_code",
        "customer_tier",
        "route_name",
        "rating",
        "sentiment",
    }
)


def _norm_key(k: str) -> str:
    return str(k).strip().lower().replace(" ", "_").replace("-", "_")


def _as_optional_str(v: Any, max_len: int | None = None) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    if max_len is not None:
        s = s[:max_len]
    return s


def _as_required_str(v: Any, *, default: str, max_len: int | None = None) -> str:
    opt = _as_optional_str(v, max_len=max_len)
    return opt if opt else default


def _as_int_or_none(v: Any) -> int | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _split_metadata_for_feedback(meta: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cột ORM (lowercase keys) vs phần còn lại cho metadata_json."""
    cols: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for k, v in meta.items():
        nk = _norm_key(str(k))
        if nk in _FEEDBACK_COLUMN_KEYS:
            cols[nk] = v
        else:
            extra[str(k)] = v
    return cols, extra


def _build_feedback_chunk(
    meta: dict[str, Any],
    content: str,
    embedding: list[float],
) -> FeedbackChunk:
    cols, extra = _split_metadata_for_feedback(meta)
    return FeedbackChunk(
        feedback_id=_as_optional_str(cols.get("feedback_id"), max_len=50),
        order_code=_as_optional_str(cols.get("order_code"), max_len=50),
        customer_code=_as_required_str(cols.get("customer_code"), default="UNKNOWN", max_len=50),
        customer_tier=_as_required_str(cols.get("customer_tier"), default="UNKNOWN", max_len=50),
        route_name=_as_required_str(cols.get("route_name"), default="UNKNOWN", max_len=100),
        rating=_as_int_or_none(cols.get("rating")),
        sentiment=_as_required_str(cols.get("sentiment"), default="neutral", max_len=50),
        content=content,
        metadata_json=extra if extra else {},
        embedding=embedding,
    )


def _build_policy_chunk(
    meta: dict[str, Any],
    content: str,
    embedding: list[float],
    file_path: str,
) -> PolicyChunk:
    extra = dict(meta)
    doc_key = None
    for k in list(extra.keys()):
        if _norm_key(k) == "document_name":
            doc_key = k
            break
    document_name = _as_optional_str(extra.pop(doc_key, None) if doc_key else None, max_len=255)
    if not document_name:
        document_name = Path(file_path).name[:255]

    section_title = None
    for k in list(extra.keys()):
        if _norm_key(k) == "section_title":
            section_title = _as_optional_str(extra.pop(k), max_len=255)
            break

    return PolicyChunk(
        document_name=document_name,
        section_title=section_title,
        content=content,
        metadata_json=extra if extra else {},
        embedding=embedding,
    )


async def persist_ingest_chunk_dicts(
    file_path: str,
    db: AsyncSession,
    chunks: list[dict[str, Any]],
) -> tuple[int, int]:
    """
    Nhúng và ghi từng chunk (FeedbackChunk / PolicyChunk). Mỗi chunk commit riêng.
    Trả (committed, skipped_or_failed).
    """
    path = Path(file_path)
    ok, skipped = 0, 0

    for i, chunk in enumerate(chunks):
        try:
            content = (chunk.get("content") or "").strip()
            if not content:
                logger.warning("[%s] chunk %d: empty content, skip", path.name, i)
                skipped += 1
                continue

            embedding = await asyncio.to_thread(generate_embedding, content)
            source_type = str(chunk.get("source_type") or "").lower().strip()
            meta = dict(chunk.get("metadata") or {})

            if source_type == "csv":
                obj = _build_feedback_chunk(meta, content, embedding)
            elif source_type in ("docx", "policy"):
                obj = _build_policy_chunk(meta, content, embedding, str(path))
            else:
                logger.warning("[%s] chunk %d: unknown source_type %r, skip", path.name, i, source_type)
                skipped += 1
                continue

            db.add(obj)
            await db.commit()
            ok += 1
            if (ok + skipped) % 10 == 0 or ok == 1:
                logger.info("[%s] progress: committed=%d skipped=%d", path.name, ok, skipped)
        except Exception:
            await db.rollback()
            logger.exception("[%s] chunk %d: failed, rolled back this chunk only", path.name, i)
            skipped += 1

    logger.info("Done %s: committed=%d skipped/failed=%d", path.name, ok, skipped)
    return ok, skipped


async def process_and_store_file(
    file_path: str,
    db: AsyncSession,
    *,
    raise_ingest_errors: bool = False,
) -> None:
    """
    Ingest một file, nhúng từng chunk, ghi DB. Lỗi từng chunk không làm dừng cả file.
    Mỗi chunk commit riêng để hàng trước đó vẫn được lưu khi một hàng lỗi.

    `raise_ingest_errors=True` (HTTP): lỗi parse/đọc file → ném exception, client nhận 5xx.
    Mặc định `False` (CLI quét thư mục): log và bỏ qua file lỗi, tiếp tục file sau.
    """
    path = Path(file_path)
    ingestor = UniversalIngestor()
    try:
        chunks = await ingestor.ingest_async(str(path))
    except Exception:
        logger.exception("Ingest failed for file %s", path)
        if raise_ingest_errors:
            raise
        return

    logger.info("Processing %s (%d chunks)", path.name, len(chunks))
    await persist_ingest_chunk_dicts(str(path), db, chunks)


_DATA_EXTENSIONS = frozenset(
    {
        ".csv",
        ".pdf",
        ".doc",
        ".docx",
        ".pptx",
        ".ppt",
        ".html",
        ".htm",
        ".xlsx",
        ".xls",
        ".md",
    }
)


def _collect_data_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in _DATA_EXTENSIONS],
        key=lambda p: str(p).lower(),
    )


async def _run_sequential(paths: list[Path]) -> None:
    async with AsyncSessionLocal() as session:
        for p in paths:
            await process_and_store_file(str(p), session)


async def _run_concurrent(paths: list[Path], max_concurrent: int) -> None:
    sem = asyncio.Semaphore(max(1, max_concurrent))

    async def one(path: Path) -> None:
        async with sem:
            async with AsyncSessionLocal() as session:
                await process_and_store_file(str(path), session)

    await asyncio.gather(*(one(p) for p in paths))


async def _async_main() -> None:
    parser = argparse.ArgumentParser(description="Embed UniversalIngestor chunks and store in PostgreSQL.")
    parser.add_argument(
        "folder",
        type=Path,
        help="Thư mục quét đệ quy CSV + tài liệu (pdf/docx/pptx/…, LlamaParse nếu có API key)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=1,
        help="Số file xử lý song song (mỗi file một AsyncSession). Mặc định 1 (tuần tự).",
    )
    args = parser.parse_args()
    folder: Path = args.folder
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")

    paths = _collect_data_files(folder)
    if not paths:
        logger.warning("No ingestable files under %s", folder)
        return

    logger.info("Found %d files under %s", len(paths), folder)
    if args.concurrent <= 1:
        await _run_sequential(paths)
    else:
        await _run_concurrent(paths, args.concurrent)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(_async_main())
