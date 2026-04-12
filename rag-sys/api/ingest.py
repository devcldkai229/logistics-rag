"""
HTTP entry cho ingestion: upload .csv / .docx → UniversalIngestor → embed → PostgreSQL.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from data.database import AsyncSessionLocal
from pipeline.db_populator import process_and_store_file

logger = logging.getLogger(__name__)

router = APIRouter()

_ALLOWED = frozenset(
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


@router.post("/file")
async def ingest_upload_file(file: UploadFile = File(...)) -> dict:
    """
    Upload một file CSV (feedback) hoặc DOCX (chính sách).
    File được lưu tạm, chạy cùng pipeline với CLI `python -m pipeline.db_populator`.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Thiếu tên file.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in _ALLOWED:
        raise HTTPException(
            status_code=400,
            detail=f"Chỉ hỗ trợ {sorted(_ALLOWED)}. PDF/PPTX/… cần LLAMA_CLOUD_API_KEY (LlamaParse). File: {suffix}",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="File rỗng.")

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        Path(tmp_path).write_bytes(raw)
        async with AsyncSessionLocal() as db:
            await process_and_store_file(tmp_path, db, raise_ingest_errors=True)
    except Exception as e:
        logger.exception("Ingest upload failed: %s", file.filename)
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return {"status": "ok", "filename": file.filename}


class IngestPathBody(BaseModel):
    path: str = Field(min_length=1, description="Đường dẫn file .csv hoặc .docx trên server")


@router.post("/path")
async def ingest_server_path(body: IngestPathBody) -> dict:
    """
    Ingest file đã có sẵn trên **server** (đường dẫn tuyệt đối), ví dụ thư mục dữ liệu nội bộ.
    Chỉ dùng trong môi trường tin cậy (không expose ra internet công khai).
    """
    p = Path(body.path).resolve()
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file: {p}")

    if p.suffix.lower() not in _ALLOWED:
        raise HTTPException(status_code=400, detail=f"Định dạng không hỗ trợ: {p.suffix}")

    try:
        async with AsyncSessionLocal() as db:
            await process_and_store_file(str(p), db, raise_ingest_errors=True)
    except Exception as e:
        logger.exception("Ingest path failed: %s", p)
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {"status": "ok", "path": str(p)}
