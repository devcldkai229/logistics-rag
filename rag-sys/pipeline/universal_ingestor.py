"""
Đọc CSV / tài liệu chính sách → chunk thống nhất cho pipeline DB.

- CSV: pandas (feedback).
- PDF / DOCX / PPTX / …: LlamaParse (Llama Cloud) khi có LLAMA_CLOUD_API_KEY — bảng + ảnh (inline) tốt hơn.
- DOCX không có key: fallback python-docx (chất lượng hạn chế).
"""

from __future__ import annotations

import asyncio
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.semantic_chunker import split_markdown_to_chunks

logger = logging.getLogger(__name__)

_CONTENT_COLUMN_CANDIDATES = (
    "content",
    "feedback_text",
    "text",
    "comment",
    "review",
    "noi_dung",
    "nội_dung",
    "feedback",
)

# Định dạng đưa qua LlamaParse (layout phức tạp: bảng, ảnh, slide…).
_LLAMA_PARSE_EXTENSIONS = frozenset(
    {
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


def build_policy_chunks_from_markdown(
    markdown_content: str,
    *,
    law_name: str,
    source_filename: str,
    ingest_pipeline: str,
) -> list[dict[str, Any]]:
    """
    Markdown đã chuẩn hóa → danh sách chunk dict (source_type=policy) cho DB.
    `law_name` dùng làm document_name trong metadata (giống tên file hiển thị).
    """
    base_meta: dict[str, Any] = {
        "document_name": law_name,
        "parser": ingest_pipeline,
        "source_format": Path(source_filename).suffix.lower().lstrip(".") or "unknown",
    }
    text = (markdown_content or "").strip()
    if not text:
        return []
    pieces = split_markdown_to_chunks(text) if len(text) > 2000 else [text]
    return [
        {"content": c, "metadata": {**base_meta}, "source_type": "policy"}
        for c in pieces
        if c and c.strip()
    ]


def _json_safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "item"):  # numpy scalar
        try:
            return v.item()
        except Exception:
            pass
    return v


class UniversalIngestor:
    async def ingest_async(self, file_path: str) -> list[dict[str, Any]]:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        ext = path.suffix.lower()

        if ext == ".csv":
            return self._ingest_csv(path)

        if ext in _LLAMA_PARSE_EXTENSIONS:
            from pipeline.llama_parse import llamaparse_configured

            if llamaparse_configured():
                return await self._ingest_llamaparse(path)
            if ext == ".docx":
                logger.warning(
                    "Không có LLAMA_CLOUD_API_KEY — ingest .docx bằng python-docx (không tối ưu cho bảng/ảnh)."
                )
                return self._ingest_docx_local(path)
            raise ValueError(
                f"Định dạng {ext} cần LlamaParse. Đặt LLAMA_CLOUD_API_KEY trong .env "
                f"(https://cloud.llamaindex.ai/)."
            )

        raise ValueError(
            f"Unsupported file type: {ext!r}. "
            f"CSV hoặc một trong: {sorted(_LLAMA_PARSE_EXTENSIONS)}"
        )

    def ingest(self, file_path: str) -> list[dict[str, Any]]:
        """Sync wrapper; không gọi từ bên trong event loop đang chạy."""
        return asyncio.run(self.ingest_async(file_path))

    def _ingest_csv(self, path: Path) -> list[dict[str, Any]]:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = [str(c).strip() for c in df.columns]
        content_col: str | None = None
        for cand in _CONTENT_COLUMN_CANDIDATES:
            for col in df.columns:
                if col.lower().replace(" ", "_") == cand:
                    content_col = col
                    break
            if content_col:
                break

        out: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            meta: dict[str, Any] = {}
            for k, v in row.items():
                meta[str(k)] = _json_safe(v)

            if content_col and content_col in row.index:
                raw = row[content_col]
                content = "" if raw is None or (isinstance(raw, float) and math.isnan(raw)) else str(raw).strip()
            else:
                parts: list[str] = []
                for k, v in row.items():
                    if k == content_col:
                        continue
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        continue
                    parts.append(str(v).strip())
                content = " ".join(p for p in parts if p).strip()

            out.append(
                {
                    "content": content,
                    "metadata": meta,
                    "source_type": "csv",
                }
            )
        return out

    async def _ingest_llamaparse(self, path: Path) -> list[dict[str, Any]]:
        from pipeline.llama_parse import parse_file_to_markdown

        md = await parse_file_to_markdown(path)
        chunks = build_policy_chunks_from_markdown(
            md,
            law_name=path.name,
            source_filename=path.name,
            ingest_pipeline="llamaparse",
        )
        if not chunks and not (md or "").strip():
            return [
                {
                    "content": "",
                    "metadata": {
                        "document_name": path.name,
                        "parser": "llamaparse",
                        "source_format": path.suffix.lower().lstrip(".") or "unknown",
                    },
                    "source_type": "policy",
                }
            ]
        return chunks

    def _ingest_docx_local(self, path: Path) -> list[dict[str, Any]]:
        try:
            from docx import Document
        except ImportError as e:
            raise ImportError(
                "python-docx is required for local .docx ingestion. Install with: pip install python-docx"
            ) from e

        doc = Document(str(path))
        full = "\n".join(p.text.strip() for p in doc.paragraphs if p.text and p.text.strip())
        chunks = build_policy_chunks_from_markdown(
            full,
            law_name=path.name,
            source_filename=path.name,
            ingest_pipeline="python-docx",
        )
        if not chunks and not full.strip():
            return [
                {
                    "content": "",
                    "metadata": {
                        "document_name": path.name,
                        "parser": "python-docx",
                        "source_format": "docx",
                    },
                    "source_type": "policy",
                }
            ]
        return chunks
