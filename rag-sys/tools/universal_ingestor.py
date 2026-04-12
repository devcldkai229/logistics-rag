"""
Universal Data Ingestor — quét thư mục thô, parse theo loại file, thống nhất Markdown,
chunk + embed + lưu PostgreSQL (pgvector).

Chạy từ thư mục rag-sys (cwd chứa data, pipeline, tools):
  python -m tools.universal_ingestor "E:/path/to/raw_folder"

- PDF/DOC/PPTX/…: LlamaParse (cần LLAMA_CLOUD_API_KEY hoặc LLAMA_PARSE_API_KEY).
- XLSX/XLS: pandas → Markdown bảng (luôn, không qua LlamaParse).
- TXT/MD: đọc UTF-8.
- CSV: UniversalIngestor (feedback) như API/db_populator.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

import data.models  # noqa: F401 — đăng ký ORM khi chạy standalone

from data.database import session_scope
from pipeline.db_populator import persist_ingest_chunk_dicts
from pipeline.llama_parse import LlamaParseIngestError, llamaparse_configured, parse_file_to_markdown_sync
from pipeline.universal_ingestor import UniversalIngestor, build_policy_chunks_from_markdown

logger = logging.getLogger("maca.universal_ingestor")

# Đuôi file (CLI + tương thích ingest rộng)
_LLAMA_EXTENSIONS = frozenset({".pdf", ".doc", ".docx", ".pptx", ".ppt", ".html", ".htm"})
_EXCEL_EXTENSIONS = frozenset({".xlsx", ".xls"})
_TEXT_EXTENSIONS = frozenset({".txt", ".md"})
_CSV_EXTENSIONS = frozenset({".csv"})

ALLOWED_INGEST_EXTENSIONS = frozenset(
    _LLAMA_EXTENSIONS | _EXCEL_EXTENSIONS | _TEXT_EXTENSIONS | _CSV_EXTENSIONS
)


def clean_excel_to_markdown(file_path: str) -> str:
    """
    Đọc mọi sheet Excel, bỏ hàng/cột toàn NaN, điền NaN còn lại bằng 'Trống',
    xuất Markdown table (index=False).
    """
    import pandas as pd

    path = Path(file_path)
    suffix = path.suffix.lower()
    engine: str | None = "xlrd" if suffix == ".xls" else None

    xl = pd.ExcelFile(file_path, engine=engine)
    parts: list[str] = []

    for sheet_name in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet_name)
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")
        df = df.fillna("Trống")
        if df.empty:
            parts.append(f"## Sheet: {sheet_name}\n\n_(Trống sau khi làm sạch)_")
            continue
        md_table = df.to_markdown(index=False)
        parts.append(f"## Sheet: {sheet_name}\n\n{md_table}")

    return "\n\n".join(parts).strip()


def _docx_plain_sync(file_path: str) -> str:
    from docx import Document

    doc = Document(file_path)
    return "\n".join(p.text.strip() for p in doc.paragraphs if p.text and p.text.strip())


def file_to_markdown_sync(file_path: str) -> tuple[str, str]:
    """
    Một file trên đĩa -> (markdown hoặc text thuần, ingest_pipeline_tag). Toàn sync —
    gọi trong asyncio.to_thread từ FastAPI / asyncio CLI.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    fp = str(path.resolve())

    if suffix in _LLAMA_EXTENSIONS:
        if llamaparse_configured():
            return parse_file_to_markdown_sync(fp), "api_llamaparse"
        if suffix == ".docx":
            text = _docx_plain_sync(fp)
            return text, "api_docx_local"
        raise ValueError(
            f"llamaparse_required: Cần LLAMA_CLOUD_API_KEY (hoặc LLAMA_PARSE_API_KEY) để ingest {suffix}."
        )

    if suffix in _EXCEL_EXTENSIONS:
        return clean_excel_to_markdown(fp), "api_excel"

    if suffix in _TEXT_EXTENSIONS:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        return text, "api_text"

    raise ValueError(f"unsupported_extension:{suffix}")


def _iter_files(folder_path: Path) -> list[Path]:
    out: list[Path] = []
    for root, _dirs, files in os.walk(folder_path):
        for name in files:
            out.append(Path(root) / name)
    return sorted(out, key=lambda p: str(p).lower())


_TAG_CLI = {
    "api_llamaparse": "universal_llamaparse",
    "api_excel": "universal_excel",
    "api_text": "universal_text",
    "api_docx_local": "universal_docx_local",
}


async def _markdown_for_path(path: Path) -> tuple[str, str]:
    fp = str(path.resolve())
    md, tag = await asyncio.to_thread(file_to_markdown_sync, fp)
    return md, _TAG_CLI.get(tag, tag)


async def process_raw_data_folder(folder_path: str) -> dict[str, Any]:
    root = Path(folder_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Khong phai thu muc hop le: {root}")

    errors_list: list[dict[str, str]] = []
    stats: dict[str, Any] = {
        "folder": str(root),
        "ok": 0,
        "failed": 0,
        "skipped_unsupported": 0,
        "skipped_empty": 0,
        "errors": errors_list,
    }

    print(f"Universal Ingestor — thư mục: {root}")

    if not llamaparse_configured():
        print(
            "Cảnh báo: LLAMA_CLOUD_API_KEY chưa set — file .pdf/.doc/.pptx/… sẽ lỗi "
            "(trừ .docx có thể dùng python-docx)."
        )

    all_files = _iter_files(root)
    print(f"Tìm thấy {len(all_files)} file (mọi loại).")

    for path in all_files:
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = Path(path.name)
        suffix = path.suffix.lower()
        basename = path.name
        law_label = basename

        allowed = _LLAMA_EXTENSIONS | _EXCEL_EXTENSIONS | _TEXT_EXTENSIONS | _CSV_EXTENSIONS
        if suffix not in allowed:
            print(f"Bỏ qua (không hỗ trợ): {rel} [{suffix or 'no-ext'}]")
            stats["skipped_unsupported"] += 1
            continue

        try:
            if suffix in _CSV_EXTENSIONS:
                print(f"CSV (feedback): {rel}")
                ingestor = UniversalIngestor()
                chunks = await ingestor.ingest_async(str(path))
                if not chunks:
                    print(f"Nội dung rỗng sau ingest, bỏ qua: {rel}")
                    stats["skipped_empty"] += 1
                    continue
                print(f"Chunk + embed: {rel} …")
                async with session_scope() as db:
                    ok, _sk = await persist_ingest_chunk_dicts(str(path), db, chunks)
                if ok > 0:
                    print(f"Đã lưu (CSV) — {rel}")
                    stats["ok"] += 1
                else:
                    print(f"Lỗi: không commit chunk nào — {rel}")
                    errors_list.append({"file": str(rel), "error": "no_chunks_committed"})
                    stats["failed"] += 1
                continue

            if suffix in _LLAMA_EXTENSIONS:
                print(f"LlamaParse / tài liệu: {rel}")
            elif suffix in _EXCEL_EXTENSIONS:
                print(f"Excel → Markdown: {rel}")
            else:
                print(f"Đọc text: {rel}")

            markdown_content, pipeline_tag = await _markdown_for_path(path)

            if not markdown_content or not markdown_content.strip():
                print(f"Nội dung rỗng sau parse, bỏ qua: {rel}")
                stats["skipped_empty"] += 1
                continue

            print(f"Chunk + embed: {rel} …")

            def _build_chunks() -> list[dict[str, Any]]:
                return build_policy_chunks_from_markdown(
                    markdown_content,
                    law_name=law_label,
                    source_filename=basename,
                    ingest_pipeline=pipeline_tag,
                )

            chunks = await asyncio.to_thread(_build_chunks)
            if not chunks:
                print(f"Không tạo được chunk, bỏ qua: {rel}")
                stats["skipped_empty"] += 1
                continue

            async with session_scope() as db:
                ok, _sk = await persist_ingest_chunk_dicts(str(path), db, chunks)

            if ok > 0:
                print(f"Đã lưu {ok} chunk — {rel}")
                stats["ok"] += 1
            else:
                print(f"Lỗi: không commit chunk nào — {rel}")
                errors_list.append({"file": str(rel), "error": "no_chunks_committed"})
                stats["failed"] += 1

        except LlamaParseIngestError as e:
            msg = str(e)
            print(f"LlamaParse lỗi [{rel}]: {msg}")
            errors_list.append({"file": str(rel), "error": msg})
            stats["failed"] += 1
            logger.exception("LlamaParse failed for %s", str(rel))
        except ValueError as e:
            es = str(e)
            if es.startswith("unsupported_extension"):
                print(f"Bỏ qua {rel}: {e}")
                stats["skipped_unsupported"] += 1
            elif es.startswith("llamaparse_required"):
                print(f"Lỗi [{rel}]: {e}")
                errors_list.append({"file": str(rel), "error": es})
                stats["failed"] += 1
            else:
                print(f"ValueError [{rel}]: {e}")
                errors_list.append({"file": str(rel), "error": es})
                stats["failed"] += 1
                logger.exception("ValueError for %s", str(rel))
        except Exception as e:
            msg = str(e)
            print(f"Lỗi [{rel}]: {msg}")
            errors_list.append({"file": str(rel), "error": msg})
            stats["failed"] += 1
            logger.exception("Ingest failed for %s", str(rel))

    print(
        f"\nHoàn tất — OK: {stats['ok']}, lỗi: {stats['failed']}, "
        f"bỏ qua (loại file): {stats['skipped_unsupported']}, rỗng: {stats['skipped_empty']}"
    )
    return stats


async def _amain() -> None:
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print('Usage: python -m tools.universal_ingestor "<folder_path>"')
        sys.exit(1)
    await process_raw_data_folder(sys.argv[1])


if __name__ == "__main__":
    asyncio.run(_amain())
