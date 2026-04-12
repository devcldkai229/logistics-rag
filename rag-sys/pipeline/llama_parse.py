"""
LlamaParse (Llama Cloud): PDF/DOCX/PPTX… → Markdown (bảng + ảnh inline tùy cấu hình).

Cần `LLAMA_CLOUD_API_KEY` hoặc `LLAMA_PARSE_API_KEY` trong môi trường.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, cast

from llama_cloud import AsyncLlamaCloud, LlamaCloud
from llama_cloud.types.parsing_create_params import OutputOptions
from llama_cloud.types.parsing_get_response import ParsingGetResponse

ParseTier = Literal["fast", "cost_effective", "agentic", "agentic_plus"]

logger = logging.getLogger(__name__)


class LlamaParseIngestError(Exception):
    """Lỗi tầng LlamaParse (API, job FAILED, không có markdown)."""


def _api_key() -> str | None:
    return os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMA_PARSE_API_KEY")


def llamaparse_configured() -> bool:
    return bool(_api_key())


def _output_options() -> OutputOptions:
    inline_img = os.getenv("MACA_LLAMA_PARSE_INLINE_IMAGES", "true").lower() in ("1", "true", "yes")
    return {
        "markdown": {
            "tables": {
                "output_tables_as_markdown": True,
                "merge_continued_tables": True,
            },
            "inline_images": inline_img,
            "annotate_links": True,
        }
    }


def _tier_timeout() -> tuple[ParseTier, float]:
    tier_raw = os.getenv("MACA_LLAMA_PARSE_TIER", "agentic").strip().lower()
    if tier_raw not in ("fast", "cost_effective", "agentic", "agentic_plus"):
        tier_raw = "agentic"
    tier = cast(ParseTier, tier_raw)
    timeout = float(os.getenv("MACA_LLAMA_PARSE_TIMEOUT", "800"))
    return tier, timeout


def _extract_markdown_from_result(result: ParsingGetResponse) -> str:
    st = result.job.status
    if st == "FAILED":
        msg = result.job.error_message or "LlamaParse FAILED"
        raise LlamaParseIngestError(msg)
    if st in ("PENDING", "RUNNING", "CANCELLED"):
        raise LlamaParseIngestError(f"LlamaParse chưa hoàn tất hoặc đã hủy: status={st}")
    if st != "COMPLETED":
        raise LlamaParseIngestError(f"LlamaParse trạng thái không xác định: status={st}")

    if result.markdown_full and str(result.markdown_full).strip():
        return str(result.markdown_full).strip()

    if result.markdown and result.markdown.pages:
        parts: list[str] = []
        for page in sorted(result.markdown.pages, key=lambda p: p.page_number):
            if getattr(page, "success", False) is True:
                parts.append(page.markdown)
            else:
                err = getattr(page, "error", "unknown")
                logger.warning("LlamaParse page %s failed: %s", page.page_number, err)
        text = "\n\n".join(parts).strip()
        if text:
            return text

    raise LlamaParseIngestError("LlamaParse không trả về markdown (kiểm tra tier / định dạng file).")


async def parse_file_to_markdown(file_path: Path) -> str:
    """Async: upload + parse → markdown."""
    key = _api_key()
    if not key:
        raise ValueError(
            "Thiếu LLAMA_CLOUD_API_KEY (hoặc LLAMA_PARSE_API_KEY). "
            "Lấy key tại https://cloud.llamaindex.ai/"
        )

    tier, timeout = _tier_timeout()
    client = AsyncLlamaCloud(api_key=key)
    logger.info("LlamaParse: uploading %s (tier=%s)", file_path.name, tier)

    try:
        result = await client.parsing.parse(
            upload_file=file_path,
            tier=tier,
            version="latest",
            expand=["markdown", "markdown_full"],
            output_options=_output_options(),
            timeout=timeout,
        )
    except Exception as e:
        raise LlamaParseIngestError(str(e)) from e
    return _extract_markdown_from_result(result)


def parse_file_to_markdown_sync(file_path: str | Path) -> str:
    """
    Sync parse (dùng trong asyncio.to_thread hoặc script đồng bộ).
    Ném LlamaParseIngestError khi parse thất bại.
    """
    path = Path(file_path)
    key = _api_key()
    if not key:
        raise LlamaParseIngestError(
            "Thiếu LLAMA_CLOUD_API_KEY (hoặc LLAMA_PARSE_API_KEY). "
            "Lấy key tại https://cloud.llamaindex.ai/"
        )

    tier, timeout = _tier_timeout()
    client = LlamaCloud(api_key=key)
    logger.info("LlamaParse (sync): uploading %s (tier=%s)", path.name, tier)

    try:
        result = client.parsing.parse(
            upload_file=path,
            tier=tier,
            version="latest",
            expand=["markdown", "markdown_full"],
            output_options=_output_options(),
            timeout=timeout,
        )
    except Exception as e:
        raise LlamaParseIngestError(str(e)) from e
    return _extract_markdown_from_result(result)
