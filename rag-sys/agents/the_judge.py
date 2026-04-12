"""
The Judge: tự kiểm (self-reflection) bản nháp so với retrieved_chunks — chuẩn enterprise.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.demo_log import log_agent_flow
from core.json_structured import ainvoke_json_model
from core.llm_factory import get_deepseek_flash
from core.state import MacaState

logger = logging.getLogger(__name__)

get_llm = get_deepseek_flash


class JudgeVerdict(BaseModel):
    is_valid: bool = Field(
        description=(
            "True only if draft_response strictly complies with retrieved evidence, tone, and completeness rules."
        )
    )
    feedback: str = Field(
        description=(
            "If invalid: concise explanation in Vietnamese (e.g. 'Email hứa hoàn tiền 100% nhưng chính sách chỉ 50%'). "
            "If valid: brief confirmation in Vietnamese."
        )
    )


_SYSTEM_JUDGE = """Bạn là QA Manager (MACA — logistics / marketing). Tiêu chí RẤT CHẶT CHẼ, ưu tiên an toàn doanh nghiệp.

So sánh `draft_response` với `retrieved_chunks` (đây là nguồn sự kiện được phép dùng).

Kiểm tra:
1) Độ chính xác sự kiện: Mọi con số, mức đền bù, thời hạn, cam kết pháp lý / hoàn tiền / bảo hiểm phải KHỚP với nội dung chunk. Nếu chunk không đề cập mà bản nháp khẳng định → KHÔNG HỢP LỆ.
2) Giọng văn công ty: trang trọng, tiếng Việt chuẩn, không slang, không hứa hẹn vượt quá bằng chứng.
3) Phản hồi khách (feedback_analysis): Nếu có chunk phản hồi, bản nháp phải phản ánh các khiếu nại / chủ đề chính; bỏ sót nghiêm trọng → KHÔNG HỢP LỆ.
4) policy_qa: Nếu chunk rỗng hoặc không chứa câu trả lời mà bản nháp vẫn đưa chi tiết cụ thể như đúng tài liệu → KHÔNG HỢP LỆ (coi như bịa).
5) general_chat: ít ràng buộc bằng chứng; chỉ cần lịch sự, không toxic, không lộ dữ liệu nội bộ giả định.

Chỉ đặt is_valid=true khi bạn CHẮC CHẮN không có vi phạm nào ở trên. Khi nghi ngờ → is_valid=false.

Trả về theo schema: is_valid (bool), feedback (str, tiếng Việt)."""


def _format_chunks_for_judge(chunks: list[dict]) -> str:
    if not chunks:
        return "(Không có đoạn truy hồi.)"
    parts: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        cit = ch.get("citation", f"Nguồn {i}")
        body = (ch.get("content") or "").strip()
        parts.append(f"[{i}] {cit}\n{body}")
    return "\n\n---\n\n".join(parts)


async def run_judge(state: MacaState) -> dict:
    """
    So sánh draft_response với retrieved_chunks; tăng retry_count chỉ khi is_valid=False (hoặc lỗi/bản nháp rỗng).
    """
    route = (state.get("current_route") or "general_chat").strip().lower()
    draft = (state.get("draft_response") or "").strip()
    chunks = list(state.get("retrieved_chunks") or [])
    prev_retry = int(state.get("retry_count") or 0)

    if not draft:
        logger.warning("Judge: empty draft_response → invalid")
        log_agent_flow("judge", "bản nháp rỗng — từ chối", route=route, is_valid=False)
        return {
            "is_valid": False,
            "judge_feedback": "Bản nháp trống; không thể duyệt.",
            "retry_count": prev_retry + 1,
        }

    context = _format_chunks_for_judge(chunks)
    re = (state.get("retrieval_error") or "").strip()
    re_line = f"\n[Lỗi truy hồi] {re}\n" if re else ""
    outage_hint = ""
    if re:
        outage_hint = (
            "\nLưu ý: Nếu có lỗi truy hồi/DB và bản nháp thành thật thông báo hệ thống tra cứu tạm ngưng, "
            "trả lời kiến thức chung có điều kiện và không giả đã có tài liệu nội bộ — có thể coi là hợp lệ.\n"
        )

    human = (
        f"Route hiện tại: {route}\n{re_line}{outage_hint}\n"
        f"=== DRAFT_RESPONSE ===\n{draft}\n\n"
        f"=== RETRIEVED_CHUNKS (nguồn kiểm chứng) ===\n{context}\n\n"
        "Hãy so sánh và phán quyết theo schema."
    )

    llm = get_llm().bind(max_tokens=512, temperature=0.0)

    log_agent_flow(
        "judge",
        "đang phán quyết",
        route=route,
        draft_chars=len(draft),
        chunk_count=len(chunks),
        retrieval_error=re or None,
        retry_count_before=prev_retry,
    )

    try:
        result = await ainvoke_json_model(
            llm,
            [
                SystemMessage(content=_SYSTEM_JUDGE),
                HumanMessage(content=human),
            ],
            JudgeVerdict,
        )
    except Exception:
        logger.exception("Judge: LLM call failed")
        log_agent_flow("judge", "lỗi LLM", route=route, is_valid=False)
        return {
            "is_valid": False,
            "judge_feedback": "Lỗi hệ thống khi chạy Judge; vui lòng thử lại.",
            "retry_count": prev_retry + 1,
        }

    if result is None:
        logger.warning("Judge: JSON parse failed")
        log_agent_flow("judge", "parse JSON thất bại", route=route, is_valid=False)
        return {
            "is_valid": False,
            "judge_feedback": "Phản hồi Judge không hợp lệ; từ chối duyệt an toàn.",
            "retry_count": prev_retry + 1,
        }

    feedback = (result.feedback or "").strip() or (
        "Hợp lệ." if result.is_valid else "Không đạt; cần sửa theo bằng chứng trong retrieved_chunks."
    )

    # Chỉ tăng retry_count khi chưa đạt (số lần cần sửa lại bản nháp).
    next_retry = prev_retry if result.is_valid else prev_retry + 1

    logger.info(
        "Judge: is_valid=%s retry_count=%d (was %d)",
        result.is_valid,
        next_retry,
        prev_retry,
    )
    log_agent_flow(
        "judge",
        "kết quả",
        is_valid=result.is_valid,
        retry_count_after=next_retry,
        feedback_preview=feedback[:400] + ("…" if len(feedback) > 400 else ""),
    )
    return {
        "is_valid": result.is_valid,
        "judge_feedback": feedback,
        "retry_count": next_retry,
    }


async def judge_node(state: MacaState) -> dict:
    return await run_judge(state)
