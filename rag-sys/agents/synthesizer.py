"""
Synthesizer: tổng hợp câu trả lời cuối từ retrieved_chunks + persona theo route.
"""

from __future__ import annotations

import logging
import os

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agents.orchestrator import _last_user_text
from core.demo_log import log_agent_flow
from core.llm_factory import get_deepseek_flash
from core.state import MacaState

logger = logging.getLogger(__name__)

get_llm = get_deepseek_flash

_SYSTEM_POLICY_QA = """Bạn là chuyên viên tư vấn chính sách nội bộ (logistics, bảo hiểm hàng hóa, đền bù, đóng gói).

QUY TẮC BẮT BUỘC:
- Chỉ được phép dựa vào các "Đoạn truy hồi" đính kèm. Không bịa, không bổ sung kiến thức bên ngoài.
- Nếu các đoạn không chứa đủ thông tin để trả lời, hãy nói rõ bằng tiếng Việt rằng bạn không tìm thấy trong tài liệu đã truy hồi và gợi ý người dùng liên hệ bộ phận phụ trách hoặc cung cấp thêm ngữ cảnh.
- Khi trả lời, phải trích dẫn nguồn: ví dụ "Theo chính sách [tên tài liệu / mục]…", hoặc "Theo đoạn trích [citation]…".
- Giọng điệu trang trọng, súc tích. Toàn bộ nội dung trả lời bằng tiếng Việt."""

_SYSTEM_FEEDBACK = """Bạn là Trưởng nhóm Chăm sóc khách hàng (CS Manager): đồng cảm, chuyên nghiệp, rõ ràng.

Nhiệm vụ (theo thứ tự):
1) Tóm tắt insight từ các đoạn phản hồi khách hàng đã truy hồi (ví dụ: chủ đề lặp lại, sentiment, số lượng / tỷ lệ nếu có cơ sở trong đoạn). Nếu không có đoạn truy hồi, nêu rõ "chưa có dữ liệu phản hồi được truy hồi từ kho".
2) Soạn phần tiếp theo theo đúng yêu cầu của người dùng: email xin lỗi, đề xuất xử lý, hoặc giải thích — luôn lịch sự và có thể hành động.
- Khi dựa vào nội dung chunk, hãy gắn với nguồn (citation) khi phù hợp.
- Toàn bộ bằng tiếng Việt."""

_SYSTEM_GENERAL = """Bạn là trợ lý MACA (Marketing & Customer Assistant) cho doanh nghiệp logistics / marketing.
Trả lời ngắn gọn, thân thiện, lịch sự, hoàn toàn bằng tiếng Việt. Không cần trích dẫn tài liệu nội bộ trừ khi người dùng hỏi cụ thể."""

# Khi tầng truy hồi (PostgreSQL / pgvector) lỗi — không có chunk đáng tin.
_SYSTEM_OUTAGE_POLICY = """Bạn là chuyên viên tư vấn MACA (logistics / chính sách).

[TÌNH HUỐNG] Hệ thống tra cứu tài liệu nội bộ (cơ sở dữ liệu / vector) hiện không khả dụng hoặc đang bảo trì. Bạn KHÔNG có đoạn trích chính sách thực tế từ công ty.

Nhiệm vụ (bắt buộc):
1) Mở đầu lịch sự, tiếng Việt: ví dụ nhắc rằng tra cứu tài liệu nội bộ đang tạm ngưng / bảo trì nên chưa thể lấy văn bản chính thức.
2) Trả lời câu hỏi ở mức kiến thức chung, thực hành tốt trong ngành (nếu phù hợp), nhưng phải nói rõ đây KHÔNG phải trích dẫn từ tài liệu nội bộ và cần xác minh với bộ phận phụ trách.
3) Tuyệt đối không giả đã đọc hay trích từ policy cụ thể của công ty.
4) Ngắn gọn, chuyên nghiệp."""

_SYSTEM_OUTAGE_FEEDBACK = """Bạn là Trưởng nhóm Chăm sóc khách hàng MACA.

[TÌNH HUỐNG] Kho phản hồi khách hàng (database) hiện không truy vấn được — có thể do bảo trì hoặc lỗi kết nối. Bạn KHÔNG có đoạn phản hồi thực tế từ hệ thống.

Nhiệm vụ (bắt buộc):
1) Thông báo nhẹ nhàng, tiếng Việt: tra cứu dữ liệu phản hồi đang tạm không dùng được; xin lỗi vì bất tiện.
2) Vẫn có thể gợi ý khung xử lý / email mẫu chung theo yêu cầu người dùng, nhưng không được giả đã thấy feedback cụ thể của khách.
3) Khuyên người dùng thử lại sau hoặc cung cấp mã đơn / kênh để xử lý thủ công nếu cần.
4) Giọng đồng cảm, chuyên nghiệp."""


def _format_retrieved_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(Không có đoạn truy hồi nào.)"
    blocks: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        citation = ch.get("citation") or f"Nguồn {i}"
        content = (ch.get("content") or "").strip()
        src = ch.get("source", "")
        label = f"[{i}] {citation}"
        if src:
            label += f" (loại: {src})"
        blocks.append(f"{label}\n{content}")
    return "\n\n---\n\n".join(blocks)


def _ai_text(message: BaseMessage) -> str:
    raw = message.content
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts: list[str] = []
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts).strip()
    return str(raw).strip()


async def run_synthesizer(state: MacaState) -> dict:
    """
    Sinh `draft_response` theo route: policy nghiêm ngặt, feedback đồng cảm + email/đề xuất, general_chat thân thiện.
    """
    route = (state.get("current_route") or "general_chat").strip().lower()
    messages = state.get("messages") or []
    user_text = _last_user_text(messages) or ""

    chunks = list(state.get("retrieved_chunks") or [])
    context = _format_retrieved_context(chunks)

    judge_note = state.get("judge_feedback")
    retrieval_err = (state.get("retrieval_error") or "").strip()
    retrieval_outage = bool(retrieval_err) and route in ("policy_qa", "feedback_analysis")

    system_note = ""
    if retrieval_err and not retrieval_outage:
        system_note += (
            f"\n[Lỗi truy hồi — nội bộ] {retrieval_err} "
            "Hãy trả lời thận trọng, không giả định có dữ liệu DB.\n"
        )
    if judge_note:
        system_note += f"\n[Ghi chú Judge / chỉnh sửa bản nháp] {judge_note}\n"

    if retrieval_outage:
        if route == "policy_qa":
            system = _SYSTEM_OUTAGE_POLICY
        else:
            system = _SYSTEM_OUTAGE_FEEDBACK
        system += (
            f"\n\n[Chi tiết kỹ thuật nội bộ — không trích nguyên văn cho user] {retrieval_err}"
        )
    elif route == "policy_qa":
        system = _SYSTEM_POLICY_QA
        if not chunks:
            system += (
                "\nHiện không có đoạn truy hồi. Bạn chỉ được trả lời rằng không tìm thấy thông tin trong "
                "tài liệu chính sách đã truy hồi và không được bịa chi tiết."
            )
    elif route == "feedback_analysis":
        system = _SYSTEM_FEEDBACK
        if not chunks:
            system += (
                "\nKhông có đoạn phản hồi khách hàng nào được truy hồi: phần (1) phải nêu rõ điều đó; "
                "phần (2) vẫn có thể soạn khung email / hướng xử lý chung theo yêu cầu người dùng, "
                "hoặc đề nghị họ cung cấp thêm mã đơn / kênh / thời gian."
            )
    else:
        system = _SYSTEM_GENERAL

    human_body = (
        f"{system_note}"
        f"Yêu cầu / tin nhắn người dùng:\n{user_text or '(trống)'}\n\n"
        f"=== ĐOẠN TRUY HỒI (dùng khi route là policy_qa hoặc feedback_analysis) ===\n{context}"
    )

    llm = get_llm().bind(max_tokens=int(os.getenv("MACA_SYNTH_MAX_TOKENS", "2048")))

    log_agent_flow(
        "synthesizer",
        "tổng hợp bản nháp",
        route=route,
        retrieval_error=retrieval_err or None,
        retrieval_outage_mode=retrieval_outage,
        chunk_count=len(chunks),
        has_judge_note=bool(judge_note),
    )

    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=human_body),
            ]
        )
    except Exception:
        logger.exception("Synthesizer: LLM invoke failed")
        return {
            "draft_response": (
                "Xin lỗi, hệ thống tạm thời không tạo được câu trả lời. "
                "Vui lòng thử lại sau hoặc liên hệ hỗ trợ kỹ thuật."
            )
        }

    if not isinstance(response, AIMessage):
        logger.warning("Synthesizer: unexpected message type %s", type(response))

    text = _ai_text(response)
    if not text:
        text = (
            "Không tạo được nội dung. Vui lòng thử lại với câu hỏi ngắn hơn hoặc kiểm tra kết nối mô hình."
        )

    logger.info("Synthesizer: draft_response length=%d chars (route=%s, chunks=%d)", len(text), route, len(chunks))
    return {"draft_response": text}


async def synthesizer_node(state: MacaState) -> dict:
    return await run_synthesizer(state)
