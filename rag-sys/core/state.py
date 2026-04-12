from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class MacaState(TypedDict, total=False):
    """State dùng chung giữa các node trong StateGraph của hệ thống MACA."""

    # 1. LỊCH SỬ VÀ PHÂN LUỒNG
    # Lịch sử hội thoại chuẩn LangGraph.
    messages: Annotated[list[BaseMessage], add_messages]

    # Kết quả phân luồng từ Orchestrator (VD: "policy_qa", "feedback_analysis", "general_chat")
    current_route: str


    # 2. XỬ LÝ CÂU HỎI & TRUY HỒI (QUERY FORMULATOR & RESEARCHER)
    # Câu hỏi đã được chuẩn hóa, loại bỏ từ ngữ nhiễu.
    standalone_query: str

    # Kỹ thuật HyDE: Đoạn văn giả định (Ví dụ: một câu luật mẫu hoặc một báo cáo mẫu)
    hypothetical_answer: str

    # Siêu dữ liệu trích xuất được dùng để Pre-filtering (Lọc JSONB trong Postgres)
    # VD: {"customer_code": "KH001", "sentiment": "negative", "route_name": "HN-HCM"}
    extracted_metadata: dict[str, Any]

    # Chỉ định đích đến của Retriever: "feedback" (bảng CSV) hoặc "policy" (bảng DOCX)
    target_table: str

    # Giới hạn top-K truy hồi (từ API / Streamlit).
    retrieval_limit: int

    # Danh sách các chunk dữ liệu truy hồi được từ DB (Sau khi đã qua Cross-Encoder Rerank)
    retrieved_chunks: list[dict[str, Any]]

    # Lỗi tầng truy hồi (DB / embedding); None hoặc bỏ key khi thành công.
    retrieval_error: str | None


    # 3. PHÂN TÍCH & TỔNG HỢP (ANALYST & CS MANAGER)
    # Báo cáo nội bộ do Agent Analyst sinh ra sau khi tổng hợp hàng chục dòng CSV.
    # (Agent CS Manager sẽ đọc cái này để biết đường xin lỗi khách).
    analysis_insight: str

    # Bản nháp cuối cùng (Có thể là Email CSKH, Báo cáo Marketing, hoặc câu trả lời trực tiếp)
    draft_response: str


    # 4. KIỂM DUYỆT & VÒNG LẶP (THE JUDGE)
    # Cờ kiểm duyệt từ The Judge (True = Hợp lệ cho xuất, False = Bắt lỗi, phải sửa)
    is_valid: bool
    
    # Lời phê của The Judge (VD: "Mức đền bù bị sai, sửa lại thành 50% theo chính sách")
    judge_feedback: str
    
    # Đếm số lần đã lặp lại để tránh infinite loop (thường set max = 3)
    retry_count: int