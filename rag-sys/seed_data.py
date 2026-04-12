"""
Nạp dữ liệu mẫu FeedbackChunk + PolicyChunk (embedding BGE-M3, vector 1024).

Chạy từ thư mục rag-sys:
  python seed_data.py
"""

from __future__ import annotations

import asyncio

import data.models  # noqa: F401 — đăng ký ORM với Base

from data.database import session_scope
from data.models import FeedbackChunk, PolicyChunk
from pipeline.embedder import generate_embedding


async def seed_maca_data() -> None:
    print("Dang chuan bi du lieu mau cho MACA...")

    sample_feedbacks = [
        {
            "feedback_id": "FB0000150",
            "order_code": "ORD0003989",
            "customer_code": "KH00242",
            "customer_tier": "Silver",
            "route_name": "HN - Shenzhen",
            "rating": 4,
            "sentiment": "positive",
            "content": "Khách hàng KH00242 (Hạng Silver) phản hồi về đơn ORD0003989 tuyến HN - Shenzhen. Cảm xúc: positive. Nội dung: Hàng về an toàn, phí cước ổn. Thời gian xử lý hơi lâu hơn mong đợi.",
            "metadata": {"order_type": "Express", "product_name": "Áo hoodie Uniqlo fleece"},
        },
        {
            "feedback_id": "FB0000272",
            "order_code": "ORD0003081",
            "customer_code": "KH00469",
            "customer_tier": "Standard",
            "route_name": "HCM - HN",
            "rating": 2,
            "sentiment": "negative",
            "content": "Khách hàng KH00469 (Hạng Standard) phản hồi về đơn ORD0003081 tuyến HCM - HN. Cảm xúc: negative. Nội dung: Nhân viên hỗ trợ phản hồi chậm, phải nhắn nhiều lần.",
            "metadata": {"order_type": "Standard", "product_name": "Đồ chơi Montessori gỗ 6 món"},
        },
    ]

    sample_policies = [
        {
            "document_name": "Chinh_sach_hop_dong.docx",
            "section_title": "1. Hình thành hợp đồng",
            "content": "Hợp đồng dịch vụ giữa khách hàng và [Công ty] được hình thành khi khách hàng xác nhận sử dụng dịch vụ chuyển hàng từ [Nước xuất] sang [Nước nhập], hoàn tất thanh toán và đồng ý với các điều khoản dịch vụ.",
            "metadata": {"effective_date": "2024-04-11", "category": "Legal"},
        },
        {
            "document_name": "Chinh_sach_hop_dong.docx",
            "section_title": "2. Trách nhiệm điều chỉnh thông tin",
            "content": "Khách hàng chịu trách nhiệm đảm bảo tính chính xác và đầy đủ của thông tin hàng hóa. [Công ty] chỉ hỗ trợ điều chỉnh khi hàng hóa chưa được bàn giao cho đơn vị vận chuyển hoặc chưa phát sinh thủ tục hải quan.",
            "metadata": {"importance": "High", "scope": "Customer Responsibility"},
        },
    ]

    async with session_scope() as db:
        print("Dang nhung vector cho feedback...")
        for item in sample_feedbacks:
            vector = await asyncio.to_thread(generate_embedding, item["content"])
            db.add(
                FeedbackChunk(
                    feedback_id=item["feedback_id"],
                    order_code=item["order_code"],
                    customer_code=item["customer_code"],
                    customer_tier=item["customer_tier"],
                    route_name=item["route_name"],
                    rating=item["rating"],
                    sentiment=item["sentiment"],
                    content=item["content"],
                    metadata_json=item["metadata"],
                    embedding=vector,
                )
            )

        print("Dang nhung vector cho policy...")
        for item in sample_policies:
            vector = await asyncio.to_thread(generate_embedding, item["content"])
            db.add(
                PolicyChunk(
                    document_name=item["document_name"],
                    section_title=item["section_title"],
                    content=item["content"],
                    metadata_json=item["metadata"],
                    embedding=vector,
                )
            )

        await db.commit()
        print(
            f"Da nap {len(sample_feedbacks)} feedback va {len(sample_policies)} policy "
            "(chay lai script se them ban ghi trung lap neu chua xoa DB)."
        )


if __name__ == "__main__":
    asyncio.run(seed_maca_data())
