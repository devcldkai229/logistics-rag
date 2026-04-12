"""
Query formulator: standalone query, HyDE, target_table và metadata lọc Postgres.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from agents.orchestrator import _last_user_text
from core.demo_log import log_agent_flow
from core.json_structured import ainvoke_json_model
from core.llm_factory import get_deepseek_flash
from core.state import MacaState

logger = logging.getLogger(__name__)

get_llm = get_deepseek_flash

_PROMPT_FEEDBACK = """You are preparing search parameters for a customer feedback / VoC database (PostgreSQL).

User route: feedback_analysis.

Instructions:
- Produce a concise standalone_query: the user's request cleaned up for search (no fluff).
- Extract customer codes, routes, sentiments, ratings, order codes, or feedback IDs mentioned in the message into extracted_metadata. Use keys like customer_code, route_name, sentiment, rating, order_code, feedback_id when applicable. Values must be JSON-friendly (e.g. rating as integer).
- Set hypothetical_answer to exactly an empty string "". DO NOT generate a hypothetical document.
- Set target_table to exactly "feedback".
"""

_PROMPT_POLICY = """You are preparing search parameters for company policy / logistics rule documents (RAG over Markdown chunks).

User route: policy_qa.

Instructions:
- Produce a concise standalone_query: the user's question cleaned up for retrieval.
- Generate hypothetical_answer: a short, highly professional policy-style excerpt (as if from an internal handbook) that would answer the query. This is for HyDE-style dense retrieval only — it must sound plausible but can be synthetic.
- Leave extracted_metadata empty: use an empty object {{}} — no pre-filters.
- Set target_table to exactly "policy".
"""


class QuerySchema(BaseModel):
    standalone_query: str = Field(description="Cleaned up user query")
    hypothetical_answer: str = Field(
        default="",
        description=(
            "A fake, highly professional document excerpt that answers the query. "
            "Leave empty if route is feedback_analysis."
        ),
    )
    target_table: str = Field(
        description="Must be 'policy' if current_route is 'policy_qa', else 'feedback'"
    )
    extracted_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extract any specific filters mentioned, e.g., "
            "{'customer_code': 'KH001', 'route_name': 'HN-HCM', 'sentiment': 'negative', 'rating': 1}. "
            "Leave empty if none found."
        ),
    )

    @field_validator("standalone_query", "hypothetical_answer", "target_table", mode="before")
    @classmethod
    def _strip_str(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v


def _coerce_metadata(raw: object) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    return {}


async def run_query_formulator(state: MacaState) -> dict:
    """
    Cập nhật standalone_query, hypothetical_answer, target_table, extracted_metadata.
    `general_chat` → không chỉnh state (trả dict rỗng).
    """
    route = (state.get("current_route") or "").strip().lower()
    if route == "general_chat":
        logger.info("Query formulator: route=general_chat — skip formulation")
        log_agent_flow("query_formulator", "bỏ qua — general_chat")
        return {}

    if route not in ("policy_qa", "feedback_analysis"):
        logger.warning("Query formulator: unknown route %r — skip formulation", route)
        log_agent_flow("query_formulator", "bỏ qua — route không hỗ trợ", current_route=route)
        return {}

    messages = state.get("messages") or []
    user_text = _last_user_text(messages) or ""
    if not user_text.strip():
        logger.warning("Query formulator: no user text; applying minimal defaults")
        target = "policy" if route == "policy_qa" else "feedback"
        log_agent_flow(
            "query_formulator",
            "mặc định tối thiểu (không có text user)",
            standalone_query="",
            hypothetical_answer="",
            target_table=target,
            extracted_metadata={},
        )
        return {
            "standalone_query": "",
            "hypothetical_answer": "",
            "target_table": target,
            "extracted_metadata": {},
        }

    if route == "feedback_analysis":
        system_content = _PROMPT_FEEDBACK
    else:
        system_content = _PROMPT_POLICY

    llm = get_llm().bind(max_tokens=768)
    try:
        result = await ainvoke_json_model(
            llm,
            [
                SystemMessage(content=system_content),
                HumanMessage(content=user_text),
            ],
            QuerySchema,
        )
    except Exception:
        logger.exception("Query formulator: LLM call failed; using fallback fields")
        target = "policy" if route == "policy_qa" else "feedback"
        log_agent_flow(
            "query_formulator",
            "fallback sau lỗi LLM",
            standalone_query=user_text,
            hypothetical_answer="",
            target_table=target,
            extracted_metadata={},
        )
        return {
            "standalone_query": user_text,
            "hypothetical_answer": "",
            "target_table": target,
            "extracted_metadata": {},
        }

    if result is None:
        logger.warning("Query formulator: JSON parse failed; using fallback fields")
        target = "policy" if route == "policy_qa" else "feedback"
        log_agent_flow(
            "query_formulator",
            "fallback sau parse JSON thất bại",
            standalone_query=user_text,
            hypothetical_answer="",
            target_table=target,
            extracted_metadata={},
        )
        return {
            "standalone_query": user_text,
            "hypothetical_answer": "",
            "target_table": target,
            "extracted_metadata": {},
        }

    # Áp rule nghiệp vụ theo route (không tin hoàn toàn vào model)
    if route == "feedback_analysis":
        hypothetical = ""
        metadata = _coerce_metadata(result.extracted_metadata)
        target_table = "feedback"
    else:
        hypothetical = (result.hypothetical_answer or "").strip()
        metadata = {}
        target_table = "policy"

    standalone = (result.standalone_query or user_text).strip()

    out = {
        "standalone_query": standalone,
        "hypothetical_answer": hypothetical,
        "target_table": target_table,
        "extracted_metadata": metadata,
    }

    logger.info(
        "Query formulator: target_table=%s extracted_metadata=%s standalone_preview=%r",
        out["target_table"],
        out["extracted_metadata"],
        standalone[:120] + ("..." if len(standalone) > 120 else ""),
    )
    log_agent_flow(
        "query_formulator",
        "chuẩn bị truy hồi (trước Researcher)",
        current_route=route,
        standalone_query=standalone,
        hypothetical_answer=hypothetical or "(rỗng — đúng kỳ vọng cho feedback_analysis)",
        target_table=target_table,
        extracted_metadata=metadata,
    )
    return out


async def query_formulator_node(state: MacaState) -> dict:
    return await run_query_formulator(state)
