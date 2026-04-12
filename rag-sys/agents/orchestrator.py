"""
Orchestrator node: phân luồng ý định người dùng (policy / feedback / general).
"""

from __future__ import annotations

import logging
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from core.demo_log import log_agent_flow
from core.json_structured import ainvoke_json_model
from core.llm_factory import get_deepseek_flash
from core.state import MacaState

logger = logging.getLogger(__name__)

# Alias theo convention tài liệu (LLM singleton).
get_llm = get_deepseek_flash

_VALID_ROUTES = frozenset({"policy_qa", "feedback_analysis", "general_chat"})

_ORCHESTRATOR_SYSTEM = """You are the routing manager for a Logistics/Marketing company.
Classify the user's message into exactly ONE route. Output JSON with key "route" only.

Allowed values for "route" (use EXACTLY one of these strings, nothing else):
- policy_qa — questions about company rules, insurance, compensation limits, liability, packing, shipping terms, internal policies.
- feedback_analysis — customer complaints, VoC, sentiment, apology emails, analyzing reviews.
- general_chat — greetings, small talk, unrelated chitchat.

If the user asks about money limits, SDR, insurance, đền bù, bảo hiểm hàng, trách nhiệm công ty → policy_qa.
"""


class RouteSchema(BaseModel):
    """`route` là str (không dùng Literal) để model không bị fail khi trả tên tự đặt như compensation_inquiry."""

    route: str = Field(
        description="Exactly one of: policy_qa, feedback_analysis, general_chat",
    )

    @field_validator("route", mode="before")
    @classmethod
    def _strip_route(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v


def _content_from_human(msg: HumanMessage) -> str | None:
    c = msg.content
    if isinstance(c, str):
        t = c.strip()
        return t if t else None
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        t = " ".join(parts).strip()
        return t if t else None
    return None


def _last_user_text(messages: list[BaseMessage]) -> str | None:
    """
    Lấy nội dung HumanMessage gần nhất (duyệt từ cuối danh sách).
    Nếu tin cuối là assistant, vẫn dùng user gần nhất trước đó — client nên kết thúc luôn bằng user.
    """
    if messages and not isinstance(messages[-1], HumanMessage):
        logger.debug("messages[-1] không phải HumanMessage — dùng user gần nhất phía trước.")
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            text = _content_from_human(msg)
            if text:
                return text
    return None


def _normalize_route(raw: str | None) -> str:
    if not raw:
        return "general_chat"
    r = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if r in _VALID_ROUTES:
        return r
    # Một vài biến thể thường gặp từ model
    aliases = {
        "policy": "policy_qa",
        "policyqa": "policy_qa",
        "qa": "policy_qa",
        "policy_question": "policy_qa",
        "policy_inquiry": "policy_qa",
        "compensation_inquiry": "policy_qa",
        "compensation_question": "policy_qa",
        "insurance_inquiry": "policy_qa",
        "insurance_question": "policy_qa",
        "logistics_policy": "policy_qa",
        "legal_question": "policy_qa",
        "rules_question": "policy_qa",
        "feedback": "feedback_analysis",
        "complaint": "feedback_analysis",
        "analysis": "feedback_analysis",
        "voc": "feedback_analysis",
        "customer_feedback": "feedback_analysis",
        "general": "general_chat",
        "chitchat": "general_chat",
        "other": "general_chat",
    }
    if r in aliases:
        return aliases[r]
    # Heuristic: câu kiểu chính sách / bồi thường (model đặt tên route lạ)
    policy_tokens = (
        "compensat",
        "insurance",
        "liability",
        "indemn",
        "policy",
        "claim",
        "sdr",
        "premium",
    )
    if any(tok in r for tok in policy_tokens):
        logger.info("Unknown route label %r — mapped to policy_qa via heuristic", raw)
        return "policy_qa"
    if "feedback" in r or "complaint" in r or "customer" in r and "analys" in r:
        logger.info("Unknown route label %r — mapped to feedback_analysis via heuristic", raw)
        return "feedback_analysis"
    logger.warning("Unknown route from model: %r — falling back to general_chat", raw)
    return "general_chat"


async def run_orchestrator(state: MacaState) -> dict:
    """
    Đọc tin nhắn user cuối cùng, gọi LLM (structured output), cập nhật `current_route`.
    Lỗi / rỗng → `general_chat`.
    """
    messages = state.get("messages") or []
    user_text = _last_user_text(messages)
    if not user_text:
        logger.info("Orchestrator: no user message; route=general_chat")
        log_agent_flow("orchestrator", "skip — không có tin user", route="general_chat")
        return {"current_route": "general_chat"}

    llm = get_llm().bind(max_tokens=256)
    try:
        result = await ainvoke_json_model(
            llm,
            [
                SystemMessage(content=_ORCHESTRATOR_SYSTEM),
                HumanMessage(content=user_text),
            ],
            RouteSchema,
        )
    except Exception:
        logger.exception("Orchestrator: LLM call failed; route=general_chat")
        log_agent_flow("orchestrator", "lỗi LLM — fallback", route="general_chat", reason="exception")
        return {"current_route": "general_chat"}

    if result is None:
        logger.warning("Orchestrator: JSON parse failed; route=general_chat")
        log_agent_flow("orchestrator", "parse JSON thất bại — fallback", route="general_chat")
        return {"current_route": "general_chat"}

    route = _normalize_route(result.route)
    logger.info("Orchestrator: phân luồng → route=%s (raw_model=%r)", route, result.route)
    log_agent_flow(
        "orchestrator",
        "phân luồng ý định",
        route=route,
        raw_route_from_model=result.route,
        user_message_preview=user_text[:500] + ("…" if len(user_text) > 500 else ""),
    )
    return {"current_route": route}


# Tên thường đăng ký trong LangGraph
async def orchestrator_node(state: MacaState) -> dict:
    return await run_orchestrator(state)
