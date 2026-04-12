"""
MACA master LangGraph: orchestrator → (query → research →) synthesizer → judge (+ retry).
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from agents.orchestrator import run_orchestrator
from agents.query_formulator import run_query_formulator
from agents.researcher import run_researcher
from agents.synthesizer import run_synthesizer
from agents.the_judge import run_judge
from core.state import MacaState


def route_after_orchestrator(
    state: MacaState,
) -> Literal["query_formulator", "synthesizer"]:
    if (state.get("current_route") or "").strip().lower() == "general_chat":
        return "synthesizer"
    return "query_formulator"


def route_after_judge(state: MacaState) -> Literal["synthesizer", "__end__"]:
    if state.get("is_valid") is True:
        return "__end__"
    # retry_count = số lần Judge trả về không hợp lệ (mỗi lần +1).
    if int(state.get("retry_count") or 0) >= 3:
        return "__end__"
    return "synthesizer"


def _build_graph() -> StateGraph:
    g = StateGraph(MacaState)

    g.add_node("orchestrator", run_orchestrator)
    g.add_node("query_formulator", run_query_formulator)
    g.add_node("researcher", run_researcher)
    g.add_node("synthesizer", run_synthesizer)
    g.add_node("judge", run_judge)

    g.add_edge(START, "orchestrator")

    g.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "query_formulator": "query_formulator",
            "synthesizer": "synthesizer",
        },
    )

    g.add_edge("query_formulator", "researcher")
    g.add_edge("researcher", "synthesizer")
    g.add_edge("synthesizer", "judge")

    g.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "synthesizer": "synthesizer",
            "__end__": END,
        },
    )

    return g


app = _build_graph().compile()
