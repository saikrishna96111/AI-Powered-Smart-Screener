# agent/graph.py

from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    intent_node,
    requirements_node,
    extract_node,
    missing_node,
    question_node,
    explain_node,
    approval_node,
    cds_node,
)


def build_graph():

    workflow = StateGraph(AgentState)

    # -----------------------
    # Add Nodes
    # -----------------------

    workflow.add_node("intent", intent_node)
    workflow.add_node("requirements", requirements_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("missing", missing_node)
    workflow.add_node("question", question_node)
    workflow.add_node("explain", explain_node)
    workflow.add_node("approval", approval_node)
    workflow.add_node("cds", cds_node)

    # -----------------------
    # Entry Point
    # -----------------------

    workflow.set_entry_point("intent")

    # -----------------------
    # Initial Flow
    # -----------------------

    workflow.add_edge("intent", "requirements")
    workflow.add_edge("requirements", "missing")

    # -----------------------
    # Missing Fields Branch
    # -----------------------

    workflow.add_conditional_edges(
        "missing",
        lambda state: "question" if state["missing_fields"] else "explain"
    )

    workflow.add_edge("question", END)

    # -----------------------
    # When User Replies
    # Graph restarts → intent runs again
    # If intent already exists, it skips
    # Then we extract new info
    # -----------------------

    workflow.add_edge("intent", "extract")
    workflow.add_edge("extract", "missing")

    # -----------------------
    # After Explanation
    # -----------------------

    workflow.add_edge("explain", "approval")

    workflow.add_conditional_edges(
        "approval",
        lambda state: "cds" if state.get("approved") else END
    )

    workflow.add_edge("cds", END)

    return workflow.compile()