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
    cds_review_node,
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
    workflow.add_node("cds_review", cds_review_node)

    # -----------------------
    # Entry Point
    # -----------------------

    workflow.set_entry_point("intent")

    # -----------------------
    # First time: ask for requirements. Later: extract from user reply.
    # -----------------------

    # First time (no required_fields yet) → requirements; user replying → extract
    workflow.add_conditional_edges(
        "intent",
        lambda state: (
            "extract"
            if state.get("required_fields") is not None
            else "requirements"
        ),
    )
    workflow.add_edge("requirements", "missing")
    workflow.add_edge("extract", "missing")

    # -----------------------
    # Missing Fields Branch
    # -----------------------

    workflow.add_conditional_edges(
        "missing",
        lambda state: "question" if state["missing_fields"] else "explain"
    )

    workflow.add_edge("question", END)

    # -----------------------
    # After Explanation
    # -----------------------

    workflow.add_edge("explain", "approval")

    # Only generate CDS when approved and not already delivered
    workflow.add_conditional_edges(
        "approval",
        lambda state: "cds" if (state.get("approved") and not state.get("cds_delivered")) else END
    )

    workflow.add_edge("cds", "cds_review")
    workflow.add_edge("cds_review", END)

    return workflow.compile()