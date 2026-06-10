# agent/graph.py

from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    conversation_gate_node,
    intent_node,
    requirements_node,
    extract_node,
    missing_node,
    question_node,
    parameters_node,
    explain_node,
    approval_node,
    retrieve_examples_node,
    cds_node,
    syntax_review_node,
    cds_review_node,
)


def build_graph():

    workflow = StateGraph(AgentState)

    # -----------------------
    # Add Nodes
    # -----------------------

    workflow.add_node("intent", intent_node)
    workflow.add_node("conversation_gate", conversation_gate_node)
    workflow.add_node("requirements", requirements_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("missing", missing_node)
    workflow.add_node("question", question_node)
    workflow.add_node("parameters", parameters_node)
    workflow.add_node("explain", explain_node)
    workflow.add_node("approval", approval_node)
    workflow.add_node("retrieve_examples", retrieve_examples_node)
    workflow.add_node("cds", cds_node)
    workflow.add_node("syntax_review", syntax_review_node)
    workflow.add_node("cds_review", cds_review_node)

    # -----------------------
    # Entry Point
    # -----------------------

    workflow.set_entry_point("conversation_gate")

    workflow.add_conditional_edges(
        "conversation_gate",
        lambda state: "intent" if state.get("cds_flow_started") else END
    )

    # -----------------------
    # PARAMETERS FIRST: as soon as intent is captured, collect the CDS view
    # parameters (mandatory date + optional extras). Only AFTER that do we plan
    # / ask the remaining business-design questions, and the planner will skip
    # any field the user already declared as a parameter.
    # -----------------------

    def _route_after_intent(state):
        if not state.get("parameters_collection_done"):
            return "parameters"
        if state.get("required_fields") is None:
            return "requirements"
        return "extract"

    workflow.add_conditional_edges("intent", _route_after_intent)

    # parameters_node either emitted a question (turn ends) or finished collection
    # (fall through to requirements/extract in the same invoke so the user sees
    # the first design question immediately after the last parameter reply).
    def _route_after_parameters(state):
        if not state.get("parameters_collection_done"):
            return END
        if state.get("required_fields") is None:
            return "requirements"
        return "extract"

    workflow.add_conditional_edges("parameters", _route_after_parameters)

    workflow.add_edge("requirements", "missing")
    workflow.add_edge("extract", "missing")

    # -----------------------
    # Missing Fields Branch
    # -----------------------

    workflow.add_conditional_edges(
        "missing",
        lambda state: "question" if state["missing_fields"] else "explain",
    )

    workflow.add_edge("question", END)

    # -----------------------
    # After Explanation
    # -----------------------

    workflow.add_edge("explain", "approval")

    # Only generate CDS when approved and not already delivered. We first
    # hop through retrieve_examples to pull the closest gold CDS view(s) from
    # the shared Chroma store so cds_node has a working template to mirror.
    workflow.add_conditional_edges(
        "approval",
        lambda state: "retrieve_examples" if (state.get("approved") and not state.get("cds_delivered")) else END
    )
    workflow.add_edge("retrieve_examples", "cds")

    # Syntax-correctness comes BEFORE the engineering/perf review so cds_review
    # always operates on a CDS that already activates in ADT.
    workflow.add_edge("cds", "syntax_review")
    workflow.add_edge("syntax_review", "cds_review")
    workflow.add_edge("cds_review", END)

    return workflow.compile()