from langgraph.graph import END, StateGraph

from .nodes import fix_cds_node
from .retrieval.retriever_node import retrieve_docs_node
from .state import ErrorAgentState


def build_graph():
    """Two-node graph: retrieve grounding docs, then ask the LLM for a fix."""
    workflow = StateGraph(ErrorAgentState)

    workflow.add_node("retrieve_docs", retrieve_docs_node)
    workflow.add_node("fix_cds", fix_cds_node)

    workflow.set_entry_point("retrieve_docs")
    workflow.add_edge("retrieve_docs", "fix_cds")
    workflow.add_edge("fix_cds", END)

    return workflow.compile()
