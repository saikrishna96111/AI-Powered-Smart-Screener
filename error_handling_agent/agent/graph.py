from langgraph.graph import END, StateGraph

from .nodes import fix_cds_node
from .state import ErrorAgentState


def build_graph():
    workflow = StateGraph(ErrorAgentState)
    workflow.add_node("fix_cds", fix_cds_node)
    workflow.set_entry_point("fix_cds")
    workflow.add_edge("fix_cds", END)
    return workflow.compile()
