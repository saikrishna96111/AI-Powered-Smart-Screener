from langgraph.graph import END, StateGraph

from .nodes import generate_node
from .state import GenerateTestDataState


def build_graph():
    workflow = StateGraph(GenerateTestDataState)
    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", END)
    return workflow.compile()
