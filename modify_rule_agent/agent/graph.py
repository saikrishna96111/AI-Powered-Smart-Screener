from langgraph.graph import END, StateGraph

from .nodes import step_node
from .state import ModifyRuleState


def build_graph():
    workflow = StateGraph(ModifyRuleState)
    workflow.add_node("step", step_node)
    workflow.set_entry_point("step")
    workflow.add_edge("step", END)
    return workflow.compile()
