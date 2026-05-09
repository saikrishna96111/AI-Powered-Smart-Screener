from langgraph.graph import END, StateGraph

from .nodes import conversation_gate_node, step_node
from .state import ModifyRuleState


def build_graph():
    workflow = StateGraph(ModifyRuleState)
    workflow.add_node("conversation_gate", conversation_gate_node)
    workflow.add_node("step", step_node)
    workflow.set_entry_point("conversation_gate")
    workflow.add_conditional_edges(
        "conversation_gate",
        lambda state: "step" if state.get("modify_flow_started") else END,
    )
    workflow.add_edge("step", END)
    return workflow.compile()
