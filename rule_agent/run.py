# run.py

from agent.graph import build_graph
from langchain_core.messages import HumanMessage

graph = build_graph()

state = {
    "messages": [],
    "cds_flow_started": False,
    "intent": None,
    "required_fields": None,
    "collected_fields": {},
    "missing_fields": [],
    "explained": False,
    "approved": False,
    "cds_delivered": False,
    "cds_review_done": False,
    "session_ended": False,
    "cds_code": None,
    "summary_human_count": 0,
    "cds_syntax_status": None,
    "cds_syntax_issues": [],
    "cds_syntax_retries": 0,
    "cds_syntax_review_done": False,
    # CDS view parameter collection (date_parameter mandatory, additional optional).
    "params_phase": None,
    "cds_parameter_inputs": {},
    "cds_parameters": [],
    "parameters_collection_done": False,
}

print("Rule Architect Agent Started\n")

while True:
    user_input = input("You: ").strip()

    # Without a checkpointer each invoke starts the state fresh — explicitly carry the
    # prior message history forward so add_messages appends rather than wipes. Skipping
    # this makes extract_node see only the new "yes" with no prior assistant example,
    # which is what caused the yes-loop on key_tables / output_grain.
    prior_messages = list(state.get("messages") or [])
    state = graph.invoke({
        **state,
        "messages": prior_messages + [HumanMessage(content=user_input)],
    })

    if state.get("messages"):
        last = state["messages"][-1]
        print("Agent:", last.content)

    if state.get("session_ended"):
        break