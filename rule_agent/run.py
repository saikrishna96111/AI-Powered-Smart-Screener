# run.py

from agent.graph import build_graph
from langchain_core.messages import HumanMessage

graph = build_graph()

state = {
    "messages": [],
    "intent": None,
    "required_fields": [],
    "collected_fields": {},
    "missing_fields": [],
    "explained": False,
    "cds_code": None
}

print("🚀 Rule Architect Agent Started\n")

while True:
    user_input = input("You: ")

    state = graph.invoke({
        **state,
        "messages": [HumanMessage(content=user_input)]
    })

    if state.get("messages"):
        last = state["messages"][-1]
        print("Agent:", last.content)