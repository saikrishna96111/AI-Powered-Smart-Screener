# run.py — modify existing CDS: summarize, then apply change requests in a loop

from langchain_core.messages import HumanMessage

from agent.graph import build_graph

SENTINEL_CDS = "###END_CDS###"


def read_cds_block() -> str:
    print("--- Paste your CDS DDL ---")
    print(
        f"End with {SENTINEL_CDS} (own line or at end of last line). "
        "Type quit on the first line only to exit.\n"
    )
    parts: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        joined = "\n".join(parts + [line])
        if SENTINEL_CDS in joined:
            idx = joined.find(SENTINEL_CDS)
            return joined[:idx].strip()
        if len(parts) == 0 and line.strip().lower() in ("quit", "exit", "q"):
            return "quit"
        parts.append(line)
    return "\n".join(parts).strip()


def main() -> None:
    graph = build_graph()

    print("Modify Rule Agent — summarize CDS, then describe changes.\n")

    cds = read_cds_block()
    if cds.strip().lower() == "quit":
        print("Goodbye.")
        return
    if not cds:
        print("No CDS pasted. Exiting.")
        return

    state = graph.invoke(
        {
            "messages": [],
            "modify_flow_started": True,
            "cds_original": cds,
            "cds_working": cds,
            "summary_sent": False,
            "session_ended": False,
        }
    )

    if state.get("messages"):
        print("\nAgent:\n", state["messages"][-1].content, "\n", sep="")

    print(
        "Describe what to change (natural language). "
        "The agent edits the **current** working copy each time.\n"
        "Say **done** when finished.\n"
    )

    while not state.get("session_ended"):
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        # Without a checkpointer, invoke starts state fresh — carry prior history
        # forward so add_messages appends. Otherwise nodes only see the new turn.
        prior_messages = list(state.get("messages") or [])
        state = graph.invoke(
            {
                **state,
                "messages": prior_messages + [HumanMessage(content=user_input)],
            }
        )

        if state.get("messages"):
            print("\nAgent:\n", state["messages"][-1].content, "\n", sep="")

        if state.get("session_ended"):
            break


if __name__ == "__main__":
    main()
