# run.py — CDS error-handling agent (paste DDL + error, get corrected CDS)

from agent.graph import build_graph

SENTINEL_CDS = "###END_CDS###"
SENTINEL_ERR = "###END_ERR###"


def read_block(title: str, sentinel: str, seed: str = "", allow_quit: bool = False) -> tuple[str, str]:
    """
    Read lines until `sentinel` appears anywhere in the accumulated text (not only on its own line).

    Returns (body_before_sentinel, text_after_sentinel). Trailing text is fed into the next read_block
    via `seed` so one paste can be: ...DDL...###END_CDS###...error...###END_ERR###
    """
    print(title)
    print(
        f"End the CDS or error section with {sentinel}.\n"
        f"It can be on its own line or at the end of the last line (same line is OK).\n"
        f"You can paste CDS and error in one go if both sentinels appear.\n"
    )

    parts: list[str] = []
    if seed.strip():
        parts.append(seed)

    while True:
        joined = "\n".join(parts)
        if sentinel in joined:
            idx = joined.find(sentinel)
            body = joined[:idx].strip()
            tail = joined[idx + len(sentinel) :].strip()
            return body, tail

        if allow_quit and len(parts) == 1 and parts[0].strip().lower() in ("quit", "exit", "q"):
            return "quit", ""

        try:
            line = input()
        except EOFError:
            joined = "\n".join(parts)
            if sentinel in joined:
                idx = joined.find(sentinel)
                return joined[:idx].strip(), joined[idx + len(sentinel) :].strip()
            return joined.strip(), ""

        parts.append(line)


def main() -> None:
    graph = build_graph()

    print("CDS Error-Handling Agent\n")
    print(
        f"Use {SENTINEL_CDS} after your DDL and {SENTINEL_ERR} after the error text.\n"
        "Sentinels can sit at the end of a line (no need for a blank line).\n"
        "At the CDS prompt, a lone line quit / exit / q exits.\n"
    )

    while True:
        cds, after_cds = read_block(
            "--- CDS source ---", SENTINEL_CDS, seed="", allow_quit=True
        )
        if cds.strip().lower() == "quit":
            print("Goodbye.")
            break

        err, _after_err = read_block(
            "--- Error message ---", SENTINEL_ERR, seed=after_cds, allow_quit=False
        )

        if not cds or not err:
            print("Skipping: empty CDS or empty error.\n")
            continue

        state = graph.invoke(
            {
                "messages": [],
                "cds_source": cds,
                "error_text": err,
                "fixed_cds": None,
                "session_ended": False,
            }
        )

        if state.get("messages"):
            last = state["messages"][-1]
            print("\nAgent:\n", last.content, "\n", sep="")

        if state.get("fixed_cds"):
            n = len(state["fixed_cds"])
            print(
                f"\n(Parsed updated DDL from fence: {n} characters. "
                "Full source is in the ```abap``` block above.)\n"
            )

        again = input("\nAnother fix? [y/N]: ").strip().lower()
        if again not in ("y", "yes"):
            print("Goodbye.")
            break
        print()


if __name__ == "__main__":
    main()
