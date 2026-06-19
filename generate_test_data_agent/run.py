# run.py — synthetic test data + visual walkthrough for a CDS view (e.g. after rule_agent)

from agent.graph import build_graph

SENTINEL_CDS = "###END_CDS###"
SENTINEL_SRVD = "###END_SRVD###"
SENTINEL_CTX = "###END_CTX###"


def read_until(
    sentinel: str,
    title: str,
    *,
    allow_quit_first_line: bool = False,
    seed: str = "",
) -> str:
    print(title)
    print(f"End with {sentinel} on its own line or at the end of the last line.\n")
    parts: list[str] = []
    if seed.strip():
        parts.append(seed)
        joined = seed
        if sentinel in joined:
            return joined[: joined.find(sentinel)].strip()
    while True:
        try:
            line = input()
        except EOFError:
            break
        joined = "\n".join(parts + [line])
        if sentinel in joined:
            idx = joined.find(sentinel)
            return joined[:idx].strip()
        if allow_quit_first_line and len(parts) == 0 and line.strip().lower() in (
            "quit",
            "exit",
            "q",
        ):
            return "quit"
        parts.append(line)
    return "\n".join(parts).strip()


def read_optional_srvd() -> str:
    print(
        "\n--- Optional: RAP service definition (.srvd.xml from rule_agent) ---\n"
        "Leave the first line **empty** to skip.\n"
        "Otherwise paste the service-definition source and end with ###END_SRVD### (same line OK).\n"
    )
    first = input()
    if not first.strip():
        return ""
    if SENTINEL_SRVD in first:
        return first[: first.find(SENTINEL_SRVD)].strip()
    return read_until(SENTINEL_SRVD, "(continue .srvd.xml)", seed=first.rstrip("\n"))


def read_optional_rule_context() -> str:
    print(
        "\n--- Optional: rule / control description (from rule_agent summary or Q&A) ---\n"
        "Leave the first line **empty** to skip.\n"
        "Otherwise paste text and end with ###END_CTX### (same line OK).\n"
    )
    first = input()
    if not first.strip():
        return ""
    if SENTINEL_CTX in first:
        return first[: first.find(SENTINEL_CTX)].strip()
    return read_until(SENTINEL_CTX, "(continue context)", seed=first.rstrip("\n"))


def main() -> None:
    graph = build_graph()

    print("Generate Test Data Agent\n")
    print(
        "Use this after rule_agent (or any CDS). Paste the CDS DDL and optionally the "
        "`.srvd.xml` service definition. You get **synthetic** SAP rows plus **investigation "
        "case** payloads (case list + case detail) in Markdown, aligned with the cases UI mockup.\n"
    )

    while True:
        cds = read_until(
            SENTINEL_CDS,
            "--- CDS view (paste DDL) ---",
            allow_quit_first_line=True,
        )
        if cds.strip().lower() == "quit":
            print("Goodbye.")
            return
        if not cds:
            print("No CDS. Try again or quit.\n")
            continue

        srvd = read_optional_srvd()
        rule_ctx = read_optional_rule_context()

        state = graph.invoke(
            {
                "messages": [],
                "cds_source": cds,
                "srvd_source": srvd,
                "rule_context": rule_ctx,
                "session_ended": False,
            }
        )

        if state.get("messages"):
            print("\n", state["messages"][-1].content, "\n", sep="")

        again = input("Another CDS? [y/N]: ").strip().lower()
        if again not in ("y", "yes"):
            print("Goodbye.")
            return
        print()


if __name__ == "__main__":
    main()
