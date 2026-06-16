from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .llm import get_llm
from .prompt_loader import load_prompt
from .utils import extract_text


def generate_node(state: dict) -> dict:
    cds = (state.get("cds_source") or "").strip()
    if not cds:
        return {
            "messages": [
                AIMessage(
                    content="No CDS was provided. Paste your view DDL ending with ###END_CDS###."
                )
            ],
        }

    ctx = (state.get("rule_context") or "").strip()
    srvd = (state.get("srvd_source") or "").strip()
    data = load_prompt("generate")
    user_block = (
        data["user"]
        .replace("{{cds_source}}", cds)
        .replace("{{srvd_source}}", srvd or "(none — infer service from CDS only)")
        .replace("{{rule_context}}", ctx or "(none)")
    )

    resp = get_llm().invoke(
        [
            SystemMessage(content=data["system"]),
            HumanMessage(content=user_block),
        ]
    )
    text = extract_text(resp).strip()

    return {
        "messages": [AIMessage(content=text)],
    }
