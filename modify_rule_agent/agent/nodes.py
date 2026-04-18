import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .llm import get_llm
from .prompt_loader import load_prompt
from .utils import extract_text


def _last_human_instruction(messages) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            t = (m.content or "").strip()
            if t:
                return t
    return ""


def _extract_abap_fence(text: str) -> str | None:
    m = re.search(r"```(?:abap|cds)?\s*\n([\s\S]*?)```", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def step_node(state: dict) -> dict:
    """
    First graph call (no new human line): summarize CDS and ask what to modify.
    Later calls: last human message = modification instruction; emit updated CDS.
    """
    cds = (state.get("cds_working") or state.get("cds_original") or "").strip()
    if not cds:
        return {
            "messages": [
                AIMessage(content="No CDS was provided. Restart and paste your DDL.")
            ],
        }

    messages = state.get("messages") or []
    instruction = _last_human_instruction(messages)
    summary_sent = bool(state.get("summary_sent"))

    # First pass: only summarize (ignore empty or missing human instruction for modify)
    if not summary_sent:
        data = load_prompt("summarize")
        user_block = data["user"].replace("{{cds_source}}", cds)
        resp = get_llm().invoke(
            [
                SystemMessage(content=data["system"]),
                HumanMessage(content=user_block),
            ]
        )
        text = extract_text(resp).strip()
        return {
            "messages": [AIMessage(content=text)],
            "summary_sent": True,
        }

    # Modification pass
    if not instruction:
        return {
            "messages": [
                AIMessage(
                    content="Describe what you want changed (e.g. add company code filter, "
                    "change time window, add a field). Say **done** to exit."
                )
            ],
        }

    if instruction.lower() in ("done", "quit", "exit", "bye", "goodbye"):
        return {
            "messages": [AIMessage(content="Session ended.")],
            "session_ended": True,
        }

    data = load_prompt("modify")
    user_block = (
        data["user"]
        .replace("{{instruction}}", instruction)
        .replace("{{cds_source}}", cds)
    )

    resp = get_llm().invoke(
        [
            SystemMessage(content=data["system"]),
            HumanMessage(content=user_block),
        ]
    )
    body = extract_text(resp).strip()
    updated = _extract_abap_fence(body) or cds

    out: dict = {
        "messages": [AIMessage(content=body)],
        "cds_working": updated,
    }
    return out
