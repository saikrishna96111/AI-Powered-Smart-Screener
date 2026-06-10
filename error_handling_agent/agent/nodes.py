import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .llm import get_llm
from .prompt_loader import load_prompt
from .utils import extract_text


def _extract_abap_fence(text: str) -> str | None:
    """First ```abap``` block = primary (minimal) fix per prompt contract."""
    m = re.search(r"```(?:abap|cds)?\s*\n([\s\S]*?)```", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def fix_cds_node(state: dict) -> dict:
    cds = (state.get("cds_source") or "").strip()
    err = (state.get("error_text") or "").strip()

    if not cds or not err:
        return {
            "messages": [
                AIMessage(
                    content="Both CDS source and error text are required. "
                    "Restart and paste each block ending with the sentinel line."
                )
            ],
        }

    references_text = (state.get("references_text") or "").strip()
    if not references_text:
        references_text = "(no reference excerpts retrieved)"

    data = load_prompt("fix")
    user_block = (
        data["user"]
        .replace("{{cds_source}}", cds)
        .replace("{{error_text}}", err)
        .replace("{{references}}", references_text)
    )

    resp = get_llm().invoke(
        [
            SystemMessage(content=data["system"]),
            HumanMessage(content=user_block),
        ]
    )
    body = extract_text(resp).strip()

    fixed = _extract_abap_fence(body)

    out: dict = {
        "messages": [AIMessage(content=body)],
    }
    if fixed:
        out["fixed_cds"] = fixed

    return out
