# agent/nodes.py

import json
import os
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .llm import llm
from .prompt_loader import load_prompt
from .schemas import ExtractBatchSchema, IntentSchema, RequiredFieldsPlan
from .utils import extract_text


def _all_user_text(messages) -> str:
    parts = []
    for m in messages or []:
        if isinstance(m, HumanMessage):
            t = (m.content or "").strip()
            if t:
                parts.append(t)
    return "\n\n".join(parts)


def _format_params(collected: dict) -> str:
    return json.dumps(collected or {}, indent=2, ensure_ascii=False)


# -----------------------------
# INTENT NODE
# -----------------------------


def intent_node(state: dict):

    if state.get("intent"):
        return {}

    last_user_msg = state["messages"][-1].content

    structured_llm = llm.with_structured_output(
        IntentSchema,
        method="function_calling",
    )

    result = structured_llm.invoke(
        "In one short phrase, name the SAP S/4HANA exception or monitoring control "
        "(FI/MM/SD/GL; e.g. duplicate invoice, vendor bank change vs payments, "
        "3-way match tolerance breach). User message:\n"
        + (last_user_msg or "")
    )

    return {
        "intent": result.intent,
    }


# -----------------------------
# REQUIREMENTS NODE
# -----------------------------


def requirements_node(state: dict):

    if state.get("required_fields") is not None:
        return {}

    user_text = _all_user_text(state.get("messages", []))
    intent = state.get("intent") or ""

    structured_llm = llm.with_structured_output(
        RequiredFieldsPlan,
        method="function_calling",
    )

    result = structured_llm.invoke(
        f"""Plan clarifying snake_case fields to gather before generating a SAP CDS exception view.

Control intent: {intent}

User message(s):
{user_text}

Return required_fields: 3–7 short names for facts still needed (examples: key_tables,
exception_or_match_logic, time_window_days, amount_threshold, tolerance_percent,
company_code_scope, output_grain, exclusions).

If the user already gave enough detail on involved tables, how to detect the exception,
time scope, and thresholds/tolerances when relevant, return an empty list.

Do not ask about tools, transport, or non-SAP configuration."""
    )

    return {
        "required_fields": list(result.required_fields or []),
    }


# -----------------------------
# EXTRACT NODE
# -----------------------------


def extract_node(state: dict):

    missing = state.get("missing_fields") or []
    if not missing:
        return {}

    last_user_msg = state["messages"][-1].content

    structured_llm = llm.with_structured_output(
        ExtractBatchSchema,
        method="function_calling",
    )

    result = structured_llm.invoke(
        f"""Extract values the user provided only for these keys (exact names): {missing}

The user may answer with lines like key=value or key: value (possibly several lines).

User message:
{last_user_msg}

Return pairs only for keys you can answer from this message; omit unknown keys."""
    )

    updated_fields = dict(state.get("collected_fields", {}))
    missing_set = set(missing)

    for p in result.pairs or []:
        name = (p.field_name or "").strip()
        val = (p.value or "").strip()
        if name in missing_set and val:
            updated_fields[name] = val

    return {
        "collected_fields": updated_fields,
    }


# -----------------------------
# MISSING NODE
# -----------------------------


def missing_node(state: dict):

    required = state.get("required_fields") or []
    collected = state.get("collected_fields", {})

    missing = [f for f in required if f not in collected]

    return {
        "missing_fields": missing,
    }


# -----------------------------
# QUESTION NODE
# -----------------------------


def question_node(state: dict):

    if not state["missing_fields"]:
        return {}

    data = load_prompt("question")
    missing = state["missing_fields"]
    lines = "\n".join(f"{i + 1}. {key}" for i, key in enumerate(missing))
    user_block = (
        data["user"]
        .replace("{{intent}}", state.get("intent") or "")
        .replace("{{missing_fields_lines}}", lines)
        .replace(
            "{{missing_fields_json}}",
            json.dumps(missing, ensure_ascii=False),
        )
        .replace("{{user_so_far}}", _all_user_text(state.get("messages", [])))
    )

    resp = llm.invoke(
        [
            SystemMessage(content=data["system"]),
            HumanMessage(content=user_block),
        ]
    )
    text = extract_text(resp).strip()

    return {
        "messages": [AIMessage(content=text)],
    }


# -----------------------------
# EXPLAIN NODE
# -----------------------------


def explain_node(state: dict):

    if state.get("explained"):
        return {}

    data = load_prompt("explain")
    user_block = (
        data["user"]
        .replace("{{intent}}", state.get("intent") or "")
        .replace("{{params}}", _format_params(state.get("collected_fields", {})))
        .replace("{{description}}", _all_user_text(state.get("messages", [])))
    )

    resp = llm.invoke(
        [
            SystemMessage(content=data["system"]),
            HumanMessage(content=user_block),
        ]
    )
    text = extract_text(resp).strip()

    return {
        "messages": [AIMessage(content=text)],
        "explained": True,
    }


# -----------------------------
# APPROVAL NODE
# -----------------------------


def approval_node(state: dict):

    if not state.get("explained"):
        return {}

    last_user_msg = (state["messages"][-1].content or "").lower().strip()

    if state.get("cds_delivered") and last_user_msg in (
        "no",
        "nope",
        "nothing",
        "exit",
        "quit",
        "done",
        "goodbye",
        "that's all",
    ):
        return {
            "messages": [AIMessage(content="Thanks. Goodbye!")],
            "approved": False,
            "session_ended": True,
        }

    if len(last_user_msg) > 25:
        return {}
    if any(
        phrase in last_user_msg
        for phrase in ["not approve", "don't approve", "reject"]
    ):
        return {}
    if not any(
        word in last_user_msg for word in ["approve", "approved", "yes", "ok", "okay"]
    ):
        return {}

    if state.get("cds_delivered"):
        return {
            "messages": [
                AIMessage(content="Rule was already generated. Anything else?")
            ],
            "approved": False,
        }

    return {"approved": True}


# -----------------------------
# CDS GENERATION NODE
# -----------------------------


def cds_node(state: dict):

    if not state.get("approved"):
        return {}

    data = load_prompt("cds")
    user_block = (
        data["user"]
        .replace("{{intent}}", state.get("intent") or "")
        .replace("{{params}}", _format_params(state.get("collected_fields", {})))
        .replace("{{description}}", _all_user_text(state.get("messages", [])))
    )

    resp = llm.invoke(
        [
            SystemMessage(content=data["system"]),
            HumanMessage(content=user_block),
        ]
    )
    body = extract_text(resp).strip()

    fence = re.search(r"```(?:abap|cds)?\s*\n([\s\S]*?)```", body, re.IGNORECASE)
    cds_code = fence.group(1).strip() if fence else body

    confirmation = "Rule approved. Here is your CDS view:\n\n" + body

    return {
        "messages": [AIMessage(content=confirmation)],
        "cds_delivered": True,
        "cds_code": cds_code,
        "cds_review_done": False,
    }


# -----------------------------
# CDS REVIEW NODE (post-generation QA)
# -----------------------------


def _last_abap_fence(text: str) -> str | None:
    matches = list(
        re.finditer(r"```(?:abap|cds)?\s*\n([\s\S]*?)```", text, re.IGNORECASE)
    )
    if not matches:
        return None
    return matches[-1].group(1).strip()


def cds_review_node(state: dict):
    """Second pass: checklist review; optional revised DDL. Skip with RULE_AGENT_SKIP_CDS_REVIEW=1."""

    if os.getenv("RULE_AGENT_SKIP_CDS_REVIEW", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return {"cds_review_done": True}

    if not state.get("cds_code") or not state.get("cds_delivered"):
        return {}

    if state.get("cds_review_done"):
        return {}

    data = load_prompt("cds_review")
    cds_code = state["cds_code"] or ""
    user_block = (
        data["user"]
        .replace("{{intent}}", state.get("intent") or "")
        .replace("{{params}}", _format_params(state.get("collected_fields", {})))
        .replace("{{description}}", _all_user_text(state.get("messages", [])))
        .replace("{{cds_code}}", cds_code)
    )

    resp = llm.invoke(
        [
            SystemMessage(content=data["system"]),
            HumanMessage(content=user_block),
        ]
    )
    body = extract_text(resp).strip()

    out: dict = {
        "messages": [AIMessage(content="**Engineering review**\n\n" + body)],
        "cds_review_done": True,
    }

    if "revised cds" in body.lower():
        refined = _last_abap_fence(body)
        if refined and len(refined) > 80:
            out["cds_code"] = refined

    return out
