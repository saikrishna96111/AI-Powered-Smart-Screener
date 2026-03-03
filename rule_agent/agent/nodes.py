# agent/nodes.py

from langchain_core.messages import AIMessage
from .llm import llm
from pydantic import BaseModel
from typing import Dict
from langchain_core.messages import AIMessage

# -----------------------------
# SCHEMAS
# -----------------------------

class IntentSchema(BaseModel):
    intent: str


class ExtractSchema(BaseModel):
    extracted_fields: Dict[str, str]


# -----------------------------
# INTENT NODE
# -----------------------------

def intent_node(state: dict):

    if state.get("intent"):
        return {}

    last_user_msg = state["messages"][-1].content

    structured_llm = llm.with_structured_output(IntentSchema)

    result = structured_llm.invoke(
        f"Identify fraud rule intent for: {last_user_msg}"
    )

    return {
        "intent": result.intent
    }


# -----------------------------
# REQUIREMENTS NODE
# (No LLM here — fixed business inputs)
# -----------------------------

def requirements_node(state: dict):

    if state.get("required_fields"):
        return {}

    return {
        "required_fields": [
            "duplicate_definition",
            "time_window",
            "exclusions"
        ]
    }


# -----------------------------
# EXTRACT NODE
# -----------------------------

def extract_node(state: dict):

    if not state.get("missing_fields"):
        return {}

    last_user_msg = state["messages"][-1].content

    structured_llm = llm.with_structured_output(ExtractSchema)

    result = structured_llm.invoke(
        f"""
        Missing fields: {state['missing_fields']}
        User response: {last_user_msg}

        Extract values for missing fields only.
        """
    )

    updated_fields = state.get("collected_fields", {})
    updated_fields.update(result.extracted_fields)

    return {
        "collected_fields": updated_fields
    }


# -----------------------------
# MISSING NODE
# -----------------------------

def missing_node(state: dict):

    required = state.get("required_fields", [])
    collected = state.get("collected_fields", {})

    missing = [f for f in required if f not in collected]

    return {
        "missing_fields": missing
    }


# -----------------------------
# QUESTION NODE
# -----------------------------

def question_node(state: dict):

    if not state["missing_fields"]:
        return {}

    question_text = f"""
    Please provide values for:
    {state['missing_fields']}
    """

    return {
        "messages": [AIMessage(content=question_text)]
    }


# -----------------------------
# EXPLAIN NODE
# -----------------------------

def explain_node(state: dict):

    if state.get("explained"):
        return {}

    explanation = f"""
    Duplicate Invoice Rule Definition:

    Match on:
    - Vendor (LIFNR)
    - Reference Number (XBLNR)
    - Amount (DMBTR)
    - Currency (WAERS)
    - Company Code (BUKRS)

    Time window: {state['collected_fields'].get('time_window')}

    Exclusions: {state['collected_fields'].get('exclusions')}

    If approved, I will generate the CDS View.
    """

    return {
        "messages": [AIMessage(content=explanation)],
        "explained": True
    }

def approval_node(state: dict):

    if not state.get("explained"):
        return {}

    last_user_msg = state["messages"][-1].content.lower()

    if any(word in last_user_msg for word in ["approve", "yes", "generate"]):
        return {"approved": True}

    return {}

def cds_node(state: dict):

    if not state.get("approved"):
        return {}

    cds_code = f"""
    @AbapCatalog.sqlViewName: 'Z_DUP_INV'
    define view Z_Duplicate_Invoice_Check as
    select from BKPF
    inner join BSEG on BKPF.BELNR = BSEG.BELNR
    {{
        BKPF.BUKRS,
        BSEG.LIFNR,
        BKPF.XBLNR,
        BSEG.DMBTR,
        BKPF.WAERS,
        BKPF.BUDAT
    }}
    where
        BKPF.BUDAT >= add_days(current_date, -90)
        and BKPF.STBLG is initial
    """

    return {
        "messages": [AIMessage(content=cds_code)]
    }