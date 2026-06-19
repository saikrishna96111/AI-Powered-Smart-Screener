from pydantic import BaseModel, Field
from typing import List


class IntentSchema(BaseModel):
    intent: str = Field(
        description=(
            "A SHORT business-domain label for the SAP control the user described "
            "(2-6 words). Derive ONLY from the user's message — do not assume "
            "duplicate invoice, vendor invoice, or any default control if they "
            "did not describe one. Examples (use only when they match what the "
            "user said): 'PO creator approver SoD', 'GR/IR clearing exception', "
            "'Vendor bank change vs payment', '3-way match tolerance breach'. "
            "NEVER repeat the prompt instruction or use phrases like "
            "'name the control' or 'from user message'."
        )
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How sure the model is about the intent label",
    )


class RequirementsSchema(BaseModel):
    required_fields: List[str]


class RequiredFieldsPlan(BaseModel):
    """LLM decides which clarifications are still needed before CDS generation."""

    required_fields: List[str] = Field(
        default_factory=list,
        description="snake_case field names still missing; empty if user already gave enough detail",
    )


class ExtractFieldPair(BaseModel):
    field_name: str
    value: str


class ExtractBatchSchema(BaseModel):
    pairs: List[ExtractFieldPair] = Field(default_factory=list)


class ReadinessSchema(BaseModel):
    ready: bool
    missing_fields: List[str]


class SyntaxReviewSchema(BaseModel):
    """Output of the ABAP CDS syntax validator (used by syntax_review_node)."""

    syntax_status: str = Field(
        description=(
            "Either 'PASSED' (the CDS is valid ABAP CDS that activates in ADT) or "
            "'FAILED' (the CDS contains constructs unsupported in ABAP CDS). "
            "Use uppercase exactly."
        )
    )
    issues: List[str] = Field(
        default_factory=list,
        description=(
            "When syntax_status is FAILED, list each ABAP CDS rule that was violated "
            "as one bullet (e.g. 'EXISTS subquery is not supported in ABAP CDS — replace "
            "with a LEFT OUTER JOIN'). Empty when PASSED."
        ),
    )
    corrected_cds: str = Field(
        default="",
        description=(
            "When syntax_status is FAILED, return the FULL corrected CDS DDL — same "
            "view name, same business logic, but only using ABAP-CDS-supported syntax. "
            "Wrap in nothing — emit raw CDS source. Empty string when PASSED."
        ),
    )