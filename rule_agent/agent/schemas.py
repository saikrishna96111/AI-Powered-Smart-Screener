from pydantic import BaseModel, Field
from typing import List


class IntentSchema(BaseModel):
    intent: str = Field(
        description=(
            "A SHORT business-domain label for the SAP control the user described "
            "(2-6 words). Examples: 'Duplicate vendor invoice check', "
            "'GR/IR clearing exception', 'Vendor bank change vs payment', "
            "'3-way match tolerance breach'. NEVER repeat the prompt instruction "
            "or use phrases like 'name the control' or 'from user message'."
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