from pydantic import BaseModel, Field
from typing import List


class IntentSchema(BaseModel):
    intent: str
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