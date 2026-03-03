from pydantic import BaseModel
from typing import List


class IntentSchema(BaseModel):
    intent: str
    confidence: float


class RequirementsSchema(BaseModel):
    required_fields: List[str]


class ReadinessSchema(BaseModel):
    ready: bool
    missing_fields: List[str]