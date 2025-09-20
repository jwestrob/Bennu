from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional


class JsonPatchOp(BaseModel):
    op: str
    path: str
    value: Optional[Any] = None

    @validator('op')
    def _validate_op(cls, v: str) -> str:
        allowed = {"add", "remove", "replace", "test"}
        if v not in allowed:
            raise ValueError(f"Invalid patch op '{v}', allowed: {sorted(allowed)}")
        return v


class PatchEnvelope(BaseModel):
    anchor: str
    obligations: List[str] = []
    patch: List[JsonPatchOp]
    evidence: Dict[str, Any]
    rationale: str
    risk: str = "low"  # low|medium|high
