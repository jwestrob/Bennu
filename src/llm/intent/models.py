from __future__ import annotations
from enum import Enum
from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field, ConfigDict


class TaskFamily(str, Enum):
    FIND_LOCI_BY_MARKER = "FIND_LOCI_BY_MARKER"
    FIND_LOCI_BY_SIGNATURE = "FIND_LOCI_BY_SIGNATURE"
    LANCEDB_KNN = "LANCEDB_KNN"
    SUMMARIZE_NEIGHBORHOOD = "SUMMARIZE_NEIGHBORHOOD"


class KnnAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["LANCEDB_KNN"] = "LANCEDB_KNN"
    top_k: int = Field(default=10, ge=1)
    exclude_pfam: List[str] = Field(default_factory=list)


class SummarizeAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["SUMMARIZE_NEIGHBORHOOD"] = "SUMMARIZE_NEIGHBORHOOD"


Action = Union[KnnAction, SummarizeAction]


class FindByMarker(BaseModel):
    model_config = ConfigDict(extra="forbid")
    marker: str
    flank_k: int = Field(default=2, ge=0)


class FindBySignature(BaseModel):
    model_config = ConfigDict(extra="forbid")
    signature_name: str  # e.g., "PROPHAGE"
    flank_k: int = Field(default=5, ge=0)


class CanonicalIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: TaskFamily
    n: int = Field(ge=1)
    find_by_marker: Optional[FindByMarker] = None
    find_by_signature: Optional[FindBySignature] = None
    actions: List[Action] = Field(default_factory=list)
    version_tag: str = "v1"

    def to_minimal_dict(self) -> dict:
        return self.model_dump(by_alias=False, exclude_none=True)

