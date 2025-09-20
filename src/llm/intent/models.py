from __future__ import annotations
from enum import Enum
from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field, ConfigDict, model_validator


class TaskFamily(str, Enum):
    FIND_LOCI_BY_MARKER = "FIND_LOCI_BY_MARKER"
    FIND_LOCI_BY_SIGNATURE = "FIND_LOCI_BY_SIGNATURE"
    LANCEDB_KNN = "LANCEDB_KNN"
    SUMMARIZE_NEIGHBORHOOD = "SUMMARIZE_NEIGHBORHOOD"
    PATHWAY_COMPLETENESS = "PATHWAY_COMPLETENESS"


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
    # Optional scope for pathway completeness
    genome_ids: Optional[List[str]] = None
    min_completeness: Optional[float] = None
    pathways: Optional[List[str]] = None  # Optional filter: KEGG map IDs (e.g., "map00010")

    def to_minimal_dict(self) -> dict:
        return self.model_dump(by_alias=False, exclude_none=True)

    @model_validator(mode="before")
    def _adjust_for_pathway(cls, data):
        """Ensure valid defaults for PATHWAY_COMPLETENESS (n>=1)."""
        try:
            if isinstance(data, dict) and str(data.get("task")) == TaskFamily.PATHWAY_COMPLETENESS:
                n = data.get("n")
                if not isinstance(n, int) or n < 1:
                    data["n"] = 1
        except Exception:
            pass
        return data
