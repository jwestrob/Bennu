from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Op(str, Enum):
    EQ = "=="
    GE = ">="
    LE = "<="


class Cardinality(BaseModel):
    value: Optional[int] = None
    op: Op = Op.EQ


class LanceDBObligation(BaseModel):
    required: bool = False
    nn: Optional[int] = None
    exclude_markers: List[str] = Field(default_factory=list)
    exclude_namespace: Optional[str] = None  # "pfam" | "kofam" | None
    distance: str = "cosine"


class Obligations(BaseModel):
    lancedb_knn: LanceDBObligation = Field(default_factory=LanceDBObligation)
    literature: bool = False


class Intent(BaseModel):
    option_name: str = "LocusDiscovery"
    marker: str
    N: Cardinality = Field(default_factory=Cardinality)
    flank: Cardinality = Field(default_factory=Cardinality)
    nn: Cardinality = Field(default_factory=Cardinality)
    obligations: Obligations = Field(default_factory=Obligations)
    raw_text: str
