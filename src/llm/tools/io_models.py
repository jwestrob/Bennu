from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ConfigDict, ValidationError, model_validator


class ToolReturn(BaseModel):
    """
    Uniform envelope for tool outputs to support downstream provenance and tracing.
    """
    model_config = ConfigDict(extra="forbid")

    data: Any = Field(..., description="Primary return payload from the tool")
    provenance: Optional[Dict[str, Any]] = Field(
        default=None, description="Provenance info (e.g., software, params, versions)"
    )
    params_used: Dict[str, Any] = Field(
        default_factory=dict, description="Effective parameters used by the tool"
    )
    timings: Dict[str, float] = Field(
        default_factory=dict, description="Timing metrics in seconds by phase"
    )
    trace_id: Optional[str] = Field(
        default=None, description="Trace or correlation id for cross-component debugging"
    )


# -------------------------
# Database Query
# -------------------------
class DatabaseQueryArgs(BaseModel):
    """
    Arguments for database_query tool.
    """
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="Cypher or SQL query text")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Bound parameters for the query"
    )
    limit: Optional[int] = Field(
        default=None, ge=1, le=10000, description="Optional max rows to return"
    )
    database: Optional[str] = Field(
        default=None, description="Optional target database name/alias"
    )


class DatabaseQueryResult(BaseModel):
    """
    Results from database_query tool.
    """
    model_config = ConfigDict(extra="forbid")

    rows: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of result records (row dicts)"
    )
    summary: Optional[Dict[str, Any]] = Field(
        default=None, description="Driver-specific summary/metadata (optional)"
    )


# -------------------------
# Vector Search
# -------------------------
class VectorHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Identifier of the retrieved entity")
    score: float = Field(..., description="Similarity score (higher = more similar)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Attached metadata")


class VectorSearchArgs(BaseModel):
    """
    Arguments for vector_search tool.
    """
    model_config = ConfigDict(extra="forbid")

    query: Union[str, List[float]] = Field(
        ..., description="Text query or embedding vector"
    )
    top_k: int = Field(10, ge=1, le=1000, description="Number of nearest neighbors")
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata filters"
    )
    collection: Optional[str] = Field(
        default=None, description="Optional collection/table name"
    )


class VectorSearchResult(BaseModel):
    """
    Results from vector_search tool.
    """
    model_config = ConfigDict(extra="forbid")

    hits: List[VectorHit] = Field(default_factory=list, description="Ranked retrieval hits")


# -------------------------
# Whole Genome Reader
# -------------------------
class Locus(BaseModel):
    """
    Compact locus description used by WholeGenomeReaderResult.
    """
    model_config = ConfigDict(extra="forbid")

    contig_id: str = Field(..., min_length=1)
    start: int = Field(..., ge=1)
    end: int = Field(..., ge=1)
    gene_ids: List[str] = Field(..., min_length=1, description="Gene IDs within the locus")
    locus_label: Optional[str] = Field(default=None, description="Optional human label")

    @model_validator(mode="after")
    def _validate_coordinates(self) -> "Locus":
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class WholeGenomeReaderArgs(BaseModel):
    """
    Arguments for whole_genome_reader tool.
    """
    model_config = ConfigDict(extra="forbid")

    genome_id: str = Field(..., min_length=1)
    max_genes_per_contig: int = Field(1000, ge=1)
    json_only: bool = Field(True, description="Return results in compact JSON form only")


class WholeGenomeReaderResult(BaseModel):
    """
    Results from whole_genome_reader tool.
    """
    model_config = ConfigDict(extra="forbid")

    loci: List[Locus] = Field(default_factory=list, description="Loci identified/read")
