from __future__ import annotations

from typing import Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, ConfigDict


# Tools supported by the unified router
ToolName = Literal[
    "database_query",
    "whole_genome_reader",
    "similarity_search",
    "code_interpreter",
    "literature_search",
    "synthesize",
]


class DBQueryParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template: Literal[
        "protein_by_id",
        "proteins_with_ko",
        "neighbors_by_window",
        "pathway_membership",
        "count_by_label",
        "cazy_family",
        "proteins_by_genome",
        "genes_on_contig",
        "proteins_with_pfam",
        "count_proteins_with_ko",
        "count_proteins_with_pfam",
        "count_proteins_in_pathway",
        "gene_neighbors_k",
        "protein_neighbors_k",
    ]
    slots: Dict[str, Any] = Field(default_factory=dict)


class SimilarityParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["by_id", "by_sequence"]
    k: int = Field(ge=1, le=1000)
    id: Optional[str] = None
    sequence: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)


class SpatialGenomeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_bp: int = Field(ge=100, le=2_000_000)
    loci_limit: int = Field(ge=1, le=5000)


RouterParams = Union[DBQueryParams, SimilarityParams, SpatialGenomeParams]


class RouterToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: ToolName
    params: RouterParams


# JSON Schema for LLM validation (must match Pydantic models; no cardinality)
TOOLCALL_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["tool", "params"],
    "additionalProperties": False,
    "properties": {
        "tool": {
            "enum": [
                "database_query",
                "whole_genome_reader",
                "similarity_search",
                "code_interpreter",
                "literature_search",
                "synthesize",
            ]
        },
        "params": {
            "oneOf": [
                {
                    "type": "object",
                    "required": ["template", "slots"],
                    "additionalProperties": False,
                    "properties": {
                        "template": {
                            "enum": [
                                "protein_by_id",
                                "proteins_with_ko",
                                "neighbors_by_window",
                                "pathway_membership",
                                "count_by_label",
                                "cazy_family",
                                "proteins_by_genome",
                                "genes_on_contig",
                                "proteins_with_pfam",
                                "count_proteins_with_ko",
                                "count_proteins_with_pfam",
                                "count_proteins_in_pathway",
                                "gene_neighbors_k",
                                "protein_neighbors_k",
                            ]
                        },
                        "slots": {"type": "object"},
                    },
                },
                {
                    "type": "object",
                    "required": ["mode", "k"],
                    "additionalProperties": False,
                    "properties": {
                        "mode": {"enum": ["by_id", "by_sequence"]},
                        "id": {"type": "string"},
                        "sequence": {"type": "string"},
                        "k": {"type": "integer", "minimum": 1, "maximum": 1000},
                        "filters": {"type": "object"},
                    },
                },
                {
                    "type": "object",
                    "required": ["window_bp", "loci_limit"],
                    "additionalProperties": False,
                    "properties": {
                        "window_bp": {
                            "type": "integer",
                            "minimum": 100,
                            "maximum": 2_000_000,
                        },
                        "loci_limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5000,
                        },
                    },
                },
            ]
        },
    },
}
