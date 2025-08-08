from __future__ import annotations

from typing import Any, Dict, List, Tuple, Type, cast, Literal

from pydantic import BaseModel, Field, ConfigDict

from .io_models import (
    DatabaseQueryArgs,
    DatabaseQueryResult,
    VectorSearchArgs,
    VectorSearchResult,
    WholeGenomeReaderArgs,
    WholeGenomeReaderResult,
)


class ToolIO(BaseModel):
    """
    JSON Schemas for tool I/O contracts that can be embedded in prompts.
    """
    model_config = ConfigDict(extra="forbid")

    input_schema: Dict[str, Any] = Field(..., description="JSON Schema for the tool's input model")
    output_schema: Dict[str, Any] = Field(..., description="JSON Schema for the tool's output model")


ToolName = Literal["database_query", "vector_search", "whole_genome_reader"]


class ToolSpec(BaseModel):
    """
    Minimal tool spec for LLM selection.
    """
    model_config = ConfigDict(extra="forbid")

    name: ToolName
    description: str
    io: ToolIO


# Registry maps a canonical tool name to (InputModel, OutputModel, description)
REGISTRY: Dict[str, Tuple[Type[BaseModel], Type[BaseModel], str]] = {
    "database_query": (
        DatabaseQueryArgs,
        DatabaseQueryResult,
        "Run a structured database query (e.g., Neo4j Cypher) and return rows.",
    ),
    "vector_search": (
        VectorSearchArgs,
        VectorSearchResult,
        "Find proteins or entities similar to a text or embedding query using vector search.",
    ),
    "whole_genome_reader": (
        WholeGenomeReaderArgs,
        WholeGenomeReaderResult,
        "Read genome(s) spatially and return compact loci summaries for analysis.",
    ),
}


def _get_models_and_description(name: str) -> Tuple[Type[BaseModel], Type[BaseModel], str]:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown tool name: {name!r}. Available: {sorted(REGISTRY.keys())}")


def build_tool_specs(names: List[str]) -> List[ToolSpec]:
    """
    Build ToolSpec objects with JSON Schemas suitable for compact prompt inclusion.
    """
    specs: List[ToolSpec] = []
    for n in names:
        in_model, out_model, desc = _get_models_and_description(n)
        specs.append(
            ToolSpec(
                name=cast(ToolName, n),
                description=desc,
                io=ToolIO(
                    input_schema=in_model.model_json_schema(),
                    output_schema=out_model.model_json_schema(),
                ),
            )
        )
    return specs


def get_input_model(name: str) -> Type[BaseModel]:
    return _get_models_and_description(name)[0]


def get_output_model(name: str) -> Type[BaseModel]:
    return _get_models_and_description(name)[1]
