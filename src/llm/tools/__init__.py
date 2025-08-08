"""
Typed I/O contracts and tool specifications for LLM tool selection and execution.

This package provides:
- io_models: Pydantic v2 models for tool inputs/outputs and common envelopes
- spec: Registry and ToolSpec builder that emits JSON Schemas
- selection_models: ToolSelection schema and validation helpers
"""
from .io_models import (
    ToolReturn,
    DatabaseQueryArgs,
    DatabaseQueryResult,
    VectorSearchArgs,
    VectorSearchResult,
    Locus,
    WholeGenomeReaderArgs,
    WholeGenomeReaderResult,
)
from .spec import (
    ToolIO,
    ToolSpec,
    REGISTRY,
    build_tool_specs,
    get_input_model,
    get_output_model,
)
from .selection_models import (
    ToolSelection,
    validate_tool_params,
    validate_tool_output,
)

__all__ = [
    # io models
    "ToolReturn",
    "DatabaseQueryArgs",
    "DatabaseQueryResult",
    "VectorSearchArgs",
    "VectorSearchResult",
    "Locus",
    "WholeGenomeReaderArgs",
    "WholeGenomeReaderResult",
    # spec/registry
    "ToolIO",
    "ToolSpec",
    "REGISTRY",
    "build_tool_specs",
    "get_input_model",
    "get_output_model",
    # selection models
    "ToolSelection",
    "validate_tool_params",
    "validate_tool_output",
]
