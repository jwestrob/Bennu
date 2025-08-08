from __future__ import annotations

from typing import Any, Dict, Optional, Literal

from pydantic import BaseModel, Field, ConfigDict, ValidationError

from .spec import get_input_model, get_output_model, ToolName


class ToolSelection(BaseModel):
    """
    Selector's decision envelope.

    The LLM must return ONLY a JSON object validating this schema.
    """
    model_config = ConfigDict(extra="forbid")

    selected_tool: ToolName = Field(..., description="Name of the chosen tool")
    tool_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the chosen tool; validated against the tool's InputModel",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Selector's confidence [0..1]")
    trace_id: Optional[str] = Field(
        default=None, description="Optional trace/correlation id echoed from the system"
    )


def validate_tool_params(selection: ToolSelection):
    """
    Validate tool parameters against the selected tool's InputModel.
    Returns a typed InputModel instance or raises ValidationError.
    """
    InputModel = get_input_model(selection.selected_tool)
    return InputModel.model_validate(selection.tool_parameters)


def validate_tool_output(tool_name: str, raw_result: Any):
    """
    Validate a raw tool result-like object against the tool's OutputModel.
    Returns a typed OutputModel instance or raises ValidationError.
    """
    OutputModel = get_output_model(tool_name)
    return OutputModel.model_validate(raw_result)
