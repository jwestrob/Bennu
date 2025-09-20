from __future__ import annotations

from typing import Any, Dict, List, Tuple
from pydantic import ValidationError

from .schemas import RouterToolCall, TOOLCALL_JSON_SCHEMA


def validate_with_pydantic(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    try:
        RouterToolCall.model_validate(obj)
        return True, []
    except ValidationError as ve:
        return False, [e['msg'] for e in ve.errors()]


def validate_with_jsonschema(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    try:
        import jsonschema
    except Exception as e:  # pragma: no cover - optional at runtime
        # Fallback: rely on Pydantic if jsonschema is unavailable
        return validate_with_pydantic(obj)

    try:
        jsonschema.validate(instance=obj, schema=TOOLCALL_JSON_SCHEMA)
        return True, []
    except jsonschema.ValidationError as ve:  # type: ignore
        return False, [ve.message]


def validate_toolcall(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a router toolcall dict against strict schema (reject unknown fields)."""
    ok, errs = validate_with_jsonschema(obj)
    if ok:
        # Double-check with Pydantic to ensure extra fields are rejected consistently
        return validate_with_pydantic(obj)
    return ok, errs


def make_repair_prompt(obj: Dict[str, Any], errors: List[str]) -> str:
    return (
        "Your last toolcall JSON did not validate. Fix strictly.\n"
        "Errors: " + "; ".join(errors) + "\n"
        "Constraints: no extra fields; match enums; adhere to integer bounds.\n"
        "Return ONLY a JSON object matching the schema."
    )

