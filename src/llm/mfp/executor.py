from __future__ import annotations
from typing import Any, Dict, List
from .operators.base import OperatorContext, get_operator


class PlanValidationError(Exception):
    pass


def validate_plan(plan: Dict[str, Any]) -> None:
    if not isinstance(plan, dict) or 'steps' not in plan or not isinstance(plan['steps'], list):
        raise PlanValidationError("Plan must be a dict with a list 'steps'")
    for i, step in enumerate(plan['steps']):
        if not isinstance(step, dict):
            raise PlanValidationError(f"Step {i} must be an object")
        if 'op' not in step:
            raise PlanValidationError(f"Step {i} missing 'op'")
        # Check operator exists
        get_operator(step['op'])
        # Optional fields: inputs, params, bind


def execute_plan(plan: Dict[str, Any], ctx: OperatorContext) -> Dict[str, Any]:
    validate_plan(plan)
    env: Dict[str, Any] = {}
    for i, step in enumerate(plan['steps']):
        name = step['op']
        spec = get_operator(name)
        inputs_ref = step.get('inputs', {}) or {}
        params = step.get('params', {}) or {}
        # Materialize inputs
        provided: Dict[str, Any] = {}
        def _normalize_ref(r: Any) -> str:
            if not isinstance(r, str):
                return str(r)
            s = r.strip()
            # Accept ${name}, $name, {name}, name
            if s.startswith('${') and s.endswith('}'):
                return s[2:-1].strip()
            if s.startswith('{') and s.endswith('}'):
                return s[1:-1].strip()
            if s.startswith('$'):
                return s[1:].strip()
            return s

        # Flexible input provisioning:
        # 1) If the plan provides an explicit mapping for a key and it's in env, pass it.
        # 2) Otherwise, fall back to env[key] when available (implicit same-name binding).
        # 3) If neither exist, omit the key (operators should handle missing/empty inputs gracefully).
        for key in spec.inputs:
            if isinstance(inputs_ref, dict) and key in inputs_ref:
                ref_name = _normalize_ref(inputs_ref.get(key))
                if ref_name and ref_name in env:
                    provided[key] = env[ref_name]
                else:
                    # Leave missing key out; downstream operator can interpret as absent
                    pass
            else:
                if key in env:
                    provided[key] = env[key]
                else:
                    # No binding; omit
                    pass
        # Execute
        result = spec.run(ctx, provided, params)
        if not isinstance(result, dict):
            raise PlanValidationError(f"Operator '{name}' returned non-dict result")
        # Bind outputs
        bind = step.get('bind')
        if bind and isinstance(bind, str):
            env[bind] = result
        # Also expose each declared output at top-level for convenience
        for out_key in spec.outputs:
            if out_key in result:
                env[out_key] = result[out_key]
    return env
