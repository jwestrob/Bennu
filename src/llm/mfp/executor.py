from __future__ import annotations
from typing import Any, Dict, List
from .operators.base import OperatorContext, get_operator
import time
import logging


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
    logger = logging.getLogger(__name__)
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
        # Execute with timing and robust logging
        t0 = time.perf_counter()
        try:
            result = spec.run(ctx, provided, params)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            # Compact params for logging
            def _compact(obj: Any, lim: int = 160) -> str:
                s = str(obj)
                return (s[:lim] + '…') if len(s) > lim else s
            logger.error(
                f"PLAN STEP {i+1}/{len(plan['steps'])} op={name} FAILED after {elapsed_ms:.0f} ms; "
                f"params={_compact(params)} inputs={list(provided.keys())}")
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
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
        # Log result sizes per output key
        def _size(val: Any) -> str:
            try:
                if isinstance(val, list):
                    return f"list:{len(val)}"
                if isinstance(val, dict):
                    return f"dict:{len(val)}"
                if val is None:
                    return "None"
                return type(val).__name__
            except Exception:
                return type(val).__name__
        sizes = {k: _size(result.get(k)) for k in spec.outputs if k in result}
        # Compact params for logging
        def _compact(obj: Any, lim: int = 200) -> str:
            try:
                s = str(obj)
            except Exception:
                s = type(obj).__name__
            return (s[:lim] + '…') if len(s) > lim else s
        logger.info(
            f"PLAN STEP {i+1}/{len(plan['steps'])} op={name} OK in {elapsed_ms:.0f} ms; "
            f"params={_compact(params)} outputs={sizes}")
    return env
