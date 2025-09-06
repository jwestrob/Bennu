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
        # Structural rule: AnnotationDiscovery must include a selection (keyword or ID inputs)
        if step['op'] == 'AnnotationDiscovery':
            params = (step.get('params') or {})
            # Require an explicit keyword/q for AnnotationDiscovery to function
            kw = params.get('keyword') or params.get('q')
            if not (isinstance(kw, str) and kw.strip() != ''):
                raise PlanValidationError(
                    "AnnotationDiscovery requires params.keyword (or params.q); calling it without a keyword yields empty results."
                )
        # Structural rule: NeighborhoodContext must have explicit seeds
        if step['op'] == 'NeighborhoodContext':
            params = (step.get('params') or {})
            inputs = (step.get('inputs') or {})
            has_param_seeds = any(
                isinstance(params.get(k), list) and len(params.get(k)) > 0
                for k in ('protein_ids', 'seed_pfam_ids', 'seed_ko_ids')
            )
            has_input_rowset = isinstance(inputs, dict) and isinstance(inputs.get('discovered_proteins'), str) and inputs.get('discovered_proteins').strip() != ''
            if not (has_param_seeds or has_input_rowset):
                raise PlanValidationError(
                    "NeighborhoodContext requires seeds: provide inputs.discovered_proteins (from a bound rowset) or params (protein_ids or seed_pfam_ids/seed_ko_ids)."
                )


def execute_plan(plan: Dict[str, Any], ctx: OperatorContext) -> Dict[str, Any]:
    validate_plan(plan)
    env: Dict[str, Any] = {}
    call_log: List[Dict[str, Any]] = []
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
                    # Fallback: if an explicit ref was provided but not found, use env[key] when available
                    if key in env:
                        provided[key] = env[key]
                    else:
                        # Leave missing key out; downstream operator can interpret as absent
                        pass
            else:
                if key in env:
                    provided[key] = env[key]
                else:
                    # No binding; omit
                    pass
        # Warn if plan provided inputs that are not declared by op spec
        if isinstance(inputs_ref, dict) and inputs_ref:
            extra = [k for k in inputs_ref.keys() if k not in (spec.inputs or [])]
            if extra:
                logging.getLogger(__name__).info(
                    f"execute_plan: ignoring undeclared input keys for op={name}: {extra}")
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

        # Append a structured record of the tool call for session diagnostics
        def _preview(val: Any, list_sample: int = 3, str_lim: int = 300, key: str | None = None) -> Any:
            try:
                if isinstance(val, list):
                    # For catalog-oriented outputs, include full lists (typically small: ≤30)
                    if key in {"pfam_catalog_hits", "pfam_ids", "pfam_terms", "ko_catalog_hits", "ko_ids"}:
                        return {"type": "list", "len": len(val), "items": val}
                    samp = val[:list_sample]
                    return {"type": "list", "len": len(val), "sample": samp}
                if isinstance(val, dict):
                    keys = list(val.keys())
                    return {"type": "dict", "len": len(keys), "keys_sample": keys[:10]}
                if isinstance(val, str):
                    return val if len(val) <= str_lim else (val[:str_lim] + '…')
                return val
            except Exception:
                return type(val).__name__

        outputs_preview: Dict[str, Any] = {}
        for k in spec.outputs:
            if k in result:
                outputs_preview[k] = _preview(result.get(k), key=k)
        provided_preview: Dict[str, Any] = {k: _preview(v, key=k) for k, v in provided.items()}
        call_log.append({
            "step": i + 1,
            "op": name,
            "elapsed_ms": int(elapsed_ms),
            "inputs_ref": inputs_ref,
            "params": params,
            "provided": provided_preview,
            "outputs": outputs_preview,
        })
    # Expose tool call records for session diagnostics
    try:
        env['__tool_calls'] = call_log
    except Exception:
        pass
    return env
