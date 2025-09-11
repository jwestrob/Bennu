from __future__ import annotations
from typing import Any, Dict, List

from .base import OperatorContext, OperatorSpec, register_operator
from ...kg.cypher_templates import registry as kg_tpl_registry


def _normalize_template_name(name: str) -> str:
    n = name.strip().lower().replace(' ', '_').replace('-', '_')
    # Minimal aliasing for common LLM variants (no biology hard-coding)
    aliases = {
        'crisprarraygenewindow': 'anchor_gene_window',
        'crispr_array_gene_window': 'anchor_gene_window',
        'genes_in_crispr_window': 'anchor_gene_window',
        'genes_in_crispr_window_any': 'anchor_gene_window',
        'crispr_arrays_window': 'crispr_arrays_by_contig',  # generic fallback guess
    }
    return aliases.get(n, n)


def _resolve_slots_with_inputs(raw_slots: Dict[str, Any] | None, rows: List[Dict[str, Any]] | None) -> Dict[str, Any]:
    """Resolve slot values that reference prior rows via a tiny declarative form.

    Supported mapping form per slot:
      {"from": "rows", "field": "<column>", "index": <int default 0>}

    Anything else is passed through unchanged.
    """
    slots: Dict[str, Any] = dict(raw_slots or {})
    if not slots:
        return {}
    out: Dict[str, Any] = {}
    for k, v in slots.items():
        try:
            if isinstance(v, dict) and v.get("from") == "rows" and isinstance(rows, list) and rows:
                idx = int(v.get("index", 0))
                fld = str(v.get("field") or "").strip()
                if 0 <= idx < len(rows) and fld:
                    out[k] = rows[idx].get(fld)
                else:
                    out[k] = None
            else:
                out[k] = v
        except Exception:
            out[k] = v
    return out


def _execute_db_template(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    name = params.get("name")
    # Optional bound rows for slot resolution
    rows_in = None
    try:
        ri = inputs.get("rows") or inputs.get("ModuleRows")
        if isinstance(ri, list):
            rows_in = [dict(x) for x in ri if isinstance(x, dict)]
    except Exception:
        rows_in = None

    slots = _resolve_slots_with_inputs(params.get("slots") or {}, rows_in)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("ExecuteDBTemplate: params.name is required")
    name = _normalize_template_name(name)
    # Back-compat: if legacy names were mapped to anchor_gene_window, default anchor_type to 'crispr'
    if name == 'anchor_gene_window' and 'anchor_type' not in slots:
        slots = dict(slots)
        slots['anchor_type'] = 'crispr'

    # Robust BGC fallback: if planner forgot to pass a real bgc_id for genes_in_bgc, try deriving from ModuleRows
    try:
        if name == 'genes_in_bgc':
            bgc_id = (slots.get('bgc_id') or '').strip() if isinstance(slots.get('bgc_id'), str) else None
            placeholder = {None, '', 'unknown_product_bgc_id', 'UNKNOWN_PRODUCT_BGC_ID'}
            if (bgc_id in placeholder) and isinstance(rows_in, list) and rows_in:
                # Prefer Unknown product rows when available
                def _prod(r: Dict[str, Any]) -> str:
                    for k in ('bgc_product', 'bgcProduct', 'product', 'cluster_type'):
                        if k in r and isinstance(r[k], str):
                            return r[k].strip()
                    return ''
                def _id(r: Dict[str, Any]) -> str | None:
                    for k in ('bgc_id', 'bgcId', 'id'):
                        v = r.get(k)
                        if isinstance(v, str) and v.strip():
                            return v.strip()
                    return None
                unknown_rows = [r for r in rows_in if _prod(r).lower() == 'unknown']
                chosen = unknown_rows[0] if unknown_rows else rows_in[0]
                cid = _id(chosen)
                if cid:
                    slots = dict(slots)
                    slots['bgc_id'] = cid
    except Exception:
        pass
    # Compile (supports dynamic compilers) or read static file
    cypher, cy_params = kg_tpl_registry.compile_query(name, dict(slots))
    rows: list[dict] = []
    with ctx.neo4j_driver.session() as session:
        res = session.run(cypher, cy_params)
        rows = [dict(r) for r in res]
    # Provide a compact preview for planner/debug
    preview = rows[:3]
    macro = {"type": "macro_result", "name": name, "rows": rows}
    return {"structured_data": rows, "preview": preview, "macro_result": macro}


register_operator(OperatorSpec(
    name="ExecuteDBTemplate",
    inputs=["rows", "ModuleRows"],  # optional; enables slot piping from previous results (BGCs via ModuleRows)
    outputs=["structured_data", "preview", "macro_result"],
    params={"name": "string", "slots": "object"},
    run=_execute_db_template,
    description="Execute a named DB template with provided slots. Supports slot piping via inputs.rows or ModuleRows; includes robust BGC id fallback.",
))
