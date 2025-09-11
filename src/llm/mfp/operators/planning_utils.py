from __future__ import annotations
from typing import Any, Dict, List

from .base import OperatorContext, OperatorSpec, register_operator


def _assess_evidence(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    data = inputs.get("data")
    try:
        min_rows = int(params.get("min_rows", 5))
    except Exception:
        min_rows = 5
    count = 0
    if isinstance(data, list):
        count = len(data)
    elif isinstance(data, dict):
        # Common shapes: {rows: [...]}, {name: rows}
        if isinstance(data.get("rows"), list):
            count = len(data.get("rows", []))
        else:
            for v in data.values():
                if isinstance(v, list):
                    count += len(v)
    metrics = {
        "rows": count,
        "threshold": min_rows,
        "meets_threshold": bool(count >= min_rows),
    }
    return {"evidence_metrics": metrics}


def _propose_followup(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    metrics = inputs.get("evidence_metrics") or {}
    data_in = inputs.get("data")
    question = str(params.get("question") or "").strip()
    try:
        top_n = int(params.get("top_n", 25))
    except Exception:
        top_n = 25
    rows = int(metrics.get("rows", 0)) if isinstance(metrics, dict) else 0
    thr = int(metrics.get("threshold", 5)) if isinstance(metrics, dict) else 5
    meets = bool(metrics.get("meets_threshold")) if isinstance(metrics, dict) else (rows >= thr)
    reason = f"insufficient_evidence: rows={rows} < threshold={thr}" if not meets else f"sufficient_evidence: rows={rows} >= threshold={thr}"

    # Normalize data rows (if provided) for schema-driven branching
    first_keys = set()
    norm_rows = []
    try:
        if isinstance(data_in, list) and data_in and isinstance(data_in[0], dict):
            norm_rows = [dict(r) for r in data_in]
            first_keys = set(norm_rows[0].keys())
        elif isinstance(data_in, dict) and isinstance(data_in.get("rows"), list) and data_in["rows"]:
            norm_rows = [dict(r) for r in data_in["rows"] if isinstance(r, dict)]
            if norm_rows:
                first_keys = set(norm_rows[0].keys())
    except Exception:
        norm_rows = []
        first_keys = set()

    # Data-driven next task selection (no prompt heuristics)
    next_task_steps = []
    inputs_needed = []

    if norm_rows:
        # Case A: arrays_per_genome shape → list arrays for top genome then fetch ±5kb window
        if {"genome_id", "arrays"}.issubset(first_keys):
            # Consolidated: go straight to anchor_gene_window, scoped to the top genome (no intermediate listing)
            next_task_steps = [
                {
                    "op": "DBTemplateCall",
                    "params": {
                        "name": "anchor_gene_window",
                        "slots": {"anchor_type": "crispr", "genome_id": {"from": "rows", "field": "genome_id", "index": 0}, "margin_bp": 5000}
                    }
                }
            ]
        # Case B: direct CRISPR array listing present → fetch ±5kb window for first array
        elif "crispr_id" in first_keys:
            next_task_steps = [
                {
                    "op": "DBTemplateCall",
                    "params": {
                        "name": "anchor_gene_window",
                        "slots": {"anchor_type": "crispr", "anchor_id": {"from": "rows", "field": "crispr_id", "index": 0}, "margin_bp": 5000}
                    }
                }
            ]
        # Case C: BGC rows present → pick an unknown-product BGC and list genes in that BGC
        elif ("bgc_id" in first_keys or "bgcId" in first_keys) and ("bgc_product" in first_keys or "bgcProduct" in first_keys):
            # Prefer Unknown product rows; else take the first
            # Runtime filter happens inside DB stage via slot piping index; here we just propose an index of 0,
            # assuming upstream table ordering already groups Unknowns or the UI will select one.
            field_id = "bgc_id" if "bgc_id" in first_keys else "bgcId"
            next_task_steps = [
                {
                    "op": "DBTemplateCall",
                    "inputs": {"rows": "ModuleRows"},
                    "params": {
                        "name": "genes_in_bgc",
                        "slots": {"bgc_id": {"from": "rows", "field": field_id, "index": 0}}
                    }
                }
            ]

    # Fallback: generic discovery plan when schema is unknown AND evidence is insufficient
    if not next_task_steps and not meets:
        next_task_steps = [
            {"op": "SearchPfamCatalogFuzzy", "params": {"q": question, "top_n": top_n}, "bind": "pfam_hits"},
            {"op": "SearchKoCatalogFuzzy", "params": {"q": question, "top_n": top_n}, "bind": "ko_hits"},
            {"op": "ExtractIdsFromCatalogHits", "inputs": {"pfam_catalog_hits": "pfam_hits", "ko_catalog_hits": "ko_hits"}, "bind": "id_lists"},
            {"op": "QueryProteinsByIds", "inputs": {"pfam_ids": "id_lists", "ko_ids": "id_lists"}, "params": {"limit": 1000}, "bind": "discovered_proteins"},
        ]
        inputs_needed = [
            {"name": "genome_ids", "desc": "Restrict to specific genomes?", "examples": ["G0012345", "G009999"]},
            {"name": "aliases", "desc": "Additional symbols/synonyms to prioritize in catalog search", "examples": ["gene symbols", "common abbreviations"]},
            {"name": "pfam_ids", "desc": "Optional PFAM IDs to search (if already known)", "examples": ["PF00016", "PF00485"]},
            {"name": "ko_ids", "desc": "Optional KO IDs to search (if already known)", "examples": ["K01601", "K00855"]},
        ]

    followup = {
        "type": "followup_request",
        "reason": reason,
        "next_task": {"steps": next_task_steps},
        "inputs_needed": inputs_needed,
    }
    return {"followup_request": followup}


register_operator(OperatorSpec(
    name="AssessEvidence",
    inputs=["data"],
    outputs=["evidence_metrics"],
    params={"min_rows": "int (default 5)"},
    run=_assess_evidence,
    description="Compute simple evidence metrics (row counts) for a bound result",
))

register_operator(OperatorSpec(
    name="ProposeFollowup",
    inputs=["evidence_metrics", "data"],  # 'data' optional; enables schema-driven branching
    outputs=["followup_request"],
    params={"question": "string", "top_n": "int (default 25)"},
    run=_propose_followup,
    description="Emit a data-driven follow-up proposal. If input rows look like CRISPR tables, propose DBTemplateCall steps; else fall back to generic discovery.",
))
