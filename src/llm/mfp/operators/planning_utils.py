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
    question = str(params.get("question") or "").strip()
    try:
        top_n = int(params.get("top_n", 25))
    except Exception:
        top_n = 25
    rows = int(metrics.get("rows", 0)) if isinstance(metrics, dict) else 0
    thr = int(metrics.get("threshold", 5)) if isinstance(metrics, dict) else 5
    reason = f"insufficient_evidence: rows={rows} < threshold={thr}" if rows < thr else f"followup_requested: rows={rows} >= threshold={thr}"

    # Generic next task: two-stage search then exact retrieval (catalog → IDs → query)
    next_task = {
        "steps": [
            {"op": "SearchPfamCatalogFuzzy", "params": {"q": question, "top_n": top_n}, "bind": "pfam_hits"},
            {"op": "SearchKoCatalogFuzzy", "params": {"q": question, "top_n": top_n}, "bind": "ko_hits"},
            {"op": "ExtractIdsFromCatalogHits", "inputs": {"pfam_catalog_hits": "pfam_hits", "ko_catalog_hits": "ko_hits"}, "bind": "id_lists"},
            {"op": "QueryProteinsByIds", "inputs": {"pfam_ids": "id_lists", "ko_ids": "id_lists"}, "params": {"limit": 1000}, "bind": "discovered_proteins"},
        ]
    }
    # Inputs needed: minimal, generic, not domain-specific
    inputs_needed = [
        {"name": "genome_ids", "desc": "Restrict to specific genomes?", "examples": ["G0012345", "G009999"]},
        {"name": "aliases", "desc": "Additional symbols/synonyms to prioritize in catalog search", "examples": ["gene symbols", "common abbreviations"]},
        {"name": "pfam_ids", "desc": "Optional PFAM IDs to search (if already known)", "examples": ["PF00016", "PF00485"]},
        {"name": "ko_ids", "desc": "Optional KO IDs to search (if already known)", "examples": ["K01601", "K00855"]},
    ]
    followup = {
        "type": "followup_request",
        "reason": reason,
        "next_task": next_task,
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
    inputs=["evidence_metrics"],
    outputs=["followup_request"],
    params={"question": "string", "top_n": "int (default 25)"},
    run=_propose_followup,
    description="Emit a generic follow-up proposal with minimal inputs requested",
))
