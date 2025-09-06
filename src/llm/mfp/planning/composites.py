# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List, Callable, TypedDict


class CompositeContext(TypedDict, total=False):
    question: str


Expansion = Callable[[Dict[str, Any], CompositeContext], List[Dict[str, Any]]]


def expand_feature_discovery(params: Dict[str, Any], ctx: CompositeContext) -> List[Dict[str, Any]]:
    """Composite: FeatureDiscovery

    Params (advisory; planner-provided):
      - feature_selector: {keyword?|pfam_ids?|ko_ids?}
      - feature_types: ["pfam","ko"] (optional; defaults both)
      - output_profile: "rowset"|"facet_summary"|"ids_only" (optional)
      - limits: {top_k,row_cap}
    """
    fs = params.get("feature_selector") or {}
    feature_types = params.get("feature_types") or ["pfam", "ko"]
    out_profile = params.get("output_profile") or "rowset"
    limits = params.get("limits") or {}
    keyword = fs.get("keyword") or params.get("keyword") or ""

    steps: List[Dict[str, Any]] = []

    # Exact ID route when IDs specified
    if (fs.get("pfam_ids") or fs.get("ko_ids")):
        steps.append({
            "op": "QueryProteinsByIds",
            "params": {
                "pfam_ids": fs.get("pfam_ids", []),
                "ko_ids": fs.get("ko_ids", []),
                "limit": limits.get("row_cap", 500)
            }
        })
    else:
        # Keyword route: catalog search → direct ids (skip ExtractIds; search ops already emit pfam_ids/ko_ids)
        if "pfam" in feature_types:
            # Default to a small PFAM probe (top_n≈5) unless planner provided limits.top_k
            steps.append({"op": "SearchPfamCatalogFuzzy", "params": {"q": keyword, "top_n": limits.get("top_k", 5)}, "bind": "pf_hits"})
        if "ko" in feature_types:
            steps.append({"op": "SearchKoCatalogFuzzy", "params": {"q": keyword, "top_n": limits.get("top_k", 25)}, "bind": "ko_hits"})
        steps.append({
            "op": "QueryProteinsByIds",
            # Both search ops produce pfam_ids/ko_ids; use them directly
            "inputs": {"pfam_ids": "pfam_ids", "ko_ids": "ko_ids"},
            "params": {"limit": limits.get("row_cap", 500)}
        })

    # Optional facet summaries
    if out_profile == "facet_summary":
        # Emit facet steps ONLY when keyword is provided; otherwise skip to avoid validation failure
        if isinstance(keyword, str) and keyword.strip():
            steps.append({
                "op": "AnnotationDiscovery",
                "params": {
                    "keyword": keyword,
                    "output_profile": "facet_summary",
                    "group_by": "pfam",
                    "return_mode": "top_k",
                    "pfam_top_k": limits.get("top_k", 20)
                },
                "bind": "pf_facet"
            })
            steps.append({
                "op": "AnnotationDiscovery",
                "params": {
                    "keyword": keyword,
                    "output_profile": "facet_summary",
                    "group_by": "ko",
                    "return_mode": "top_k",
                    "ko_top_k": limits.get("top_k", 20)
                },
                "bind": "ko_facet"
            })

    # Final materializer (typed outputs)
    mat_inputs = {"discovered_proteins": "discovered_proteins"}
    if out_profile == "facet_summary":
        mat_inputs.update({"pf_facet": "pf_facet", "ko_facet": "ko_facet"})
    steps.append({"op": "MaterializeFeatureDiscovery", "inputs": mat_inputs, "params": {"output_profile": out_profile}})
    return steps


def expand_gene_context(params: Dict[str, Any], ctx: CompositeContext) -> List[Dict[str, Any]]:
    """Composite: GeneContext

    Params:
      - seeds: {protein_ids?|pfam_ids?|ko_ids?|keyword?}
      - context: {seeds_limit, limit, span_fallback_bp, k, include_degree_zero_seeds}
      - output_profile: "rowset"|"macro_summary"
    """
    seeds = params.get("seeds") or {}
    context = params.get("context") or {}
    out_profile = params.get("output_profile") or "rowset"

    n_params: Dict[str, Any] = {
        "output_profile": "rowset" if out_profile == "rowset" else "summary",
        "seeds_limit": context.get("seeds_limit", 10),
        "limit": context.get("limit", 200),
        "fallback_window_bp": context.get("span_fallback_bp", 10000),
        "include_degree_zero_seeds": bool(context.get("include_degree_zero_seeds", False)),
    }
    # k-step adjacency if provided
    if "k" in context or "k_step" in context:
        n_params["k"] = int(context.get("k", context.get("k_step", 0) or 0)) or None
    # Seeding options
    if seeds.get("protein_ids"):
        n_params["protein_ids"] = list(seeds.get("protein_ids") or [])
    else:
        if seeds.get("pfam_ids"):
            n_params["seed_pfam_ids"] = list(seeds.get("pfam_ids") or [])
        if seeds.get("ko_ids"):
            n_params["seed_ko_ids"] = list(seeds.get("ko_ids") or [])

    steps: List[Dict[str, Any]] = []
    steps.append({"op": "NeighborhoodContext", "params": n_params, "inputs": {"discovered_proteins": "discovered_proteins"}})
    steps.append({"op": "MaterializeGeneContext", "inputs": {"neighborhoods": "neighborhoods", "neighborhood_summary": "neighborhood_summary"}, "params": {"output_profile": out_profile}})
    return steps


def expand_pathway_profile(params: Dict[str, Any], ctx: CompositeContext) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    steps.append({"op": "FetchPresentKOs", "params": {"genome_ids": params.get("genomes", [])}})
    steps.append({"op": "LoadKoPathwayTotals", "params": {}})
    steps.append({
        "op": "ComputePathwayCompleteness",
        "inputs": {"present": "present", "totals": "totals"},
        "params": {"min_completeness": params.get("min_completeness", 0.0), "pathways": params.get("pathway_filter", [])}
    })
    steps.append({"op": "MaterializePathwayProfile", "inputs": {"present": "present", "pathway_completeness": "pathway_completeness"}, "params": {}})
    return steps


def expand_module_profile(params: Dict[str, Any], ctx: CompositeContext) -> List[Dict[str, Any]]:
    module = (params.get("module") or "cazy").strip().lower()
    out = params.get("output_profile") or "per_genome"
    steps: List[Dict[str, Any]] = []
    if module == "cazy":
        # Counts-only path: do not fetch rowsets
        if out == "global_counts":
            steps.append({"op": "CountCazymeFamilies", "params": {}})
        else:
            steps.append({"op": "QueryCazymesByGenome", "params": {"genome_ids": params.get("genomes", [])}})
    elif module == "bgc":
        steps.append({"op": "QueryBGCsByGenome", "params": {"genome_ids": params.get("genomes", [])}})
    steps.append({"op": "MaterializeModuleProfile", "inputs": {"cazymes": "cazymes", "cazyme_family_counts": "cazyme_family_counts", "bgcs": "bgcs"}, "params": {"module": module, "output_profile": out}})
    return steps


def expand_evidence_and_next(params: Dict[str, Any], ctx: CompositeContext) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    # Evidence assessment on last bound result when available; if not, will compute with zero rows
    steps.append({"op": "AssessEvidence", "inputs": {"data": "discovered_proteins"}, "params": {"min_rows": params.get("min_rows", 5)}})
    steps.append({"op": "ProposeFollowup", "inputs": {"evidence_metrics": "evidence_metrics"}, "params": {"question": params.get("question", ctx.get("question", "")), "top_n": params.get("top_n", 10)}})
    steps.append({"op": "MaterializeEvidenceAndNext", "inputs": {"evidence_metrics": "evidence_metrics", "followup_request": "followup_request"}, "params": {}})
    return steps


COMPOSITE_EXPANDERS: Dict[str, Expansion] = {
    "FeatureDiscovery": expand_feature_discovery,
    "GeneContext": expand_gene_context,
    "PathwayProfile": expand_pathway_profile,
    "ModuleProfile": expand_module_profile,
    "EvidenceAndNext": expand_evidence_and_next,
}


def planner_catalog_overlay() -> Dict[str, Any]:
    """Return a minimal planner-visible catalog describing only the 5 composites.

    This is used to restrict the planner's choice set without changing the runtime registry.
    """
    return {
        "operators": [
            {
                "name": "FeatureDiscovery",
                "description": "Find proteins via PFAM/KO keywords or exact IDs; outputs a typed ProteinSet and optional facet summaries.",
                "inputs": ["feature_selector", "feature_types", "output_profile", "limits"],
                "params": {
                    "feature_selector": "{keyword?: str, pfam_ids?: List[str], ko_ids?: List[str]}",
                    "feature_types": "List[str] one or both of ['pfam','ko'] (default both)",
                    "output_profile": "'rowset' (default) | 'facet_summary' | 'ids_only'",
                    "limits": "{top_k?: int, row_cap?: int}"
                },
                "outputs": ["FeatureSet", "ProteinSet", "FacetSummary"],
            },
            {
                "name": "GeneContext",
                "description": "Neighborhoods around seed proteins (k-step or flanking); seeds can come from FeatureDiscovery.",
                "inputs": ["seeds", "context", "output_profile"],
                "params": {
                    "seeds": "{protein_ids?: List[str], pfam_ids?: List[str], ko_ids?: List[str]}",
                    "context": "{seeds_limit?: int, limit?: int, span_fallback_bp?: int, k?: int, include_degree_zero_seeds?: bool}",
                    "output_profile": "'rowset' (default) | 'macro_summary'"
                },
                "outputs": ["NeighborhoodSet", "NeighborhoodSummary"],
            },
            {
                "name": "PathwayProfile",
                "description": "Compute per-genome KO presence and KEGG pathway completeness.",
                "inputs": ["genomes", "pathway_filter", "min_completeness"],
                "params": {
                    "genomes": "List[str] genome IDs (optional; default all)",
                    "pathway_filter": "List[str] map IDs (optional)",
                    "min_completeness": "float filter (0.0–1.0)"
                },
                "outputs": ["PresentKOsByGenome", "CompletenessMatrix", "CompletenessSummary"],
            },
            {
                "name": "ModuleProfile",
                "description": "CAZy or BGC profiling. Use output_profile='global_counts' for counts-only (no row fetch); per_genome/rowset for detailed rows.",
                "inputs": ["genomes", "module", "output_profile"],
                "params": {
                    "genomes": "List[str] genome IDs (optional; default all)",
                    "module": "'cazy' | 'bgc'",
                    "output_profile": "'global_counts' | 'per_genome' (default) | 'rowset'"
                },
                "outputs": ["ModuleRows", "GlobalCounts"],
            },
            {
                "name": "EvidenceAndNext",
                "description": "Assess evidence sufficiency and propose follow-up actions.",
                "inputs": ["bound_result_ref", "min_rows", "question", "top_n"],
                "params": {
                    "bound_result_ref": "string binding name for data to assess (optional)",
                    "min_rows": "int threshold (default 5)",
                    "question": "original question (for follow-up)",
                    "top_n": "int for catalog search in follow-up"
                },
                "outputs": ["EvidenceMetrics", "FollowupPlan"],
            },
        ]
    }
