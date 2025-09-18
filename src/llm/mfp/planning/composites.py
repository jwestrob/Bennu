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
    # Gating: require explicit feature_selector.keyword or explicit ID lists
    kw = fs.get("keyword") if isinstance(fs.get("keyword"), str) else ""
    has_kw = isinstance(kw, str) and kw.strip() != ""
    has_ids = bool(fs.get("pfam_ids") or fs.get("ko_ids"))
    if not (has_kw or has_ids):
        # Do not run discovery unless the planner explicitly provided a keyword or IDs
        return []

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
            steps.append({"op": "SearchPfamCatalogFuzzy", "params": {"q": kw, "top_n": limits.get("top_k", 5)}, "bind": "pf_hits"})
        if "ko" in feature_types:
            steps.append({"op": "SearchKoCatalogFuzzy", "params": {"q": kw, "top_n": limits.get("top_k", 25)}, "bind": "ko_hits"})
        steps.append({
            "op": "QueryProteinsByIds",
            # Both search ops produce pfam_ids/ko_ids; use them directly
            "inputs": {"pfam_ids": "pfam_ids", "ko_ids": "ko_ids"},
            "params": {"limit": limits.get("row_cap", 500)}
        })

    # Optional facet summaries
    if out_profile == "facet_summary":
        # Emit facet steps ONLY when keyword is provided; otherwise skip to avoid validation failure
        if has_kw:
            steps.append({
                "op": "AnnotationDiscovery",
                "params": {
                    "keyword": kw,
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
                    "keyword": kw,
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


def expand_db_template_call(params: Dict[str, Any], ctx: CompositeContext) -> List[Dict[str, Any]]:
    """Composite: DBTemplateCall — execute a named DB template with slots.

    Params:
      - name: str (template name registered in kg/cypher_templates/registry)
      - slots: dict (template parameters)
    """
    name = params.get("name")
    slots = params.get("slots") or {}
    if not isinstance(name, str) or not name.strip():
        return []
    return [{"op": "ExecuteDBTemplate", "params": {"name": name, "slots": slots}}]


def expand_feature_profile(params: Dict[str, Any], ctx: CompositeContext) -> List[Dict[str, Any]]:
    """Composite: FeatureProfile

    Per‑genome PFAM+KO counts derived from keyword → IDs via local catalogs.

    Params:
      - genomes?: List[str]
      - keyword?: str (defaults to ctx.question)
      - pfam_top_k?: int (PFAM catalog hits cap; default 12)
      - ko_top_k?: int (KO catalog hits cap; default 25)
    """
    steps: List[Dict[str, Any]] = []
    kw = (params.get("keyword") or ctx.get("question", "")).strip()
    try:
        pfam_top_k = int(params.get("pfam_top_k", params.get("top_k", 12)))
    except Exception:
        pfam_top_k = 12
    try:
        ko_top_k = int(params.get("ko_top_k", params.get("top_k", 25)))
    except Exception:
        ko_top_k = 25
    if kw:
        steps.append({"op": "SearchPfamCatalogFuzzy", "params": {"q": kw, "top_n": pfam_top_k}, "bind": "pfam_search"})
        steps.append({"op": "SearchKoCatalogFuzzy", "params": {"q": kw, "top_n": ko_top_k}, "bind": "ko_search"})
        # Use the direct id outputs from the catalog searches (avoid nested binding pitfalls)
        steps.append({
            "op": "CountByIdsPerGenome",
            "inputs": {"pfam_ids": "pfam_ids", "ko_ids": "ko_ids"},
            "params": {"genome_ids": params.get("genomes", [])},
        })
    else:
        # If no keyword, do nothing (planner should provide keyword for this composite)
        return []
    steps.append({"op": "MaterializeFeatureProfile", "inputs": {"pfam_counts": "pfam_counts", "ko_counts": "ko_counts"}, "params": {}})
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
    # KO presence per genome (scoped via DatasetContext by operator default)
    steps.append({"op": "FetchPresentKOs", "params": {"genome_ids": params.get("genomes", [])}})
    # Load KO→pathway totals (for mapping)
    steps.append({"op": "LoadKoPathwayTotals", "params": {}})
    # Derive pathway filter from KO keywords in the question when not provided
    pw_filter = params.get("pathway_filter") or []
    if not pw_filter:
        steps.append({"op": "SearchKoCatalogFuzzy", "params": {"q": ctx.get("question", ""), "top_n": 25}, "bind": "ko_hits"})
        steps.append({
            "op": "MapKOsToPathways",
            "inputs": {"ko_ids": "ko_hits", "totals": "totals"},
            "params": {"top_n": 25},
            "bind": "pw_list"
        })
        steps.append({
            "op": "ComputePathwayCompleteness",
            "inputs": {"present": "present", "totals": "totals"},
            # slots mapping: pathways taken from pw_list.pathways
            "params": {"min_completeness": params.get("min_completeness", 0.0), "pathways": {"from": "rows", "field": "pathways", "index": 0}}
        })
    else:
        steps.append({
            "op": "ComputePathwayCompleteness",
            "inputs": {"present": "present", "totals": "totals"},
            "params": {"min_completeness": params.get("min_completeness", 0.0), "pathways": pw_filter}
        })
    steps.append({"op": "MaterializePathwayProfile", "inputs": {"present": "present", "pathway_completeness": "pathway_completeness"}, "params": {}})
    return steps


def expand_functional_profile(params: Dict[str, Any], ctx: CompositeContext) -> List[Dict[str, Any]]:
    """Composite: FunctionalProfile

    Unified pathways (KO completeness) + modules (CAZy/BGC) profiling.

    Params:
      - genomes?: List[str]
      - include?: List[str] in {'pathways','cazy','bgc'} (default ['pathways'])
      - pathway_filter?: List[str]
      - min_completeness?: float
      - cazy_output?: 'per_genome' | 'global_counts' (default 'per_genome')
    """
    steps: List[Dict[str, Any]] = []
    include = params.get("include") or ["pathways"]
    # Normalize include list
    try:
        include = [str(x).strip().lower() for x in include if isinstance(x, (str,))]
    except Exception:
        include = ["pathways"]

    genomes = params.get("genomes", [])

    if "pathways" in include:
        steps.append({"op": "FetchPresentKOs", "params": {"genome_ids": genomes}})
        steps.append({"op": "LoadKoPathwayTotals", "params": {}})
        pw_filter = params.get("pathway_filter") or []
        if not pw_filter:
            steps.append({"op": "SearchKoCatalogFuzzy", "params": {"q": ctx.get("question", ""), "top_n": 25}, "bind": "ko_hits"})
            steps.append({
                "op": "MapKOsToPathways",
                "inputs": {"ko_ids": "ko_hits", "totals": "totals"},
                "params": {"top_n": 25},
                "bind": "pw_list"
            })
            steps.append({
                "op": "ComputePathwayCompleteness",
                "inputs": {"present": "present", "totals": "totals"},
                "params": {"min_completeness": params.get("min_completeness", 0.0), "pathways": {"from": "rows", "field": "pathways", "index": 0}}
            })
        else:
            steps.append({
                "op": "ComputePathwayCompleteness",
                "inputs": {"present": "present", "totals": "totals"},
                "params": {"min_completeness": params.get("min_completeness", 0.0), "pathways": pw_filter}
            })

    if "cazy" in include:
        cazy_out = str(params.get("cazy_output") or "per_genome").strip().lower()
        if cazy_out == "global_counts":
            steps.append({"op": "CountCazymeFamilies", "params": {}})
        else:
            steps.append({"op": "QueryCazymesByGenome", "params": {"genome_ids": genomes}})

    if "bgc" in include:
        steps.append({"op": "QueryBGCsByGenome", "params": {"genome_ids": genomes}})

    steps.append({
        "op": "MaterializeFunctionalProfile",
        "inputs": {
            "present": "present",
            "pathway_completeness": "pathway_completeness",
            "cazymes": "cazymes",
            "cazyme_family_counts": "cazyme_family_counts",
            "bgcs": "bgcs"
        },
        "params": {"include": include}
    })
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
    # Evidence assessment: prefer a bound result reference if provided; else default to 'structured_data'
    bound = params.get("bound_result_ref")
    input_ref = bound if (isinstance(bound, str) and bound.strip()) else "structured_data"
    steps.append({"op": "AssessEvidence", "inputs": {"data": input_ref}, "params": {"min_rows": params.get("min_rows", 5)}})
    # Pass the same bound data into ProposeFollowup for data-driven branching
    steps.append({
        "op": "ProposeFollowup",
        "inputs": {"evidence_metrics": "evidence_metrics", "data": input_ref},
        "params": {"question": params.get("question", ctx.get("question", "")), "top_n": params.get("top_n", 10)}
    })
    steps.append({"op": "MaterializeEvidenceAndNext", "inputs": {"evidence_metrics": "evidence_metrics", "followup_request": "followup_request"}, "params": {}})
    return steps


COMPOSITE_EXPANDERS: Dict[str, Expansion] = {
    "FeatureDiscovery": expand_feature_discovery,
    "FeatureProfile": expand_feature_profile,
    "FunctionalProfile": lambda p, c: expand_functional_profile(p, c),
    "GeneContext": expand_gene_context,
    "PathwayProfile": expand_pathway_profile,
    "ModuleProfile": expand_module_profile,
    "DBTemplateCall": expand_db_template_call,
}


def planner_catalog_overlay() -> Dict[str, Any]:
    """Return a minimal planner-visible catalog describing only the 5 composites.

    This is used to restrict the planner's choice set without changing the runtime registry.
    """
    return {
        "operators": [
            {
                "name": "FeatureProfile",
                "description": "Per-genome PFAM+KO counts from a keyword using catalog outputs directly. Canonical flow: SearchPfamCatalogFuzzy → SearchKoCatalogFuzzy → CountByIdsPerGenome → MaterializeFeatureProfile. Do not insert ExtractIdsFromCatalogHits in this flow.",
                "inputs": ["genomes", "keyword", "pfam_top_k", "ko_top_k"],
                "params": {
                    "genomes": "List[str] genome IDs (optional; default dataset sample)",
                    "keyword": "string (defaults to user question)",
                    "pfam_top_k": "int cap for PFAM catalog hits (default 12)",
                    "ko_top_k": "int cap for KO catalog hits (default 25)"
                },
                "outputs": ["PerGenomeFeatureCounts", "FeatureProfileSummary", "PerGenomeTopMatrix"],
            },
            {
                "name": "FunctionalProfile",
                "description": "Unified pathways+modules profiling. Include any of ['pathways','cazy','bgc']; gates pathway completeness on non-empty KO→pathway mapping or explicit request.",
                "inputs": ["genomes", "include", "pathway_filter", "min_completeness"],
                "params": {
                    "genomes": "List[str] genome IDs (optional; default dataset sample)",
                    "include": "List of 'pathways'|'cazy'|'bgc' (default ['pathways'])",
                    "pathway_filter": "List[str] map IDs (optional)",
                    "min_completeness": "float filter (0.0–1.0)"
                },
                "outputs": ["PresentKOsByGenome", "CompletenessMatrix", "CompletenessSummary", "CAZyRowsByGenome", "CazymeFamilyCounts", "BGCsByGenome", "ProfileKinds"],
            },
            {
                "name": "FeatureDiscovery",
                "description": "Find proteins via PFAM/KO when an explicit feature_selector is provided. Use ONLY when params.feature_selector contains a non-empty keyword or explicit pfam_ids/ko_ids. Do not infer or invent keywords from the user question.",
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
                "name": "DBTemplateCall",
                "description": "Execute a named Neo4j DB template with slots (advanced/explicit-ID or window queries). Use when identifiers/windows are explicit; avoid for generic discovery.",
                "inputs": [],
                "params": {
                    "name": "string (template name)",
                    "slots": "object (template parameters)"
                },
                "outputs": ["structured_data"],
            },
        ]
    }
