from __future__ import annotations
from typing import Any, Dict
from pathlib import Path

from .base import OperatorContext, OperatorSpec, register_operator
from ...options.template_runner import FileCypherRunner
from .catalog_search import _search_pfam, _search_ko
from ...kegg.pathway_mapping import load_ko_pathway_maps


def _fetch_present_kos(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    runner = FileCypherRunner(ctx.neo4j_driver)
    genome_ids = params.get("genome_ids") or []
    rows = runner.run_template("present_kos_by_genome.cypher", {"genome_ids": genome_ids})
    present: Dict[str, Any] = {}
    for r in rows or []:
        gid = str(r.get("genome_id"))
        kos = [str(k) for k in (r.get("present_ko_ids") or [])]
        present[gid] = kos
    return {"present": present}


def _load_ko_totals(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    # Loads native ko_pathway.list mapping
    pw_to_kos, _ = load_ko_pathway_maps()
    return {"totals": {k: sorted(list(v)) for k, v in pw_to_kos.items()}}


def _compute_pathway_completeness(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    # Inputs: present (dict genome->list KO), totals (dict pathway->list KO)
    present = inputs.get("present") or {}
    totals = inputs.get("totals") or {}
    pathways = params.get("pathways")  # optional list of map IDs
    min_c = params.get("min_completeness")
    try:
        min_c = float(min_c) if min_c is not None else None
    except Exception:
        min_c = None
    # Default behavior: filter out pathways with zero representatives unless explicitly overridden with 0.0
    if min_c is None:
        min_c = 1e-6

    # optional filter
    allowed = set(pathways) if isinstance(pathways, list) and pathways else None

    out_rows = []
    for gid, kos in present.items():
        s = set(k.lstrip('ko:') for k in kos)
        for pw, all_k in totals.items():
            if allowed is not None and pw not in allowed:
                continue
            all_set = set(all_k)
            tot = len(all_set)
            if tot == 0:
                continue
            pc = len(all_set & s)
            comp = pc / float(tot)
            if min_c is not None and comp < min_c:
                continue
            out_rows.append({
                "genome_id": gid,
                "pathway_id": pw,
                "pathway_name": pw,
                "present_kos": pc,
                "total_kos": tot,
                "completeness": comp,
            })
    # Stable sort
    out_rows.sort(key=lambda r: (r["genome_id"], -float(r["completeness"]), -int(r["present_kos"]), r["pathway_id"]))
    return {"pathway_completeness": out_rows}


def _bgcs_by_genome(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    genome_id = params.get("genome_id")
    genome_ids = params.get("genome_ids") or []
    runner = FileCypherRunner(ctx.neo4j_driver)
    rows = runner.run_template(
        "bgcs_by_genome.cypher",
        {
            "genome_id": genome_id,
            "genome_ids": genome_ids,
            # Dynamic property key preferences to avoid UnknownPropertyKey warnings
            "id_keys": ["bgcId", "bgc_id", "id"],
            "product_keys": ["bgcProduct", "bgc_product", "product", "cluster_type"],
            "contig_keys": ["contig", "scaffold", "seqid"],
            "start_keys": ["startCoordinate", "start", "begin", "start_position"],
            "end_keys": ["endCoordinate", "end", "finish", "end_position"],
            "length_keys": ["lengthNt", "length"],
            "protein_keys": ["proteinCount", "proteins", "protein_count"],
            "avg_prob_keys": ["averageProbability", "avg_probability", "average_p"],
            "max_prob_keys": ["maxProbability", "max_probability", "max_p"],
        },
    )
    return {"bgcs": rows or []}


def _cazymes_by_genome(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    genome_id = params.get("genome_id")
    genome_ids = params.get("genome_ids") or []
    runner = FileCypherRunner(ctx.neo4j_driver)
    rows = runner.run_template("cazymes_by_genome.cypher", {"genome_id": genome_id, "genome_ids": genome_ids})
    return {"cazymes": rows or []}


def _cazyme_family_counts(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    runner = FileCypherRunner(ctx.neo4j_driver)
    rows = runner.run_template("cazyme_family_counts.cypher", {})
    return {"cazyme_family_counts": rows or []}


def _annotation_discovery(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """PFAM+KO discovery via two-stage flow (catalog → IDs → exact retrieval).

    - Uses local catalogs (PFAM and KO) to map natural-language keywords to precise IDs
    - Retrieves proteins by exact ID matches (IN filters), deduplicating with PFAM/KO provenance
    """
    runner = FileCypherRunner(ctx.neo4j_driver)
    q = str(params.get("keyword") or params.get("q") or "").strip()
    if not q:
        return {"discovered_proteins": []}
    try:
        limit = int(params.get("limit", 1000))
    except Exception:
        limit = 1000
    return_full = bool(params.get("return_full_rows") or False)
    genome_ids = params.get("genome_ids") or []

    # Stage 1: catalog fuzzy search
    pf_hits = _search_pfam(q, ctx.project_root, top_n=50)
    ko_hits = _search_ko(q, ctx.project_root, top_n=50)
    # Accept both accessions (PFxxxxx) and short names for robust matching
    pfam_ids = []
    seen = set()
    for h in pf_hits:
        acc = (h.get("pfam_id") or "").strip()
        short = (h.get("short") or "").strip()
        if acc and acc not in seen:
            pfam_ids.append(acc)
            seen.add(acc)
        if short and short not in seen:
            pfam_ids.append(short)
            seen.add(short)
    ko_ids = [h.get("ko_id") for h in ko_hits if h.get("ko_id")]

    # Stage 2: exact ID retrieval
    pf_rows = []
    ko_rows = []
    if pfam_ids:
        pf_rows = runner.run_template(
            "proteins_by_pfam_ids.cypher",
            {"pfam_ids": pfam_ids, "genome_ids": genome_ids, "limit": limit},
        ) or []
    if ko_ids:
        ko_rows = runner.run_template(
            "proteins_by_ko_ids.cypher",
            {"ko_ids": ko_ids, "genome_ids": genome_ids, "limit": limit},
        ) or []

    # Merge with provenance; key by (genome_id, protein_id)
    merged: Dict[str, Dict[str, Any]] = {}
    for r in pf_rows:
        gid = str(r.get("genome_id"))
        pid = str(r.get("protein_id"))
        key = f"{gid}\t{pid}"
        entry = merged.setdefault(key, {"genome_id": gid, "protein_id": pid, "pfams": [], "kos": []})
        pf = r.get("pfam_id") or r.get("domain_id")
        if pf and pf not in entry["pfams"]:
            entry["pfams"].append(pf)
    for r in ko_rows:
        gid = str(r.get("genome_id"))
        pid = str(r.get("protein_id"))
        key = f"{gid}\t{pid}"
        entry = merged.setdefault(key, {"genome_id": gid, "protein_id": pid, "pfams": [], "kos": []})
        koid = r.get("ko_id")
        if koid and koid not in entry["kos"]:
            entry["kos"].append(koid)

    out = list(merged.values())
    out.sort(key=lambda x: (x.get("genome_id", ""), x.get("protein_id", "")))
    res = {"discovered_proteins": out}
    if return_full:
        res["_format"] = "full"
    return res

# Register operators
register_operator(OperatorSpec(
    name="FetchPresentKOs",
    inputs=[],
    outputs=["present"],
    params={"genome_ids": "List[str] | null"},
    run=_fetch_present_kos,
    description="Present KO IDs per genome (global or filtered).",
))

register_operator(OperatorSpec(
    name="LoadKoPathwayTotals",
    inputs=[],
    outputs=["totals"],
    params={"source": "native|db (native default)"},
    run=_load_ko_totals,
    description="Load KO→pathway totals from ko_pathway.list (native mode).",
))

register_operator(OperatorSpec(
    name="ComputePathwayCompleteness",
    inputs=["present", "totals"],
    outputs=["pathway_completeness"],
    params={
        "pathways": "List[str] | null",
        "min_completeness": "float | null (default 1e-6; set 0.0 to include empty pathways)",
    },
    run=_compute_pathway_completeness,
    description="Compute per-genome KEGG pathway completeness from present KOs.",
))

register_operator(OperatorSpec(
    name="QueryBGCsByGenome",
    inputs=[],
    outputs=["bgcs"],
    params={"genome_id": "str | null", "genome_ids": "List[str] | null"},
    run=_bgcs_by_genome,
    description="List predicted BGC clusters (global or per-genome); schema-tolerant.",
))

register_operator(OperatorSpec(
    name="QueryCazymesByGenome",
    inputs=[],
    outputs=["cazymes"],
    params={"genome_id": "str | null", "genome_ids": "List[str] | null"},
    run=_cazymes_by_genome,
    description="List CAZyme-annotated proteins (global or per-genome).",
))

register_operator(OperatorSpec(
    name="CountCazymeFamilies",
    inputs=[],
    outputs=["cazyme_family_counts"],
    params={},
    run=_cazyme_family_counts,
    description="Global count of CAZy families across proteins.",
))

## Removed legacy keyword operators in favor of catalog→ID→exact retrieval inside AnnotationDiscovery

register_operator(OperatorSpec(
    name="AnnotationDiscovery",
    inputs=[],
    outputs=["discovered_proteins"],
    params={"keyword": "str", "limit": "int | null", "genome_ids": "List[str] | null", "return_full_rows": "bool | null (default False)"},
    run=_annotation_discovery,
    description="Two-stage discovery (catalog→IDs→exact) across PFAM+KO; returns union with PFAM/KO provenance.",
))
