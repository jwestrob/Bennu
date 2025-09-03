from __future__ import annotations
from typing import Any, Dict, List, Tuple
import os
from pathlib import Path
import time
import logging

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
    # Build aggregated summary (per-KO present genome counts)
    present_counts: Dict[str, int] = {}
    for gid, ko_list in present.items():
        for ko in (ko_list or []):
            if not isinstance(ko, str) or not ko:
                continue
            present_counts[ko] = present_counts.get(ko, 0) + 1
    present_summary = [{"ko_id": k, "present_genome_count": v} for k, v in sorted(present_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return {"present": present, "present_summary": present_summary}


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
        # Normalize KO ids: accept 'Kxxxxx', 'ko:Kxxxxx', or odd types defensively
        norm_kos = set()
        try:
            for k in (kos or []):
                try:
                    ks = str(k).strip()
                except Exception:
                    continue
                if not ks:
                    continue
                ks_u = ks.upper()
                if ks_u.startswith('KO:'):
                    ks_u = ks_u[3:]
                norm_kos.add(ks_u)
        except Exception:
            norm_kos = set()
        for pw, all_k in totals.items():
            if allowed is not None and pw not in allowed:
                continue
            all_set = set(all_k)
            tot = len(all_set)
            if tot == 0:
                continue
            pc = len(all_set & norm_kos)
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
    # New static, explicit controls
    output_profile = str(params.get("output_profile") or "facet_summary").strip().lower()  # facet_summary|rowset
    return_mode = str(params.get("return_mode") or "top_k").strip().lower()  # top_k|all
    try:
        ko_top_k = int(params.get("ko_top_k", 30))
    except Exception:
        ko_top_k = 30
    try:
        pfam_top_k = int(params.get("pfam_top_k", 20))
    except Exception:
        pfam_top_k = 20
    fields = params.get("fields") or []
    group_by = str(params.get("group_by") or "both").strip().lower()  # ko|pfam|both
    include_examples = str(params.get("include_examples") or "counts").strip().lower()  # none|counts|ids
    return_full = bool(params.get("return_full_rows") or False)
    genome_ids = params.get("genome_ids") or []

    # Stage 1: catalog fuzzy search (timed)
    _t0 = time.perf_counter()
    pf_hits = _search_pfam(q, ctx.project_root, top_n=50)
    _t1 = time.perf_counter()
    ko_hits = _search_ko(q, ctx.project_root, top_n=50)
    _t2 = time.perf_counter()
    try:
        import logging
        logging.getLogger(__name__).info(
            f"AnnotationDiscovery: catalog keyword='{q}' pf_hits={len(pf_hits)} in {(_t1-_t0)*1000:.0f} ms; "
            f"ko_hits={len(ko_hits)} in {(_t2-_t1)*1000:.0f} ms")
    except Exception:
        pass
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
        __tp = time.perf_counter()
        pf_rows = runner.run_template(
            "proteins_by_pfam_ids.cypher",
            {"pfam_ids": pfam_ids, "genome_ids": genome_ids, "limit": limit},
        ) or []
        try:
            import logging
            logging.getLogger(__name__).info(
                f"AnnotationDiscovery: proteins_by_pfam_ids ids={len(pfam_ids)} genomes={len(genome_ids)} "
                f"rows={len(pf_rows)} in {(time.perf_counter()-__tp)*1000:.0f} ms")
        except Exception:
            pass
    if ko_ids:
        __tk = time.perf_counter()
        ko_rows = runner.run_template(
            "proteins_by_ko_ids.cypher",
            {"ko_ids": ko_ids, "genome_ids": genome_ids, "limit": limit},
        ) or []
        try:
            import logging
            logging.getLogger(__name__).info(
                f"AnnotationDiscovery: proteins_by_ko_ids ids={len(ko_ids)} genomes={len(genome_ids)} "
                f"rows={len(ko_rows)} in {(time.perf_counter()-__tk)*1000:.0f} ms")
        except Exception:
            pass

    # Merge with provenance; key by (genome_id, protein_id)
    debug_ann = str(os.getenv('DEBUG_ANN_DISCOVERY', '')).lower() not in ('', '0', 'false')
    merged: Dict[str, Dict[str, Any]] = {}
    for r in pf_rows:
        gid = str(r.get("genome_id"))
        pid = str(r.get("protein_id"))
        key = f"{gid}\t{pid}"
        entry = merged.setdefault(key, {"genome_id": gid, "protein_id": pid, "pfams": [], "kos": []})
        pf_label = r.get("pfam_name") or r.get("domain_desc") or r.get("domain_id") or r.get("pfam_id")
        if pf_label and pf_label not in entry["pfams"]:
            entry["pfams"].append(pf_label)
        # Optionally capture internal PFAM accessions for debugging/hallmark detection (never printed by default)
        if debug_ann:
            pf_acc = r.get("pfam_id")
            if pf_acc:
                ids = entry.setdefault("pfam_ids", [])
                if pf_acc not in ids:
                    ids.append(pf_acc)
    for r in ko_rows:
        gid = str(r.get("genome_id"))
        pid = str(r.get("protein_id"))
        key = f"{gid}\t{pid}"
        entry = merged.setdefault(key, {"genome_id": gid, "protein_id": pid, "pfams": [], "kos": []})
        koid = r.get("ko_id")
        if koid and koid not in entry["kos"]:
            entry["kos"].append(koid)

    # Aggregate facets
    def _aggregate_counts(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        ko_counts: Dict[str, int] = {}
        pf_counts: Dict[str, int] = {}
        for r in rows:
            for ko in (r.get("kos") or []):
                if isinstance(ko, str) and ko:
                    ko_counts[ko] = ko_counts.get(ko, 0) + 1
            for pf in (r.get("pfams") or []):
                if isinstance(pf, str) and pf:
                    pf_counts[pf] = pf_counts.get(pf, 0) + 1
        kos = sorted(({"id": k, "count": v} for k, v in ko_counts.items()), key=lambda d: (-d["count"], d["id"]))
        pfs = sorted(({"id": k, "count": v} for k, v in pf_counts.items()), key=lambda d: (-d["count"], d["id"]))
        return kos, pfs

    kos_facets, pfam_facets = _aggregate_counts(list(merged.values()))
    max_server_cap = 10000
    def _apply_limit(items: List[Dict[str, Any]], top_k: int):
        total = len(items)
        applied_k = total if return_mode == 'all' else min(top_k, total)
        applied_k = min(applied_k, max_server_cap)
        clamped = (return_mode == 'all' and total > max_server_cap)
        return items[:applied_k], {
            "requested": {"return_mode": return_mode, "top_k": top_k, "fields": fields},
            "applied": {"top_k_applied": applied_k, "order_by": "count_desc,id_asc"},
            "total_available": total,
            "max_server_cap": max_server_cap,
            "clamped": clamped,
            "estimated_tokens": applied_k * 6,
        }

    selection_meta = {"groups": {}}
    facets: Dict[str, Any] = {}
    if group_by in ("ko", "both"):
        sel, sm = _apply_limit(kos_facets, ko_top_k)
        facets["kos"] = sel
        selection_meta["groups"]["ko"] = sm
    if group_by in ("pfam", "both"):
        sel, sm = _apply_limit(pfam_facets, pfam_top_k)
        facets["pfams"] = sel
        selection_meta["groups"]["pfam"] = sm

    # Compose result
    out_rows = list(merged.values())
    out_rows.sort(key=lambda x: (x.get("genome_id", ""), x.get("protein_id", "")))
    res: Dict[str, Any] = {"facet_summary": facets, "selection_metadata": selection_meta}
    if output_profile == 'rowset' or return_full:
        res["discovered_proteins"] = out_rows
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
    outputs=["facet_summary", "selection_metadata", "discovered_proteins"],
    params={
        "keyword": "str (free text)",
        "limit": "int (row budget for legacy rowset mode; default 1000)",
        "output_profile": "facet_summary|rowset (default facet_summary)",
        "return_mode": "top_k|all (default top_k)",
        "ko_top_k": "int (default 30)",
        "pfam_top_k": "int (default 20)",
        "fields": "List[str] (requested fields for summaries; optional)",
        "group_by": "ko|pfam|both (default both)",
        "include_examples": "none|counts|ids (ids only for contig/locus anchors)",
        "return_full_rows": "bool (legacy rowset include)",
        "genome_ids": "List[str] | null",
    },
    run=_annotation_discovery,
    description="Facet-first annotation discovery: keyword→IDs→exact retrieval→KO/PFAM summaries (counts, top_k or all). Rowset optional.",
))
