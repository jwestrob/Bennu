from __future__ import annotations
from typing import Any, Dict, List, Tuple
import os
from pathlib import Path
import time
import logging

from .base import OperatorContext, OperatorSpec, register_operator
from ...options.template_runner import FileCypherRunner
from .catalog_search import _search_pfam, _search_ko, _load_pfam_catalog, _load_ko_catalog
from ...kegg.pathway_mapping import load_ko_pathway_maps
from ...kg.cypher_templates import registry as kg_tpl_registry
from ..types import FeatureSet, ProteinSet, assert_featureset, assert_proteinset


def _fetch_present_kos(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    runner = FileCypherRunner(ctx.neo4j_driver)
    genome_ids = params.get("genome_ids") or []
    if not genome_ids and ctx.dataset_context:
        genome_ids = ctx.dataset_context.get('genome_ids_sample') or []
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


def _map_kos_to_pathways(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Map a list of KO ids to KEGG pathway ids using a totals mapping.

    Inputs:
      - ko_ids: list[str]
      - totals: { pathway_id -> [KO ids] }
    Params:
      - top_n (optional): cap the number of pathways returned (default 25)
    """
    ko_ids_in = inputs.get("ko_ids") or params.get("ko_ids") or []
    totals_in = inputs.get("totals") or params.get("totals") or {}
    try:
        top_n = int(params.get("top_n", 25))
    except Exception:
        top_n = 25
    # Normalize KO ids: accept 'Kxxxxx' or 'ko:Kxxxxx'
    kos = []
    try:
        for k in (ko_ids_in or []):
            s = str(k).strip()
            if not s:
                continue
            u = s.upper()
            if u.startswith('KO:'):
                u = u[3:]
            kos.append(u)
    except Exception:
        kos = []
    if not kos or not isinstance(totals_in, dict):
        return {"pathways": []}
    kos_set = set(kos)
    out = []
    for pw, ko_list in totals_in.items():
        try:
            has = any((str(x).upper() in kos_set) for x in (ko_list or []))
        except Exception:
            has = False
        if has:
            out.append(str(pw))
    # Deterministic order and clamp
    out = sorted(set(out))[: max(1, top_n)]
    return {"pathways": out}


def _compute_pathway_completeness(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    # Inputs expected:
    #  - present: { genome_id -> [KO ids] }
    #  - totals:  { pathway_id -> [KO ids] }
    # Be defensive: unwrap common envelope shapes like {'present': {...}, 'present_summary': [...]}
    raw_present = inputs.get("present") or {}
    if isinstance(raw_present, dict) and isinstance(raw_present.get("present"), dict):
        present = raw_present.get("present") or {}
    else:
        present = raw_present

    raw_totals = inputs.get("totals") or {}
    if isinstance(raw_totals, dict) and isinstance(raw_totals.get("totals"), dict):
        totals = raw_totals.get("totals") or {}
    else:
        totals = raw_totals
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
    # Guard: avoid computing ALL pathways by default unless explicitly allowed
    if allowed is None and not bool(params.get("allow_all_pathways", False)):
        try:
            import logging
            logging.getLogger(__name__).info("ComputePathwayCompleteness skipped: pathways filter empty and allow_all_pathways not set")
        except Exception:
            pass
        return {"pathway_completeness": []}

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
    if not genome_id and not genome_ids and ctx.dataset_context:
        genome_ids = ctx.dataset_context.get('genome_ids_sample') or []
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
    if not genome_id and not genome_ids and ctx.dataset_context:
        genome_ids = ctx.dataset_context.get('genome_ids_sample') or []
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
    # Sanitize requested fields to known facet fields
    fields_in = params.get("fields") or []
    fields_allowed = {"id", "name", "count"}
    fields = [f for f in fields_in if isinstance(f, str) and f in fields_allowed]
    if not fields:
        fields = ["id", "name", "count"]
    group_by = str(params.get("group_by") or "both").strip().lower()  # ko|pfam|both
    include_examples = str(params.get("include_examples") or "counts").strip().lower()  # none|counts|ids
    return_full = bool(params.get("return_full_rows") or False)
    genome_ids = params.get("genome_ids") or []
    if not genome_ids and ctx.dataset_context:
        genome_ids = ctx.dataset_context.get('genome_ids_sample') or []

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

    # Decide which facets are needed and whether to use counts or rowsets
    pfam_needed = group_by in ("pfam", "both")
    ko_needed = group_by in ("ko", "both")
    use_counts = (output_profile == 'facet_summary') and (not return_full)

    kos_facets: List[Dict[str, Any]] = []
    pfam_facets: List[Dict[str, Any]] = []
    merged: Dict[str, Dict[str, Any]] = {}

    # Clamp catalog IDs to requested top_k before DB calls (facet path)
    pfam_inputs = pfam_ids
    ko_inputs = ko_ids
    # Determine token clamps and optional explicit allowlists (agent-visible knobs)
    # Apply these consistently to both counts and rowset paths (no hidden behavior).
    # Optional planner-provided token caps
    try:
        pfam_tokens_top_n = int(params.get("pfam_tokens_top_n")) if params.get("pfam_tokens_top_n") is not None else None
    except Exception:
        pfam_tokens_top_n = None
    try:
        ko_tokens_top_n = int(params.get("ko_tokens_top_n")) if params.get("ko_tokens_top_n") is not None else None
    except Exception:
        ko_tokens_top_n = None

    # Optional explicit allowlists to restrict rowset retrieval (params or inputs)
    pfam_ids_override = None
    if isinstance(params.get("pfam_ids"), list):
        pfam_ids_override = [str(x) for x in params.get("pfam_ids")]
    elif isinstance(inputs.get("pfam_ids"), list):
        pfam_ids_override = [str(x) for x in inputs.get("pfam_ids")]
    ko_ids_override = None
    if isinstance(params.get("ko_ids"), list):
        ko_ids_override = [str(x) for x in params.get("ko_ids")]
    elif isinstance(inputs.get("ko_ids"), list):
        ko_ids_override = [str(x) for x in inputs.get("ko_ids")]

    # Build rowset token lists respecting caps/overrides
    def _clamp(tokens: List[str], n: int | None) -> List[str]:
        if n is None or n <= 0:
            return tokens
        return tokens[:n]

    rowset_pfam_ids = pfam_ids
    rowset_ko_ids = ko_ids
    if pfam_ids_override:
        rowset_pfam_ids = [str(x) for x in pfam_ids_override if isinstance(x, str)]
    else:
        rowset_pfam_ids = _clamp(rowset_pfam_ids, pfam_tokens_top_n)
    if ko_ids_override:
        rowset_ko_ids = [str(x) for x in ko_ids_override if isinstance(x, str)]
    else:
        rowset_ko_ids = _clamp(rowset_ko_ids, ko_tokens_top_n)

    if use_counts:
        # Optional planner-provided token caps
        try:
            pfam_tokens_top_n = int(params.get("pfam_tokens_top_n")) if params.get("pfam_tokens_top_n") is not None else None
        except Exception:
            pfam_tokens_top_n = None
        try:
            ko_tokens_top_n = int(params.get("ko_tokens_top_n")) if params.get("ko_tokens_top_n") is not None else None
        except Exception:
            ko_tokens_top_n = None

        if pfam_needed:
            pfam_inputs = pfam_ids[:pfam_tokens_top_n] if (pfam_tokens_top_n and pfam_tokens_top_n > 0) else pfam_ids
        else:
            pfam_inputs = []
        if ko_needed:
            ko_inputs = ko_ids[:ko_tokens_top_n] if (ko_tokens_top_n and ko_tokens_top_n > 0) else ko_ids
        else:
            ko_inputs = []

    if use_counts:
        # Run count templates for selected facets only
        pf_provenance: Dict[str, set] = {}
        if pfam_needed and pfam_inputs:
            __tpf = time.perf_counter()
            # Candidate cap for domains per token (default 200)
            try:
                pfam_candidate_cap = int(params.get("pfam_candidate_cap")) if params.get("pfam_candidate_cap") is not None else 200
            except Exception:
                pfam_candidate_cap = 200
            cnt_pf = runner.run_template(
                "count_proteins_by_pfam_tokens.cypher",
                {"tokens": pfam_inputs, "genome_ids": genome_ids, "candidate_cap": pfam_candidate_cap},
            ) or []
            try:
                logging.getLogger(__name__).info(
                    f"AnnotationDiscovery: count_pf tokens={len(pfam_inputs)} candidates={len(cnt_pf)} cap={pfam_candidate_cap} genomes={len(genome_ids)} in {(time.perf_counter()-__tpf)*1000:.0f} ms")
            except Exception:
                pass
            # Build facet entries from counts with stable id (PFxxxxx) and name label
            # Deduplicate by id if multiple tokens resolve to same PFAM; keep max count
            pf_map: Dict[str, Dict[str, Any]] = {}
            for r in cnt_pf:
                pid = r.get("pfam_id") or r.get("token")
                lbl = r.get("label") or r.get("token")
                cnt = int(r.get("count") or 0)
                tok = r.get("token")
                if not pid:
                    continue
                prev = pf_map.get(pid)
                if (prev is None) or (cnt > int(prev.get("count", 0))):
                    pf_map[pid] = {"id": pid, "name": lbl, "count": cnt}
                if isinstance(tok, str) and tok:
                    s = pf_provenance.setdefault(pid, set())
                    s.add(tok)
            pfam_facets = sorted(pf_map.values(), key=lambda d: (-int(d.get("count",0)), str(d.get("id",""))))
        if ko_needed and ko_inputs:
            __tko = time.perf_counter()
            cnt_ko = runner.run_template(
                "count_proteins_by_ko_ids.cypher",
                {"ko_ids": ko_inputs, "genome_ids": genome_ids},
            ) or []
            try:
                logging.getLogger(__name__).info(
                    f"AnnotationDiscovery: count_ko ids={len(ko_inputs)} genomes={len(genome_ids)} in {(time.perf_counter()-__tko)*1000:.0f} ms")
            except Exception:
                pass
            # Include KO description as name when available
            kos_facets = []
            for r in cnt_ko:
                kid = r.get("ko_id")
                if not kid:
                    continue
                kos_facets.append({"id": kid, "name": r.get("label"), "count": int(r.get("count") or 0)})
            kos_facets.sort(key=lambda d: (-d["count"], str(d["id"])) )
    else:
        # Rowset path: fetch rows using clamped/overridden token sets and aggregate locally
        pf_rows: List[Dict[str, Any]] = []
        ko_rows: List[Dict[str, Any]] = []
        if pfam_needed and rowset_pfam_ids:
            __tp = time.perf_counter()
            pf_rows = runner.run_template(
                "proteins_by_pfam_ids.cypher",
                {"pfam_ids": rowset_pfam_ids, "genome_ids": genome_ids, "limit": limit},
            ) or []
            try:
                logging.getLogger(__name__).info(
                    f"AnnotationDiscovery: proteins_by_pfam_ids ids={len(rowset_pfam_ids)} genomes={len(genome_ids)} rows={len(pf_rows)} in {(time.perf_counter()-__tp)*1000:.0f} ms")
            except Exception:
                pass
        if ko_needed and rowset_ko_ids:
            __tk = time.perf_counter()
            ko_rows = runner.run_template(
                "proteins_by_ko_ids.cypher",
                {"ko_ids": rowset_ko_ids, "genome_ids": genome_ids, "limit": limit},
            ) or []
            try:
                logging.getLogger(__name__).info(
                    f"AnnotationDiscovery: proteins_by_ko_ids ids={len(rowset_ko_ids)} genomes={len(genome_ids)} rows={len(ko_rows)} in {(time.perf_counter()-__tk)*1000:.0f} ms")
            except Exception:
                pass

        # Merge with provenance; key by (genome_id, protein_id)
        debug_ann = str(os.getenv('DEBUG_ANN_DISCOVERY', '')).lower() not in ('', '0', 'false')
        for r in pf_rows:
            gid = str(r.get("genome_id"))
            pid = str(r.get("protein_id"))
            key = f"{gid}\t{pid}"
            entry = merged.setdefault(key, {"genome_id": gid, "protein_id": pid, "pfams": [], "kos": []})
            pf_label = r.get("pfam_name") or r.get("domain_desc") or r.get("domain_id") or r.get("pfam_id")
            if pf_label and pf_label not in entry["pfams"]:
                entry["pfams"].append(pf_label)
            # Always include accession tokens for downstream filtering (version-tolerant: keep base PFxxxxx)
            pf_acc = r.get("pfam_id")
            if pf_acc:
                base = str(pf_acc).split('.')[0]
                ids = entry.setdefault("pfam_ids", [])
                if base not in ids:
                    ids.append(base)
        for r in ko_rows:
            gid = str(r.get("genome_id"))
            pid = str(r.get("protein_id"))
            key = f"{gid}\t{pid}"
            entry = merged.setdefault(key, {"genome_id": gid, "protein_id": pid, "pfams": [], "kos": []})
            koid = r.get("ko_id")
            if koid and koid not in entry["kos"]:
                entry["kos"].append(koid)

        # Aggregate facets from merged rows
        def _aggregate_counts(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            ko_counts: Dict[str, int] = {}
            pf_counts: Dict[str, int] = {}
            for r in rows:
                if ko_needed:
                    for ko in (r.get("kos") or []):
                        if isinstance(ko, str) and ko:
                            ko_counts[ko] = ko_counts.get(ko, 0) + 1
                if pfam_needed:
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
    if ko_needed:
        sel, sm = _apply_limit(kos_facets, ko_top_k)
        facets["kos"] = sel
        selection_meta["groups"]["ko"] = sm
    if pfam_needed:
        sel, sm = _apply_limit(pfam_facets, pfam_top_k)
        facets["pfams"] = sel
        # Include token provenance for PFAMs when counts path was used
        try:
            if use_counts and pf_provenance and isinstance(sel, list):
                prov = []
                for it in sel:
                    pid = it.get("id")
                    if isinstance(pid, str) and pid in pf_provenance:
                        prov.append({"id": pid, "tokens": sorted(list(pf_provenance.get(pid, set())))})
                if prov:
                    sm["token_provenance"] = prov
        except Exception:
            pass
        selection_meta["groups"]["pfam"] = sm

    # Compose result
    res: Dict[str, Any] = {"facet_summary": facets, "selection_metadata": selection_meta}
    if not use_counts:
        out_rows = list(merged.values())
        out_rows.sort(key=lambda x: (x.get("genome_id", ""), x.get("protein_id", "")))
        if output_profile == 'rowset' or return_full:
            res["discovered_proteins"] = out_rows
            if return_full:
                res["_format"] = "full"
    return res


def _neighborhood_context(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Batch neighborhood extraction for seed proteins using curated KG templates.

    Inputs:
      - discovered_proteins (optional): rows with at least protein_id
    Params:
      - protein_ids: explicit seed list (overrides discovered_proteins)
      - k: optional k-step adjacency (int). If absent, use flanking genes (±5)
      - limit: per-seed cap on neighbor rows (default 200)
      - seeds_limit: max number of seeds to process (default 10)
      - fallback_window_bp: bp window for span fallback when adjacency/flanking yields 0 rows (default 10000)
      - output_profile: summary|rowset (default summary)

    Outputs:
      - neighborhoods: list of per-seed neighborhood payloads [{seed_protein_id, template, rows, debug}]
      - neighborhood_summary: {total_rows, summary_table:[{seed,row_count}]}
      - seeds_used: list of seed protein IDs actually processed
      - neighborhood_macro_result: compact macro_result for reporter context
    """
    # Parameter parsing
    try:
        k = int(params.get("k")) if params.get("k") is not None else None
    except Exception:
        k = None
    try:
        limit = int(params.get("limit", 200))
    except Exception:
        limit = 200
    try:
        seeds_limit = int(params.get("seeds_limit", 10))
    except Exception:
        seeds_limit = 10
    try:
        fallback_window_bp = int(params.get("fallback_window_bp", 10000))
    except Exception:
        fallback_window_bp = 10000
    output_profile = str(params.get("output_profile") or "summary").strip().lower()
    include_degree_zero = bool(params.get("include_degree_zero_seeds", False))

    # Optional explicit seeding via PFAM/KO ids (agent-visible)
    seed_pfam_ids = params.get("seed_pfam_ids") if isinstance(params.get("seed_pfam_ids"), list) else None
    seed_ko_ids = params.get("seed_ko_ids") if isinstance(params.get("seed_ko_ids"), list) else None
    seed_scope_genome_ids = params.get("seed_scope_genome_ids") if isinstance(params.get("seed_scope_genome_ids"), list) else []
    try:
        seed_fetch_limit = int(params.get("seed_fetch_limit", seeds_limit))
    except Exception:
        seed_fetch_limit = seeds_limit

    # Seed selection: explicit protein_ids first, else from discovered_proteins
    seeds_in = params.get("protein_ids") or []
    if not (isinstance(seeds_in, list) and seeds_in):
        dp = inputs.get("discovered_proteins") or []
        # Allow bound dict envelopes: {"discovered_proteins": [...]} (from a bound rowset)
        if isinstance(dp, dict):
            inner = dp.get("discovered_proteins")
            if isinstance(inner, list):
                dp = inner
        try:
            # Optional filter by PFAM accession(s) if provided
            filt = params.get("seed_filter_pfam_ids") if isinstance(params.get("seed_filter_pfam_ids"), list) else None
            filt_norm = None
            if filt:
                filt_norm = set(str(x).split('.')[0].upper() for x in filt if isinstance(x, str) and x)
            seeds_in = []
            for r in dp:
                if not (isinstance(r, dict) and r.get("protein_id")):
                    continue
                if filt_norm:
                    pfids = r.get("pfam_ids") or []
                    pfids_norm = set(str(x).split('.')[0].upper() for x in pfids if isinstance(x, str) and x)
                    if not pfids_norm.intersection(filt_norm):
                        continue
                seeds_in.append(str(r.get("protein_id")))
        except Exception:
            seeds_in = []

    # Self-seed STRICTLY via explicit PFAM/KO params (no implicit fallbacks)
    if not seeds_in:
        if seed_pfam_ids or seed_ko_ids:
            runner = FileCypherRunner(ctx.neo4j_driver)
            try:
                if seed_pfam_ids:
                    rows = runner.run_template(
                        "proteins_by_pfam_ids.cypher",
                        {"pfam_ids": [str(x) for x in seed_pfam_ids], "genome_ids": seed_scope_genome_ids, "limit": seed_fetch_limit},
                    ) or []
                    seeds_in.extend([str(r.get("protein_id")) for r in rows if r.get("protein_id")])
                if seed_ko_ids:
                    rows = runner.run_template(
                        "proteins_by_ko_ids.cypher",
                        {"ko_ids": [str(x) for x in seed_ko_ids], "genome_ids": seed_scope_genome_ids, "limit": seed_fetch_limit},
                    ) or []
                    seeds_in.extend([str(r.get("protein_id")) for r in rows if r.get("protein_id")])
            except Exception:
                # Do not hide errors with implicit alternatives; keep strict behavior
                pass
    # Deduplicate, drop empties
    seen = set()
    seeds: List[str] = []
    for pid in (seeds_in or []):
        try:
            s = str(pid).strip()
        except Exception:
            continue
        if not s or s.lower() in ("example", "placeholder", "sample"):
            continue
        if s in seen:
            continue
        seen.add(s)
        seeds.append(s)
        if len(seeds) >= seeds_limit:
            break

    # Enforce explicit seeds: if still empty, fail fast (no implicit fallbacks)
    if not (isinstance(seeds_in, list) and seeds_in):
        raise ValueError(
            "NeighborhoodContext requires explicit seeds: provide discovered_proteins input or params (protein_ids or seed_pfam_ids/seed_ko_ids)."
        )

    # Degree-aware seed filtering (exclude nextDegree=0 by default)
    def _filter_seeds_by_degree(pids: List[str], include_zero: bool) -> Dict[str, int]:
        if not pids:
            return {}
        cypher = (
            "UNWIND $pids AS pid "
            "MATCH (p:Protein {id: pid})-[:ENCODEDBY]->(g:Gene) "
            "OPTIONAL MATCH (g)-[:NEXT]-(:Gene) "
            "WITH pid, g, count(*) AS c "
            "WITH pid, coalesce(g.nextDegree, c) AS deg "
            "RETURN pid AS protein_id, toInteger(deg) AS deg"
        )
        deg_map: Dict[str, int] = {}
        with ctx.neo4j_driver.session() as s:
            rows = s.run(cypher, pids=pids)
            for r in rows:
                pid = r.get("protein_id")
                deg = int(r.get("deg") or 0)
                if include_zero or deg > 0:
                    deg_map[pid] = deg
        return deg_map

    seed_degrees = _filter_seeds_by_degree(seeds, include_degree_zero)
    if seed_degrees and not include_degree_zero:
        before = len(seeds)
        seeds = [pid for pid in seeds if pid in seed_degrees]
        dropped = before - len(seeds)
        try:
            logging.getLogger(__name__).info(
                f"NeighborhoodContext: filtered {dropped} degree-zero seeds; using {len(seeds)} seeds")
        except Exception:
            pass

    neighborhoods: List[Dict[str, Any]] = []
    total_rows = 0

    # Helpers to execute KG templates (from src/llm/kg/cypher_templates)
    def _run_tpl(filename: str, p: Dict[str, Any]) -> List[Dict[str, Any]]:
        cypher_path = kg_tpl_registry.TEMPLATES_DIR / filename
        cypher = cypher_path.read_text(encoding="utf-8")
        with ctx.neo4j_driver.session() as s:
            res = s.run(cypher, p)
            return [dict(r) for r in res]

    def _run_k_step(protein_id: str, k_steps: int, limit_x: int) -> List[Dict[str, Any]]:
        # Return neighbor genes/proteins with annotation summaries (no APOC dependency)
        cypher = (
            "MATCH (p:Protein {id:$protein_id})-[:ENCODEDBY]->(g:Gene) "
            f"CALL (g) {{ MATCH pth=(g)-[:NEXT*..{k_steps}]-(ng:Gene) RETURN DISTINCT ng }} "
            "OPTIONAL MATCH (np:Protein)-[:ENCODEDBY]->(ng) "
            "OPTIONAL MATCH (np)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain) "
            "OPTIONAL MATCH (np)-[:HASFUNCTION]->(ko:KEGGOrtholog) "
            "OPTIONAL MATCH (ng)-[f:FLANKS_CRISPR]->(ca:CrisprArray) "
            "WITH ng, np, f, ca, "
            "collect(DISTINCT CASE "
            "  WHEN coalesce(d.pfamAccession, d.id) IS NOT NULL AND coalesce(d.name, d.description) IS NOT NULL AND coalesce(d.name, d.description) <> '' "
            "    THEN coalesce(d.pfamAccession, d.id) + ': ' + coalesce(d.name, d.description) "
            "  WHEN coalesce(d.pfamAccession, d.id) IS NOT NULL "
            "    THEN coalesce(d.pfamAccession, d.id) "
            "  ELSE coalesce(d.name, d.description) "
            "END) AS pfams, "
            "collect(DISTINCT ko.description) AS kos "
            "RETURN ng.id AS gene_id, ng.contig AS contig, toInteger(ng.startCoordinate) AS start, "
            "toInteger(ng.endCoordinate) AS end, ng.strand AS strand, np.id AS protein_id, pfams, kos, "+
            "ca.id AS crispr_id, toInteger(f.distanceBp) AS crispr_distance_bp "
            "ORDER BY start LIMIT $limit"
        )
        with ctx.neo4j_driver.session() as s:
            res = s.run(cypher, {"protein_id": protein_id, "limit": int(limit_x)})
            rows = [dict(r) for r in res]
        # Normalize description strings for reporter convenience
        for r in rows:
            try:
                pfams = [x for x in (r.get("pfams") or []) if isinstance(x, str) and x]
                kos = [x for x in (r.get("kos") or []) if isinstance(x, str) and x]
                r["pfam_desc"] = "; ".join(sorted(set(pfams))) if pfams else None
                r["ko_desc"] = "; ".join(sorted(set(kos))) if kos else None
            except Exception:
                r["pfam_desc"] = r.get("pfam_desc")
                r["ko_desc"] = r.get("ko_desc")
        return rows

    def _run_flanking_annotated(protein_id: str, flank_n: int, limit_x: int) -> List[Dict[str, Any]]:
        # Fetch ±flank_n genes (by contig order) AND include any CRISPR arrays flanking those genes.
        # Arrays do not count toward the gene limit; they are attached as extras per neighbor.
        cypher = (
            "MATCH (p:Protein {id:$protein_id})-[:ENCODEDBY]->(seed:Gene) "
            "MATCH (g:Gene {contig: seed.contig}) WITH seed, g ORDER BY toInteger(g.startCoordinate) "
            "WITH seed, collect(g) AS gs "
            "WITH seed, gs, [i IN range(0, size(gs)-1) WHERE gs[i].id = seed.id][0] AS idx "
            "WITH seed, gs, idx, range(-$flank_n, $flank_n) AS offsets "
            "UNWIND offsets AS off WITH seed, gs, idx, off WHERE off <> 0 "
            "WITH gs[(idx + off)] AS ng WHERE (idx + off) >= 0 AND (idx + off) < size(gs) "
            "OPTIONAL MATCH (np:Protein)-[:ENCODEDBY]->(ng) "
            "OPTIONAL MATCH (np)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain) "
            "OPTIONAL MATCH (np)-[:HASFUNCTION]->(ko:KEGGOrtholog) "
            "OPTIONAL MATCH (ng)-[f:FLANKS_CRISPR]->(ca:CrisprArray) "
            "WITH ng, np, f, ca, "
            "collect(DISTINCT CASE "
            "  WHEN coalesce(d.pfamAccession, d.id) IS NOT NULL AND coalesce(d.name, d.description) IS NOT NULL AND coalesce(d.name, d.description) <> '' "
            "    THEN coalesce(d.pfamAccession, d.id) + ': ' + coalesce(d.name, d.description) "
            "  WHEN coalesce(d.pfamAccession, d.id) IS NOT NULL "
            "    THEN coalesce(d.pfamAccession, d.id) "
            "  ELSE coalesce(d.name, d.description) "
            "END) AS pfams, "
            "collect(DISTINCT ko.description) AS kos "
            "RETURN ng.id AS gene_id, ng.contig AS contig, toInteger(ng.startCoordinate) AS start, "
            "toInteger(ng.endCoordinate) AS end, ng.strand AS strand, np.id AS protein_id, "+
            "pfams, kos, ca.id AS crispr_id, toInteger(f.distanceBp) AS crispr_distance_bp "
            "ORDER BY start LIMIT $limit"
        )
        with ctx.neo4j_driver.session() as s:
            res = s.run(cypher, {"protein_id": protein_id, "flank_n": int(flank_n), "limit": int(limit_x)})
            rows = [dict(r) for r in res]
        for r in rows:
            try:
                pfams = [x for x in (r.get("pfams") or []) if isinstance(x, str) and x]
                kos = [x for x in (r.get("kos") or []) if isinstance(x, str) and x]
                r["pfam_desc"] = "; ".join(sorted(set(pfams))) if pfams else None
                r["ko_desc"] = "; ".join(sorted(set(kos))) if kos else None
            except Exception:
                r["pfam_desc"] = r.get("pfam_desc")
                r["ko_desc"] = r.get("ko_desc")
        return rows

    def _run_span_window(contig: str, start: int, end: int, limit_x: int) -> List[Dict[str, Any]]:
        cypher = (
            "MATCH (g:Gene {contig:$contig}) "
            "WHERE toInteger(g.startCoordinate) >= $start AND toInteger(g.endCoordinate) <= $end "
            "OPTIONAL MATCH (np:Protein)-[:ENCODEDBY]->(g) "
            "RETURN g.id AS gene_id, g.contig AS contig, toInteger(g.startCoordinate) AS start, "
            "toInteger(g.endCoordinate) AS end, g.strand AS strand, np.id AS protein_id "
            "ORDER BY start LIMIT $limit"
        )
        with ctx.neo4j_driver.session() as s:
            res = s.run(cypher, {"contig": contig, "start": int(start), "end": int(end), "limit": int(limit_x)})
            return [dict(r) for r in res]

    for pid in seeds[:seeds_limit]:
        seed_debug: Dict[str, Any] = {"seed_protein_id": pid}
        # Fetch seed context (contig/coordinates)
        try:
            ctx_rows = _run_tpl("protein_gene_context.cypher", {"protein_id": pid})
        except Exception:
            ctx_rows = []
        if ctx_rows:
            s0 = ctx_rows[0]
            seed_debug.update({
                "seed_gene_id": s0.get("gene_id"),
                "seed_contig": s0.get("contig"),
                "seed_start": s0.get("start"),
                "seed_end": s0.get("end"),
                "seed_strand": s0.get("strand"),
            })
            try:
                # Attach nextDegree for visibility (use filtered value if available)
                if pid in seed_degrees:
                    seed_debug["seed_next_degree"] = seed_degrees[pid]
                else:
                    nxt = _run_tpl("gene_next_degree.cypher", {"gene_id": s0.get("gene_id")})
                    if nxt:
                        seed_debug["seed_next_degree"] = nxt[0].get("next_degree")
            except Exception:
                pass
            # Optional: quick NEXT degree for the seed gene
            try:
                nxt = _run_tpl("gene_next_degree.cypher", {"gene_id": s0.get("gene_id")})
                if nxt:
                    seed_debug["seed_next_degree"] = nxt[0].get("next_degree")
            except Exception:
                pass

        # Primary neighborhood: k-step or flanking
        rows: List[Dict[str, Any]] = []
        template_name = "protein_flanking_genes_5"
        try:
            if isinstance(k, int) and k > 0:
                template_name = "protein_neighbors_k"
                rows = _run_k_step(pid, k, limit)
                # If adjacency returns 0 rows, also try flanking (±5) to surface context on short contigs
                if not rows:
                    try:
                        flk_rows = _run_flanking_annotated(pid, 5, limit)
                        if flk_rows:
                            rows = flk_rows
                            template_name = "protein_flanking_genes"
                    except Exception:
                        pass
            else:
                template_name = "protein_flanking_genes"
                rows = _run_flanking_annotated(pid, 5, limit)
        except Exception:
            rows = []

        # Fallback to span window around seed gene when nothing found
        if not rows and seed_debug.get("seed_contig") is not None:
            try:
                cx = int(seed_debug.get("seed_start") or 0)
                cy = int(seed_debug.get("seed_end") or 0)
                w = int(fallback_window_bp)
                start_w = max(0, min(cx, cy) - w)
                end_w = max(cx, cy) + w
                rows = _run_span_window(str(seed_debug.get("seed_contig")), start_w, end_w, limit)
                template_name = "neighbors_by_window"
                seed_debug["fallback"] = {"start": start_w, "end": end_w, "window_bp": w, "row_count": len(rows)}
            except Exception:
                pass

        neighborhoods.append({
            "seed_protein_id": pid,
            "template": template_name,
            "rows": rows,
            "debug": seed_debug,
        })
        total_rows += len(rows)

    summary_table = [{"seed": n.get("seed_protein_id"), "row_count": len(n.get("rows") or [])} for n in neighborhoods]
    seeds_used = [n.get("seed_protein_id") for n in neighborhoods]

    # Compact macro_result for reporter context
    macro_rows: List[Dict[str, Any]] = []
    for n in neighborhoods:
        sd = n.get("debug") or {}
        rs = n.get("rows") or []
        examples = []
        examples_ann = []
        try:
            for r in rs:
                pid2 = r.get("protein_id")
                if isinstance(pid2, str) and pid2:
                    examples.append(pid2)
                    examples_ann.append({
                        "protein_id": pid2,
                        "pfam_desc": r.get("pfam_desc"),
                        "ko_desc": r.get("ko_desc"),
                    })
                    if len(examples) >= 3:
                        break
        except Exception:
            examples = []
            examples_ann = []
        macro_rows.append({
            "seed": n.get("seed_protein_id"),
            "contig": sd.get("seed_contig"),
            "start": sd.get("seed_start"),
            "end": sd.get("seed_end"),
            "row_count": len(rs),
            "example_neighbors": examples,
            "example_neighbors_ann": examples_ann,
        })

    macro_result = {"type": "macro_result", "name": "neighborhoods", "rows": macro_rows}

    out: Dict[str, Any] = {
        "neighborhoods": neighborhoods if output_profile == "rowset" else [],
        "neighborhood_summary": {"total_rows": total_rows, "summary_table": summary_table},
        "seeds_used": seeds_used,
        "neighborhood_macro_result": macro_result,
    }
    return out

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
    inputs=["pfam_ids", "ko_ids"],
    outputs=["facet_summary", "selection_metadata", "discovered_proteins"],
    params={
        "keyword": "str (free text)",
        "limit": "int (row budget for legacy rowset mode; default 1000)",
        "output_profile": "facet_summary|rowset (default facet_summary)",
        "return_mode": "top_k|all (default top_k)",
        "ko_top_k": "int (default 30)",
        "pfam_top_k": "int (default 20)",
        "fields": "List[str] (requested fields for summaries; optional; valid: id,name,count)",
        "group_by": "ko|pfam|both (default both)",
        "include_examples": "none|counts|ids (ids only for contig/locus anchors)",
        "return_full_rows": "bool (legacy rowset include)",
        "genome_ids": "List[str] | null",
        "pfam_tokens_top_n": "int | null (cap catalog PFAM tokens; applies to counts and rowset)",
        "ko_tokens_top_n": "int | null (cap catalog KO tokens; applies to counts and rowset)",
        "pfam_candidate_cap": "int | null (cap candidate PFAM domains per token in counts; default 200)",
        "pfam_ids": "List[str] | null (restrict rowset PFAM tokens to these accessions)",
        "ko_ids": "List[str] | null (restrict rowset KO tokens to these ids)",
    },
    run=_annotation_discovery,
    description="Facet-first annotation discovery: keyword→IDs→exact retrieval→KO/PFAM summaries (counts, top_k or all). Rowset optional.",
))

# Neighborhoods: batch per-seed neighborhoods from discovered proteins or explicit seeds
register_operator(OperatorSpec(
    name="NeighborhoodContext",
    inputs=["discovered_proteins"],
    outputs=["neighborhoods", "neighborhood_summary", "neighborhood_macro_result", "seeds_used"],
    params={
        "protein_ids": "List[str] | null (explicit seeds)",
        "k": "int | null (k-step adjacency; default None → flanking)",
        "limit": "int (per-seed cap; default 200)",
        "seeds_limit": "int (max seeds; default 10)",
        "fallback_window_bp": "int (bp window for span fallback; default 10000)",
        "seed_filter_pfam_ids": "List[str] | null (only seed proteins that carry any of these PFAM accessions; uses discovered_proteins.pfam_ids)",
        "seed_pfam_ids": "List[str] | null (self-seed by fetching proteins with these PFAMs when discovered_proteins/protein_ids are absent)",
        "seed_ko_ids": "List[str] | null (self-seed by fetching proteins with these KO ids when discovered_proteins/protein_ids are absent)",
        "seed_scope_genome_ids": "List[str] | null (restrict self-seeding to these genomes; default global)",
        "seed_fetch_limit": "int (max proteins to fetch during self-seeding; default = seeds_limit)",
        "include_degree_zero_seeds": "bool (default false; exclude seeds with nextDegree=0 by default)",
        "output_profile": "summary|rowset (default summary)",
    },
    run=_neighborhood_context,
    description="Neighborhoods around seed proteins: k-step or ±5 flanking; span-window fallback. Returns compact macro_result for synthesis.",
))

# --- Materializers for composite outputs (lightweight adapters) ---

def _materialize_feature_discovery(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    proteins = inputs.get("discovered_proteins") or []
    pf_facet_in = inputs.get("pf_facet")
    ko_facet_in = inputs.get("ko_facet")
    # Extract facet_summary if wrapped via bind
    def _facet(x: Any):
        if isinstance(x, dict) and isinstance(x.get("facet_summary"), dict):
            return x.get("facet_summary")
        return x
    pf_facet = _facet(pf_facet_in)
    ko_facet = _facet(ko_facet_in)
    feature_set: FeatureSet = {"source": "mixed", "ids": [], "terms": []}
    protein_set: ProteinSet = {"proteins": proteins if isinstance(proteins, list) else []}
    # Minimal validation
    try:
        assert_proteinset(protein_set)
        assert_featureset(feature_set)
    except Exception:
        pass
    return {"FeatureSet": feature_set, "ProteinSet": protein_set, "FacetSummary": {"pfam": pf_facet, "ko": ko_facet}}


def _count_by_ids_per_genome(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Count proteins per genome for provided PFAM and/or KO ids.

    Params:
      - pfam_ids?: List[str]
      - ko_ids?: List[str]
      - genome_ids?: List[str] (defaults to dataset_context.genome_ids_sample)
    Outputs:
      - pfam_counts: [{genome_id, pfam_id, count}]
      - ko_counts:   [{genome_id, ko_id, count}]
    """
    runner = FileCypherRunner(ctx.neo4j_driver)
    pfam_in = inputs.get("pfam_ids") or params.get("pfam_ids") or []
    ko_in = inputs.get("ko_ids") or params.get("ko_ids") or []
    # Unwrap when passed as a dict payload from ExtractIdsFromCatalogHits
    if isinstance(pfam_in, dict) and "pfam_ids" in pfam_in:
        pfam_ids = pfam_in.get("pfam_ids") or []
    else:
        pfam_ids = pfam_in or []
    if isinstance(ko_in, dict) and "ko_ids" in ko_in:
        ko_ids = ko_in.get("ko_ids") or []
    else:
        ko_ids = ko_in or []
    genome_ids = params.get("genome_ids") or []
    if not genome_ids and ctx.dataset_context:
        genome_ids = ctx.dataset_context.get('genome_ids_sample') or []

    pfam_counts: List[Dict[str, Any]] = []
    ko_counts: List[Dict[str, Any]] = []
    warnings: List[str] = []
    if (not pfam_ids) and (not ko_ids):
        warnings.append("empty_id_lists")
    if pfam_ids:
        try:
            pfam_counts = runner.run_template(
                "count_proteins_by_pfam_ids_per_genome.cypher",
                {"pfam_ids": pfam_ids, "genome_ids": genome_ids},
            ) or []
        except Exception:
            pfam_counts = []
    if ko_ids:
        try:
            ko_counts = runner.run_template(
                "count_proteins_by_ko_ids_per_genome.cypher",
                {"ko_ids": ko_ids, "genome_ids": genome_ids},
            ) or []
        except Exception:
            ko_counts = []

    macro = {"type": "macro_result", "name": "CountByIdsPerGenome", "rows": {"pfam_counts": pfam_counts, "ko_counts": ko_counts}}
    return {"pfam_counts": pfam_counts, "ko_counts": ko_counts, "warnings": warnings, "macro_result": macro}


def _materialize_feature_profile(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Materialize a compact per-genome feature profile from count rows.

    Inputs (currently supported):
      - pfam_counts: [{genome_id,pfam_id,count}]
      - ko_counts:   [{genome_id,ko_id,count}]
    The implementation is generic over present feature types so that additional
    types can be added later without changing this function.
    """
    # Gather available count tables by feature type
    counts_by_type: Dict[str, List[Dict[str, Any]]] = {
        'pfam': inputs.get('pfam_counts') or [],
        'ko': inputs.get('ko_counts') or [],
    }
    present_types = [t for t, rows in counts_by_type.items() if rows]

    # Build per-genome structure generically
    per_genome: Dict[str, Dict[str, Any]] = {}
    for ftype, rows in counts_by_type.items():
        id_key = f'{ftype}_id'
        for r in rows:
            gid = str(r.get('genome_id'))
            fid = str(r.get(id_key))
            cnt = int(r.get('count') or 0)
            if not gid or not fid:
                continue
            e = per_genome.setdefault(gid, {'genome_id': gid})
            e.setdefault(ftype, []).append({'id': fid, 'count': cnt})
    # Ensure all present types are present in each row (empty lists ok)
    for row in per_genome.values():
        for t in present_types:
            row.setdefault(t, [])
    rows = sorted(per_genome.values(), key=lambda d: d.get('genome_id', ''))

    # Global summaries per feature type
    from collections import Counter
    top_by_type: Dict[str, List[Dict[str, Any]]] = {}
    def _top(cnt: Counter, n: int = 20):
        out = []
        for k, v in cnt.most_common(n):
            if isinstance(k, str) and k:
                out.append({'id': k, 'total': int(v)})
        return out
    for ftype, rows in counts_by_type.items():
        c = Counter()
        id_key = f'{ftype}_id'
        for r in rows:
            fid = r.get(id_key)
            if isinstance(fid, str) and fid:
                c[fid] += int(r.get('count') or 0)
        top_by_type[ftype] = _top(c)

    # Label enrichment (kept per-type; safe if catalogs missing)
    labels: Dict[str, Dict[str, str]] = {t: {} for t in present_types}
    try:
        for pfid, short, desc in _load_pfam_catalog(getattr(ctx, 'project_root', None)):
            base = (pfid or '').split('.')[0].upper()
            if base:
                labels.setdefault('pfam', {})[base] = short or desc or base
    except Exception:
        pass
    try:
        for kid, label in _load_ko_catalog(getattr(ctx, 'project_root', None)):
            if kid:
                labels.setdefault('ko', {})[kid] = label or kid
    except Exception:
        pass

    def _annotate(items: List[Dict[str, Any]], ftype: str) -> List[Dict[str, Any]]:
        out = []
        lab = labels.get(ftype) or {}
        for it in items:
            i = dict(it)
            fid = i.get('id') or i.get(f'{ftype}_id')
            if isinstance(fid, str):
                name = lab.get(fid)
                if name:
                    i['name'] = name
            out.append(i)
        return out

    summary = {
        'feature_types': present_types,
        'top': {t: _annotate(top_by_type.get(t, []), t) for t in present_types},
        'labels': {t: labels.get(t) or {} for t in present_types},
    }
    # Backward-compat aliases for existing consumers
    if 'pfam' in present_types:
        summary['top_pfam'] = summary['top']['pfam']
    if 'ko' in present_types:
        summary['top_ko'] = summary['top']['ko']

    # Build compact matrices per feature type (top features only)
    def _matrix(count_rows: List[Dict[str, Any]], id_key: str, top_ids: List[str]) -> List[Dict[str, Any]]:
        by_g: Dict[str, Dict[str, int]] = {}
        for r in count_rows:
            gid = str(r.get('genome_id'))
            fid = str(r.get(id_key))
            c = int(r.get('count') or 0)
            if gid in (None, '', 'None') or fid not in top_ids:
                continue
            e = by_g.setdefault(gid, {k: 0 for k in top_ids})
            e[fid] += c
        out = []
        for gid in sorted(by_g.keys()):
            row = {'genome_id': gid}
            row.update({fid: by_g[gid].get(fid, 0) for fid in top_ids})
            out.append(row)
        return out

    feature_order: Dict[str, List[str]] = {t: [x['id'] for x in summary['top'].get(t, [])] for t in present_types}
    per_genome_matrix: Dict[str, Any] = {'feature_order': feature_order, 'feature_types': present_types}
    for t in present_types:
        top_ids = feature_order.get(t) or []
        per_genome_matrix[t] = _matrix(counts_by_type.get(t, []), f'{t}_id', top_ids)

    return {
        'PerGenomeFeatureCounts': rows,
        'FeatureProfileSummary': summary,
        'PerGenomeTopMatrix': per_genome_matrix,
    }


def _materialize_gene_context(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    neighborhoods = inputs.get("neighborhoods") or []
    n_summary = inputs.get("neighborhood_summary")
    return {"NeighborhoodSet": {"neighborhoods": neighborhoods}, "NeighborhoodSummary": n_summary}


def _materialize_pathway_profile(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    present = inputs.get("present") or {}
    completeness = inputs.get("pathway_completeness") or []
    # Optional compact summary (per-pathway overall completeness); keep simple
    try:
        from collections import defaultdict
        agg: Dict[str, Dict[str, Any]] = {}
        for r in completeness:
            pw = str(r.get("pathway_id"))
            comp = float(r.get("completeness") or 0.0)
            e = agg.setdefault(pw, {"pathway_id": pw, "max_completeness": 0.0, "examples": 0})
            if comp > e["max_completeness"]:
                e["max_completeness"] = comp
            e["examples"] += 1
        c_summary = sorted(agg.values(), key=lambda x: (-x["max_completeness"], x["pathway_id"]))
    except Exception:
        c_summary = None
    return {"PresentKOsByGenome": present, "CompletenessMatrix": completeness, "CompletenessSummary": c_summary}


def _materialize_module_profile(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    module = (params.get("module") or "cazy").strip().lower()
    if module == "cazy":
        return {"ModuleRows": inputs.get("cazymes"), "GlobalCounts": inputs.get("cazyme_family_counts")}
    elif module == "bgc":
        return {"ModuleRows": inputs.get("bgcs"), "GlobalCounts": None}
    return {"ModuleRows": None, "GlobalCounts": None}


def _materialize_evidence_and_next(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    return {"EvidenceMetrics": inputs.get("evidence_metrics"), "FollowupPlan": inputs.get("followup_request")}


def _plan_similarity_search(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Derive LanceDB similarity seeds and plan metadata from discovery outputs."""

    def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            return max(minimum, min(int(value), maximum))
        except Exception:
            return default

    def _normalize_pid(candidate: str) -> str:
        if not isinstance(candidate, str):
            return ""
        cid = candidate.strip()
        if not cid:
            return ""
        base = cid.split(':', 1)[-1]
        return base

    seed_limit = _coerce_int(params.get("seed_limit"), default=1, minimum=1, maximum=10)
    nn_value = _coerce_int(params.get("nn", params.get("top_k", 10)), default=10, minimum=1, maximum=50)

    raw_seeds: List[str] = []
    normalized_seeds: List[str] = []
    seen: set[str] = set()

    def _add_seed(candidate: Any) -> None:
        if not isinstance(candidate, str):
            return
        cid = candidate.strip()
        if not cid:
            return
        if cid not in seen:
            seen.add(cid)
            raw_seeds.append(cid)
            norm = _normalize_pid(cid)
            if norm:
                normalized_seeds.append(norm)
            else:
                normalized_seeds.append(cid)

    explicit_ids = params.get("protein_ids") if isinstance(params.get("protein_ids"), list) else None
    if explicit_ids:
        for pid in explicit_ids:
            _add_seed(pid)

    if len(raw_seeds) < seed_limit:
        discovered = inputs.get("discovered_proteins")
        rows: List[Dict[str, Any]] = []
        if isinstance(discovered, dict):
            if isinstance(discovered.get("discovered_proteins"), list):
                rows = discovered.get("discovered_proteins")  # type: ignore[assignment]
            elif isinstance(discovered.get("proteins"), list):
                rows = discovered.get("proteins")  # type: ignore[assignment]
        elif isinstance(discovered, list):
            rows = discovered  # type: ignore[assignment]
        for row in rows:
            if len(raw_seeds) >= seed_limit:
                break
            if isinstance(row, dict):
                _add_seed(row.get("protein_id"))

    raw_seeds = raw_seeds[:seed_limit]
    normalized_seeds = normalized_seeds[:seed_limit]

    filters = params.get("filters") if isinstance(params.get("filters"), dict) else {}
    annotate = bool(params.get("annotate", True))

    plan = None
    if normalized_seeds:
        plan = {
            "type": "similarity_plan",
            "seeds": normalized_seeds,
            "raw_seeds": raw_seeds,
            "nn": nn_value,
            "filters": filters,
            "annotate": annotate,
        }

    macro = None
    if raw_seeds:
        macro = {
            "type": "macro_result",
            "name": "similarity_seeds",
            "rows": [{"seed_protein_id": pid} for pid in raw_seeds],
            "row_count": len(raw_seeds),
        }

    result: Dict[str, Any] = {
        "SimilarityPlan": plan,
        "SimilaritySeedSet": raw_seeds,
    }
    if macro:
        result["SimilaritySeedMacro"] = macro
    return result


register_operator(OperatorSpec(
    name="MaterializeFeatureDiscovery",
    inputs=["discovered_proteins", "pf_facet", "ko_facet"],
    outputs=["FeatureSet", "ProteinSet", "FacetSummary"],
    params={"output_profile": "facet_summary|rowset|ids_only"},
    run=_materialize_feature_discovery,
    description="Package discovery results into typed records (FeatureSet, ProteinSet, optional facets)",
))

register_operator(OperatorSpec(
    name="CountByIdsPerGenome",
    inputs=["pfam_ids", "ko_ids"],
    outputs=["pfam_counts", "ko_counts", "warnings", "macro_result"],
    params={"pfam_ids": "List[str] | null", "ko_ids": "List[str] | null", "genome_ids": "List[str] | null"},
    run=_count_by_ids_per_genome,
    description="Count proteins per genome matching provided PFAM/KO IDs. Emits non-fatal warnings when ID lists are empty.",
))

register_operator(OperatorSpec(
    name="MaterializeFeatureProfile",
    inputs=["pfam_counts", "ko_counts"],
    outputs=["PerGenomeFeatureCounts", "FeatureProfileSummary", "PerGenomeTopMatrix"],
    params={},
    run=_materialize_feature_profile,
    description="Package per-genome PFAM/KO counts, labeled summaries, and a compact top-feature matrix",
))

register_operator(OperatorSpec(
    name="MaterializeGeneContext",
    inputs=["neighborhoods", "neighborhood_summary"],
    outputs=["NeighborhoodSet", "NeighborhoodSummary"],
    params={"output_profile": "rowset|macro_summary"},
    run=_materialize_gene_context,
    description="Package neighborhoods into typed records",
))

register_operator(OperatorSpec(
    name="MaterializePathwayProfile",
    inputs=["present", "pathway_completeness"],
    outputs=["PresentKOsByGenome", "CompletenessMatrix", "CompletenessSummary"],
    params={},
    run=_materialize_pathway_profile,
    description="Package KO presence and pathway completeness into typed records",
))

register_operator(OperatorSpec(
    name="PlanSimilaritySearch",
    inputs=["discovered_proteins"],
    outputs=["SimilarityPlan", "SimilaritySeedSet", "SimilaritySeedMacro"],
    params={
        "protein_ids": "List[str] | null",
        "seed_limit": "int (default 1)",
        "nn": "int (default 10)",
        "filters": "dict | null",
        "annotate": "bool (default true)",
    },
    run=_plan_similarity_search,
    description="Prepare LanceDB similarity plan metadata from seeds",
))

register_operator(OperatorSpec(
    name="MapKOsToPathways",
    inputs=["ko_ids", "totals"],
    outputs=["pathways"],
    params={"top_n": "int (default 25)"},
    run=_map_kos_to_pathways,
    description="Map KO ids to KEGG pathway ids using totals mapping; returns a capped pathway list.",
))

register_operator(OperatorSpec(
    name="MaterializeModuleProfile",
    inputs=["cazymes", "cazyme_family_counts", "bgcs"],
    outputs=["ModuleRows", "GlobalCounts"],
    params={"module": "cazy|bgc", "output_profile": "per_genome|global_counts|rowset"},
    run=_materialize_module_profile,
    description="Package CAZy or BGC module outputs into typed records",
))

register_operator(OperatorSpec(
    name="MaterializeEvidenceAndNext",
    inputs=["evidence_metrics", "followup_request"],
    outputs=["EvidenceMetrics", "FollowupPlan"],
    params={},
    run=_materialize_evidence_and_next,
    description="Package evidence metrics and follow-up plan into typed records",
))

# --- FunctionalProfile materializer (aggregates pathways + modules) ---

def _materialize_functional_profile(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate optional sections from pathway and module profiling.

    Inputs (optional):
      - present: { genome_id -> [KO ids] }
      - pathway_completeness: list of rows
      - cazymes: list of rows
      - cazyme_family_counts: list of rows
      - bgcs: list of rows
    Params:
      - include: List[str] (advisory)
    Outputs:
      - PresentKOsByGenome, CompletenessMatrix, CompletenessSummary
      - CAZyRowsByGenome, CazymeFamilyCounts, BGCsByGenome
      - ProfileKinds
    """
    present = inputs.get("present") or inputs.get("PresentKOsByGenome")
    completeness = inputs.get("pathway_completeness") or inputs.get("CompletenessMatrix") or []
    cazymes = inputs.get("cazymes")
    cazy_counts = inputs.get("cazyme_family_counts")
    bgcs = inputs.get("bgcs")

    # Reuse PathwayProfile summarization for completeness
    try:
        agg: Dict[str, Dict[str, Any]] = {}
        for r in (completeness or []):
            pw = str(r.get("pathway_id"))
            comp = float(r.get("completeness") or 0.0)
            e = agg.setdefault(pw, {"pathway_id": pw, "max_completeness": 0.0, "examples": 0})
            if comp > e["max_completeness"]:
                e["max_completeness"] = comp
            e["examples"] += 1
        c_summary = sorted(agg.values(), key=lambda x: (-x["max_completeness"], x["pathway_id"]))
    except Exception:
        c_summary = None

    kinds: List[str] = []
    if isinstance(completeness, list) and completeness:
        kinds.append("pathways")
    if isinstance(cazymes, list) and cazymes:
        kinds.append("cazy")
    if isinstance(cazy_counts, list) and cazy_counts:
        if "cazy" not in kinds:
            kinds.append("cazy")
    if isinstance(bgcs, list) and bgcs:
        kinds.append("bgc")

    return {
        "PresentKOsByGenome": present or {},
        "CompletenessMatrix": completeness or [],
        "CompletenessSummary": c_summary,
        "CAZyRowsByGenome": cazymes or [],
        "CazymeFamilyCounts": cazy_counts or [],
        "BGCsByGenome": bgcs or [],
        "ProfileKinds": kinds,
    }

register_operator(OperatorSpec(
    name="MaterializeFunctionalProfile",
    inputs=["present", "pathway_completeness", "cazymes", "cazyme_family_counts", "bgcs"],
    outputs=["PresentKOsByGenome", "CompletenessMatrix", "CompletenessSummary", "CAZyRowsByGenome", "CazymeFamilyCounts", "BGCsByGenome", "ProfileKinds"],
    params={"include": "List[str] | null"},
    run=_materialize_functional_profile,
    description="Aggregate pathways (KO completeness) and modules (CAZy/BGC) into one profile",
))
