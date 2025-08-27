from __future__ import annotations
from typing import Any, Dict
from pathlib import Path

from .base import OperatorContext, OperatorSpec, register_operator
from ...options.template_runner import FileCypherRunner
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


def _find_proteins_by_pfam_keyword(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    runner = FileCypherRunner(ctx.neo4j_driver)
    q = str(params.get("q") or params.get("keyword") or "").strip()
    if not q:
        return {"pfam_keyword_proteins": []}
    try:
        limit = int(params.get("limit", 500))
    except Exception:
        limit = 500
    genome_ids = params.get("genome_ids") or []
    def tokens(s: str) -> list[str]:
        import re
        ws = re.split(r"[^A-Za-z0-9_]+", s.lower())
        toks = [t for t in ws if len(t) >= 4]
        # keep original if specific
        if len(s) >= 4 and s.lower() not in toks:
            toks.append(s.lower())
        # dedupe preserve order
        seen = set()
        out = []
        for t in toks:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out[:5]
    all_rows = []
    for term in tokens(q):
        rr = runner.run_template(
            "proteins_by_pfam_keyword.cypher",
            {"q": term, "limit": max(50, limit // 3), "genome_ids": genome_ids},
        ) or []
        all_rows.extend(rr)
    return {"pfam_keyword_proteins": all_rows or []}


def _find_proteins_by_ko_keyword(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    runner = FileCypherRunner(ctx.neo4j_driver)
    q = str(params.get("q") or params.get("keyword") or "").strip()
    if not q:
        return {"ko_keyword_proteins": []}
    try:
        limit = int(params.get("limit", 500))
    except Exception:
        limit = 500
    genome_ids = params.get("genome_ids") or []
    def tokens(s: str) -> list[str]:
        import re
        ws = re.split(r"[^A-Za-z0-9_]+", s.lower())
        toks = [t for t in ws if len(t) >= 4]
        if len(s) >= 4 and s.lower() not in toks:
            toks.append(s.lower())
        seen = set()
        out = []
        for t in toks:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out[:5]
    all_rows = []
    for term in tokens(q):
        rr = runner.run_template(
            "proteins_by_ko_keyword.cypher",
            {"q": term, "limit": max(50, limit // 3), "genome_ids": genome_ids},
        ) or []
        all_rows.extend(rr)
    return {"ko_keyword_proteins": all_rows or []}


def _annotation_discovery(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """PFAM+KO keyword discovery (DB-backed; no external tools).

    - Searches PFAM domains and KEGG KOs by keyword
    - Returns union of proteins matched by either, deduplicated with provenance
    """
    runner = FileCypherRunner(ctx.neo4j_driver)
    q = str(params.get("keyword") or params.get("q") or "").strip()
    if not q:
        return {"discovered_proteins": []}
    try:
        limit = int(params.get("limit", 500))
    except Exception:
        limit = 500
    genome_ids = params.get("genome_ids") or []

    def tokens(s: str) -> list[str]:
        import re
        ws = re.split(r"[^A-Za-z0-9_]+", s.lower())
        toks = [t for t in ws if len(t) >= 4]
        if len(s) >= 4 and s.lower() not in toks:
            toks.append(s.lower())
        seen = set()
        out = []
        for t in toks:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out[:6]
    pfam_rows = []
    ko_rows = []
    terms = tokens(q)
    for term in terms:
        pfam_rows.extend(runner.run_template(
            "proteins_by_pfam_keyword.cypher",
            {"q": term, "limit": max(50, limit // max(1,len(terms))), "genome_ids": genome_ids},
        ) or [])
        ko_rows.extend(runner.run_template(
            "proteins_by_ko_keyword.cypher",
            {"q": term, "limit": max(50, limit // max(1,len(terms))), "genome_ids": genome_ids},
        ) or [])

    # Merge with provenance; key by (genome_id, protein_id)
    merged: Dict[str, Dict[str, Any]] = {}
    for r in pfam_rows:
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
    # Keep deterministic order
    out.sort(key=lambda x: (x.get("genome_id", ""), x.get("protein_id", "")))
    return {"discovered_proteins": out}

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

register_operator(OperatorSpec(
    name="FindProteinsByPfamKeyword",
    inputs=[],
    outputs=["pfam_keyword_proteins"],
    params={"q": "str", "limit": "int | null", "genome_ids": "List[str] | null"},
    run=_find_proteins_by_pfam_keyword,
    description="Find proteins whose PFAM domains match a keyword (id/accession/description).",
))

register_operator(OperatorSpec(
    name="FindProteinsByKoKeyword",
    inputs=[],
    outputs=["ko_keyword_proteins"],
    params={"q": "str", "limit": "int | null", "genome_ids": "List[str] | null"},
    run=_find_proteins_by_ko_keyword,
    description="Find proteins whose KEGG Ortholog annotations match a keyword (id/description).",
))

register_operator(OperatorSpec(
    name="AnnotationDiscovery",
    inputs=[],
    outputs=["discovered_proteins"],
    params={"keyword": "str", "limit": "int | null", "genome_ids": "List[str] | null"},
    run=_annotation_discovery,
    description="Keyword search across PFAM + KO; returns union of matched proteins with PFAM/KO provenance.",
))
