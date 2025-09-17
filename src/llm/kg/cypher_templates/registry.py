from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any, Callable


TEMPLATES_DIR = Path(__file__).parent


@dataclass(frozen=True)
class TemplateSpec:
    filename: str | None
    required: Dict[str, type]
    optional: Dict[str, type]
    compiler: Callable[[Dict[str, Any]], Tuple[str, Dict[str, Any]]] | None = None
    # Non-behavioral metadata for catalogs (advisory only)
    category: str = "general"       # discovery | neighborhood | count | span_window | general
    returns: str = "table"           # protein | gene | table | scalar
    cost: str = "cheap"              # cheap | moderate | expensive
    slot_hints: Dict[str, str] = None  # e.g., {"pfam": "PFxxxxx", "ko": "Kxxxxx"}


def _read(name: str) -> str:
    path = TEMPLATES_DIR / f"{name}.cypher"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _compile_count_by_label(slots: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    # Only allow safe, enumerated labels; render static query to avoid injection.
    label_in = slots.get("label")
    if not isinstance(label_in, str) or not label_in.strip():
        raise ValueError("count_by_label: 'label' must be a non-empty string")
    # Case-insensitive mapping to canonical labels present in the graph
    canon_map = {
        'protein': 'Protein',
        'gene': 'Gene',
        'crisprarray': 'CrisprArray',
        'pathway': 'Pathway',
        'keggortholog': 'KEGGOrtholog',
        'domain': 'Domain',
        'domainannotation': 'DomainAnnotation',
    }
    key = label_in.strip().lower()
    if key not in canon_map:
        allowed = ", ".join(sorted(set(canon_map.values())))
        raise ValueError(f"count_by_label: 'label' must be one of [{allowed}]")
    label = canon_map[key]
    cypher = f"MATCH (n:{label}) RETURN count(n) AS count;"
    return cypher, {}


def _compile_gene_neighbors_k(slots: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    gene_id = slots.get("gene_id")
    if not isinstance(gene_id, str):
        raise ValueError("gene_neighbors_k: 'gene_id' must be str")
    try:
        k = int(slots.get("k", 1))
    except Exception:
        k = 1
    limit = slots.get("limit")
    # Embed k directly into the pattern length; param for id/limit only
    cypher = (
        f"MATCH (g:Gene {{id:$gene_id}}) "
        f"CALL {{ WITH g MATCH p=(g)-[:NEXT*..{k}]-(n:Gene) RETURN DISTINCT n }} "
        "RETURN n ORDER BY toInteger(n.startCoordinate)"
    )
    # Let compiler append LIMIT later if provided
    params = {"gene_id": gene_id}
    if limit is not None:
        params["limit"] = int(limit)
        cypher = cypher + "\nLIMIT $limit"
    return cypher, params


def _compile_protein_neighbors_k(slots: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    protein_id = slots.get("protein_id")
    if not isinstance(protein_id, str):
        raise ValueError("protein_neighbors_k: 'protein_id' must be str")
    try:
        k = int(slots.get("k", 1))
    except Exception:
        k = 1
    limit = slots.get("limit")
    # Graph uses (Gene)-[:ENCODEDBY]->(Protein) and genomic NEXT edges between genes.
    # 1) Anchor on the seed protein and resolve its gene via incoming ENCODEDBY
    # 2) Expand k-step gene neighbors via NEXT (both directions)
    # 3) Resolve neighbor proteins via (ng)-[:ENCODEDBY]->(np)
    # 4) Order results by numeric startCoordinate of the neighbor gene
    cypher = (
        "MATCH (p:Protein {id:$protein_id})-[:ENCODEDBY]->(g:Gene) "
        f"CALL {{ WITH g MATCH pth=(g)-[:NEXT*..{k}]-(ng:Gene) RETURN DISTINCT ng }} "
        "OPTIONAL MATCH (np:Protein)-[:ENCODEDBY]->(ng) "
        "WITH DISTINCT np, ng WHERE np IS NOT NULL "
        "RETURN np AS protein ORDER BY toInteger(ng.startCoordinate)"
    )
    params = {"protein_id": protein_id}
    if limit is not None:
        params["limit"] = int(limit)
        cypher = cypher + "\nLIMIT $limit"
    return cypher, params


def _compile_anchor_gene_window(slots: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    atype = str(slots.get("anchor_type") or "").strip().lower()
    # Normalize common synonyms/variants
    syn = {
        'crispr_array': 'crispr', 'crispr_arrays': 'crispr', 'crisprarray': 'crispr', 'crisprarrays': 'crispr',
        'protein': 'protein', 'proteins': 'protein',
        'gene': 'gene', 'genes': 'gene',
        'bgc': 'bgc', 'bgcs': 'bgc', 'cluster': 'bgc', 'clusters': 'bgc', 'bgc_cluster': 'bgc',
        'coords': 'coords', 'coordinate': 'coords', 'coordinates': 'coords', 'span': 'coords', 'window': 'coords', 'region': 'coords',
    }
    if atype in syn:
        atype = syn[atype]
    elif atype.startswith('crispr'):
        atype = 'crispr'
    if atype not in {"crispr", "protein", "gene", "bgc", "coords"}:
        raise ValueError("anchor_gene_window: 'anchor_type' must be one of crispr|protein|gene|bgc|coords")
    # Normalize numeric optional params
    margin = slots.get("margin_bp")
    try:
        margin = int(margin) if margin is not None else None
    except Exception:
        margin = None
    limit = slots.get("limit")
    try:
        limit = int(limit) if limit is not None else None
    except Exception:
        limit = None

    # Optional annotation enrichment
    include_ann = bool(slots.get("include_annotations") or slots.get("annotations"))
    # Common RETURN clause (with optional annotations)
    if include_ann:
        ret = (
            "OPTIONAL MATCH (p)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain) "
            "OPTIONAL MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog) "
            "OPTIONAL MATCH (p)-[:HASCAZYME]->(ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily) "
            "WITH g, p, anchor_id, "
            "collect(DISTINCT coalesce(d.pfamAccession, d.id)) AS pfam_ids, "
            "collect(DISTINCT coalesce(d.name, d.description)) AS pfam_names, "
            "collect(DISTINCT ko.id) AS ko_ids, "
            "collect(DISTINCT ko.description) AS ko_desc, "
            "collect(DISTINCT cf.familyId) AS cazy_families "
            "RETURN g.id AS gene_id, "
            "p.id AS protein_id, "
            "g.contig AS contig, "
            "toInteger(g.startCoordinate) AS start, "
            "toInteger(g.endCoordinate) AS end, "
            "anchor_id AS anchor_id, "
            "pfam_ids AS pfam_ids, pfam_names AS pfam_names, ko_ids AS ko_ids, ko_desc AS ko_desc, cazy_families AS cazy_families "
            "ORDER BY start"
        )
    else:
        ret = (
            "RETURN g.id AS gene_id, "
            "p.id AS protein_id, "
            "g.contig AS contig, "
            "toInteger(g.startCoordinate) AS start, "
            "toInteger(g.endCoordinate) AS end, "
            "anchor_id AS anchor_id "
            "ORDER BY start"
        )
    if isinstance(limit, int) and limit > 0:
        ret += "\nLIMIT $limit"

    # Template per anchor_type
    if atype == "crispr":
        # Two modes: anchor_id provided → use it; else (optional genome_id) → pick array with richest window
        if isinstance(slots.get("anchor_id"), str) and slots.get("anchor_id").strip():
            cypher = (
                "WITH toInteger(coalesce($margin_bp, 5000)) AS M "
                "MATCH (ca:CrisprArray {id:$anchor_id}) "
                "WITH ca, M, toInteger(ca.startCoordinate)-M AS wstart, toInteger(ca.endCoordinate)+M AS wend, ca.id AS anchor_id "
                "MATCH (g:Gene {contig: ca.contig}) "
                "WHERE toInteger(g.startCoordinate) <= wend AND toInteger(g.endCoordinate) >= wstart "
                "OPTIONAL MATCH (p:Protein)-[:ENCODEDBY]->(g) "
                + ret
            )
            params = {"anchor_id": str(slots["anchor_id"])}
            if isinstance(margin, int):
                params["margin_bp"] = margin
            if isinstance(limit, int):
                params["limit"] = limit
            return cypher, params
        else:
            # Optional genome_id filter
            pre = (
                "WITH toInteger(coalesce($margin_bp, 5000)) AS M "
                "MATCH (ca:CrisprArray) "
            )
            if isinstance(slots.get("genome_id"), str) and slots.get("genome_id").strip():
                pre = (
                    "WITH toInteger(coalesce($margin_bp, 5000)) AS M "
                    "MATCH (g:Genome {id:$genome_id})<-[:BELONGSTOGENOME]-(ca:CrisprArray) "
                )
            cypher = (
                pre +
                "WITH ca, M "
                "MATCH (gg:Gene {contig: ca.contig}) "
                "WHERE toInteger(gg.startCoordinate) <= toInteger(ca.endCoordinate) + M "
                "  AND toInteger(gg.endCoordinate) >= toInteger(ca.startCoordinate) - M "
                "WITH ca, M, count(gg) AS gene_count "
                "ORDER BY gene_count DESC, ca.contig, toInteger(ca.startCoordinate) "
                "LIMIT 1 "
                "WITH ca, M, ca.id AS anchor_id, toInteger(ca.startCoordinate)-M AS wstart, toInteger(ca.endCoordinate)+M AS wend "
                "MATCH (g:Gene {contig: ca.contig}) "
                "WHERE toInteger(g.startCoordinate) <= wend AND toInteger(g.endCoordinate) >= wstart "
                "OPTIONAL MATCH (p:Protein)-[:ENCODEDBY]->(g) "
                + ret
            )
            params = {}
            if isinstance(margin, int):
                params["margin_bp"] = margin
            if isinstance(limit, int):
                params["limit"] = limit
            if isinstance(slots.get("genome_id"), str) and slots.get("genome_id").strip():
                params["genome_id"] = str(slots["genome_id"]).strip()
            return cypher, params

    if atype == "protein":
        cypher = (
            "WITH toInteger(coalesce($margin_bp, 5000)) AS M "
            "MATCH (p0:Protein {id:$anchor_id})-[:ENCODEDBY]->(seed:Gene) "
            "WITH seed, M, p0.id AS anchor_id, toInteger(seed.startCoordinate)-M AS wstart, toInteger(seed.endCoordinate)+M AS wend "
            "MATCH (g:Gene {contig: seed.contig}) "
            "WHERE toInteger(g.startCoordinate) <= wend AND toInteger(g.endCoordinate) >= wstart "
            "OPTIONAL MATCH (p:Protein)-[:ENCODEDBY]->(g) "
            + ret
        )
        params = {"anchor_id": str(slots.get("anchor_id") or "").strip()}
        if not params["anchor_id"]:
            raise ValueError("anchor_gene_window: protein requires anchor_id")
        if isinstance(margin, int):
            params["margin_bp"] = margin
        if isinstance(limit, int):
            params["limit"] = limit
        return cypher, params

    if atype == "gene":
        cypher = (
            "WITH toInteger(coalesce($margin_bp, 5000)) AS M "
            "MATCH (seed:Gene {id:$anchor_id}) "
            "WITH seed, M, seed.id AS anchor_id, toInteger(seed.startCoordinate)-M AS wstart, toInteger(seed.endCoordinate)+M AS wend "
            "MATCH (g:Gene {contig: seed.contig}) "
            "WHERE toInteger(g.startCoordinate) <= wend AND toInteger(g.endCoordinate) >= wstart "
            "OPTIONAL MATCH (p:Protein)-[:ENCODEDBY]->(g) "
            + ret
        )
        params = {"anchor_id": str(slots.get("anchor_id") or "").strip()}
        if not params["anchor_id"]:
            raise ValueError("anchor_gene_window: gene requires anchor_id")
        if isinstance(margin, int):
            params["margin_bp"] = margin
        if isinstance(limit, int):
            params["limit"] = limit
        return cypher, params

    if atype == "bgc":
        cypher = (
            "WITH toInteger(coalesce($margin_bp, 5000)) AS M "
            "MATCH (b:Bgc) WHERE b.id = $anchor_id OR b.bgcId = $anchor_id "
            "WITH b, M, coalesce(b.contig, '') AS contig, toInteger(b.startCoordinate)-M AS wstart, toInteger(b.endCoordinate)+M AS wend, coalesce(b.bgcId, b.id) AS anchor_id "
            "MATCH (g:Gene {contig: contig}) "
            "WHERE toInteger(g.startCoordinate) <= wend AND toInteger(g.endCoordinate) >= wstart "
            "OPTIONAL MATCH (p:Protein)-[:ENCODEDBY]->(g) "
            + ret
        )
        params = {"anchor_id": str(slots.get("anchor_id") or "").strip()}
        if not params["anchor_id"]:
            raise ValueError("anchor_gene_window: bgc requires anchor_id")
        if isinstance(margin, int):
            params["margin_bp"] = margin
        if isinstance(limit, int):
            params["limit"] = limit
        return cypher, params

    if atype == "coords":
        # contig,start,end required
        contig = str(slots.get("contig") or "").strip()
        try:
            s = int(slots.get("start"))
            e = int(slots.get("end"))
        except Exception:
            raise ValueError("anchor_gene_window: coords requires contig, start, end (ints)")
        cypher = (
            "WITH toInteger(coalesce($margin_bp, 5000)) AS M, $contig AS contig, toInteger($start) AS s, toInteger($end) AS e "
            "WITH contig, s, e, M, 'coords:' + contig + ':' + toString(s) + '-' + toString(e) AS anchor_id, (s - M) AS wstart, (e + M) AS wend "
            "MATCH (g:Gene {contig: contig}) "
            "WHERE toInteger(g.startCoordinate) <= wend AND toInteger(g.endCoordinate) >= wstart "
            "OPTIONAL MATCH (p:Protein)-[:ENCODEDBY]->(g) "
            + ret
        )
        params = {"contig": contig, "start": s, "end": e}
        if isinstance(margin, int):
            params["margin_bp"] = margin
        if isinstance(limit, int):
            params["limit"] = limit
        return cypher, params

    # unreachable
    raise ValueError("anchor_gene_window: unsupported configuration")


SLOT_HINTS_EMPTY: Dict[str, str] = {}

SPECS: Dict[str, TemplateSpec] = {
    "protein_by_id": TemplateSpec(
        filename="protein_by_id.cypher",
        required={"id": str},
        optional={},
        category="discovery",
        returns="protein",
        cost="cheap",
        slot_hints=SLOT_HINTS_EMPTY,
    ),
    "anchor_gene_window": TemplateSpec(
        filename=None,
        required={"anchor_type": str},
        optional={"anchor_id": str, "genome_id": str, "contig": str, "start": int, "end": int, "margin_bp": int, "limit": int, "include_annotations": bool, "annotations": bool},
        compiler=_compile_anchor_gene_window,
        category="span_window",
        returns="gene",
        cost="cheap",
        slot_hints={
            "anchor_type": "crispr|protein|gene|bgc|coords",
            "anchor_id": "optional (id for crispr/protein/gene/bgc)",
            "genome_id": "optional (crispr selection scope)",
            "contig": "coords only",
            "start": "coords only",
            "end": "coords only",
            "margin_bp": "int",
            "limit": "int",
            "include_annotations": "bool"
        },
    ),
    "arrays_per_genome": TemplateSpec(
        filename="arrays_per_genome.cypher",
        required={},
        optional={},
        category="count",
        returns="table",
        cost="cheap",
        slot_hints=SLOT_HINTS_EMPTY,
    ),
    "gene_next_degree": TemplateSpec(
        filename="gene_next_degree.cypher",
        required={"gene_id": str},
        optional={},
        category="debug",
        returns="scalar",
        cost="cheap",
        slot_hints={"gene_id": "gene:<id>"},
    ),
    "contig_gene_index": TemplateSpec(
        filename="contig_gene_index.cypher",
        required={"contig": str, "gene_id": str},
        optional={},
        category="debug",
        returns="table",
        cost="cheap",
        slot_hints={"contig": "contig:<id>", "gene_id": "gene:<id>"},
    ),
    "proteins_with_ko": TemplateSpec(
        filename="proteins_with_ko.cypher",
        required={"ko": str},
        optional={},
        category="discovery",
        returns="protein",
        cost="cheap",
        slot_hints={"ko": "Kxxxxx", "limit": "int"},
    ),
    "neighbors_by_window": TemplateSpec(
        filename="neighbors_by_window.cypher",
        required={"contig": str, "start": int, "end": int},
        optional={},
        category="span_window",
        returns="gene",
        cost="cheap",
        slot_hints={"contig": "contig:<id>", "start": "int", "end": "int", "limit": "int"},
    ),
    "pathway_membership": TemplateSpec(
        filename="pathway_membership.cypher",
        required={"pathway": str},
        optional={},
        category="discovery",
        returns="protein",
        cost="cheap",
        slot_hints={"pathway": "mapxxxxx", "limit": "int"},
    ),
    "cazy_family": TemplateSpec(
        filename="cazy_family.cypher",
        required={"family": str},
        optional={"limit": int},
        category="discovery",
        returns="table",
        cost="cheap",
        slot_hints={"family": "GHxx/GTxx/PLxx/CExx/AAxx/CBMxx", "limit": "int"},
    ),
    "cazymes_by_genome": TemplateSpec(
        filename="cazymes_by_genome.cypher",
        required={"genome_id": str},
        optional={"limit": int},
        category="discovery",
        returns="table",
        cost="cheap",
        slot_hints={"genome_id": "<Genome.id>", "limit": "int"},
    ),
    "cazyme_family_counts": TemplateSpec(
        filename="cazyme_family_counts.cypher",
        required={},
        optional={},
        category="count",
        returns="table",
        cost="cheap",
        slot_hints=SLOT_HINTS_EMPTY,
    ),
    "bgcs_by_genome": TemplateSpec(
        filename="bgcs_by_genome.cypher",
        required={"genome_id": str},
        optional={"limit": int},
        category="discovery",
        returns="table",
        cost="cheap",
        slot_hints={"genome_id": "<Genome.id>", "limit": "int"},
    ),
    "bgcs_by_product": TemplateSpec(
        filename="bgcs_by_product.cypher",
        required={"product": str},
        optional={"limit": int},
        category="discovery",
        returns="table",
        cost="cheap",
        slot_hints={"product": "e.g., Terpene/Polyketide", "limit": "int"},
    ),
    "genes_in_bgc": TemplateSpec(
        filename="genes_in_bgc.cypher",
        required={"bgc_id": str},
        optional={"limit": int},
        category="neighborhood",
        returns="table",
        cost="cheap",
        slot_hints={"bgc_id": "<Bgc.bgcId>", "limit": "int"},
    ),
    "count_by_label": TemplateSpec(
        filename=None,
        required={"label": str},
        optional={},
        compiler=_compile_count_by_label,
        category="count",
        returns="scalar",
        cost="cheap",
        slot_hints={"label": "Protein|Gene"},
    ),
    "gene_neighbors_k": TemplateSpec(
        filename=None,
        required={"gene_id": str},
        optional={"k": int, "limit": int},
        compiler=_compile_gene_neighbors_k,
        category="neighborhood",
        returns="gene",
        cost="cheap",
        slot_hints={"gene_id": "gene:<id>", "k": "int", "limit": "int"},
    ),
    "protein_neighbors_k": TemplateSpec(
        filename=None,
        required={"protein_id": str},
        optional={"k": int, "limit": int},
        compiler=_compile_protein_neighbors_k,
        category="neighborhood",
        returns="protein",
        cost="cheap",
        slot_hints={"protein_id": "protein:<id>", "k": "int", "limit": "int"},
    ),
    "protein_flanking_genes_5": TemplateSpec(
        filename="protein_flanking_genes_5.cypher",
        required={"protein_id": str},
        optional={},
        category="neighborhood",
        returns="gene",
        cost="cheap",
        slot_hints={"protein_id": "protein:<id>"},
    ),
    "proteins_by_genome": TemplateSpec(
        filename="proteins_by_genome.cypher",
        required={"genome_id": str},
        optional={},
        category="discovery",
        returns="protein",
        cost="cheap",
        slot_hints={"genome_id": "genome:<id>", "limit": "int"},
    ),
    "protein_gene_context": TemplateSpec(
        filename="protein_gene_context.cypher",
        required={"protein_id": str},
        optional={},
        category="neighborhood",
        returns="table",
        cost="cheap",
        slot_hints={"protein_id": "protein:<id>"},
    ),
    # --- CRISPR templates ---
    "crispr_arrays_by_contig": TemplateSpec(
        filename="crispr_arrays_by_contig.cypher",
        required={"contig": str},
        optional={"start": int, "end": int, "limit": int},
        category="span_window",
        returns="table",
        cost="cheap",
        slot_hints={"contig": "<contig id>", "start": "int", "end": "int", "limit": "int"},
    ),
    "crispr_arrays_by_genome": TemplateSpec(
        filename="crispr_arrays_by_genome.cypher",
        required={"genome_id": str},
        optional={"limit": int},
        category="span_window",
        returns="table",
        cost="cheap",
        slot_hints={"genome_id": "<Genome.id>", "limit": "int"},
    ),
    "crispr_arrays_global": TemplateSpec(
        filename="crispr_arrays_global.cypher",
        required={},
        optional={"limit": int},
        category="span_window",
        returns="table",
        cost="cheap",
        slot_hints={"limit": "int"},
    ),
    "protein_crispr_context": TemplateSpec(
        filename="protein_crispr_context.cypher",
        required={"protein_id": str},
        optional={"flank_n": int, "limit": int},
        category="neighborhood",
        returns="table",
        cost="cheap",
        slot_hints={"protein_id": "protein:<id>", "flank_n": "int", "limit": "int"},
    ),
    "next_edges_crossing_crispr": TemplateSpec(
        filename="next_edges_crossing_crispr.cypher",
        required={},
        optional={"contig": str, "limit": int},
        category="debug",
        returns="table",
        cost="cheap",
        slot_hints={"contig": "<contig id>", "limit": "int"},
    ),
    "genes_on_contig": TemplateSpec(
        filename="genes_on_contig.cypher",
        required={"contig": str},
        optional={},
        category="span_window",
        returns="gene",
        cost="cheap",
        slot_hints={"contig": "contig:<id>", "limit": "int"},
    ),
    "proteins_with_pfam": TemplateSpec(
        filename="proteins_with_pfam.cypher",
        required={"pfam": str},
        optional={"limit": int, "exact": bool},
        category="discovery",
        returns="protein",
        cost="cheap",
        slot_hints={"pfam": "PFxxxxx or name", "limit": "int", "exact": "bool"},
    ),
    "proteins_with_pfams": TemplateSpec(
        filename="proteins_with_pfams.cypher",
        required={"pfams": list},
        optional={"limit": int},
        category="discovery",
        returns="protein",
        cost="cheap",
        slot_hints={"pfams": "[PFxxxxx,...]", "limit": "int"},
    ),
    "proteins_with_kos": TemplateSpec(
        filename="proteins_with_kos.cypher",
        required={"kos": list},
        optional={"limit": int},
        category="discovery",
        returns="protein",
        cost="cheap",
        slot_hints={"kos": "[Kxxxxx,...]", "limit": "int"},
    ),
    "pfam_search": TemplateSpec(
        filename="pfam_search.cypher",
        required={"q": str},
        optional={"limit": int},
        category="discovery",
        returns="table",
        cost="cheap",
        slot_hints={"q": "substring", "limit": "int"},
    ),
    "kofam_search": TemplateSpec(
        filename="kofam_search.cypher",
        required={"q": str},
        optional={"limit": int},
        category="discovery",
        returns="table",
        cost="cheap",
        slot_hints={"q": "substring", "limit": "int"},
    ),
    "count_proteins_with_ko": TemplateSpec(
        filename="count_proteins_with_ko.cypher",
        required={"ko": str},
        optional={},
        category="count",
        returns="scalar",
        cost="cheap",
        slot_hints={"ko": "Kxxxxx"},
    ),
    "count_proteins_with_pfam": TemplateSpec(
        filename="count_proteins_with_pfam.cypher",
        required={"pfam": str},
        optional={"exact": bool},
        category="count",
        returns="scalar",
        cost="cheap",
        slot_hints={"pfam": "PFxxxxx or name", "exact": "bool"},
    ),
    "count_proteins_in_pathway": TemplateSpec(
        filename="count_proteins_in_pathway.cypher",
        required={"pathway": str},
        optional={},
        category="count",
        returns="scalar",
        cost="cheap",
        slot_hints={"pathway": "mapxxxxx"},
    ),
    "count_proteins_by_pfam_ids": TemplateSpec(
        filename="count_proteins_by_pfam_ids.cypher",
        required={"pfams": list},
        optional={"genome_ids": list},
        category="count",
        returns="table",
        cost="cheap",
        slot_hints={"pfams": "[PFxxxxx or tokens]", "genome_ids": "[genome_ids]"},
    ),
    "count_proteins_by_ko_ids": TemplateSpec(
        filename="count_proteins_by_ko_ids.cypher",
        required={"ko_ids": list},
        optional={"genome_ids": list},
        category="count",
        returns="table",
        cost="cheap",
        slot_hints={"ko_ids": "[Kxxxxx,...]", "genome_ids": "[genome_ids]"},
    ),
    "count_proteins_by_pfam_ids_per_genome": TemplateSpec(
        filename="count_proteins_by_pfam_ids_per_genome.cypher",
        required={"pfam_ids": list},
        optional={"genome_ids": list},
        category="count",
        returns="table",
        cost="cheap",
        slot_hints={"pfam_ids": "[PFxxxxx or tokens]", "genome_ids": "[genome_ids]"},
    ),
    "count_proteins_by_ko_ids_per_genome": TemplateSpec(
        filename="count_proteins_by_ko_ids_per_genome.cypher",
        required={"ko_ids": list},
        optional={"genome_ids": list},
        category="count",
        returns="table",
        cost="cheap",
        slot_hints={"ko_ids": "[Kxxxxx,...]", "genome_ids": "[genome_ids]"},
    ),
}


def validate_slots(name: str, slots: Dict[str, Any]) -> None:
    if name not in SPECS:
        raise ValueError(f"Unknown template: {name}")
    spec = SPECS[name]
    for key, typ in spec.required.items():
        if key not in slots:
            raise ValueError(f"Missing required slot '{key}' for template '{name}'")
        if not isinstance(slots[key], typ):
            raise ValueError(f"Slot '{key}' must be of type {typ.__name__}")
    for key in slots:
        if key not in spec.required and key not in spec.optional:
            # Allow a universal optional 'limit' slot for deterministic caps
            if key == "limit":
                continue
            raise ValueError(f"Unknown slot '{key}' for template '{name}'")


def compile_query(name: str, slots: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    # Lightweight, template-aware slot normalization before validation
    if name == "proteins_with_pfams":
        # Accept common variants and coerce to list
        if "pfams" not in slots and "pfam" in slots:
            slots = dict(slots)
            slots["pfams"] = [slots.pop("pfam")]
        elif "pfams" in slots and not isinstance(slots["pfams"], list):
            slots = dict(slots)
            slots["pfams"] = [slots["pfams"]]
        # Coerce limit to int if present
        if "limit" in slots and not isinstance(slots["limit"], int):
            try:
                slots = dict(slots)
                slots["limit"] = int(slots["limit"])  # type: ignore
            except Exception:
                pass
    elif name == "proteins_with_kos":
        if "kos" not in slots and "ko" in slots:
            slots = dict(slots)
            slots["kos"] = [slots.pop("ko")]
        elif "kos" in slots and not isinstance(slots["kos"], list):
            slots = dict(slots)
            slots["kos"] = [slots["kos"]]
        if "limit" in slots and not isinstance(slots["limit"], int):
            try:
                slots = dict(slots)
                slots["limit"] = int(slots["limit"])  # type: ignore
            except Exception:
                pass

    # Provide template-specific defaults for optional parameters
    if name == "proteins_with_pfam" and "exact" not in slots:
        # Default to flexible matching unless explicitly overridden
        slots = dict(slots)
        slots["exact"] = False
    if name == "count_proteins_with_pfam" and "exact" not in slots:
        slots = dict(slots)
        slots["exact"] = False
    validate_slots(name, slots)
    spec = SPECS[name]
    if spec.compiler:
        return spec.compiler(slots)
    text = _read(name)
    # Ensure $limit parameter exists when referenced in the template text
    # Neo4j requires parameters to be provided even when wrapped in coalesce()
    if "$limit" in text and "limit" not in slots:
        slots = dict(slots)
        slots["limit"] = None
    # Optionally append LIMIT if provided and not already present
    limit = slots.get("limit")
    try:
        limit_int = int(limit) if limit is not None else None
    except Exception:
        limit_int = None
    if isinstance(limit_int, int) and limit_int > 0 and "LIMIT" not in text.upper():
        text = text.rstrip().rstrip(";") + "\nLIMIT $limit;\n"
    # Use parameterized execution via Neo4j; slots are params directly.
    return text, slots
