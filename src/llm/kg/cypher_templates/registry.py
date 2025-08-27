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
    label = slots.get("label")
    if label not in ("Protein", "Gene"):
        raise ValueError("count_by_label: 'label' must be one of ['Protein','Gene']")
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
        optional={},
        category="count",
        returns="scalar",
        cost="cheap",
        slot_hints={"pfam": "PFxxxxx"},
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
    validate_slots(name, slots)
    spec = SPECS[name]
    if spec.compiler:
        return spec.compiler(slots)
    text = _read(name)
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
