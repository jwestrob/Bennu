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
        "RETURN n ORDER BY n.start"
    )
    # Let compiler append LIMIT later if provided
    params = {"gene_id": gene_id}
    if limit is not None:
        params["limit"] = limit
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
    cypher = (
        "MATCH (p:Protein {id:$protein_id})-[:ENCODEDBY]->(g:Gene) "
        f"CALL {{ WITH g MATCH pth=(g)-[:NEXT*..{k}]-(ng:Gene) RETURN DISTINCT ng }} "
        "OPTIONAL MATCH (np:Protein)-[:ENCODEDBY]->(ng) "
        "WITH DISTINCT np WHERE np IS NOT NULL RETURN np as protein"
    )
    params = {"protein_id": protein_id}
    if limit is not None:
        params["limit"] = limit
    return cypher, params


SPECS: Dict[str, TemplateSpec] = {
    "protein_by_id": TemplateSpec(
        filename="protein_by_id.cypher",
        required={"id": str},
        optional={},
    ),
    "proteins_with_ko": TemplateSpec(
        filename="proteins_with_ko.cypher",
        required={"ko": str},
        optional={},
    ),
    "neighbors_by_window": TemplateSpec(
        filename="neighbors_by_window.cypher",
        required={"contig": str, "start": int, "end": int},
        optional={},
    ),
    "pathway_membership": TemplateSpec(
        filename="pathway_membership.cypher",
        required={"pathway": str},
        optional={},
    ),
    "cazy_family": TemplateSpec(
        filename="cazy_family.cypher",
        required={"family": str},
        optional={},
    ),
    "count_by_label": TemplateSpec(
        filename=None,
        required={"label": str},
        optional={},
        compiler=_compile_count_by_label,
    ),
    "gene_neighbors_k": TemplateSpec(
        filename=None,
        required={"gene_id": str},
        optional={"k": int, "limit": int},
        compiler=_compile_gene_neighbors_k,
    ),
    "protein_neighbors_k": TemplateSpec(
        filename=None,
        required={"protein_id": str},
        optional={"k": int, "limit": int},
        compiler=_compile_protein_neighbors_k,
    ),
    "proteins_by_genome": TemplateSpec(
        filename="proteins_by_genome.cypher",
        required={"genome_id": str},
        optional={},
    ),
    "genes_on_contig": TemplateSpec(
        filename="genes_on_contig.cypher",
        required={"contig": str},
        optional={},
    ),
    "proteins_with_pfam": TemplateSpec(
        filename="proteins_with_pfam.cypher",
        required={"pfam": str},
        optional={},
    ),
    "count_proteins_with_ko": TemplateSpec(
        filename="count_proteins_with_ko.cypher",
        required={"ko": str},
        optional={},
    ),
    "count_proteins_with_pfam": TemplateSpec(
        filename="count_proteins_with_pfam.cypher",
        required={"pfam": str},
        optional={},
    ),
    "count_proteins_in_pathway": TemplateSpec(
        filename="count_proteins_in_pathway.cypher",
        required={"pathway": str},
        optional={},
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
