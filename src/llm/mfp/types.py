# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Literal, TypedDict, Tuple, Optional

FeatureSource = Literal["pfam", "ko", "mixed"]


class FeatureSet(TypedDict, total=False):
    source: FeatureSource         # "pfam" | "ko" | "mixed"
    ids: List[str]                # accessions or IDs (may be empty)
    terms: List[str]              # optional human-readable terms


class ProteinRecord(TypedDict, total=False):
    protein_id: str
    genome_id: str
    contig_id: str
    coords: Tuple[int, int]
    features: Dict[str, List[str]]  # e.g., {"pfams": [...], "kos": [...]} (optional)


class ProteinSet(TypedDict, total=False):
    proteins: List[ProteinRecord]


class NeighborhoodRecord(TypedDict, total=False):
    seed_protein_id: str
    nodes: List[ProteinRecord]
    edges: List[Tuple[str, str, str]]   # (protein_id_a, protein_id_b, "adjacency|co-annotation")
    span_bp: int


class NeighborhoodSet(TypedDict, total=False):
    neighborhoods: List[NeighborhoodRecord]


class CompletenessMatrix(TypedDict, total=False):
    rows: List[str]       # genome_ids
    cols: List[str]       # pathway_ids
    values: List[List[float]]  # per-genome per-pathway completeness


# Minimal runtime validators (no new deps)
def assert_featureset(fs: FeatureSet) -> None:
    assert "ids" in fs and isinstance(fs["ids"], list)


def assert_proteinset(ps: ProteinSet) -> None:
    assert "proteins" in ps and isinstance(ps["proteins"], list)

