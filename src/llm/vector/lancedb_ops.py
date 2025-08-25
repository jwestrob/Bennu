from __future__ import annotations
from typing import Dict, List, Tuple, Literal, Any


class LanceDBConfigError(RuntimeError):
    ...


def ensure_manifest_parity(lancedb) -> None:
    manifest = getattr(lancedb, "manifest", None)
    # Best-effort check: require presence of dim/version, do not fabricate
    if not manifest or not hasattr(manifest, "dim") or not hasattr(manifest, "version"):
        raise LanceDBConfigError("LanceDB manifest missing dim/version")


async def batched_knn_and_filter(
    lancedb_processor,
    seed_ids: List[str],           # stable external protein IDs (NOT Neo4j internal id)
    topk: int,
    distance: Literal["cosine", "l2", "dot"],
    pfam_filter_ids: List[str],    # e.g., ["PF00589", ...]
    pfam_filter_needle: str,       # e.g., "integrase" (lowercase)
    neo4j_runner,                  # object with run_template(name, params)
    pfam_include_ids: List[str] | None = None,
    pfam_include_needle: str | None = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Returns: mapping seed_id -> top filtered neighbors (still oversampled; caller can slice to nn).
    Steps:
      1) parity check
      2) single batched LanceDB query (exclude self)
      3) single compiled Cypher to get Pfam flags for ALL neighbor ids
      4) filter in-memory; do NOT make a second LanceDB call
    """
    # 1) parity check: rely on processor having manifest attribute if available
    if hasattr(lancedb_processor, "manifest"):
        ensure_manifest_parity(lancedb_processor)

    if not seed_ids:
        return {}

    # 2) single batched kNN using existing LanceDBQueryProcessor API
    qr = await lancedb_processor.execute_similarity_batch(seed_ids, max(1, int(topk)))
    # Shape: QueryResult.results = [ {seed_id: [ {protein_id, distance, ...}, ...]} ]
    base_map: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(qr.results, list) and qr.results and isinstance(qr.results[0], dict):
        base_map = qr.results[0]

    # 2b) gather unique neighbor ids (exclude self)
    all_ids: List[str] = []
    for sid, items in base_map.items():
        for r in (items or []):
            nid = r.get("protein_id")
            if nid and nid != sid:
                all_ids.append(nid)
    uniq_ids = list(dict.fromkeys(all_ids))

    if not uniq_ids:
        return {sid: [] for sid in seed_ids}

    # 3) one KG join to get Pfam flags
    rows = neo4j_runner.run_template(
        "pfam_flags_for_protein_ids.cypher",
        {
            "protein_ids": uniq_ids,
            "exclude_needle": pfam_filter_needle or "",
            "exclude_markers": pfam_filter_ids or [],
            "include_needle": (pfam_include_needle or ""),
            "include_markers": (pfam_include_ids or []),
        },
    )
    is_marker = {r.get("protein_id"): bool(r.get("is_marker")) for r in (rows or [])}
    matches_include = {r.get("protein_id"): bool(r.get("matches_include")) for r in (rows or [])}

    # 4) filter
    out: Dict[str, List[Tuple[str, float]]] = {}
    for sid, items in base_map.items():
        kept: List[Tuple[str, float]] = []
        for r in (items or []):
            nid = r.get("protein_id")
            if not nid or nid == sid:
                continue
            if is_marker.get(nid, False):
                continue
            # If include criteria present, require matches_include == True
            if (pfam_include_ids or pfam_include_needle) and not matches_include.get(nid, False):
                continue
            kept.append((nid, float(r.get("distance", 0.0))))
        out[sid] = kept
    return out
