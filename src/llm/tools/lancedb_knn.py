from __future__ import annotations
from typing import List, Dict, Tuple, Literal, Any
from pydantic import BaseModel, Field, conint
import logging
from ..vector.lancedb_ops import batched_knn_and_filter
from ..options.template_runner import FileCypherRunner


class LanceDbKnnInput(BaseModel):
    seed_ids: List[str] = Field(..., description="Stable protein IDs present in LanceDB")
    topk: conint(gt=0, le=200) = 50
    distance: Literal["cosine", "l2", "dot"] = "cosine"
    exclude_namespace: Literal["pfam", "kofam", "none"] = "pfam"
    exclude_markers: List[str] = Field(default_factory=list)  # e.g., ["integrase","PF00589"]
    include_namespace: Literal["pfam", "kofam", "none"] = "none"
    include_markers: List[str] = Field(default_factory=list)
    include_text: str | None = None


class LanceDbKnnOutput(BaseModel):
    neighbors: Dict[str, List[Tuple[str, float]]]
    picked: Dict[str, List[Tuple[str, float]]]
    stats: Dict[str, Any]


async def lancedb_knn_tool(
    rag_system,
    seed_ids: List[str],
    nn: int,
    topk: int = 50,
    distance: str = "cosine",
    exclude_namespace: str = "pfam",
    exclude_markers: List[str] | None = None,
    **kwargs,
) -> Dict[str, Any]:
    """First-class LanceDB kNN tool (batched, filtered by KG Pfam flags)."""
    logger = logging.getLogger(__name__)
    params = LanceDbKnnInput(
        seed_ids=seed_ids,
        topk=topk,
        distance=distance,  # type: ignore
        exclude_namespace=exclude_namespace if exclude_namespace in ("pfam", "kofam", "none") else "pfam",
        exclude_markers=exclude_markers or [],
    )
    # Build KG runner from Neo4j driver
    runner = FileCypherRunner(rag_system.neo4j_processor.driver)
    ldb = rag_system.lancedb_processor

    # Derive needle for Pfam filter (lowercase, prefer singular form) and accessions list
    needle = ""
    pfam_ids: List[str] = []
    if params.exclude_namespace == "pfam" and params.exclude_markers:
        # Normalize markers:
        # - keep PF accessions intact for direct matching
        # - for text markers, prefer a singular variant if available (e.g., integrase)
        lowers = [(m or "").lower() for m in params.exclude_markers]
        pfam_ids = [m for m in lowers if m.upper().startswith("PF")]
        text_markers = [m for m in lowers if not m.startswith("pf")]
        # Build candidates with simple singularization heuristics
        cand: List[str] = []
        for t in text_markers:
            cand.append(t)
            if t.endswith("es"):
                cand.append(t[:-2])
            elif t.endswith("s"):
                cand.append(t[:-1])
        # Prefer exact common forms if present, otherwise shortest candidate
        pref = None
        for p in ("integrase", "terminase"):
            if p in cand:
                pref = p
                break
        if pref is None and cand:
            pref = sorted(set(cand), key=len)[0]
        needle = pref or (text_markers[0] if text_markers else "")

    # Primary: simple, filtered neighbors (seed -> [(protein_id, distance), ...])
    filtered = await batched_knn_and_filter(
        ldb,
        seed_ids=params.seed_ids,
        topk=int(params.topk),
        distance=params.distance,  # type: ignore
        pfam_filter_ids=pfam_ids,
        pfam_filter_needle=needle,
        neo4j_runner=runner,
    )
    picked = {sid: arr[: int(nn)] for sid, arr in filtered.items()}

    # Enriched: include genome_id/similarity/length (seed -> [ {...}, ... ])
    neighbors_full: Dict[str, List[Dict[str, Any]]] = {}
    try:
        # Base LanceDB results (unfiltered, with metadata)
        qr = await ldb.execute_similarity_batch(params.seed_ids, int(params.topk))
        base_map = qr.results[0] if isinstance(qr.results, list) and qr.results and isinstance(qr.results[0], dict) else {}
        # Build allowed set using single KG join (same as in lancedb_ops)
        uniq_ids: List[str] = []
        for sid, items in (base_map or {}).items():
            for r in (items or []):
                nid = r.get("protein_id")
                if nid and nid != sid:
                    uniq_ids.append(nid)
        uniq_ids = list(dict.fromkeys(uniq_ids))
        allowed: Dict[str, bool] = {}
        # Derive include filters (pfam only for now). Compute regardless of uniq_ids for logging/stats.
        inc_ids: List[str] = []
        inc_text = ""
        if params.include_namespace == "pfam":
            lowers_inc = [(m or "").lower() for m in (params.include_markers or [])]
            inc_ids = [m for m in lowers_inc if m.upper().startswith("PF")]
            # prefer explicit include_text if provided; else singularize first text marker
            include_text = params.include_text or ""
            if not include_text:
                text_markers = [m for m in lowers_inc if not m.startswith("pf")]
                if text_markers:
                    t0 = text_markers[0]
                    if t0.endswith("es"):
                        include_text = t0[:-2]
                    elif t0.endswith("s"):
                        include_text = t0[:-1]
                    else:
                        include_text = t0
            inc_text = (include_text or "").lower()

        include_ok: Dict[str, bool] = {}
        if uniq_ids:
            rows = runner.run_template(
                "pfam_flags_for_protein_ids.cypher",
                {
                    "protein_ids": uniq_ids,
                    "exclude_needle": needle or "",
                    "exclude_markers": pfam_ids or [],
                    "include_needle": inc_text or "",
                    "include_markers": inc_ids or [],
                },
            )
            # Keep those NOT flagged as marker
            allowed = {r.get("protein_id"): (not bool(r.get("is_marker"))) for r in (rows or [])}
            # Inclusion map for full neighbors
            include_ok = {r.get("protein_id"): bool(r.get("matches_include")) for r in (rows or [])}
        # Construct full neighbors
        for sid, items in (base_map or {}).items():
            arr: List[Dict[str, Any]] = []
            for r in (items or []):
                nid = r.get("protein_id")
                if not nid or nid == sid:
                    continue
                if allowed and not allowed.get(nid, True):
                    continue
                # If include criteria present, require matches_include == True
                if (params.include_namespace == "pfam") and (inc_ids or inc_text):
                    if not include_ok.get(nid, False):
                        continue
                arr.append({
                    "protein_id": nid,
                    "distance": float(r.get("distance", 0.0)),
                    "similarity": float(r.get("similarity", 1.0 - float(r.get("distance", 0.0)))),
                    "genome_id": r.get("genome_id"),
                    "sequence_length": r.get("sequence_length"),
                })
            neighbors_full[sid] = arr
    except Exception as e:
        logger.debug(f"lancedb_knn_tool: enrichment skipped: {e}")
    # Debug log of effective filter
    try:
        logger.info(
            "KNN_FILTER: ns=%s needle=%s markers=%s topk=%s nn=%s seeds=%s",
            params.exclude_namespace,
            needle,
            pfam_ids,
            int(params.topk),
            int(nn),
            len(params.seed_ids),
        )
    except Exception:
        pass

    out = LanceDbKnnOutput(
        neighbors=filtered,
        picked=picked,
        stats={
            "queried_seeds": len(params.seed_ids),
            "topk": int(params.topk),
            "neighbors_full_present": bool(neighbors_full),
            "neighbors_counts": {sid: len(v) for sid, v in filtered.items()},
            "exclude_namespace": params.exclude_namespace,
            "filter_needle": needle,
            "filter_markers": pfam_ids,
            "include_namespace": params.include_namespace,
            "include_needle": inc_text,
            "include_markers": inc_ids,
        },
    )
    # Envelope consistent with external tool patterns
    return {
        "tool_name": "lancedb_knn",
        "success": True,
        "display_text": f"LanceDB kNN executed for {len(params.seed_ids)} seeds (topk={params.topk}, nn={nn})",
        "structured_data": [out.dict(), {"neighbors_full": neighbors_full}],
    }
