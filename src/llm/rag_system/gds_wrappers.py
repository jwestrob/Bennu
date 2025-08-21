from __future__ import annotations

import os
from typing import Any, Dict


class GDSDisabledError(RuntimeError):
    pass


def _ensure_enabled():
    if os.getenv("AGENT_ENABLE_GDS", "0") != "1":
        raise GDSDisabledError("GDS wrappers are disabled. Set AGENT_ENABLE_GDS=1 to enable.")


def k_step_neighborhood(tx, start_label: str, start_id_key: str, start_id: str, k: int = 1) -> Dict[str, Any]:
    """Fetch k-step neighborhood using Cypher expansions (no CALL).*"""
    _ensure_enabled()
    cypher = (
        f"MATCH (s:{start_label} {{{start_id_key}:$start_id}}) "
        f"CALL {{ WITH s MATCH p=(s)-[*..{k}]-(n) RETURN nodes(p) AS ns }} "
        "UNWIND ns AS node RETURN DISTINCT node"
    )
    result = tx.run(cypher, start_id=start_id)
    return {"nodes": [dict(r["node"]) for r in result]}


def bgc_community_detection(tx, community_label: str = "BGCCommunity") -> Dict[str, Any]:
    """Placeholder for curated community detection; requires backend precomputation."""
    _ensure_enabled()
    cypher = f"MATCH (c:{community_label}) RETURN count(c) AS count"
    result = tx.run(cypher)
    count = result.single()["count"] if result else 0
    return {"communities_indexed": int(count)}

