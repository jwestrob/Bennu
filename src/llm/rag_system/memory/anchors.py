from __future__ import annotations
from typing import Iterable, Dict, List


def assign_anchor(item: dict) -> str:
    """Assign a stable, type-aware anchor string for a result row.

    Heuristics:
    - If 'locus_id' present → sec:locus:<id>
    - Else if 'contig' present → sec:contig:<contig>
    - Else if KO/PFAM present →
        • Prefer KO when available → sec:ko:<first_ko>
        • Else PFAM → sec:pfam:<first_pfam>
        • If both missing but marker-like, fall back to sec:marker:<capability>
    - Else fallback → sec:community:default
    """
    if not isinstance(item, dict):
        return "sec:community:default"
    if item.get("locus_id"):
        return f"sec:locus:{item['locus_id']}"
    if item.get("contig") or item.get("scaffold"):
        contig = item.get("contig") or item.get("scaffold")
        return f"sec:contig:{contig}"
    kos = item.get("kos") or []
    pfs = item.get("pfams") or []
    if isinstance(kos, list) and kos:
        return f"sec:ko:{str(kos[0])}"
    if isinstance(pfs, list) and pfs:
        return f"sec:pfam:{str(pfs[0])}"
    # Try capability hint when present
    cap = item.get("_capability")
    if isinstance(cap, str) and cap:
        return f"sec:marker:{cap}"
    return "sec:community:default"


def group_by_anchor(items: Iterable[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for it in items:
        a = assign_anchor(it)
        out.setdefault(a, []).append(it)
    return out
