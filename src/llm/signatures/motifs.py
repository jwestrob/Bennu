from __future__ import annotations
from typing import Iterable, Set, Dict, Any


def any_in(set_ids: Set[str], present: Set[str]) -> bool:
    return bool(set_ids & present)


def count_domain_sets_at_least(domain_sets: Iterable[Set[str]], present: Set[str], m: int) -> bool:
    hits = sum(1 for ds in domain_sets if any_in(ds, present))
    return hits >= m


def evaluate_motif(motif_spec: Dict[str, Any], present_pfam: Set[str], present_kegg: Set[str], domain_sets: Dict[str, Dict[str, Dict[str, list]]]) -> bool:
    all_present = present_pfam | present_kegg

    def resolve(ds_key: str) -> Set[str]:
        group, name = ds_key.split(".")
        group_map = domain_sets.get(group, {})
        entry = group_map.get(name, {})
        pf = set((entry.get("pfam") or []))
        kk = set((entry.get("kegg") or []))
        return {str(x).lower() for x in (pf | kk)}

    if "all_any" in motif_spec:
        for elem in motif_spec["all_any"]:
            ds = resolve(elem["domain_set"])
            if not any_in(ds, {x.lower() for x in all_present}):
                return False
        return True

    if "any_any" in motif_spec:
        return any(any_in(resolve(elem["domain_set"]), {x.lower() for x in all_present}) for elem in motif_spec["any_any"])

    if "any_at_least" in motif_spec:
        m = int(motif_spec.get("count", 1))
        dsets = [resolve(e["domain_set"]) for e in motif_spec["any_at_least"]]
        return count_domain_sets_at_least(dsets, {x.lower() for x in all_present}, m)

    raise ValueError(f"Unsupported motif spec: {motif_spec}")

