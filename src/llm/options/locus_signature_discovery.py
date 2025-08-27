from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Any
import logging

from .locus_discovery import LocusCard
from ..signatures.registry import SignatureRegistry
from ..signatures.motifs import evaluate_motif


logger = logging.getLogger(__name__)


def _gather_ids(resolved_sets: Dict[str, Dict[str, Dict[str, list]]], group: str, names: List[str]) -> Tuple[List[str], List[str]]:
    pf, kk = set(), set()
    grp = resolved_sets.get(group, {})
    for nm in names:
        entry = grp.get(nm, {})
        pf.update((entry.get("pfam") or []))
        kk.update((entry.get("kegg") or []))
    return [str(x).lower() for x in pf], [str(x).lower() for x in kk]


def find_loci_by_signature(signature_name: str, n: int, flank_k: int, db_runner, registry: SignatureRegistry) -> Tuple[List[LocusCard], Dict[str, Any]]:
    spec = registry.get(signature_name)

    # Resolve all domain sets dynamically from anchors
    resolved_sets = registry.resolve_all(db_runner)

    # Build seeds from resolved sets: integration, structural, lysis at minimum
    seed_groups = {
        "integration": ["integration"],
        "structural": ["structural_capsid", "structural_portal", "terminase", "tail"],
        "lysis": ["lysis_holin", "lysis_endolysin"],
    }

    seeds: List[Dict[str, Any]] = []
    for label, names in seed_groups.items():
        pfam_ids, ko_ids = _gather_ids(resolved_sets, "phage", names)
        if not pfam_ids and not ko_ids:
            # Fail fast: anchors likely too narrow; ask user to broaden signature anchors
            raise RuntimeError(
                f"No domain IDs resolved for seed group '{label}'. Refine anchors in config/signatures/prophage.yml (pfam_query/kegg_query)."
            )
        rows = db_runner.run_template(
            "seeds_by_domain_set.cypher",
            {"pfam_ids": [str(x).lower() for x in pfam_ids], "ko_ids": [str(x).lower() for x in ko_ids], "limit": max(50, 5 * n)},
        )
        seeds.extend(rows or [])

    # Deduplicate seeds by protein id
    seen = set()
    uniq_seeds: List[Dict[str, Any]] = []
    for r in seeds:
        pid = r.get("seed_protein_id")
        if pid and pid not in seen:
            seen.add(pid)
            uniq_seeds.append(r)

    if not uniq_seeds:
        raise RuntimeError("No seeds produced; populate phage.* domain sets in config/domain_sets.yml")

    # Neighborhoods ±k (reuse gated template without extra EVI logic)
    neigh = db_runner.run_template(
        "batched_neighborhoods_gated.cypher",
        {
            "seeds": uniq_seeds[: max(5 * n, 50)],
            "min_contig_len": 0,
            "min_orf": 0,
            "k_window": int(flank_k),
        },
    )

    # Evaluate motifs and clauses per neighborhood window
    candidates: List[Tuple[LocusCard, Dict[str, Any]]] = []
    for row in neigh:
        neighbors = row.get("neighbors", []) or []
        # Present domain ids/ko ids across window (include seed annos)
        pfams: Set[str] = set(map(str, row.get("seed_pfams", []) or []))
        kos: Set[str] = set(map(str, row.get("seed_kos", []) or []))
        for nb in neighbors:
            pfams.update(map(str, nb.get("pfams", []) or []))
            kos.update(map(str, nb.get("kos", []) or []))
        motifs_true: Dict[str, bool] = {}
        for mname, mspec in spec.motifs.items():
            motifs_true[mname] = evaluate_motif(mspec, {x.lower() for x in pfams}, {x.lower() for x in kos}, registry.domain_sets)

        satisfied = []
        for idx, clause in enumerate(spec.clauses):
            all_ok = all(motifs_true.get(m, False) for m in clause.get("all_of", []))
            any_list = clause.get("any_of", [])
            any_ok = (not any_list) or any(motifs_true.get(m, False) for m in any_list)
            if all_ok and any_ok:
                satisfied.append(f"Clause{idx+1}")

        if satisfied:
            # Build card and witness
            card = LocusCard(
                seed_protein_id=row.get("seed_protein_id"),
                contig_id=row.get("contig_id") or "",
                genome_id=row.get("genome_id", ""),
                contig_len=row.get("contig_len"),
                neighbors=neighbors,
                verdict="CONTEXTUALIZED",
            )
            witness = {
                "clauses": satisfied,
                "motifs_true": [k for k, v in motifs_true.items() if v],
                "pfams_present": sorted({x.lower() for x in pfams}),
                "kos_present": sorted({x.lower() for x in kos}),
            }
            candidates.append((card, witness))

    # Deduplicate overlapping windows by ORF ids; greedy lexicographic selection
    def window_key(card: LocusCard, wit: Dict[str, Any]) -> tuple:
        # Higher is better: more clauses, more motifs_true; tie-break by smaller neighbor count
        return (
            len(wit.get("clauses", [])),
            len(wit.get("motifs_true", [])),
            -len(card.neighbors or []),
            card.seed_protein_id or "",
        )

    picked: List[Tuple[LocusCard, Dict[str, Any]]] = []
    used_orfs: Set[str] = set()
    for card, wit in sorted(candidates, key=lambda t: window_key(t[0], t[1]), reverse=True):
        # Collect ORF ids in window
        orfs = set()
        if card.seed_protein_id:
            orfs.add(card.seed_protein_id)
        for nb in card.neighbors or []:
            pid = nb.get("protein_id")
            if pid:
                orfs.add(pid)
        # Skip if overlaps with any picked window
        if orfs & used_orfs:
            continue
        picked.append((card, wit))
        used_orfs |= orfs
        if len(picked) >= n:
            break

    cards = [c for c, _ in picked]
    witness_map = {c.seed_protein_id: w for c, w in picked if c.seed_protein_id}
    meta = {
        "escalate": False,
        "signature": spec.name,
        "k": int(flank_k),
        "witness": witness_map,
        "resolved_sets_summary": {k: {kk: {"pfam": len(vv.get("pfam", [])), "kegg": len(vv.get("kegg", []))} for kk, vv in v.items()} for k, v in resolved_sets.items()},
    }

    return cards, meta
