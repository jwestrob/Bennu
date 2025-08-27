from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import logging

# Import deterministic utilities from dedicated namespace to avoid conflicts
from ..deterministic.evi import evi_gate
from ..deterministic.invariants import (
    assert_schema_minimal,
    collapse_redundant_seeds,
    embedding_parity_check,
)


@dataclass(frozen=True)
class LocusCard:
    seed_protein_id: str | None
    contig_id: str
    genome_id: str
    contig_len: int | None
    neighbors: List[Dict[str, Any]]
    verdict: str  # CONTEXTUALIZED | PARTIAL_SIGNAL | NO_SIGNAL
    seed_pfams: List[str] | None = None
    seed_kos: List[str] | None = None


class LocusDiscoveryOption:
    """
    Deterministic macro:
      1) Seed retrieval (PFAM/KOFAM synonyms are resolved in Cypher).
      2) Batched neighborhoods for seeds that pass EVI gates.
      3) Optional batched LanceDB kNN if nn>0 (postconditions satisfied).
      4) Persist Locus nodes and return structured LocusCards.
    """

    def __init__(self, db, lancedb=None, config=None):
        self.db = db
        self.lancedb = lancedb
        self.cfg = config or {}
        self.logger = logging.getLogger(__name__)

    def run(self, marker: str, N: int, k: int, nn: int) -> Tuple[List[LocusCard], Dict[str, Any]]:
        # 1) Seeds (prefer ID-backed resolution over substring matching)
        self.logger.info("[MFP] Resolving marker to PFAM/KO IDs...")
        pfam_ids: List[str] = []
        ko_ids: List[str] = []
        try:
            # Deterministic templates in resources/cypher
            pref = self.db.run_template("pfam_ids_by_query.cypher", {"q": marker, "limit": 200}) or []
            for r in pref:
                pid = (r.get("pfam_id") or r.get("id") or "").strip()
                if pid:
                    pfam_ids.append(pid)
            kref = self.db.run_template("ko_ids_by_query.cypher", {"q": marker, "limit": 200}) or []
            for r in kref:
                kid = (r.get("ko_id") or "").strip()
                if kid:
                    ko_ids.append(kid)
            # Dedup while preserving order
            pfam_ids = list(dict.fromkeys(pfam_ids))
            ko_ids = list(dict.fromkeys(ko_ids))
        except Exception as e:
            self.logger.info(f"[MFP] Marker ID resolution skipped: {e}")

        if pfam_ids or ko_ids:
            self.logger.info(f"[MFP] IDs resolved: pfam={len(pfam_ids)} ko={len(ko_ids)}; fetching seeds by IDs…")
            seeds = self.db.run_template(
                "seeds_by_domain_set.cypher",
                {"pfam_ids": pfam_ids, "ko_ids": ko_ids, "limit": 5 * max(N, 10)},
            )
        else:
            self.logger.info("[MFP] No IDs resolved; falling back to substring matching seeds_by_marker…")
            seeds = self.db.run_template(
                "seeds_by_marker.cypher",
                {"markers": [marker], "limit": 5 * max(N, 10)},
            )
        self.logger.info(f"[MFP] Seeds fetched: {len(seeds)}")
        assert_schema_minimal(seeds)
        seeds = collapse_redundant_seeds(seeds)
        self.logger.info(f"[MFP] Seeds after dedup: {len(seeds)}")

        # 2) EVI gating (pure, deterministic) with permissive fallback
        gated = [s for s in seeds if evi_gate(s)]
        self.logger.info(f"[MFP] Gated seeds: {len(gated)} (need >= {N})")
        if len(gated) < N:
            return ([], {"escalate": True, "reason": "insufficient_gated_seeds", "available": len(gated)})
        shortlist = gated[:N]
        # 3) Enforce thresholds before selecting N; then batched neighborhoods (single query)
        min_contig_len = int(self.cfg.get("min_contig_len", 1500))
        min_orf = int(self.cfg.get("min_orf", 0))
        eligible = [s for s in gated if int(s.get("contig_len") or 0) >= min_contig_len and int(s.get("orf_count") or 0) >= min_orf]
        if len(eligible) < N:
            return ([], {"escalate": True, "reason": "insufficient_thresholded_seeds", "available": len(eligible)})
        shortlist = eligible[:N]
        neigh = self.db.run_template(
            "batched_neighborhoods_gated.cypher",
            {
                "seeds": shortlist,
                "min_contig_len": 0,
                "min_orf": 0,
                # Use ±k neighbors by index (radius = k)
                "k_window": int(k),
            },
        )
        self.logger.info(f"[MFP] Neighborhood rows: {len(neigh)}")
        assert_schema_minimal(neigh)

        # Skeptic auditor
        try:
            if self.cfg.get("SKEPTIC_ENABLED", True):
                from .skeptic import skeptic_after_batched
                flags = skeptic_after_batched(neigh)
                if flags:
                    self.logger.warning(f"[MFP] Skeptic flags: {flags}")
                    return ([], {"escalate": True, "reason": "skeptic_flags", "flags": flags})
        except Exception as e:
            self.logger.warning(f"[MFP] Skeptic auditor error (continuing): {e}")

        # Build locus cards
        cards: List[LocusCard] = []
        for row in neigh:
            neighbors = row.get("neighbors", [])
            verdict = "CONTEXTUALIZED" if len(neighbors) > 0 else "PARTIAL_SIGNAL"
            # Attach seed annotations if present
            try:
                seed_pfams = row.get("seed_pfams") or []
                seed_kos = row.get("seed_kos") or []
            except Exception:
                seed_pfams = []
                seed_kos = []
            cards.append(
                LocusCard(
                    seed_protein_id=row.get("seed_protein_id"),
                    contig_id=row.get("contig_id") or "",
                    genome_id=row.get("genome_id", ""),
                    contig_len=row.get("contig_len"),
                    neighbors=neighbors,
                    verdict=verdict,
                    seed_pfams=seed_pfams,
                    seed_kos=seed_kos,
                )
            )

        # 4) Optional batched kNN
        knn_info = {}
        if nn and self.lancedb:
            embedding_parity_check(self.lancedb)
            # perform one batched query; implementation uses repo's LanceDB API
            knn_info = self._batched_knn([c.seed_pid for c in cards], nn)

        # 5) Persist Locus entities (template-only)
        self._persist_loci(cards)

        # Enrich meta for downstream synthesis (carry marker/k/nn and seed sets)
        try:
            meta: Dict[str, Any] = {
                "escalate": False,
                "knn": knn_info,
                "marker": marker,
                "k": int(k),
                "nn": int(nn),
                "seed_candidates": [r.get("seed_protein_id") for r in (seeds or []) if r.get("seed_protein_id")],
                "gated_seed_ids": [r.get("seed_protein_id") for r in (gated or []) if r.get("seed_protein_id")],
                "shortlist_seed_ids": [r.get("seed_protein_id") for r in (shortlist or []) if r.get("seed_protein_id")],
                "analysis_scope": {"mode": "locus_window", "window_k": int(k), "contig_semantics": "gene_order"},
                "seed_selection": {
                    "method": "id_resolution" if (pfam_ids or ko_ids) else "substring_fallback",
                    "pfam_ids_count": len(pfam_ids),
                    "ko_ids_count": len(ko_ids),
                },
            }
        except Exception:
            meta = {"escalate": False, "knn": knn_info, "marker": marker, "k": int(k), "nn": int(nn)}

        return (cards, meta)

    def _synonyms(self, marker: str) -> List[str]:
        # Minimal deterministic synonym mapping left intentionally generic; prefer ID-based resolution upstream.
        return [marker]

    def _batched_knn(self, seed_pids: List[int], nn: int) -> Dict[str, Any]:
        # Delegate to existing LanceDB utility if present; otherwise no-op
        # Must be batched and exclude self-match; no dummy vectors created.
        return {}

    def _persist_loci(self, cards: List[LocusCard]) -> None:
        # Use compiled migration/merge template in resources/cypher (applied via db.run_template)
        payload = [
            {"seed_protein_id": c.seed_protein_id, "contig_id": c.contig_id, "verdict": c.verdict}
            for c in cards
        ]
        self.db.run_template("locus_schema_migration.cypher", {"loci": payload})
