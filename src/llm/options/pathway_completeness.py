from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Iterable
import logging

from .template_runner import FileCypherRunner
from ..kegg.pathway_mapping import load_ko_pathway_maps, filter_pathway_ids


@dataclass(frozen=True)
class PathwayCompletenessRow:
    genome_id: str
    pathway_id: str
    pathway_name: str
    present_kos: int
    total_kos: int
    completeness: float
    missing_ko_ids: List[str]
    present_ko_ids: List[str]


class PathwayCompletenessOption:
    """Deterministic KEGG pathway completeness computation by genome.

    Uses a single Cypher template to compute per-(genome, pathway):
      - present KOs, total KOs, completeness ratio, missing KO IDs, present KO IDs
    """

    def __init__(self, db: FileCypherRunner):
        self.db = db
        self.logger = logging.getLogger(__name__)

    def run(
        self,
        genome_ids: Optional[List[str]] = None,
        min_completeness: Optional[float] = None,
        pathways: Optional[List[str]] = None,
        use_native_totals: Optional[bool] = None,
    ) -> Tuple[List[PathwayCompletenessRow], Dict[str, Any]]:
        """Compute pathway completeness per genome.

        If use_native_totals is True (default via env flag), compute totals from ko_pathway.list
        and present KOs via a single Cypher. Otherwise, fall back to DB template which may
        compute totals from graph edges.
        """
        # Decide execution mode
        if use_native_totals is None:
            import os as _os
            use_native_totals = _os.getenv("USE_NATIVE_TOTALS_FOR_PATHWAYS", "1").lower() in ("1", "true", "yes")

        if use_native_totals:
            return self._run_native(genome_ids, min_completeness, pathways)
        else:
            return self._run_via_db(genome_ids, min_completeness, pathways)

    def _run_via_db(
        self,
        genome_ids: Optional[List[str]],
        min_completeness: Optional[float],
        pathways: Optional[List[str]],
    ) -> Tuple[List[PathwayCompletenessRow], Dict[str, Any]]:
        params: Dict[str, Any] = {
            "genome_ids": list(genome_ids) if genome_ids else [],
            "min_completeness": float(min_completeness) if isinstance(min_completeness, (int, float)) else None,
        }
        rows = self.db.run_template("pathway_completeness_by_genome.cypher", params) or []
        out: List[PathwayCompletenessRow] = []
        totals_by_genome: Dict[str, int] = {}
        complete_by_genome: Dict[str, int] = {}

        # Optional filtering by pathways (IDs expected like map00010)
        allowed: Optional[set] = None
        if pathways:
            allowed = {p.split(":")[-1] for p in pathways if isinstance(p, str)}

        for r in rows:
            try:
                pid = str(r.get("pathway_id"))
                if allowed is not None and pid not in allowed:
                    continue
                row = PathwayCompletenessRow(
                    genome_id=str(r.get("genome_id")),
                    pathway_id=pid,
                    pathway_name=str(r.get("pathway_name")),
                    present_kos=int(r.get("present_kos") or 0),
                    total_kos=int(r.get("total_kos") or 0),
                    completeness=float(r.get("completeness") or 0.0),
                    missing_ko_ids=list(r.get("missing_ko_ids") or []),
                    present_ko_ids=list(r.get("present_ko_ids") or []),
                )
                out.append(row)
                totals_by_genome[row.genome_id] = 1 + totals_by_genome.get(row.genome_id, 0)
                if row.total_kos > 0 and abs(row.completeness - 1.0) < 1e-9:
                    complete_by_genome[row.genome_id] = 1 + complete_by_genome.get(row.genome_id, 0)
            except Exception:
                continue
        # Sort for stability
        out.sort(key=lambda r: (r.genome_id, -float(r.completeness), -int(r.present_kos), r.pathway_id))
        meta = {
            "genome_scope": list(genome_ids) if genome_ids else "ALL",
            "min_completeness": float(min_completeness) if isinstance(min_completeness, (int, float)) else None,
            "pathway_scope": list(allowed) if allowed is not None else "ALL",
            "pathways_evaluated": len(out),
            "complete_counts": complete_by_genome,
            "totals_source": "db_graph",
        }
        return out, meta

    def _run_native(
        self,
        genome_ids: Optional[List[str]],
        min_completeness: Optional[float],
        pathways: Optional[List[str]],
    ) -> Tuple[List[PathwayCompletenessRow], Dict[str, Any]]:
        # Step 1: get present KO sets per genome
        params: Dict[str, Any] = {"genome_ids": list(genome_ids) if genome_ids else []}
        rr = self.db.run_template("present_kos_by_genome.cypher", params) or []
        # Early-out if nothing present
        if not rr:
            return [], {
                "genome_scope": list(genome_ids) if genome_ids else "ALL",
                "min_completeness": float(min_completeness) if isinstance(min_completeness, (int, float)) else None,
                "pathway_scope": list(pathways) if pathways else "ALL",
                "pathways_evaluated": 0,
                "complete_counts": {},
                "totals_source": "native_ko_pathway.list",
            }

        # Step 2: load KO->Pathway mapping
        pw_to_kos, _ko_to_pw = load_ko_pathway_maps()
        # Build pathway filter
        allowed_ids = filter_pathway_ids(pw_to_kos.keys(), pathways)
        # If user requested unknown IDs, allowed_ids may be empty; that's acceptable.

        # Step 3: compute completeness
        out: List[PathwayCompletenessRow] = []
        totals_by_genome: Dict[str, int] = {}
        complete_by_genome: Dict[str, int] = {}
        min_c = float(min_completeness) if isinstance(min_completeness, (int, float)) else None

        def _norm_ko(x: str) -> str:
            if not isinstance(x, str):
                return ""
            s = x.strip()
            if s.startswith("ko:"):
                s = s[3:]
            return s

        for rec in rr:
            gid = str(rec.get("genome_id"))
            present_kos_list = rec.get("present_ko_ids") or []
            present_set = {_norm_ko(k) for k in present_kos_list if isinstance(k, str)}
            # Choose pathway universe
            pids: Iterable[str] = allowed_ids if pathways else pw_to_kos.keys()
            count_for_gid = 0
            complete_for_gid = 0
            for pid in pids:
                all_kos = pw_to_kos.get(pid, set())
                tot = len(all_kos)
                if tot == 0:
                    continue
                pres_set = present_set & all_kos
                pc = len(pres_set)
                comp = float(pc) / float(tot) if tot > 0 else 0.0
                if min_c is not None and comp < min_c:
                    continue
                row = PathwayCompletenessRow(
                    genome_id=gid,
                    pathway_id=pid,
                    pathway_name=pid,  # Name lookup optional; default to ID
                    present_kos=pc,
                    total_kos=tot,
                    completeness=comp,
                    missing_ko_ids=sorted(list(all_kos - pres_set)),
                    present_ko_ids=sorted(list(pres_set)),
                )
                out.append(row)
                count_for_gid += 1
                if abs(comp - 1.0) < 1e-9:
                    complete_for_gid += 1
            if count_for_gid:
                totals_by_genome[gid] = count_for_gid
                if complete_for_gid:
                    complete_by_genome[gid] = complete_for_gid

        # Stable sort
        out.sort(key=lambda r: (r.genome_id, -float(r.completeness), -int(r.present_kos), r.pathway_id))
        meta = {
            "genome_scope": list(genome_ids) if genome_ids else "ALL",
            "min_completeness": min_c,
            "pathway_scope": list(allowed_ids) if pathways else "ALL",
            "pathways_evaluated": len(out),
            "complete_counts": complete_by_genome,
            "totals_source": "native_ko_pathway.list",
        }
        return out, meta
