from __future__ import annotations

import re
from typing import Optional, Tuple, Dict
import os
try:
    from .policy_engine import get_policy_engine  # type: ignore
except Exception:  # pragma: no cover
    get_policy_engine = None  # type: ignore


def map_question_to_template(question: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """Heuristically map a natural language question to a named DB template.

    Returns (template_name, slots) or None if no mapping applies.
    Patterns are intentionally conservative to avoid false positives.
    """
    q = question or ""
    # Protein by ID: explicit protein:<id>
    m = re.search(r"\bprotein:([A-Za-z0-9:_\-\.]+)\b", q)
    if m:
        pid = m.group(0)  # already includes 'protein:' prefix
        return "protein_by_id", {"id": pid}

    # Count KO proteins: e.g., "count K20469 proteins"
    m = re.search(r"\bcount\s+K(\d{5})\b", q)
    if m:
        return "count_proteins_with_ko", {"ko": f"K{m.group(1)}"}

    # KEGG KO: Kxxxxx
    m = re.search(r"\bK(\d{5})\b", q)
    if m:
        return "proteins_with_ko", {"ko": f"K{m.group(1)}", "limit": _default_limit()}

    # CAZy: GH/PL/CE digits
    m = re.search(r"\b(GH|PL|CE)(\d+)\b", q, re.IGNORECASE)
    if m:
        fam = f"{m.group(1).upper()}{m.group(2)}"
        return "cazy_family", {"family": fam, "limit": _default_limit()}

    # Generic count proteins
    if re.search(r"\bcount\s+proteins\b", q):
        return "count_by_label", {"label": "Protein"}

    # KEGG pathway map: mapxxxxx
    m = re.search(r"\bmap(\d{5})\b", q, re.IGNORECASE)
    if m:
        return "pathway_membership", {"pathway": f"map{m.group(1)}", "limit": _default_limit()}

    # PFAM accession PFxxxxx (5 digits)
    m = re.search(r"\bpf\d{5}\b", q, re.IGNORECASE)
    if m and not re.search(r"\bcount\s+pf\d{5}\b", q, re.IGNORECASE):
        return "proteins_with_pfam", {"pfam": m.group(0).upper(), "limit": _default_limit()}

    # Genome by ID: explicit genome:<id>
    m = re.search(r"\bgenome:([A-Za-z0-9:_\-\.]+)\b", q)
    if m:
        gid = m.group(0)  # already includes 'genome:' prefix
        return "proteins_by_genome", {"genome_id": gid, "limit": _default_limit()}

    # Genes on contig: explicit contig:<id>
    m = re.search(r"\bcontig:([A-Za-z0-9:_\-\.]+)\b", q)
    if m:
        return "genes_on_contig", {"contig": m.group(0), "limit": _default_limit()}

    return None


def _default_limit() -> int:
    # Prefer policy engine if available
    pe_limit = None
    if get_policy_engine is not None:
        try:
            pe = get_policy_engine()
            pe_limit = int(pe.get_max_results("database_query"))
        except Exception:
            pe_limit = None
    if pe_limit is not None:
        val = pe_limit
    else:
        try:
            val = int(os.getenv("AGENT_DEFAULT_DB_LIMIT", "100"))
        except Exception:
            val = 100
    return max(1, min(val, 5000))
