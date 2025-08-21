from __future__ import annotations

import re
from typing import Optional, Tuple, Dict


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

    # KEGG KO: Kxxxxx
    m = re.search(r"\bK(\d{5})\b", q)
    if m:
        return "proteins_with_ko", {"ko": f"K{m.group(1)}"}

    # CAZy: GH/PL/CE digits
    m = re.search(r"\b(GH|PL|CE)(\d+)\b", q, re.IGNORECASE)
    if m:
        fam = f"{m.group(1).upper()}{m.group(2)}"
        return "cazy_family", {"family": fam}

    # KEGG pathway map: mapxxxxx
    m = re.search(r"\bmap(\d{5})\b", q, re.IGNORECASE)
    if m:
        return "pathway_membership", {"pathway": f"map{m.group(1)}"}

    return None

