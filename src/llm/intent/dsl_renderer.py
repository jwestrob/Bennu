from __future__ import annotations
from typing import List
from .models import CanonicalIntent, TaskFamily, KnnAction, SummarizeAction


def _render_knn_segments(act: KnnAction) -> List[str]:
    segs: List[str] = []
    # Use nn_stage form to comply with grammar: "number closest" with optional filter
    base = f"THEN {int(act.top_k)} closest"
    if act.exclude_pfam:
        # Grammar only supports a single filter per stage; emit one per marker for accumulation
        for i, mk in enumerate(act.exclude_pfam):
            if i == 0:
                segs.append(base + f" NOT ANNOTATED AS {mk} BY PFAM")
            else:
                segs.append(f"THEN NOT ANNOTATED AS {mk} BY PFAM")
    else:
        segs.append(base)
    return segs


def render_to_dsl(plan: CanonicalIntent) -> str:
    parts: List[str] = []
    if plan.task == TaskFamily.FIND_LOCI_BY_MARKER:
        fm = plan.find_by_marker
        if not fm:
            raise ValueError("Canonical plan missing find_by_marker")
        parts.append(f"FIND {int(plan.n)} LOCI WITH {fm.marker} ± {int(fm.flank_k)}")
    elif plan.task == TaskFamily.FIND_LOCI_BY_SIGNATURE:
        fs = plan.find_by_signature
        if not fs:
            raise ValueError("Canonical plan missing find_by_signature")
        # Emit explicit SIGNATURE keyword per grammar extension
        parts.append(f"FIND {int(plan.n)} LOCI WITH {fs.signature_name} SIGNATURE ± {int(fs.flank_k)}")
    else:
        raise ValueError(f"Unsupported primary task: {plan.task}")

    # Actions: only render grammar-supported stages
    for act in plan.actions:
        if isinstance(act, KnnAction):
            parts.extend(_render_knn_segments(act))
        elif isinstance(act, SummarizeAction):
            # Not represented in current grammar; skip in DSL but keep in canonical JSON
            continue
        else:
            raise ValueError(f"Unsupported action type: {type(act)}")

    return " ".join(parts)
