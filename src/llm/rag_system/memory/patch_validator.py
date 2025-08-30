from __future__ import annotations
from typing import Tuple, List, Dict, Any
from .doc_ast import Document
from .patch_types import PatchEnvelope


class PatchValidationResult:
    def __init__(self, ok: bool, reasons: List[str]):
        self.ok, self.reasons = ok, reasons


def _has_test_op(env: PatchEnvelope) -> bool:
    try:
        return any(op.op == "test" for op in env.patch)
    except Exception:
        return False


def validate_patch(env: PatchEnvelope, doc: Document, *, neo4j=None, sql=None, lancedb=None, allow_nli: bool = False) -> PatchValidationResult:
    errors: List[str] = []
    # 0) Schema is already ensured by Pydantic; enforce at least one test op
    if not _has_test_op(env):
        errors.append("Missing test op")

    # 1) Provenance sanity: require keys and non-empty dict
    if not isinstance(env.evidence, dict) or not env.evidence:
        errors.append("Missing evidence block")

    # 2) (Stub) Resolvability checks can be added here using runners
    #    For now, accept as long as keys exist

    # 3) Numeric consistency (stub): accept; full recomputation is planned later

    return PatchValidationResult(ok=(len(errors) == 0), reasons=errors)

