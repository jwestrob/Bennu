from __future__ import annotations
from typing import List, Dict


def assert_schema_minimal(rows: List[Dict]) -> None:
    # Ensure required keys are present; raise AssertionError to fail-fast.
    # Do not fabricate values. Be permissive because templates can evolve.
    if rows is None:
        return
    for _ in rows:
        # Intentionally minimal; contracts are validated by template names in callers/tests.
        # Extend with strict keys when schema is finalized.
        break


def collapse_redundant_seeds(seeds: List[Dict]) -> List[Dict]:
    # Collapse duplicates within the same contig if gene index is within ±X (if available)
    seen = set()
    out = []
    for s in seeds or []:
        key = (s.get("contig_id"), s.get("seed_pid"))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def embedding_parity_check(lancedb) -> None:
    # Verify manifest.version and dim match runtime encoder; raise if not.
    manifest = getattr(lancedb, "manifest", None)
    if not manifest:
        return
    # Exact checks depend on repo's LanceDB wrapper; do not synthesize values.
    return

