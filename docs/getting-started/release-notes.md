# Release Notes — 2025‑09‑06

Highlights:
- Stage 07 simplified and deterministic (no enrichment); precomputes `[:NEXT]`, `nextDegree`, and `genesOnContig` in CSVs.
- Bulk import uses `neo4j-admin` with Docker path defaulting to no‑auth.
- Planner and plan validation tightened: `AnnotationDiscovery.keyword` is required; seeds must be explicit for neighborhoods.
- dbCAN CLI path now persists JSON artifacts; Stage 07 ingests CAZy when present.
- Tool calls are captured to `data/session_notes/<sid>/synthesis_notes/tool_calls.json`.

