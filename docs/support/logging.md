# Logging & Expected Messages

## Neighborhoods
- `NeighborhoodContext: filtered X degree-zero seeds; using Y seeds` — degree-aware filter summary.

## Stage 07 Import
- CSV conversion messages indicating node and relationship counts per file.
- Bulk import messages from `neo4j-admin` followed by index creation notices.

## CAZy Integration
- `Synthesized CAZyme JSON ...` — when fallback converts `overview.tsv` into JSON artifacts.
- `WARNING  CAZyme manifest not found ...` — older runs; addressed by the CLI fix.

