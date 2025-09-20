# Stage 06 — dbCAN CAZy

CAZyme annotations via dbCAN (DIAMOND mode). The CLI persists JSON artifacts required by Stage 07.

Entry: `src/ingest/dbcan_cazyme.py`

Artifacts under `data/stage06_dbcan/`:
- `processing_manifest.json`
- `dbcan_summary.json`
- `<genome>_cazyme_results.json`

Fallback: if only `overview.tsv` files exist (external runs), the module synthesizes JSON before returning.

