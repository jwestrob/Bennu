# CAZy Integration (dbCAN) — Status 2025‑09‑06

## Current Behavior

- Stage 06 runs dbCAN and produces tabular outputs per genome (e.g., `overview.tsv`, `uniInput.faa`).
- The CLI now persists JSON artifacts expected by Stage 07 under `data/stage06_dbcan/`:
  - `processing_manifest.json`
  - `dbcan_summary.json`
  - `<genome>_cazyme_results.json`

## Implementation Notes

- `src/ingest/dbcan_cazyme.py` contains `save_results(...)` and `create_processing_manifest(...)`.
- The pipeline CLI (Stage 06 in `src/cli.py`) calls these after `run_dbcan_batch_analysis(...)` to ensure JSON is present.
- A synthesis fallback converts existing `overview.tsv` to JSON when `.faa` inputs are absent or when only tabular outputs are available.

## Verify in Neo4j

- Global:
  - `MATCH (p:Protein)-[:HASCAZYME]->(:Cazymeannotation)-[:CAZYMEFAMILY]->(:Cazymefamily) RETURN count(p) AS proteins`.
- Per‑genome:
  - `MATCH (g:Genome)<-[:BELONGSTOGENOME]-(:Gene)<-[:ENCODEDBY]-(p:Protein)-[:HASCAZYME]->(:Cazymeannotation) RETURN g.id, count(p) ORDER BY count(p) DESC`.

## JSON Artifacts

- Per‑genome `<genome>_cazyme_results.json` structure (fields abbreviated):
  - `genome_id`: string
  - `total_proteins`: int
  - `cazyme_proteins`: int
  - `family_counts`: {family → count}
  - `annotations`: list of CAZymeAnnotation records:
    - `protein_id`, `cazyme_family`, `family_type`, `evalue`, `coverage`, `start_pos`, `end_pos`, `hmm_length`, optional `substrate_prediction`, optional `ec_number`
  - `processing_time`: float

