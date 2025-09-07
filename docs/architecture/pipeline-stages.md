# Pipeline Stages

High-level overview with implementation notes reflecting the current codebase.

## 0. Input Preparation (`src/ingest/00_prepare_inputs.py`)
- Validates FASTA inputs and organizes per-genome directories.

## 1. QUAST Quality Assessment (`src/ingest/01_run_quast.py`)
- Runs QUAST; parses summary metrics for later provenance.

## 2. DFAST_QC Taxonomy (`src/ingest/02_dfast_qc.py`)
- Optional via `--skip-tax`. Produces CheckM/ANI metrics when enabled.

## 3. Prodigal Gene Prediction (`src/ingest/03_prodigal.py`)
- Produces ORFs and protein FASTA; nucleotide sequences optional.

## 4. Functional Annotation (`src/ingest/04_astra_scan.py`)
- PFAM and KOFAM detection with score cutoffs. Outputs drive Stage 07 function nodes.

## 5. BGC Detection (`src/ingest/gecco_bgc.py`)
- GECCO outputs mapped into BGC nodes/edges when present.

## 6. CAZy Annotations (`src/ingest/dbcan_cazyme.py`)
- dbCAN runs over `.faa` inputs with DIAMOND. The CLI persists JSON artifacts:
  - `<genome>_cazyme_results.json`
  - `dbcan_summary.json`
  - `processing_manifest.json`
- A synthesis fallback converts `overview.tsv` to JSON if only tabular outputs exist.

## 7. Knowledge Graph Construction (`src/build_kg/rdf_builder.py` → `rdf_to_csv_converter.py` → `neo4j_bulk_loader.py`)
- Builds RDF, converts to CSV with precomputed `[:NEXT]` and `Gene.nextDegree`/`genesOnContig` embedded.
- Bulk imports with `neo4j-admin`. Constraints and indexes are created after import.
- Stage 07 no longer performs functional enrichment from reference files; only Stage 04 outputs are used.

## 8. ESM2 Embeddings (`src/ingest/06_esm2_embeddings.py`)
- Generates per-protein embeddings and writes LanceDB artifacts for vector search.

