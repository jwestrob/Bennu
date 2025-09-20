# Bulk Import Details

How the CSVs produced in Stage 07 are imported into Neo4j using `neo4j-admin`.

## Engines

- `docker` (default): uses image `neo4j:5` with `NEO4J_AUTH=none`.
- `system`: uses a locally installed `neo4j-admin` (Homebrew/apt).

## CSV Detection

- Node CSVs: any `*.csv` file that does not include `relationships` in its name.
- Relationship CSVs: `*_relationships.csv`, with header `:START_ID,:END_ID`.

## Node Label Mapping

- Canonical overrides for bulk import label naming:
  - `domainannotations.csv` → `DomainAnnotation`
  - `functionalannotations.csv` → `FunctionalAnnotation`
  - `keggorthologs.csv` → `KEGGOrtholog`
  - `qualitymetrics.csv` → `QualityMetrics`
  - `bgcs.csv` or `bgc_clusters.csv` → `Bgc`
  - `cazymeannotations.csv` → `Cazymeannotation`
  - `cazymefamilies.csv` → `Cazymefamily`

All other filenames are converted from their stem (singular, title‑cased), e.g., `proteins.csv` → `Protein`.

## Relationship Type Mapping

- Relationship filenames are converted to uppercase type names by stripping `_relationships.csv`, e.g., `next_relationships.csv` → `NEXT`.

## Post‑Import Steps

- Constraints and indexes are created by default (authless supported). See Operations → Neo4j Schema for details.
- A `data/neo4j/connection.json` is emitted for zero‑config clients (`{"uri":"bolt://localhost:7687","auth":null}`).

## Optional Post‑Load Tuning

- You can run `src/build_kg/postload_tuning.py` to recompute `[:NEXT]`, `nextDegree`, or re‑apply indexes if you attach to an older DB.

