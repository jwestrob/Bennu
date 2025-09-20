# KG Build Components

Overview of Stage 07 build modules.

## RDF Builder (`src/build_kg/rdf_builder.py`)

- Constructs RDF triples for all entities and relationships.
- Emits `(:Gene)-[:NEXT]->(:Gene)` edges and per‑gene degree properties in the RDF so they are materialized into CSVs.
- Integrates BGC and CAZy when their stage outputs are present.

## RDF→CSV Converter (`src/build_kg/rdf_to_csv_converter.py`)

- Maps RDF nodes to stable node CSVs (see Data → CSV Layout).
- Groups relationships into `*_relationships.csv` with `:START_ID,:END_ID` header.
- Preserves namespace prefixes for node IDs (e.g., `protein:...`, `gene:...`) to avoid collisions.

## Bulk Loader (`src/build_kg/neo4j_bulk_loader.py`)

- Validates CSVs, runs `neo4j-admin database import full`, and starts Neo4j (Docker engine by default).
- Applies constraints/indexes after import.
- Writes a simple connection descriptor to `data/neo4j/connection.json`.

## Post‑Load Tuning (`src/build_kg/postload_tuning.py`)

- Optional: re‑apply constraints/indexes and recompute `[:NEXT]`, `nextDegree`, `genesOnContig` for attached databases.

