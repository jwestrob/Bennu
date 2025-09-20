# Stage 07 — Build and Import

Stage 07 builds the RDF knowledge graph, converts it to CSV with precomputed genomic adjacency, then bulk imports into Neo4j in one shot.

## Run

- Build just Stage 07:
  - `python -m src.cli build -f 7 -t 7 --force`

## What It Does

- Converts RDF→CSV and writes:
  - `next_relationships.csv` containing `(:Gene)-[:NEXT]->(:Gene)` edges (directed; treat as undirected for degree).
  - `Gene.nextDegree` and `Gene.genesOnContig` properties directly into the Gene CSVs to avoid post-load computation.
- Imports CSVs via `neo4j-admin` (Docker engine by default, no auth).
- Applies constraints and useful indexes after import (authless supported).

## Logs to Expect

- `NeighborhoodContext: filtered X degree-zero seeds; using Y seeds` (degree filter summary).
- No Neo4j UnknownPropertyKey warnings for `nextDegree` (property exists on Gene from import).
- Tool-call capture at `data/session_notes/<sid>/synthesis_notes/tool_calls.json` for audit.

## Performance Notes

- Composite indexes (`:Gene(contig,startCoordinate,endCoordinate)` and `:Gene(contig,startCoordinate)`) speed locus/flanking scans.
- Total graph size from a reference run: ≈10.48M triples.

