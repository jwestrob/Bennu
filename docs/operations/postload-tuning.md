#!/usr/bin/env md
# Post‑Load Tuning (Optional)

When attaching to an existing Neo4j instead of rebuilding via Stage 07, you can apply constraints/indexes, precompute `[:NEXT]` edges, and compute `Gene.nextDegree`/`genesOnContig`.

## Commands

- Create constraints/indexes, precompute NEXT, compute degrees:
  - `python -m src.build_kg.postload_tuning`

- Neighbors only:
  - `python -m src.build_kg.postload_tuning --neighbors-only`

- Degrees only:
  - `python -m src.build_kg.postload_tuning --compute-degrees-only`

- Limit contigs (quick test):
  - `python -m src.build_kg.postload_tuning --contig-limit 50`

Environment:
- Set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` when connecting to an authenticated instance.

