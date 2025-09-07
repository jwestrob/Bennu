# Microbial Claude Matter — Genomic AI Platform

Transform microbial genome assemblies into an intelligent, queryable knowledge graph with LLM-powered biological insights.

## Quick Start

- Build Stage 07 only (RDF→CSV, NEXT edges, degrees, bulk import):
  - `python -m src.cli build -f 7 -t 7 --force`
  - Outputs: `data/stage07_kg/knowledge_graph.ttl` and `data/stage07_kg/csv/*`, then imports via `neo4j-admin`.

- Default Neo4j connection (Docker):
  - URI `bolt://localhost:7687`; auth none (`NEO4J_AUTH=none`).
  - With creds: set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` env vars.

## What’s New (2025‑09‑06)

- Stage 07 is deterministic and focused: loads Stage 04 PFAM/KO + core graph (Genome→Gene→Protein), BGC/CAZy when available, precomputes `[:NEXT]`, `Gene.nextDegree`, `Gene.genesOnContig` in CSVs.
- Bulk import path via `neo4j-admin` has no post-load fixes; constraints and helpful indexes are applied by default.
- Neighborhood operators run APOC‑free with degree-aware seed filtering (exclude contig‑isolated seeds by default).
- Planner tightened: `AnnotationDiscovery` must set `params.keyword` (or `q`) and IDs are wired via `inputs`.

## Common Tasks

- See Operations → Stage 07 Build & Import for the end‑to‑end flow and logs to expect.
- See Operations → Diagnostics for checking `[:NEXT]` counts, degree histograms, adjacency/flanking examples.
- See Operations → Neo4j Schema for labels/properties/relationships and helpful Cypher queries.

## Status Checkpoints

- After import, expect ≈10.48M triples converted, `next_relationships.csv` loaded, and `Gene.nextDegree` populated with no UnknownPropertyKey warnings.
- Tool-call capture for agent runs is written under `data/session_notes/<sid>/synthesis_notes/tool_calls.json`.

## Known Pitfalls

- Highly fragmented assemblies yield many degree‑0 seeds; default filtering reduces noise. Use `include_degree_zero_seeds=true` to include.
- Attaching to an older database lacking `nextDegree` will still work (live count fallback), but Neo4j may warn about the unfamiliar property key until a rebuild or tuning step sets it.

