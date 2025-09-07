# Architecture Overview

The platform converts microbial genome assemblies into a Neo4j-powered knowledge graph and layers a modular, deterministic RAG system on top for biologically grounded Q&A.

## Pipeline Summary

- Stage 0–3: Input prep, QUAST, DFAST_QC (optional), Prodigal.
- Stage 4: Functional annotations (PFAM, KOFAM) via Astra/PyHMMER.
- Stage 5: GECCO BGC detection.
- Stage 6: dbCAN CAZy annotations (JSON artifacts produced under `data/stage06_dbcan`).
- Stage 7: RDF/CSV build, `[:NEXT]` and gene degrees precomputed, bulk import via `neo4j-admin`.
- Stage 8: ESM2 embeddings for semantic search (LanceDB).

## Knowledge Graph

- Core: `Genome` → `Gene` → `Protein` with provenance and quality metrics.
- Function: PFAM Domains, KOs, KEGG Pathways.
- Structure: `(:Gene)-[:NEXT]->(:Gene)` directed edges (treat as undirected for degree), with `Gene.nextDegree` and `Gene.genesOnContig` properties.
- Optional: BGCs (GECCO) and CAZymes (dbCAN) when available.

## RAG System (LLM Layer)

- Modular agents (planner → executor → synthesizer) with strict plan validation.
- Key operators:
  - AnnotationDiscovery: catalog keyword → candidate IDs → bound rowsets.
  - NeighborhoodContext: degree-aware seed filtering; adjacency or flanking windows without APOC.
- Tool-call capture and session notes persisted for auditability.

