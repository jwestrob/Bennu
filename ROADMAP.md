# Roadmap: Agent + Knowledge Graph Optimization

This roadmap focuses on improving the agent system and knowledge graph (KG) stack: query performance, reliability, and prompt optimization. The build pipeline is mostly stable and out of scope except where it affects KG/agent efficiency.

## Guiding Principles

- Maximize correctness and reproducibility; minimize hallucinations.
- Prefer precomputation and indexes over repeated heavy queries.
- Store large data once and reference by ID; never truncate biological data.
- Optimize prompts with measurable, repeatable offline evaluations.
- Keep signatures generic; no dataset-specific logic.

## Immediate High-Impact Actions

1) Neo4j constraints and indexes for fast queries (integrated into loader) — DONE
2) Precompute `:NEXT` edges via CSV and bulk import for spatial scans — DONE
3) Tight JSON schemas for tool I/O + synthesis guardrails with citations
4) Agent eval harness with small curated task set and metrics
5) DSPy compile loop + GEPA search for key signatures (start with classifier and DB query generator)
6) Vector index tuning + two-stage retrieval (annotation/lexical filter → vector)

## Knowledge Graph Performance Plan

1) Constraints and Indexes (Neo4j 5.x syntax, integrated)
- Uniqueness:
  - `CREATE CONSTRAINT genome_id IF NOT EXISTS FOR (g:Genome) REQUIRE g.id IS UNIQUE;`
  - `CREATE CONSTRAINT genome_genomeId IF NOT EXISTS FOR (g:Genome) REQUIRE g.genomeId IS UNIQUE;`
  - `CREATE CONSTRAINT gene_id IF NOT EXISTS FOR (g:Gene) REQUIRE g.id IS UNIQUE;`
  - `CREATE CONSTRAINT protein_id IF NOT EXISTS FOR (p:Protein) REQUIRE p.id IS UNIQUE;`
  - `CREATE CONSTRAINT pathway_id IF NOT EXISTS FOR (pw:Pathway) REQUIRE pw.id IS UNIQUE;`
  - `CREATE CONSTRAINT bgc_id IF NOT EXISTS FOR (b:Bgc) REQUIRE b.id IS UNIQUE;`
- Composite for spatial queries:
  - `CREATE INDEX gene_contig_coords IF NOT EXISTS FOR (g:Gene) ON (g.contig, g.startCoordinate, g.endCoordinate);`
- Full-text (optional, for name/desc lookups):
  - `CREATE FULLTEXT INDEX proteinText IF NOT EXISTS FOR (p:Protein) ON EACH [p.name, p.description];`
  - `CREATE FULLTEXT INDEX domainText IF NOT EXISTS FOR (d:Domain) ON EACH [d.id, d.name, d.description];`
  - `CREATE FULLTEXT INDEX keggText IF NOT EXISTS FOR (k:KEGGOrtholog) ON EACH [k.id, k.description];`
  - `CREATE FULLTEXT INDEX pathwayText IF NOT EXISTS FOR (pw:Pathway) ON EACH [pw.id, pw.name, pw.description];`
 - Helpful single-property indexes (cheap, improves filters):
  - `CREATE INDEX protein_name IF NOT EXISTS FOR (p:Protein) ON (p.name);`
  - `CREATE INDEX domain_name IF NOT EXISTS FOR (d:Domain) ON (d.name);`
  - `CREATE INDEX kegg_desc IF NOT EXISTS FOR (k:KEGGOrtholog) ON (k.description);`

2) Precomputed Spatial Edges (performance-optimized)
- Create only `(:Gene)-[:NEXT {contig,delta,same_strand}]->(:Gene)` per contig.
- Use incoming `[:NEXT]` to traverse backwards (no separate `:PREV`).
- Filter by `same_strand` property instead of separate `_SAME_STRAND` edges.
- Delta semantics: `delta > 0` intergenic distance; `delta < 0` overlap length.

Integration (two paths):
- CSV path (primary):
  - `src/build_kg/rdf_to_csv_converter.py` emits `next_relationships.csv` during Stage 07
  - `src/build_kg/neo4j_bulk_loader.py` detects this file and skips any post-load precompute
  - Optional non-destructive loader: `scripts/neo4j/load_next_from_csv.cypher` to load with `CALL { … } IN TRANSACTIONS`
- Fallback (rare): Streaming per-contig postload in `Neo4jBulkLoader._precompute_neighbor_edges()`
  - Not recommended for large graphs; retained for completeness

3) Materialized Rollups
- Precompute `(:Genome)-[:HAS_PATHWAY]->(:Pathway)` via KOs (and counts) to accelerate presence/absence queries.
- Optionally precompute family membership edges to reduce repeated joins.

4) Cypher Hygiene
- Bound expansions with contig + coordinate filters; prefer index-backed patterns.
- Use `USING INDEX` hints where the planner struggles.
- Return small typed payloads (IDs and minimal fields) and fetch details on demand.

## Vector Retrieval Plan (LanceDB)

- Index Params: tune HNSW/IVF for 10k–100k scale; test recall/latency trade-offs.
- Storage: consider float16 vectors if acceptable to halve memory.
- Two-Stage Retrieval: filter by annotation/taxonomic context before vector search to reduce candidate set.
- Precompute top-k neighbor lists for popular classes (e.g., integrases) for instant lookups.

## Agent Reliability and Efficiency

1) Tool Gating and Planning
- Low-temp classifier decides next tool; budgeted planner caps steps/tokens.
- Value-of-information scoring: attempt cheap DB lookups before heavy vector/literature tools.

2) Strict Schemas and Recovery
- Enforce Pydantic schemas for tool inputs/outputs; reject/repair malformed JSON with a short retry prompt.

3) Synthesis Guardrails
- Require citations to `tool_result_id`s for every claim; forbid unsupported claims.
- Add an automated verifier pass that checks claimed IDs/paths exist in retrieved results; flag mismatches.

4) Caching
- Deterministic cache keys: `fingerprint(prompt, tool_params, signature_version, db_snapshot)`.
- Persist common sub-queries by session (e.g., "all integrases").

## Prompt Optimization Program (DSPy + GEPA)

Goal: Improve correctness and efficiency of signature-driven behavior (classification, retrieval, query generation, synthesis) with measurable metrics.

1) Evaluation Dataset
- 50–200 tasks spanning:
  - Graph queries with expected result sets (saved Cypher answers)
  - Spatial neighborhood tasks with expected loci bounds
  - Vector retrieval with recall@k ground truth
- Keep small at first (20–40) to iterate fast; grow once the loop is stable.

2) Metrics
- Correctness: set equality/F1 on nodes/edges; recall@k for vectors.
- Efficiency: steps, tool-switches, latency, tokens.
- Hallucinations: claim-without-evidence rate.

3) Loop (per signature)
- Start with DSPy compile methods to produce a baseline compiled signature.
- Wrap a GEPA layer to mutate:
  - Instruction wording, constraints
  - Few-shot exemplars (selection + ordering)
  - Output schema hints and rationale style
- Use bandit-style early stopping to allocate trials to promising variants and stop poor arms early.

4) Outputs
- Frozen prompt bundles per signature with semantic versioning.
- Regression tests that fail on >X% drop in metrics.

### DSPy Methods (Quick Reference)

- BootstrapFewShot: automatically selects/constructs a small set of examples to guide the model; compiles a signature into a prompt with curated exemplars and instructions.
- ProgramOfThoughts: structured intermediate reasoning (scratchpad/plan) to decompose tasks; yields more reliable multi-step decisions without long chains.
- Teleprompter/Compile: generic DSPy compilation utilities that optimize instructions/examples for a signature given an eval set.

### GEPA (Evolutionary Prompt Search)

- Treat prompt configs as genomes; mutate/crossover instruction text, example sets, and formatting; evaluate on a fixed task set.
- Multi-objective: optimize correctness first; then efficiency with a small weight.
- Keep top-k elites; random restarts to avoid local minima.

### Bandit-Style Early Stopping

- Use a multi-armed bandit (e.g., UCB1 or Thompson sampling) across candidate prompt variants.
- After each mini-batch of tasks, update per-variant reward; stop allocating trials to underperformers; concentrate budget on promising arms.
- Benefits: fewer total eval calls, faster convergence; avoids over-spending on bad variants.

## Optimizing Many Signatures in One Pipeline

1) Isolate and Optimize Locally
- Optimize each signature against tasks that isolate its responsibility (hold others fixed with current best versions).
- Examples:
  - QueryClassifier: map queries → intent/tool; metric: accuracy vs. labeled intents.
  - ContextRetriever/DB Query Generator: generate Cypher patterns; metric: result-set F1 vs. ground truth queries.
  - WholeGenomeReader/GenomicChunkAnalyzer: locus identification; metric: loci overlap/coordinates match.
  - LociPrioritizer: ranking quality; metric: nDCG/precision@k.
  - ProgressiveSynthesizer: claim coverage vs. citations; metric: supported-claim ratio.

2) Hierarchical Compile
- Start with high-influence/high-frequency signatures (classifier, DB query generator), then mid-layer (spatial reader, prioritizer), then synthesis.
- After local optima, run a small end-to-end suite to catch interactions; adjust if regressions.

3) Bundle Versions
- Define a signature bundle spec (YAML/JSON) mapping each signature to a version.
- Agent runs with a fixed bundle; upgrade bundles only after passing the regression suite.

## Eval Harness (Implementation Sketch)

Directory structure:
- `src/tests/agent_bench/`
  - `tasks/` → JSONL with inputs, expected outputs/IDs
  - `suites/` → lists of tasks per suite (core, spatial, vectors)
  - `metrics.py` → set/graph metrics, recall@k, hallucination checks
  - `runner.py` → runs tasks through signatures/tools with mocks as needed
  - `report.json` → aggregate metrics

CLI:
- `python -m src.tests.agent_bench.runner --suite core --out reports/agent_bench.json`

Gatekeeping:
- CI/PR check fails if regression > threshold.

## Synthesis Guardrails (Prompt Contract)

- Every claim must cite `tool_result_id` or specific node IDs.
- Add a “Claims Without Evidence” section that should be empty; used by verifier to fail outputs that include unsupported claims.
- Penalize returns with any unsupported claim in eval metrics.

## Suggested Implementation Order

1) Add Neo4j constraints/indexes (integrated in `src/build_kg/neo4j_bulk_loader.py` Step 6)
2) Add strict Pydantic schemas for tool I/O and citation-required synthesis
3) Implement eval harness skeleton with 5 exemplar tasks and metrics
4) Add precompute job for `:NEXT`/`:PREV` edges (integrated in loader Step 7) and pathway rollups
5) Add DSPy compile loop + GEPA layer for 2 key signatures (classifier, DB query generator)
6) Tune LanceDB index and add two-stage retrieval

## Pydantic Schema Coverage (Concrete Plan)

- New file: `src/llm/rag_system/tool_schemas.py`
  - `ToolResultEnvelope`: standard wrapper for all tool outputs
    - Fields: `tool_name, success, version, tool_result_id, summary, message, display_text, structured_data, references, timings, token_usage`
  - Genomic context models (replace or convert from dataclasses):
    - `GeneContextModel`, `ContigContextModel`, `GenomeContextModel` (omit `hypothetical_count`)
  - Tool-specific models:
    - `LiteratureArticleModel`, `DatabaseQueryResultModel`, `CodeInterpreterResultModel`, `GenomeSelectorResultModel`
  - Synthesis input guardrails:
    - `Claim`, `SynthesisInput` (requires evidence IDs for each claim)

- Integration points:
  - `src/llm/rag_system/external_tools.py`
    - Wrap all tool returns in `ToolResultEnvelope`; IMPLEMENTED for whole_genome_reader, genome_selector, literature_search, code_interpreter, report_synthesis.
    - Provide `display_text` for backwards-compatible consumption; include `structured_data` with typed payloads where applicable.
  - `src/llm/rag_system/task_executor.py`
    - Validate envelopes on receipt (non-fatal logging on failure); persist returned `.dict()` in `completed_results`. IMPLEMENTED.
    - For synthesis tasks, construct `SynthesisInput` and enforce non-empty `evidence_ids`.
  - `src/llm/rag_system/whole_genome_reader.py`
    - Convert dataclasses → Pydantic (or add converters) when producing `structured_data`.
    - Remove `hypothetical_count` from structured outputs (keep textual mention in `display_text` if helpful).
  - `src/llm/rag_system/memory/progressive_synthesizer.py`
    - Enforce citation checks: fail synthesis if any claim lacks `tool_result_id` or known node IDs.
  - `src/llm/rag_system/core.py`
    - Updated helper paths to consume `display_text` from envelopes for literature/code tools so integrated context remains string-based. IMPLEMENTED.

## Status Snapshot

- CSV-based adjacency generation and bulk load integrated; tested on existing `genes.csv` (writes ~189k edges in ~2s; DB load in seconds).
- Loader now idempotently creates all constraints/indexes and skips precompute when `next_relationships.csv` is present.
- Tool outputs standardized via `ToolResultEnvelope` with typed `structured_data` where applicable.
- Core and executor adapted to envelope flow while preserving existing context text.

## Quick Test Commands

- Build NEXT CSV from Stage 07 outputs:
  - `python -m src.build_kg.csv_neighbors --csv-dir data/stage07_kg/csv`
- Non-destructive load into Neo4j (Homebrew install):
  - `cp data/stage07_kg/csv/next_relationships.csv /opt/homebrew/opt/neo4j/libexec/import/`
  - `cypher-shell -u neo4j -p "$NEO4J_PASSWORD" -f scripts/neo4j/load_next_from_csv.cypher`
- Verify:
  - `cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "MATCH ()-[:NEXT]->() RETURN count(*) AS edges"`

## Near-Term Next Steps

- Synthesis guardrails: enforce citations using `SynthesisInput` with `evidence_ids` per claim; add verifier pass.
- Eval harness: scaffold `src/tests/agent_bench/` with 5 exemplar tasks and metrics; wire for DSPy + GEPA loop.
- Hygiene: complete removal of `hypothetical_count` in structured outputs; rename `quick_switch_to_o3` → `quick_switch_to_gpt5`.
- Vector search: tune HNSW/IVF and implement two-stage filtering for common classes.

- Testing:
  - Unit tests per tool to validate envelope shape and presence of `tool_result_id`.
  - Verifier tests to enforce claim-citation contract before synthesis.

## Pipeline Integration (No External Cypher Required)

- Post-import steps integrated into `Neo4jBulkLoader.bulk_import()`:
  - Step 6: Constraints + Indexes (`_create_constraints_and_indexes`) using env vars `NEO4J_URI/USER/PASSWORD`.
  - Step 7: Neighbor edge precompute (`_precompute_neighbor_edges`).
- Scripts under `scripts/neo4j/` are kept as references, but the pipeline now owns post-load tuning.

## Open TODOs (Repo Hygiene)

- Rename `quick_switch_to_o3` → `quick_switch_to_gpt5` and update imports
- Remove `hypothetical_count` field from loci structures and outputs
- Keep contig-only field in spatial queries (fix already applied elsewhere)
- Maintain reference-based notes; no truncation of biological data

## Optional Future Enhancements

- GDS integration for neighborhood/community detection on projected subgraphs
- Comparative genomics: synteny, HGT, pan-genome analyses at the agent layer
- Additional annotation databases: TIGRFAMs, COG/eggNOG, TCDB, MEROPS
- Smarter cost routing and caching policies per query type
