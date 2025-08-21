# Roadmap: Agent + Knowledge Graph Optimization

This roadmap focuses on improving the agent system and knowledge graph (KG) stack: query performance, reliability, and prompt optimization. The build pipeline is mostly stable and out of scope except where it affects KG/agent efficiency.

## Guiding Principles

- Maximize correctness and reproducibility; minimize hallucinations.
- Prefer precomputation and indexes over repeated heavy queries.
- Store large data once and reference by ID; never truncate biological data.
- Optimize prompts with measurable, repeatable offline evaluations.
- Keep signatures generic; no dataset-specific logic.

## Immediate High-Impact Actions

1) Neo4j constraints and indexes for fast queries (see Cypher section below)
2) Precompute `:NEXT` edges for spatial scans within contigs
3) Tight JSON schemas for tool I/O + synthesis guardrails with citations
4) Agent eval harness with small curated task set and metrics
5) DSPy compile loop + GEPA search for key signatures (start with classifier and DB query generator)
6) Vector index tuning + two-stage retrieval (annotation/lexical filter → vector)

## Knowledge Graph Performance Plan

1) Constraints and Indexes (Neo4j 5.x syntax)
- Uniqueness:
  - `CREATE CONSTRAINT genome_id IF NOT EXISTS FOR (g:Genome) REQUIRE g.id IS UNIQUE;`
  - `CREATE CONSTRAINT gene_id IF NOT EXISTS FOR (g:Gene) REQUIRE g.id IS UNIQUE;`
  - `CREATE CONSTRAINT protein_id IF NOT EXISTS FOR (p:Protein) REQUIRE p.id IS UNIQUE;`
- Composite for spatial queries:
  - `CREATE INDEX gene_contig_coords IF NOT EXISTS FOR (g:Gene) ON (g.contig, g.start, g.end);`
- Full-text (optional, for name/desc lookups):
  - `CREATE FULLTEXT INDEX proteinText IF NOT EXISTS FOR (p:Protein) ON EACH [p.name, p.description];`
  - `CREATE FULLTEXT INDEX pfamText IF NOT EXISTS FOR (d:PFAM) ON EACH [d.id, d.name];`
  - `CREATE FULLTEXT INDEX keggText IF NOT EXISTS FOR (k:KEGG) ON EACH [k.id, k.name];`

2) Precomputed Spatial Edges
- Within each contig, create `(:Gene)-[:NEXT {delta:int}]->(:Gene)` edges to enable O(steps) neighborhood/operon traversals without variable-length scans.
- Batch job sketch:
  - For each contig: sort genes by `start`; connect consecutive genes with `:NEXT` and `delta = next.start - this.end` (or 0 if overlapping).

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

1) Add Neo4j constraints/indexes; create Cypher script in `scripts/neo4j/`
2) Add strict Pydantic schemas for tool I/O and citation-required synthesis
3) Implement eval harness skeleton with 5 exemplar tasks and metrics
4) Add precompute job for `:NEXT` edges and pathway rollups
5) Add DSPy compile loop + GEPA layer for 2 key signatures (classifier, DB query generator)
6) Tune LanceDB index and add two-stage retrieval

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

