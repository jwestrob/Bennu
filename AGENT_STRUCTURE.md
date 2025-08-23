# Agent Architecture (Typed, Deterministic)

This document reflects the current state of the GenomicRAG agent after the typed router + FSM refactor. It explains how components fit together and where strictness and determinism are enforced. References include concrete file paths to the implemented code.

## High‑Level Overview

- Two-stage router + strict schemas:
  - Stage A deterministic guardrail handles obvious intents (e.g., spatial → `whole_genome_reader`) with safe defaults.
  - Stage B LLM router emits a single, strictly validated toolcall (Pydantic + JSON Schema), with one repair attempt.
- FSM-governed agent loop: An `ActionGraph` finite state machine enforces legal transitions and prevents oscillations; enabled by default. Direct `PLAN → SYN` is allowed for end-of-loop synthesis.
- Templates-only DB access: All Neo4j queries come from named, curated templates; free-form LLM Cypher generation is disabled in strict modes.
- Immutable GenomeScope: Context is attached and propagated as an immutable scope across processors.
- Observability: JSONL tracing is on by default with stubs for Langfuse/LangSmith.
- Memory and progressive synthesis: Session notes, caching, and progressive synthesis remain for scalable summarization.

## Core Components

- Orchestrator: `src/llm/rag_system/core.py`
  - `GenomicRAG.ask()` is the entry point. It integrates the router, strict DB template mode, GenomeScope derivation, compression, and synthesis.
  - Uses processors: `Neo4jQueryProcessor`, `LanceDBQueryProcessor`, `HybridQueryProcessor` (`src/llm/query_processor.py`).
  - Integrates the unified router via `src/llm/rag_system/router/get_router()` and validates Stage A/B toolcalls with `agent/tools/validate.py`.
  - Traditional path is now template-first: Stage B `database_query` runs `execute_named_template(...)`; a strict “templates-only” fast path can map NL questions → templates via `db_template_mapper.py`.
  - GenomeScope: `src/llm/rag_system/context/scope.py` provides immutable scope; `_derive_scope_from_slots()` attaches scope where possible.
  - Planning and answer generation still use DSPy signatures (`dspy_signatures.py`), but raw free-form Cypher generation is gated/disabled in strict modes.

- Router (Two-Stage, Typed): `src/llm/rag_system/router/`
  - `two_stage.py`: Stage A deterministic guardrail + Stage B LLM router orchestration; emits `RouterDecision`.
  - `llm_router.py`: Single LLM router that predicts a toolcall; validates against `TOOLCALL_JSON_SCHEMA`; performs one repair via `ToolRouteRepair`.
  - `signatures.py`: `ToolRoute` and `ToolRouteRepair` DSPy signatures, and `RouterDecision` dataclass.

- Toolcall Schemas and Validation: `src/llm/rag_system/agent/tools/`
  - `schemas.py`: Pydantic models for toolcalls (`RouterToolCall`, `DBQueryParams`, `SimilarityParams`, `SpatialGenomeParams`) and matching JSON Schema.
  - `validate.py`: Strict validator that rejects unknown fields (`extra='forbid'`) and provides a repair prompt helper.

- Query Processors: `src/llm/query_processor.py`
  - Neo4j: Free-form “auto-query” is disabled; use `execute_named_template(name, slots)` with `kg/cypher_templates/registry.py` compilers/validators.
  - LanceDB: Deterministic similarity search with runtime ESM2 embedder available for `by_sequence` mode (`embedding/runtime_embedder.py`) and manifest parity checks.
  - Optional curated GDS wrappers are behind a flag and exposed via safe call sites only.

- Cypher Template Library: `src/llm/kg/cypher_templates/*`
  - Template files and a registry with slot validation and compilers for special cases (e.g., `count_by_label`, adjacency `*_neighbors_k`).
  - Deterministic defaults: enforce or inject `LIMIT` from policy when appropriate. Template metadata (category, returns, cost, slot_hints) is exposed to the agent via a JSON catalog; compile-aware repair validates params before execution.
  - Discovery templates: `pfam_search.cypher`, `kofam_search.cypher`, `proteins_with_pfams.cypher`, `proteins_with_kos.cypher` (PFAM/KOFAM catalog search + union protein fetch).
  - Neighborhood templates: `protein_flanking_genes_5.cypher` (fixed 5 up/down neighbors), plus debug helpers `gene_next_degree.cypher`, `contig_gene_index.cypher`.
  - Compiler slot normalization: for list-based templates, singular or scalar inputs are normalized (e.g., `pfam`→`pfams=[...]`, `ko`→`kos=[...]`); coerces `limit` to int.

- Unified Agent Executor (FSM): `src/llm/rag_system/agent_executor.py`
  - FSM-enabled by default (`AGENT_FSM_STRICT=1`): states `PLAN → (DB|SIM|GENOME) → ACCUM → DECIDE → (PLAN|SYN)` (`fsm/action_graph.py`).
  - `database_query` path is template-only and returns tabular rows; a per-executor dedup cache suppresses repeated identical template+slot queries within a run.
  - `neighborhood_extractor` supports batch extraction via `protein_ids`; the executor/agent can pass seeds directly to extract all loci in one step.
  - `whole_genome_reader` uses hierarchical analysis; `literature_search`/`code_interpreter` remain available.
  - Progressive synthesis used for guidance and final answers; tool results cached via `ToolResultCache` and expanded during synthesis (notes carry references).
  - Soft progress signals passed to decisions: `progress_state` (candidates collected, loci built, last_row_count, zero_result_streak, est_chunks). Optional advisory `requires_approval` is set when heavy tools are proposed on large datasets.

- Tools + Tool Result Envelope
  - Envelope models: `src/llm/rag_system/tool_schemas.py` (`ToolResultEnvelope`, genome/literature/code models) used by external tools for stable outputs.
  - Implementations: `src/llm/rag_system/external_tools.py` with `AVAILABLE_TOOLS` and `TOOL_CAPABILITIES` for selection metadata.
    - `annotation_discovery`: keyword-driven PFAM+KOFAM discovery (case-insensitive). Uses `pfam_search` and `kofam_search` then fetches proteins via `proteins_with_pfams`/`proteins_with_kos`. Returns deduped proteins + candidate annotations.
    - `neighborhood_extractor` (DB-backed): extracts local neighborhoods via curated templates. Modes:
      - Single-seed: `protein_neighbors_k` (k-step adjacency) or (default) `protein_flanking_genes_5` (5 upstream + 5 downstream by contig order).
      - Windowed: `neighbors_by_window` (contig + start + end).
      - Batch: `protein_ids=[...]` runs per-seed neighborhoods in one call; auto-seeds from last DB result if no seeds provided (capped by `seeds_limit`).
      - Adds `summary_table` (seed → row_count) and an advisory for very large batches.
    - `whole_genome_reader`: remains for broad spatial reads (small genomes); metadata no longer recommends it for per-locus neighborhood extraction.

- Observability: `src/llm/rag_system/tracing.py`
  - JSONL tracing default-on; `get_tracer()` supports `AGENT_TRACING=jsonl:...` and stubs for Langfuse/LangSmith.
  - Reduced console noise: large debug file saves now emit at debug level; concise DB/neighborhood execution logs retained.

- Legacy Task System (gated off by default)
  - TaskGraph types remain behind `AGENT_ENABLE_LEGACY_TASKGRAPH=0` by default to avoid drift. The unified agent + FSM supersedes legacy loops.

## Execution Flows

1) Router-First Traditional Path (typed)
   - Stage A guardrail may force `whole_genome_reader` with safe defaults; toolcall is validated.
   - Stage B `database_query` uses named templates only; default `limit` injected via policy when missing.
   - Results are formatted, optionally compressed, optional tools run, then synthesized via `GenomicAnswerer`.

2) Unified Agent Path (FSM)
   - FSM governs steps: decide → execute (`database_query`/`neighborhood_extractor`/`whole_genome_reader`/`literature_search`/`code_interpreter`) → accumulate → decide → synthesize.
   - Returns a structured execution trace with steps, tools used, and final synthesis.

## Data and Graph Considerations

- Spatial adjacency and windows
  - Neo4j templates include coordinate and adjacency patterns; adjacency helpers (`gene_neighbors_k`, `protein_neighbors_k`) avoid in-graph SPARQL-like scans and use deterministic expansions.
  - `WholeGenomeReader` provides ordered per-contig contexts; hierarchical analyzers curate interesting loci for the LLM.
  - Compiled neighborhood queries use `CALL (g) { ... }` subqueries and order by `toInteger(startCoordinate)` to avoid deprecation warnings and ensure numerical sort.

- Indices and constraints
  - `scripts/neo4j/indices.cypher` and loader utilities establish constraints and indexes for predictable performance.

## Current Issues and Risks

- Tool I/O shape mismatch (minor)
  - External tools return `ToolResultEnvelope`, while `_execute_whole_genome_reader` returns a dict with `tool_output`. A thin adapter would fully unify shapes.

- Capability metadata drift
  - `TOOL_CAPABILITIES` lives alongside agent heuristics; keep a single source of truth to avoid divergence.

- Template coverage gaps
  - Free-form Cypher is disabled; uncommon queries may require adding new templates and mapper rules.
  - Domain (PFAM) matching can be versioned or name-based; use flexible matching (`STARTS WITH`/description contains) when `exact=false`.

- Runtime embedder dependencies
  - `by_sequence` similarity requires transformers+torch; ensure environment is prepared and LanceDB dimensions match the manifest.

- Tracing providers
  - JSONL tracing is default; external providers are stubs pending integration.

## Flags and Defaults

- See `docs/AGENT_FLAGS.md` for details. Key defaults:
  - `AGENT_FSM_STRICT=1` (FSM on), `AGENT_DB_TEMPLATES_ONLY=1` (templates-only DB), `AGENT_DEFAULT_DB_LIMIT=100`.
  - Legacy TaskGraph/Selectors disabled by default.
  - `AGENT_WGR_APPROVAL_CHUNKS` (default: 0 disabled): when set >0, marks `whole_genome_reader` decisions as `requires_approval=true` if estimated chunks exceed threshold (advisory only).

## File Map (Quick Reference)

- Orchestrator: `src/llm/rag_system/core.py`
- Router: `src/llm/rag_system/router/{two_stage.py,llm_router.py,signatures.py}`
- Toolcall schemas: `src/llm/rag_system/agent/tools/{schemas.py,validate.py}`
- FSM: `src/llm/rag_system/fsm/action_graph.py`
- Templates: `src/llm/kg/cypher_templates/*` + `registry.py`
- Query processors: `src/llm/query_processor.py`
- Runtime embedder: `src/llm/embedding/runtime_embedder.py`
- Agent executor: `src/llm/rag_system/agent_executor.py`
- Tool envelope + tools: `src/llm/rag_system/{tool_schemas.py,external_tools.py}`
- Tracing: `src/llm/rag_system/tracing.py`

## Template/Param Defaults and Repair

- Compile-aware defaults: `registry.compile_query` injects safe defaults for optional slots (e.g., `proteins_with_pfam.exact=false`) to prevent ParameterMissing errors.
- Execution-time limits: `_execute_database_query` injects `limit` from policy/env for list templates when missing (bounded 1–5000).
- Param repair: when a decision emits invalid/missing params, a single repair attempt is made; if compile fails, a second repair includes the compile error + catalog context. Only repaired params that compile are used.
 - Slot normalization: for list-based templates (`proteins_with_pfams`, `proteins_with_kos`), compiler tolerates scalar and singular keys, normalizing to proper list slots.

## Decision Context Enrichment

- `db_templates_catalog`: includes template metadata (category, returns, cost, slot_hints) for the LLM to choose concrete DB steps.
- `data_profile`: real contig/gene counts + `est_chunks` to inform tool cost/scale.
- `tool_costs`, `policy_hints`: neutral hints (cheap-first, templates-only-db, may require approval). No hard-coded logic.
- `functional_signatures_catalog` (optional config): alias panels (e.g., PFAM/KOFAM) passed as advisory context.

## Discovery-First + Batch Neighborhoods (Deterministic Plan)

- Discovery-first preflight (recommended):
  - Extract a neutral functional keyword from the query.
  - Run `annotation_discovery` once to collect PFAM+KOFAM candidates and union proteins; fall back to `proteins_with_pfam` if nothing found.
  - Run `neighborhood_extractor` in batch with `protein_ids` (flanks or k-step) for the top-N seeds; then synthesize.
- Two-stage router vs. unified agent:
  - A Stage‑A guard routes to `annotation_discovery` for functional keywords; the unified agent should adopt the same preflight so both paths behave identically.

## Dev Utilities

- Template smoke test: `scripts/smoke_test_templates.py` compiles all templates and can execute a safe subset (`--run`) against a dev DB to catch schema drift early.

## Consolidation Plan (Single Path)

- Goal: Remove behavioral drift between the two surfaces (TwoStageRouter and UnifiedAgentExecutor) and converge on a single, deterministic, template‑first flow.
- Approach:
  - Add a discovery‑first preflight to UnifiedAgentExecutor (extract neutral keyword → `annotation_discovery` once → seeds → batch `neighborhood_extractor`) and run it before the FSM loop.
  - Make TwoStageRouter a thin shim that delegates to the same preflight; Stage‑B LLM routing becomes advisory for rare cases only.
  - Keep one source of truth for template metadata (SPECS) and tool capabilities; both preflight and agent decisions read from it.
  - Retain compile‑time slot normalization and per‑executor DB dedup cache.
  - Outcome: Consistent, cheap, DB‑first behavior across all entry points.
