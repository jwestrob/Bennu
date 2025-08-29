# Agent Architecture (MacroPlanner + Structured Operators)

This document reflects the current, operator‑driven GenomicRAG architecture. It replaces the older typed‑router/FSM write‑up. The system now plans with a MacroPlanner over a strict operator catalog, executes deterministic DB queries and helpers, and synthesizes compact results with preserved totals.

## High‑Level Overview

- MacroPlanner‑first: A DSPy plan chooses from a fixed operator catalog (no free‑form tools). Plans are executed deterministically.
- Two‑stage annotation discovery: Catalog fuzzy search → precise IDs → exact Neo4j retrieval. Supports PFAM accessions and short names, and KEGG KOs.
- Templates‑only DB access: Cypher runs via named templates; no model‑generated Cypher.
- Compact handoff to synthesis: Discovered proteins are deduplicated and passed as structured lists; synthesizer pre‑compacts examples while preserving full counts.
- Optional completeness and neighbors: KEGG pathway completeness (native KO totals) and neighborhood/kNN tools remain available.

## Core Components

- Orchestrator: `src/llm/rag_system/core.py`
  - Entry point `GenomicRAG.ask()` handles model allocation, MacroPlanner planning, deterministic operator execution, result collection, context debug, and final synthesis.
  - Collects only whitelisted list payloads (e.g., `discovered_proteins`, `pathway_completeness`) and deduplicates proteins globally by `(genome_id, protein_id)`.
  - Emits “Context debug” and “Context trim” logs to pinpoint large contributors and dropped duplicates.

- MacroPlanner (DSPy): `src/llm/rag_system/dspy_signatures.py`
  - `MacroPlannerSignature`: Produces a strict JSON plan using only cataloged operators. Rubric encourages breadth‑first PFAM+KO exploration, two‑stage identifier retrieval, compact evidence, and optional follow‑ups.
  - Planner context includes an operator catalog and optional compact references (disabled by default to avoid token bloat).

- Operator Catalog + Execution
  - Registration and specs: `src/llm/mfp/operators/base.py`
  - Built‑ins: `src/llm/mfp/operators/builtin.py`
    - `AnnotationDiscovery`: Performs two‑stage discovery (catalog fuzzy → IDs → exact retrieval). Returns `discovered_proteins` with PFAM/KO provenance; supports `limit`, `genome_ids`, `return_full_rows`.
    - KEGG helpers: `FetchPresentKOs`, `LoadKoPathwayTotals` (native ko_pathway.list), `ComputePathwayCompleteness` with sensible defaults.
  - Catalog search: `src/llm/mfp/operators/catalog_search.py`
    - `SearchPfamCatalogFuzzy`, `SearchKoCatalogFuzzy`, `ExtractIdsFromCatalogHits`, `QueryProteinsByIds` (exact PFAM/KO ID filters). PFAM supports both accessions and short names.
  - Plan execution: `src/llm/mfp/executor.py` executes the operator list against an `OperatorContext` (Neo4j driver + project_root).

- Deterministic DB Layer
  - Template runner: `src/llm/options/template_runner.py` executes file‑based Cypher with validated params.
  - Exact retrieval templates (resources/): `resources/cypher/proteins_by_pfam_ids.cypher`, `resources/cypher/proteins_by_ko_ids.cypher` (case‑insensitive equality; list comprehensions fixed to avoid runtime errors).
  - Present KO summary: `resources/cypher/present_kos_by_genome.cypher` for completeness computation.

- Synthesis and Memory: `src/llm/rag_system/memory/`
  - `ProgressiveSynthesizer`: Map‑Reduce capable summarizer with pre‑compaction of large lists.
    - Default compact mode shows “rows=<true total>” and up to 10 examples (configurable via `SUMMARY_EXAMPLE_CAP`).
    - `return_full_rows=true` on `AnnotationDiscovery` marks a binding as `_format='full'` to include complete JSON rows for small targets (≤ 2000 rows).
  - `NoteKeeper` + `ToolResultCache`: Persist session notes and large tool outputs for reference without bloating model context.
  - Model allocation: `model_allocation.py` picks appropriate models per task; synthesizer updates model‑aware token limits.

- Completeness (native): `src/llm/options/pathway_completeness.py` and `src/llm/kegg/pathway_mapping.py`
  - Computes pathway totals from `data/reference/ko_pathway.list` and intersects with present KOs from the graph. DB‑only totals are no longer assumed.

## Execution Flow (Typical)

1) Plan: MacroPlanner proposes a sequence of operator calls (often multiple `AnnotationDiscovery` steps with distinct keywords covering PFAM+KO).
2) Execute: The MFP executor runs operators deterministically, with DB templates handling exact PFAM/KO ID matches.
3) Collect: Core collects only whitelisted list outputs. `discovered_proteins` are deduplicated across all steps.
4) Synthesize: Progressive synthesizer compacts large lists (counts + examples), preserving true totals. For small targeted pulls with `_format='full'`, full JSON rows are included.

## Context Compaction Policy (Active)

- Preserve totals: For compacted lists, the synthesizer stores `total_rows` and displays correct counts (not the capped example size).
- Default examples: Up to 10 examples per list in compact mode; override globally via `SUMMARY_EXAMPLE_CAP` env var.
- Full rows: Use `return_full_rows=true` on `AnnotationDiscovery` for small, targeted queries (≤ 2000 rows), optionally with a low `limit`.
- Catalog hits: PFAM/KO catalog hit metadata is not fed to synthesis; only probe metadata influences planning.

## Important Flags

- Planning
  - `USE_MFP_PLANNER=1` (default): Enable MacroPlanner path.
  - `INCLUDE_REFERENCE_IN_PLANNER=0` (default): Avoids stuffing KO/PFAM catalogs in planner prompts; can be enabled with optional caps.

- Synthesis
  - `SUMMARY_EXAMPLE_CAP=10`: Cap for examples per list in compact mode.
  - `EMIT_FOLLOWUP_REQUESTS=1`: Allows planner‑visible, lightweight follow‑ups when evidence is thin.

- Completeness
  - `USE_NATIVE_TOTALS_FOR_PATHWAYS=1` (default): Use native ko_pathway.list totals.
  - `USE_CI_TOTALS_FOR_PATHWAYS=0`: Legacy CI‑based totals (not needed with native mode).

## Key Files (Quick Map)

- Orchestrator: `llm/rag_system/core.py`
- Planner/signatures: `llm/rag_system/dspy_signatures.py`
- Operators: `llm/mfp/operators/{base.py,builtin.py,catalog_search.py,planning_utils.py}`
- Plan executor: `llm/mfp/executor.py`
- Template runner: `llm/options/template_runner.py`
- Synthesis & memory: `llm/rag_system/memory/{progressive_synthesizer.py,note_keeper.py,tool_result_cache.py,model_allocation.py}`
- Completeness: `llm/options/pathway_completeness.py`, `llm/kegg/pathway_mapping.py`
- Config: `llm/config.py`

## Design Notes

- Determinism at the edges: All DB interactions are through curated templates; operators are pure functions over inputs/params.
- Planner breadth over depth: Encourages PFAM+KO coverage first, then optional completeness or neighborhoods.
- Token discipline: Dedup at collection, compact at synthesis, and preserve informative counts. Full detail remains opt‑in for small targets.

## Future Directions

- Evidence matrix summary (marker → counts, per‑genome counts, examples) to standardize reporting.
- Configurable post‑merge trimming to return exactly N examples after PFAM+KO union when requested.
- Optional multi‑pass, diff‑based synthesis (incremental context ingestion) — see discussion in AGENTS.md.

