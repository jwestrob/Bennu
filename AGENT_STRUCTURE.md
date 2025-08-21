# Agent Architecture and Improvement Plan

This document explains the current agent architecture in this repository, how the pieces fit together, and concrete issues and improvements to consider. It’s based on a direct reading of the codebase (not aspirational docs), with file paths and key classes/functions referenced for clarity.

## High‑Level Overview

- Two execution modes coexist:
  - Traditional RAG path: a single pass that classifies the question, plans retrieval, runs Cypher/semantic queries, optionally uses tools, then synthesizes an answer.
  - Unified agent path: an agent loop that dynamically selects and chains tools over multiple steps, with periodic guidance synthesis and a final report.
- Tool I/O is being standardized around Pydantic envelopes so tool outputs can be integrated consistently and cached.
- DSPy signatures define planning, retrieval, and synthesis “contracts” and are used via a model allocation layer that chooses models per task.
- Memory and progressive synthesis provide scalable accumulation of findings and token‑aware summarization.

## Core Components

- Orchestrator: `src/llm/rag_system/core.py`
  - Class `GenomicRAG` is the main entry point (`ask()`), managing both execution modes and shared services.
  - Initializes processors: `Neo4jQueryProcessor`, `LanceDBQueryProcessor`, `HybridQueryProcessor` (`src/llm/query_processor.py`).
  - Configures DSPy and a model allocation layer (`memory/model_allocation.py`) to pick models per task (e.g., use o3 only for “agentic_planning” and “final_synthesis”).
  - Traditional path: `QueryClassifier` → `ContextRetriever` (Cypher) → retrieval + optional tool runs (literature/code interpreter) → `GenomicAnswerer`.
  - Agentic path: hands off to `UnifiedAgentExecutor` for multi‑step dynamic tool chaining and progressive synthesis; can fall back to traditional.
  - Utilities: context formatting, compression gating, comparative query validation (e.g., avoid LIMIT 1 for cross‑genome queries), tool integration helpers, and genome scoping hooks.

- DSPy Signatures: `src/llm/rag_system/dspy_signatures.py`
  - `PlannerAgent`: decide traditional vs agentic.
  - `QueryClassifier`, `ContextRetriever`: classify and generate Cypher (with strict formatting and domain rules such as CAZyme patterns, directional relationship hygiene).
  - `GenomicAnswerer` (and summarizer/synthesizer signatures): produce final answers with biological rigor and citation requirements.
  - File also documents an explicit Neo4j schema and “allowed” properties to constrain query generation.

- Model Allocation: `src/llm/rag_system/memory/model_allocation.py`
  - Centralized, task‑aware model picker with “optimized” vs “premium everywhere” modes.
  - Defaults to cost‑optimized (use `gpt-4.1-mini` for most tasks; reserve `o3` for a few high‑value tasks). Includes robust fallbacks.

- Query Processors: `src/llm/query_processor.py`
  - `Neo4jQueryProcessor`: raw Cypher with pre‑ and post‑repair (TaskRepairAgent), plus guardrails (strip comments, normalize, fix common relationship mistakes) and error→repair retry flow.
  - `LanceDBQueryProcessor`, `HybridQueryProcessor` exist but the agent currently leans on Neo4j for structured steps.

- Tools + Tool Schemas
  - Pydantic Envelopes: `src/llm/rag_system/tool_schemas.py`
    - `ToolResultEnvelope` provides a stable envelope: `tool_name`, `success`, `display_text`, optional `structured_data`, timing/usage, and references.
    - Supporting models for contexts (Gene/Contig/Genome), literature articles, code interpreter results, and synthesis inputs/claims (for future guardrails).
  - Tool Implementations: `src/llm/rag_system/external_tools.py`
    - `whole_genome_reader_tool(...)`: spatial genome reading; caches by normalized parameter set; returns envelope.
    - `genome_selector_tool(...)`: intelligent genome targeting; returns envelope.
    - `literature_search(...)`: PubMed via Biopython; returns envelope with article models.
    - `code_interpreter_tool(...)`: async HTTP to a sandboxed service; returns envelope with stdout/output mapped.
    - `report_synthesis_tool(...)`: signals that synthesis should run using the memory system.
    - `AVAILABLE_TOOLS` and `TOOL_CAPABILITIES` registry for agent selection.

- Unified Agent Executor: `src/llm/rag_system/agent_executor.py`
  - `UnifiedAgentExecutor` replaces the older TaskGraph executor for the agentic path.
  - Loop: make decision (LLM `AgentDecisionMaker`) → execute tool or DB query → collect results → optional guidance synthesis every N steps → final comprehensive synthesis.
  - Tools executed via internal methods:
    - `database_query`: routes through “traditional” query logic.
    - `whole_genome_reader`: currently calls `WholeGenomeReader` and a hierarchical analyzer directly (see note below), returning a dict with `tool_output`.
    - `code_interpreter`: generates analysis code from step data and executes the interpreter; expects meaningful printed output and a JSON block for structured findings when present.
    - `literature_search`: calls the external tool wrapper and returns text.
  - Step data is accumulated in `_previous_step_data` as a dict keyed by step, used to drive analysis‑code generation.
  - Progressive synthesis is used both for periodic “guidance” updates and final reporting, leveraging `NoteKeeper` and `ProgressiveSynthesizer`.

- Hierarchical Spatial Analysis: `src/llm/rag_system/whole_genome_reader.py` and `hierarchical_analysis/*`
  - `WholeGenomeReader`: pulls all genes per contig ordered by coordinates; organizes plus/minus strands and formats rich LLM‑readable context.
  - `HierarchicalGenomeAnalyzer` and `GenomicChunkAnalyzer`: chunk spatial context (token‑aware) and let sub‑agents identify “interesting loci”, then generate a curated, analyzable output with summaries and details.
  - In the agent path, `UnifiedAgentExecutor` uses this hierarchical flow instead of dumping raw spatial text.

- Memory and Synthesis: `src/llm/rag_system/memory/*`
  - `NoteKeeper`: manages session/task notes and paths; `ToolResultCache` stores large tool results off‑trace with references.
  - `ProgressiveSynthesizer`: Map‑Reduce style progressive synthesis with token‑aware decisions, guidance (“lightweight”) vs report (“comprehensive”) modes, caching, and model allocation integration.

- Legacy Task System (still present): `task_management.py`, `task_executor.py`, `agent_tool_selector.py`
  - DAG of `Task` objects with tool selection (LLM‑first), executor, and enhanced logging.
  - Intended to be superseded in the agentic path by `UnifiedAgentExecutor`, but still used by parts of the traditional flow and for compatibility.

## Execution Flows

1) Traditional Path (single pass)
   - Plan and classify (`PlannerAgent`, `QueryClassifier`).
   - Generate Cypher (`ContextRetriever`), apply genome scoping if detected, validate comparative queries.
   - Run Neo4j; compress if needed; optionally run `literature_search` and/or `code_interpreter` based on heuristics.
   - Synthesize (`GenomicAnswerer`); return structured result with metadata.

2) Unified Agent Path (multi‑step)
   - `AgentDecisionMaker` chooses next action: database query, `whole_genome_reader`, `code_interpreter`, `literature_search`, or `synthesize`.
   - Each step stores results, updates “current findings”, and optionally writes task notes.
   - Every N steps, run guidance synthesis to steer the loop; stop when the agent decides to synthesize.
   - Final report uses all notes/raw data via progressive synthesis.

## Data and Graph Considerations

- Spatial adjacency
  - KG bulk‑load now emits `next_relationships.csv` from `genes.csv` (see `src/build_kg/csv_neighbors.py` and `rdf_to_csv_converter.py`), and `Neo4jBulkLoader` supports creating indices and either precomputing or loading adjacency.
  - `WholeGenomeReader` currently orders genes by `(contig, startCoordinate)` directly; adjacency edges (`:NEXT`) can enable efficient neighborhood/window queries and multi‑hop traversals for future tools.

- Indices and constraints
  - `scripts/neo4j/indices.cypher` defines uniqueness constraints and useful indexes (composite `Gene(contig,startCoordinate,endCoordinate)`, full‑text indexes, etc.). `Neo4jBulkLoader` also applies them programmatically post‑import.

## Issues and Risks Observed

- Tool I/O contract is only partially unified
  - External tools return `ToolResultEnvelope`, but the agent’s `_execute_whole_genome_reader` method returns a dict with `tool_output` and not an envelope. Mixing shapes requires special‑case handling downstream.
  - Some integration points read `display_text` (envelope) while others look for `tool_output` (custom dict). This can cause brittle branching and missed data.

- Duplicate or divergent capability metadata
  - `TOOL_CAPABILITIES` is large and currently defined in `external_tools.py`. There’s overlap with selector logic elsewhere and the agent’s own heuristics. Risk of drift if definitions are copy‑pasted or re‑declared.

- Two systems in parallel
  - Unified agent vs TaskGraph executor both exist, plus a separate “traditional with tools” branch in `core.py`. Duplication increases maintenance and reasoning complexity.

- Code interpreter contract is implicit
  - The agent expects printed “ANALYSIS RESULTS” JSON blocks (regex‑extracted) and general output text. A defined schema for analysis outputs would reduce parsing fragility and improve downstream synthesis.

- Model allocation knobs are static
  - Current rules are hand‑tuned (e.g., keep most tasks on `gpt-4.1-mini`). We lack dynamic gating by budget, token counts, or “return‑on‑reasoning” signals from recent steps.

- Logging/telemetry not structured
  - Rich, informative logs exist (with emojis), but there’s no standardized structured telemetry or trace IDs across steps/tools. Harder to benchmark and regress.

- Validation and guardrails
  - `GenomicAnswerer` describes citation/identifier requirements, but enforcement is best‑effort. There’s scaffolding (`Claim`, `SynthesisInput`) for evidence‑backed synthesis that isn’t yet enforced end‑to‑end.

- Caching is ad‑hoc
  - Genome reading caches in an in‑module dict (no TTL, no size bound). Tool result caching exists under memory’s session dir, but not all tools integrate uniformly.

## Recommended Improvements

1) Unify tool contracts end‑to‑end
   - Ensure every tool call returns a `ToolResultEnvelope` (including the hierarchical `whole_genome_reader` path). Provide adapters so older code that expects `tool_output` reads `display_text` from the envelope.
   - Ingest envelopes centrally in `core.py` and `agent_executor.py` via a small helper that extracts `display_text`, merges `structured_data`, and records `references` into `NoteKeeper`.

2) Centralize tool capability metadata
   - Move `TOOL_CAPABILITIES` into a single JSON/YAML and reference it from both the agent and selector(s). Add a small linter/test to prevent drift.

3) Standardize analysis outputs for `code_interpreter`
   - Define a Pydantic `AnalysisResultEnvelope` with required keys (`summary`, `statistics`, `key_findings`, optional `dataframes` schema) and require the interpreter to emit machine‑readable JSON in stdout. Replace regex extraction with strict JSON parsing and validation.

4) Unify execution paths
   - Gradually retire `TaskGraph` for agentic flows and keep a single traditional path plus a single unified agent path. Reuse the same tool selection logic (LLM‑first) in the traditional path when tools are considered.

5) Tighten DSPy signatures and prompts
   - Reduce duplication across `PlannerAgent`, `QueryClassifier`, `ContextRetriever`. Explicitly thread the Neo4j schema snippet relevant to a question (rather than the entire block) to lower token use and reduce off‑schema queries.
   - Consider splitting `GenomicAnswerer` into a short “validator” signature that checks contig/scaffold citation rules before finalization, followed by a final synthesis signature.

6) Dynamic model allocation and budgeting
   - Add budget/time/token gates that can promote a task to `o3` when prior attempts on `mini` underperform, or demote when the agent’s next actions are routine. Log these decisions for evaluation.

7) Structured telemetry and evals
   - Emit a compact JSON event per step with: `session_id`, `step`, `tool`, `decision_reasoning`, `tokens_in/out`, `latency_ms`, `result_size`, `error`. Wire to simple CSV/JSONL for offline analysis.
   - Add seed evals (case studies) that measure: answer correctness, time, token cost, tool mix, and hallucination rate. Integrate into CI with small synthetic DB snapshots.

8) Graph adjacency usage
   - Add neighborhood/window queries powered by `:NEXT` (e.g., “N genes to either side”). Expose a `neighborhood_reader` tool that takes `gene_id`/`contig` + window size and returns an envelope with an ordered neighborhood.
   - This enables targeted spatial analysis without full genome dumps and improves latency.

9) Hardening and hygiene
   - Bound caches (LRU + TTL) for genome reading. Normalize and de‑duplicate common logging. Remove dead/duplicated constants. Ensure `tiktoken` fallbacks don’t degrade behavior.

## Notable File/Module Map (agent‑related)

- Orchestrator: `src/llm/rag_system/core.py`
- Agent loop: `src/llm/rag_system/agent_executor.py`
- DSPy signatures: `src/llm/rag_system/dspy_signatures.py`
- Tools: `src/llm/rag_system/external_tools.py` and schemas `tool_schemas.py`
- Hierarchical analysis: `src/llm/rag_system/whole_genome_reader.py`, `hierarchical_analysis/*`
- Memory/synthesis: `src/llm/rag_system/memory/*`
- Legacy task system: `task_management.py`, `task_executor.py`, `agent_tool_selector.py`
- Processors: `src/llm/query_processor.py`

## Concrete Next Steps (suggested order)

1) Tool envelope unification
   - Refactor `UnifiedAgentExecutor._execute_whole_genome_reader` to call the external tool wrapper or to wrap its current dict into a `ToolResultEnvelope`. Add a tiny compatibility shim in the agent to consume envelopes only.

2) Neighborhood tool
   - Implement `neighborhood_reader_tool` that leverages `:NEXT` with a bounded window, returning a concise, structured neighborhood for follow‑on analysis.

3) Code‑interpreter schema and parser
   - Define and enforce an `AnalysisResultEnvelope` and update code generator templates so the interpreter always prints strict JSON blocks. Replace regex parsing with Pydantic validation.

4) Execution path cleanup
   - Route “traditional with tools” through the same tool selection logic as the agent (or share a small selector helper). Begin deprecating the TaskGraph executor for agentic flows.

5) Telemetry + eval harness
   - Add a minimal event logger and a small eval suite (including your upcoming case study) to track accuracy/time/token cost/tool usage. Use it to drive DSPy signature tweaks and model allocation rules.

6) Prompt/signature hardening
   - Split “validation” from final synthesis and add explicit checks for required citations/identifiers. Add schema‑guided prompt snippets instead of the full block where possible.

7) Caching and policy
   - Add TTL + size bounds to genome reading cache; centralize tool result caching behind a common interface; expose policy toggles in config for guidance frequency, tool usage, and compression thresholds.

With these changes, the agent becomes simpler to reason about (one smart traditional path + one unified agent path), more reliable (typed tool contracts), and more efficient (adjacency‑powered neighborhoods, dynamic model allocation, and consistent telemetry for tuning).

