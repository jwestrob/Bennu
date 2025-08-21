# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Quick Commands

Prereq: activate the conda env before running anything
- source /Users/jacob/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh && conda activate genome-kg

Build and run the pipeline
- Build full pipeline (data/raw -> stage08): python -m src.cli build
- Resume from stage N: python -m src.cli build --from-stage 3
- Limit to stage range: python -m src.cli build --from-stage 3 --to-stage 6
- Increase workers: python -m src.cli build --threads 8
- Skip taxonomy stage: python -m src.cli build --skip-tax

Load Neo4j from CSVs (bulk import)
- python -m src.build_kg.neo4j_bulk_loader --csv-dir data/stage07_kg/csv

Ask questions over the KG with the LLM agent
- python -m src.cli ask "Find all transport proteins"
- python -m src.cli ask "Analyze genomic neighborhoods around integrases"

ESM2 embeddings on Apple Silicon (optimized)
- python scripts/run_esm2_m4_max.py
- Monitor progress: python scripts/monitor_esm2_progress.py

Run tests
- Full test suite: pytest -v
- Single file: pytest -v src/tests/test_agentic_rag_system.py
- By keyword: pytest -v -k "genomic or tool_selector"
- By marker: pytest -v -m integration
- Coverage: pytest -v --cov=src --cov-report=term-missing

Lint/format/type-check
- Format: black src && isort src
- Lint: flake8 src
- Types: mypy src

Key config locations
- Pipeline defaults: config/pipeline.yaml
- Conda env: env/environment.yml
- LLM/deps: requirements-llm.txt
- Pytest config: pytest.ini

Secrets and endpoints (set before running)
- Neo4j: export NEO4J_URI=bolt://localhost:7687; export NEO4J_USER=neo4j; export NEO4J_PASSWORD={{NEO4J_PASSWORD}}
- OpenAI: export OPENAI_API_KEY={{OPENAI_API_KEY}}
- LanceDB path aligns with data/stage08_esm2/lancedb


## Big-Picture Architecture (what matters for working across modules)

Three layers tie this repo together end-to-end:

1) Bioinformatics pipeline (src/ingest, src/build_kg, orchestrated by src/cli.py)
- Stages 0–8 produce normalized data products:
  - Stage 0: input prep → data/stage00_prepared
  - Stage 1: QUAST QC → data/stage01_quast
  - Stage 2: DFAST_QC taxonomy → data/stage02_dfast_qc
  - Stage 3: Prodigal genes/proteins → data/stage03_prodigal
  - Stage 4: Astra HMM (PFAM/KOFAM) → data/stage04_astra
  - Stage 5: GECCO BGCs → data/stage05_gecco
  - Stage 6: dbCAN CAZymes → data/stage06_dbcan
  - Stage 7: Knowledge graph (RDF + CSV) → data/stage07_kg
  - Stage 8: ESM2 embeddings + LanceDB → data/stage08_esm2
- Knowledge graph construction (src/build_kg):
  - rdf_builder.py generates triples linking Genome → Gene → Protein and annotations (PFAM, KEGG, BGC, CAZyme) and pathways
  - functional_enrichment.py attaches authoritative descriptions for PFAM/KEGG/CAZy
  - neo4j_bulk_loader.py converts RDF → CSV and uses neo4j-admin import for fast loads
- Embeddings (src/ingest/06_esm2_embeddings.py) generate 320-d vectors and LanceDB index used downstream by the LLM layer.

2) Storage and retrieval (Neo4j + LanceDB)
- Neo4j hosts the graph (48k+ nodes, 95k+ relationships). CSV import is the production path
- LanceDB stores ~10k protein embeddings for sub-ms similarity search
- src/llm/query_processor.py provides processors:
  - Neo4jQueryProcessor for Cypher
  - LanceDBQueryProcessor for vector search
  - HybridQueryProcessor for staged (graph→vector) retrieval

3) Agentic LLM system (src/llm/rag_system)
- Orchestrator: src/llm/rag_system/core.py (class GenomicRAG)
  - Decides traditional vs agentic path using PlannerAgent (DSPy)
  - Performs genome selection up front when appropriate (genome_selection.py)
  - Validates and scopes queries (query_validator.py + enforce_genome_scope)
  - Executes either a single retrieval pass or a multi-task DAG
- Tooling and execution:
  - Tool selection is LLM-first via agent_tool_selector.py (BiologicalToolSelector) with model allocation
  - TaskGraph + Task (task_management.py) represent a DAG of ATOMIC_QUERY (database) and TOOL_CALL steps
  - TaskExecutor (task_executor.py) runs tasks; ATOMIC_QUERY goes through processors; TOOL_CALL invokes external tools in external_tools.py (e.g., whole_genome_reader)
- Memory/synthesis pipeline:
  - ProgressiveSynthesizer (memory/progressive_synthesizer.py) performs token-aware Map-Reduce synthesis; switches to direct synthesis when the context is small enough
  - NoteKeeper stores compact findings and references rather than raw megabytes of tool output (memory/tool_result_cache.py)
  - Model allocation (memory/model_allocation.py) routes tasks: complex reasoning to premium model, classification/summarization to cost-effective models
- Spatial analysis path:
  - For queries that imply spatial/neighborhood/operon/prophage context, the system routes to whole_genome_reader and preserves gene order/coordinates; synthesis operates on this structured context rather than generic compressed text

Why this matters to you in Warp
- Single questions can fan out into multiple tasks spanning database queries, vector searches, and code execution; the final answer is synthesized with awareness of model limits
- Genome scoping and comparative-query validation are enforced automatically to reduce empty-result failures
- Spatial queries bypass generic compression to preserve locus-level structure


## Repo-specific rules that affect how you operate

- No truncating biological data: spatial/operon/prophage contexts must preserve coordinates and gene order; use reference-based storage in notes instead of dumping raw multi-MB blobs
- Don’t hardcode biology in DSPy signatures: avoid dataset-specific IDs or patterns; use placeholders and let runtime fill values
- Model allocation is central: complex planning/tool-selection/final synthesis use a premium model; simple classification and formatting use a cheaper model
- Genome selection happens once per agentic run and is propagated to all tasks to control scope and cost
- Comparative queries get validated to avoid LIMIT-induced empty results
- GPT-5 compatibility exists behind a wrapper; some legacy names may still say “o3” but route to configured models. Open follow-up TODO noted in CLAUDE.md: quick_switch_to_o3 should be renamed to quick_switch_to_gpt5 when convenient


## Practical debugging entry points

- Agent path and planning: src/llm/rag_system/core.py (ask(), _execute_traditional_query(), _execute_agentic_plan())
- Tool selection decisions: src/llm/rag_system/agent_tool_selector.py (BiologicalToolSelector)
- Task graph orchestration and execution: task_management.py, task_executor.py
- Memory and synthesis: memory/progressive_synthesizer.py (direct vs Map-Reduce, token thresholds); memory/note_keeper.py; memory/tool_result_cache.py
- Knowledge graph integration: src/build_kg/rdf_builder.py, functional_enrichment.py, rdf_to_csv_converter.py, neo4j_bulk_loader.py
- Pipeline stages: src/ingest/*.py (naming corresponds to stage numbers)


## When running heavy commands in Warp

- Long-running steps (build stages, large queries, synthesis) are expected. Prefer the bulk Neo4j loader when reloading the graph
- For selective iteration during development, use --from-stage/--to-stage to constrain work; re-run only the stages you’re changing
- For tests, use -k to scope to a specific behavior and -m to avoid slow/integration where not needed


## Pointers to important docs already in the repo

- README.md: end-to-end overview, quickstart, example queries, and data products
- CLAUDE.md: deep implementation notes for the agent system; includes critical rules (no truncation, signature guidelines), performance fixes, and architecture changes
- docs/LLM_SYSTEM_ARCHITECTURE.md and docs/COMPONENT_MAP.md: detailed architecture and data flow for the agentic system
- config/pipeline.yaml: single source of truth for many stage defaults and LLM/RAG knobs

