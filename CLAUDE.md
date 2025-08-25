# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with this advanced genomic AI platform.

## 🚨 **CRITICAL: USER RUNS ALL CLI COMMANDS**

**🛑 NEVER RUN src.cli COMMANDS — ALWAYS ASK THE USER TO RUN THEM! 🛑**

**IMPORTANT: Claude Code MUST NOT execute any `src.cli` commands due to timeout and execution issues!**

- **Pipeline commands**: `python -m src.cli build`, `python -m src.cli ask`, etc.
- **Testing commands**: Any command that might take >30 seconds
- **Agent queries**: All `ask` commands that invoke the LLM agent system

**✅ USER SHOULD RUN**: All CLI commands, pipeline builds, agent queries
**✅ CLAUDE CAN RUN**: File operations, quick bash utilities, git commands

The user needs to be able to test pipeline commands, especially stage 7 builds and CSV generation, and agent queries without Claude Code timing out during long-running processes.

## ✅ Recent Agent Architecture Updates (DB‑first, deterministic)

- Templates-only DB access: all queries run via curated Cypher templates; free-form Cypher is disabled.
- Macro Fast Path (MFP): typed, LLM‑free route for locus discovery with obligation gates and fail‑fast routing.
  - Grammar + normalizer: tolerates natural phrasing (e.g., “within N genes” → `± N`; strips “focusing on …”).
  - Deterministic flow: seeds_by_marker → batched_neighborhoods_gated (±k) → LanceDB kNN → heavy synthesis.
  - Neighborhoods include per‑neighbor PFAM/KO and seed PFAM/KO; flank is a true ±k radius.
  - Fail‑fast flags: `FAIL_FAST_ON_GRAMMAR_ERROR=1` and `FAIL_FAST_ON_TOOL_ERROR=1` stop early on compile errors.
  - LanceDB defaults: `nn=10` when unspecified; `topk=max(10, 10*nn)`; kNN outcome is always summarized in the final report.
- New tools:
  - `annotation_discovery`: keyword-based PFAM + KOFAM search (case-insensitive), then union proteins via `proteins_with_pfams`/`proteins_with_kos`.
  - `neighborhood_extractor`: DB-backed neighborhoods with three modes:
    - Single seed: `protein_neighbors_k` (k-step) or default `protein_flanking_genes_5` (5 upstream + 5 downstream by contig order).
    - Windowed: `neighbors_by_window` with contig+start+end.
    - Batch: `protein_ids=[...]` per-seed neighborhoods in one call; auto-seeds from last DB result if seeds are not provided.
  - Adds concise summary_table and advisory for very large batches.
- `lancedb_knn`: batched vector similarity search with PFAM/KOFAM‑aware filtering (pfam include/exclude implemented).
  - Exclude: drop neighbors whose Domain description CONTAINS exclusion text (e.g., “integrase”) or whose Domain id/acc IN exclusion markers (e.g., `PF00589`).
  - Include (pfam, new): if provided, keep only neighbors whose Domain description CONTAINS include text or id/acc IN include markers.
  - Returns `neighbors`, `picked`, `neighbors_full`, and `stats` (filter summaries, counts, topk). Filter criteria are logged and added to session notes.
- New templates: `pfam_search`, `kofam_search`, `proteins_with_pfams`, `proteins_with_kos`, `protein_flanking_genes_5`, `gene_next_degree`, `contig_gene_index`.
- Compiler normalization: scalar/singular params are normalized for list-based templates (e.g., `pfam`→`pfams=[...]`), and numeric coercions are applied where sensible.
- DB query dedup: identical template+slot calls are cached per executor instance and marked `summary.deduplicated=true`.
- Smoke test: `scripts/smoke_test_templates.py` compiles all templates and can run a safe subset (`--run`) against a dev DB to catch drift early.

Notes for Claude Code:
- Favor `annotation_discovery` → `neighborhood_extractor` (batch) → synthesis for functional queries.
- Do not hardcode specific biology (e.g., integrase IDs); use the templates/tooling above.
- Avoid redundant DB calls; rely on dedup and/or plan steps once.

## Neo4j Schema Reference (Authoritative)

Use ONLY the labels, relationships, and properties below. Property names are case-sensitive. Coordinates on Gene nodes are strings in the DB; cast with `toInteger(...)` when ordering or comparing.

### Node Labels and Properties

Genome
- id: Unique genome identifier
- genomeId: Internal genome identifier (use this for joins)

Gene
- id: Gene identifier (e.g., `gene:<contig>_<index>`)
- geneId: Identifier without prefix (may be present)
- contig: Contig/scaffold identifier string
- startCoordinate: String position; cast with `toInteger`
- endCoordinate: String position; cast with `toInteger`
- strand: String, typically `1` or `-1`

Protein
- id: Stable protein identifier (e.g., `protein:<contig>_<index>`)
- length: Amino-acid length (Integer)
- proteinId: Identifier without prefix (may be present)

Domain (PFAM Family)
- id: PFAM family ID (accession or ID string)
- pfamAccession: PFAM accession (e.g., `PF00589`), if present
- name: Family name, if present
- description: Family description

DomainAnnotation (per-protein domain hit)
- id: Annotation identifier (opaque)

KEGGOrtholog
- id: KO identifier (e.g., `K06966`)
- description: KO description

Pathway
- id: Pathway identifier (string)

Bgc (GECCO BGC)
- id, bgcId: Identifiers
- bgcProduct: Product type
- contig: Contig identifier
- startCoordinate, endCoordinate: Integers
- lengthNt, proteinCount: Integers
- averageProbability, maxProbability: Floats
- alkaloidProbability, nrpProbability, polyketideProbability, rippProbability, saccharideProbability, terpeneProbability: Floats
- domains: Semicolon-separated PFAM list

QualityMetrics (per-Genome QUAST)
- quast_totalLength: Integer
- quast_n50, quast_n90, quast_l50, quast_l90: Integers
- quast_numContigs, quast_largestContig, quast_contigs1kbPlus, quast_contigs5kbPlus, quast_contigs10kbPlus, quast_contigs25kbPlus, quast_contigs50kbPlus: Integers
- quast_gcContent, quast_auN, quast_nsPer100kb: Floats

Cazymeannotation (per-protein CAZy hit)
- id: Annotation identifier
- cazymeType: One of `GH|GT|PL|CE|AA|CBM`
- familyId: e.g., `GH3`, `GT2`
- substrateSpecificity: String
- evalue, coverage: Floats
- startPosition, endPosition, hmmLength: Integers

Cazymefamily
- familyId: e.g., `GH3`
- cazymeType: `GH|GT|PL|CE|AA|CBM`
- substrateSpecificity: String

### Relationships (Direction Is Critical)

- (p:Protein)-[:ENCODEDBY]->(g:Gene)
- (p:Protein)-[:HASDOMAIN]->(da:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
- (p:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog)
- (ko:KEGGOrtholog)-[:PARTICIPATESIN]->(pw:Pathway)
- (g:Gene)-[:BELONGSTOGENOME]->(gen:Genome)
- (gen:Genome)-[:HASQUALITYMETRICS]->(qm:QualityMetrics)
- (gen:Genome)-[:HASBGC]->(bgc:Bgc)
- (g:Gene)-[:PARTOFBGC]->(bgc:Bgc)
- (g:Gene)-[:NEXT]-(g2:Gene)  // undirected adjacency along a contig
- (p:Protein)-[:HASCAZYME]->(ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily)

### Query Tips and Gotchas

- Coordinates: store as strings; always cast for math/sorting: `ORDER BY toInteger(g.startCoordinate)`.
- Avoid `id(node)`; use the `id` property on nodes. Neo4j’s `id()`/elementId() should not be used in templates.
- There is no Contig node; contig is a string property on Gene (e.g., `g.contig`).
- Domain path ALWAYS starts at Protein going out: `(p)-[:HASDOMAIN]->(da)-[:DOMAINFAMILY]->(d)`.
- Use KO via HASFUNCTION; PFAM via HASDOMAIN/DOMAINFAMILY.
 - LanceDB filtering (pfam): exclusion and inclusion use Domain description substring matching and PFAM id/accession lists. See `resources/cypher/pfam_flags_for_protein_ids.cypher`.

## ✅ Fast Path Grammar Tolerance + Fail‑Fast

- Natural phrasing tolerance (deterministic pre‑normalizer):
  - “within N genes” → `± N` flank
  - Removes non‑semantic asides like “, focusing on …” before THEN
  - Plural normalization for common markers (e.g., “integrases” → “integrase”)
- Fail‑fast mode (on by default):
  - `FAIL_FAST_ON_GRAMMAR_ERROR=1` aborts instead of falling back to FSM on grammar parse errors.
  - `FAIL_FAST_ON_TOOL_ERROR=1` aborts FSM on first tool compile/execute error.

## ✅ LanceDB Reporting & Debugging

- kNN stage is always reflected in session notes and final synthesis:
  - Session notes: `debug_data_flow/pre_synthesis_data_*.json` carries `knn_stats` (exclude/include summaries, topk, counts).
  - Formatted synthesis input includes a LanceDB header and a compact `filter:` / `include:` line.
  - Final answer appends a deterministic LanceDB postscript (seeds, topk, neighbors after filtering) even when zero.
- Logging: tool logs `KNN_FILTER: ns=..., needle=..., markers=[...] topk=..., nn=..., seeds=...`.

### Common Patterns (Copy/Paste)

Protein → Gene coordinates
```cypher
MATCH (p:Protein {id:$protein_id})-[:ENCODEDBY]->(g:Gene)
RETURN g.id AS gene_id, g.contig AS contig,
       toInteger(g.startCoordinate) AS start,
       toInteger(g.endCoordinate) AS end,
       g.strand AS strand
LIMIT 1;
```

Proteins with PFAM (by accession/ID substring)
```cypher
MATCH (d:Domain)
WHERE toLower(d.id) CONTAINS toLower($q)
   OR toLower(d.pfamAccession) CONTAINS toLower($q)
   OR toLower(d.description) CONTAINS toLower($q)
WITH collect(d.id) AS pfams
MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(dom:Domain)
WHERE dom.id IN pfams
RETURN DISTINCT p
LIMIT $limit;
```

Proteins with KO list
```cypher
MATCH (p:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog)
WHERE ko.id IN $kos
RETURN DISTINCT p
LIMIT $limit;
```

Windowed neighborhood by contig coordinates
```cypher
MATCH (g:Gene {contig:$contig})
WHERE toInteger(g.startCoordinate) >= $start
  AND toInteger(g.endCoordinate) <= $end
RETURN g
ORDER BY toInteger(g.startCoordinate);
```

K-step neighborhood (genes → proteins)
```cypher
MATCH (p:Protein {id:$protein_id})-[:ENCODEDBY]->(g:Gene)
CALL (g) { MATCH pth=(g)-[:NEXT*..$k]-(ng:Gene) RETURN DISTINCT ng }
OPTIONAL MATCH (np:Protein)-[:ENCODEDBY]->(ng)
WITH DISTINCT np, ng WHERE np IS NOT NULL
RETURN np AS protein
ORDER BY toInteger(ng.startCoordinate)
LIMIT $limit;
```

CAZy annotations
```cypher
MATCH (p:Protein)-[:HASCAZYME]->(ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily)
RETURN p.id AS protein_id, cf.familyId AS cazyme_family, ca.substrateSpecificity AS substrate;
```

Pathway membership
```cypher
MATCH (p:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog)-[:PARTICIPATESIN]->(pw:Pathway {id:$pathway})
RETURN DISTINCT p;
```

If a query returns zero rows, verify labels/relationships match this reference and that coordinate casts are applied.

## ✅ **FIXED: Intent Classification and Database Query Issues** 

**Problem**: The dynamic agent system was incorrectly classifying complex discovery queries like "Find five loci with integrases and tell me about them" as `PRESENCE_ABSENCE` instead of `SPATIAL_NEIGHBORHOOD`, and had method name mismatches causing query failures.

**Solution Implemented**:
- **Enhanced Intent Classification**: Added priority-based pattern matching that correctly identifies discovery queries with spatial keywords as `SPATIAL_NEIGHBORHOOD`
- **Fixed Method Calls**: Updated `SchemaResolver` to use `process_query()` instead of `run_query()` and made methods properly async
- **Improved Query Logic**: Better database query generation for integrase and other biological entity searches

**Files Fixed**:
- `src/llm/rag_system/core.py:1442-1450` - Enhanced intent classification with spatial keyword detection
- `src/llm/rag_system/schema_resolver.py:246-247` - Fixed method name and async execution  
- `src/llm/rag_system/agent_executor.py:1907-1925` - Improved integrase query with genomic context

**Result**: Complex queries like "Find five loci with integrases" now correctly trigger spatial analysis instead of simple presence/absence checks.

## ✅ **FIXED: Removed All Hardcoded Cypher Queries**

**Problem**: The dynamic agent system had hardcoded Cypher queries in the database execution tool, violating the core principle of using DSPy signatures for query generation.

**Solution**: 
- **Removed ALL hardcoded queries** from `agent_executor.py:_execute_database_query()`
- **Delegated to existing DSPy system**: Now uses `_execute_traditional_query_logic()` which properly uses QueryClassifier and ContextRetriever signatures
- **No more hardcoded biology**: No hardcoded KEGG IDs, protein names, or query patterns

**Result**: Database queries are now generated dynamically via DSPy signatures as intended, following the established pattern used throughout the system.

## ✅ **HIERARCHICAL ANALYSIS SYSTEM - IMPLEMENTED**

**🎉 The hierarchical analysis system has been successfully implemented!**

The system now uses intelligent sub-agent analysis instead of brute-force context stuffing:

- **GenomicChunkAnalyzer**: LLM-guided analysis of genomic chunks based on user questions
- **LociPrioritizer**: Ranks candidate loci by biological significance and user relevance  
- **HierarchicalGenomeAnalyzer**: Orchestrates the complete workflow with chunking → analysis → prioritization
- **AgentExecutor Integration**: `whole_genome_reader` now returns curated loci findings instead of raw data dumps

**📍 Location**: `src/llm/rag_system/hierarchical_analysis/`

**🔄 Data Flow**: Raw genomic data → Biological chunks → Sub-agent analysis → Loci prioritization → Curated synthesis

This replaces the broken 4MB+ context stuffing with ~10K tokens of highly relevant loci analysis.

## ✅ **SYNTHESIS ENHANCEMENT: Premium Model Context Optimization**

**🎉 Successfully implemented high-capacity compression threshold for comprehensive synthesis reports!**

**Previously**: The synthesis pipeline used a fixed 5,000 token compression threshold, severely underutilizing premium models like O3 (200K tokens) by sending only compressed summaries instead of detailed genomic data.

**Solution Implemented**: 
- **Premium Model Optimization**: 100,000 token compression threshold (20x increase from 5K)
- **Full Context Utilization**: Assumes premium model (~200K context) and maximizes detailed data transmission
- **Enhanced Loci Preservation**: All 15 identified loci with coordinates, gene counts, and biological features reach final synthesis
- **Comprehensive Reports**: Enables verbose, detailed synthesis matching the extensive analysis performed

**Performance Gains**:
- **Context Utilization**: O3 now receives full detailed genomic data instead of compressed summaries  
- **Report Quality**: System can produce comprehensive, detailed reports reflecting all analytical work
- **Data Preservation**: Complete biological context preserved for synthesis
- **Simple Configuration**: Single threshold optimized for premium model performance

**Technical Implementation**: 
- Simplified `_get_compression_threshold()` returns 100K token threshold
- Enhanced compression logic with biological data extraction in `_extract_detailed_loci_summary()`
- Assumes premium model configuration for optimal performance

**Location**: `src/llm/rag_system/memory/progressive_synthesizer.py:481-490`

---

## ✅ **FIXED: Contig Field Database Issue**

**Problem**: The COALESCE fallback in `whole_genome_reader.py:112` was allowing gene IDs to masquerade as contig identifiers when processing spatial genomic data, leading to false genome counting (e.g., "5 genomes" when there are only 4).

**Solution**: Removed the problematic fallback:
```sql
# OLD (problematic):
COALESCE(g.contig, g.id, 'unknown_contig') AS contig_id

# NEW (clean):
g.contig AS contig_id
```

**Benefits**: 
- Gene IDs can no longer appear as contig identifiers
- Any missing contig assignments will fail explicitly (revealing corruption)
- LLM analysis sees correct genome boundaries
- Prevents false genome counting in reports

**Status**: Fixed

---

## 📋 **MULTI-STAGE SYNTHESIS IMPLEMENTATION PLAN**

### **Phase 1: Fix Current Context (In Progress)**
1. **Debug tool result cache expansion** - verify the reference loading mechanism
2. **Check synthesis input format** - ensure expanded data reaches final LLM calls
3. **Enhance synthesis prompts** - explicitly instruct LLM to use detailed data

### **Phase 2: Multi-Stage Enhancement (Planned)**
1. **Stage 1: Initial Draft** - Generate preliminary report from summaries
2. **Stage 2: Data Retrieval** - Identify gaps and retrieve specific tool results  
3. **Stage 3: Enhanced Synthesis** - Combine draft with detailed data for comprehensive report

**Implementation Strategy**:
- Add `synthesis_stage` parameter to track synthesis phases
- Create DSPy signatures for data gap identification and targeted retrieval
- Implement iterative synthesis workflow with multiple API calls
- Cache intermediate results to avoid redundant processing

---

## Project Overview

This is a next-generation genomic intelligence platform that transforms microbial genome assemblies into intelligent, queryable knowledge graphs with LLM-powered biological insights. The system combines traditional bioinformatics workflows with AI agents and embedding-based vector similarity search to create a comprehensive 8-stage pipeline culminating in an intelligent question-answering system.

### Key Achievements (Current Status)
- **373,587 RDF triples** linking genomes, proteins, domains, and functions
- **1,145 PFAM families + 813 KEGG orthologs** with authoritative functional descriptions
- **287 KEGG pathways** with 4,937 KO-pathway relationships
- **10,102 proteins** with 320-dimensional ESM2 semantic embeddings
- **Sub-millisecond vector similarity search** with LanceDB
- **Production-ready bulk Neo4j loading** (48K nodes, 95K relationships in <10 seconds)
- **Apple Silicon M4 Max optimization** (~85 proteins/second processing)
- **Intelligent model allocation** (gpt-5-2025-08-07/gpt-5-mini-2025-08-07) for cost-optimized biological reasoning

## Environment Setup

**CRITICAL: Always activate the conda environment before running any commands!**

```bash
# Activate the genome-kg conda environment (REQUIRED)
source /Users/jacob/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh && conda activate genome-kg

# Verify environment is active (should show genome-kg)
conda info --envs | grep '*'
```

**All commands below assume the `genome-kg` environment is activated.**

## Core Commands

### Pipeline Execution
```bash
# Build complete knowledge graph from genomes in data/raw/
python -m src.cli build

# Resume from specific stage
python -m src.cli build --from-stage 3

# Load knowledge graph into Neo4j (recommended bulk loader)
python -m src.build_kg.neo4j_bulk_loader --csv-dir data/stage07_kg/csv

# Query with LLM-powered insights
python -m src.cli ask "What metabolic pathways are present in Escherichia coli?"
python -m src.cli ask "Find proteins similar to heme transporters"
```

### Testing
```bash
# Run all tests
python scripts/run_tests.py

# Quick smoke tests
python scripts/run_tests.py --smoke

# Run with coverage
python scripts/run_tests.py --coverage
```

## Agent System Architecture

### Unified Agent Execution
The system uses **UnifiedAgentExecutor** with dynamic tool chaining:
1. Agent examines current state and chooses next tool
2. Tool executes and returns results
3. Agent examines results and decides next action
4. Repeat until agent decides to synthesize final answer

### Available Tools
- **whole_genome_reader**: Spatial genomic analysis, operon detection, gene clustering
- **code_interpreter**: Statistical analysis, pattern detection, quantitative assessments
- **database_query**: Direct Neo4j/LanceDB queries for specific lookups
- **literature_search**: PubMed validation of novel findings
 - **neighborhood_extractor**: DB-backed neighborhoods (single/batch/windowed)
 - **annotation_discovery**: PFAM+KOFAM discovery + union protein fetch

## 🚨 CRITICAL PERFORMANCE OPTIMIZATION: Reference-Based Note Storage

### Current Problem: Massive Context Waste
The note-taking system currently stores **9.9M+ tokens** of repetitive tool results:
- Each `whole_genome_reader` call stores ~1.6M tokens of full genomic data
- Multiple steps store identical/similar datasets
- Synthesis spends 8+ minutes compressing redundant information
- Final synthesis only uses compressed summaries anyway

### Solution: Discovery-Focused Notes + Reference Storage

#### Implementation Plan:

1. **Tool Result Caching System**
   - Store large tool results in session files once: `session_data/tool_results/`
   - Assign unique IDs: `whole_genome_spatial_4919_genes`, `code_analysis_prophage_loci`
   - Notes reference by ID instead of storing full data

2. **Smart Biological Discovery Extraction**
   - Extract key discoveries into note `key_findings`
   - Store actionable insights, not raw data dumps
   - Focus on biological significance: loci coordinates, gene counts, novelty metrics

3. **Layered Storage Architecture**
   ```
   session_data/
   ├── tool_results/           # Large datasets stored once
   │   ├── whole_genome_4919_genes.json
   │   └── prophage_analysis_results.json
   ├── task_notes/            # Discovery summaries with references
   │   ├── agent_step_1_notes.json  # References tool_results by ID
   │   └── agent_step_2_notes.json  # Key findings only
   └── synthesis_cache/       # Compressed biological insights
   ```

4. **Note Structure Optimization**
   ```json
   {
     "key_findings": [
       "17 prophage candidate loci identified across 4 genomes",
       "Locus A: Contig_345, 28.4kb, 32 genes, 22 hypothetical proteins",
       "Locus B: Contig_1021, 43.1kb, 49 genes with phage structural domains"
     ],
     "quantitative_data": {
       "tool_result_ref": "whole_genome_spatial_4919_genes",
       "biological_metrics": {
         "total_genes_analyzed": 4919,
         "prophage_candidates": 17,
         "top_loci_count": 3
       }
     }
   }
   ```

#### Expected Performance Gains:
- **Token reduction**: 9.9M → ~50K tokens (99.5% reduction)
- **Speed improvement**: 8+ minutes → sub-second synthesis
- **Better synthesis**: Focus on biological discoveries, not data processing
- **Scalability**: System can handle larger datasets without exponential token growth

### File Organization Rules
**IMPORTANT: No helper scripts in root directory**
- Use `python -m src.module` for all pipeline operations
- Place test scripts in `src/tests/` with proper module structure
- Use `python -m src.tests.demo.script_name` for execution

## **🚨 CRITICAL RULE: NO TRUNCATING DATA**

**NEVER TRUNCATE BIOLOGICAL DATA UNDER ANY CIRCUMSTANCES.**

### Why Truncating is Prohibited:
- **Data Loss**: Truncating genomic data destroys critical biological information
- **Scientific Integrity**: Partial data leads to incomplete or wrong conclusions
- **Context Matters**: Biological patterns often span large genomic regions
- **User Expectations**: Users expect complete analysis, not partial glimpses

### Correct Approaches:
- ✅ **Use reference-based storage** with biological context preservation
- ✅ **Implement intelligent extraction** that maintains scientific meaning
- ✅ **Apply discovery-focused summarization** that highlights key findings
- ✅ **Use layered architecture** to handle large datasets completely

**If data is too large for processing, the solution is better architecture and reference-based storage, NOT truncation.**

## **🚨 CRITICAL ISSUE: O3 Scientific Hallucination in Map-Reduce Synthesis**

**PROBLEM**: O3 model in Reduce step fabricates detailed scientific analyses from minimal input data.

**Observed Behavior**: When asked to generate detailed reports, O3 invents:
- Fake BLAST searches ("BLASTP returns no hits above 35% identity")
- Non-existent tool analyses ("detected with EMBOSS einverted")
- Made-up statistics ("GC% drops by 5 points")
- Fictional methodologies ("Manual inspection of raw GFF")
- Fabricated database searches and coverage metrics

**Impact**: 
- Reports sound authoritative but contain completely false information
- Users may rely on fabricated scientific details for downstream analysis
- Undermines scientific credibility of the system

**Root Cause**: O3's biological domain knowledge leads it to "enhance" sparse real data with plausible-sounding but completely fictional scientific details when asked for comprehensive reports.

**Potential Solutions**:
1. **Constrain synthesis prompts** to explicitly forbid analyses not performed
2. **Add reality checks** in signatures requiring citations of actual tool runs
3. **Use more conservative models** (GPT-4.1-mini) for final synthesis to reduce hallucination
4. **Implement data validation** that flags unsupported claims

**Priority**: High - affects scientific integrity of all detailed reports

**Location**: `src/llm/rag_system/dspy_signatures.py` - `GenomicSelector` signature

## **🚨 GPT-5 COMPATIBILITY STATUS**

**CURRENT STATUS**: GPT-5 models configured but require library upgrades to function properly.

### **Root Cause**
- DSPy 2.6.27 internally injects `max_tokens` parameter even when we specify `max_completion_tokens`
- GPT-5 models **require** `max_completion_tokens` and **reject** `max_tokens`
- Current LiteLLM version's `drop_params = True` doesn't prevent DSPy's parameter injection

### **Solutions Implemented**
✅ **Compatibility wrapper**: `src/llm/rag_system/dspy_compat.py` automatically maps parameters  
✅ **LiteLLM param dropping**: Enabled globally to prevent parameter conflicts  
✅ **Model allocation updates**: All DSPy LM creation uses compatibility wrapper  

### **Required Upgrades for Full GPT-5 Support**
```bash
# Upgrade LiteLLM to latest (≥ v1.74.x)
pip install -U litellm

# Optionally try DSPy beta (may have better GPT-5 support)
pip install -U dspy-ai==3.0.0b3
```

### **Temporary Workaround**
System is configured for GPT-5 but will fall back to working models until libraries are upgraded.

---

## **TODO: Function Name Cleanup**

**IMPORTANT**: The function `quick_switch_to_o3()` in `src/llm/rag_system/memory/model_config.py` still has the old name but now calls GPT-5 models internally. This was kept to avoid breaking imports during the GPT-5 migration.

**Action Required**: 
- Rename `quick_switch_to_o3()` → `quick_switch_to_gpt5()` 
- Update all imports in `src/llm/rag_system/memory/__init__.py`
- Update any other references throughout codebase
- The function works correctly (calls GPT-5), just has misleading name

---

## Development Guidelines

### **Session Notes Location**
Session notes are stored in `data/session_notes/[SESSION_ID]/` where SESSION_ID can be found in the CLI output. This directory contains:
- Individual task notes
- Tool result references (not full data)
- Biological discovery summaries

### **🚨 REMOVE MISLEADING HYPOTHETICAL_COUNT FIELD**

**CRITICAL: The `hypothetical_count` field in genomic loci analysis is misleading and should be removed.**

**Problem**: The system reports `hypothetical_count=0` for loci that are clearly described as containing "multiple hypothetical genes" and "clusters of hypothetical proteins". This creates contradictory information that confuses LLM analysis.

**Root Cause**: The field is populated incorrectly or using wrong criteria, leading to systematic undercounting of hypothetical proteins in identified loci.

**Action Required**: Remove `hypothetical_count` field from:
- `InterestingLocus` data structures
- Genomic analysis output formatting  
- DSPy signature descriptions
- Debug output generation

**Locations to Clean**:
- `src/llm/rag_system/hierarchical_analysis/genomic_chunk_analyzer.py`
- `src/llm/rag_system/whole_genome_reader.py:39` (`GeneContext` dataclass)
- `src/llm/rag_system/memory/progressive_synthesizer.py` (debug formatting)

**Priority**: High - this field provides no useful information and creates analytical confusion

### **CRITICAL: DSPy Signature Development Guidelines**
**NEVER hardcode behavior for particular query types or use dummy data directly within DSPy signatures.**

#### Prohibited Practices:
- **Hardcoded genome names**: Never use specific genome IDs
- **Hardcoded query patterns**: Never embed specific WHERE clauses with actual dataset values
- **Hardcoded biological IDs**: Never use specific KEGG IDs, PFAM IDs in examples
- **Dataset-specific behavior**: Never create logic that only works with current test data

#### Required Approach:
- **Generic placeholders**: Use "[SPECIFIC_GENOME_ID_PROVIDED_BY_SYSTEM]", "[KEGG_ID]", "[PFAM_ID]"
- **Pattern-based examples**: Show query patterns that work with any data
- **Flexible logic**: Create signatures that work with any genomic dataset
- **Runtime determination**: Let the system determine actual values during execution

### Testing Requirements
- **Test-Driven Development**: Write tests first, then implement
- **100% Coverage**: All tests must pass before commit
- **Component Testing**: Unit tests with mocks, integration tests

## Data Structure
```
data/
├── raw/                    # Input genome assemblies
├── stage00_prepared/       # Validated inputs
├── stage01_quast/         # Quality metrics
├── stage02_dfast_qc/      # Taxonomic classification
├── stage03_prodigal/      # Gene predictions
├── stage04_astra/         # Functional annotations
├── stage05_gecco/         # BGC detection
├── stage06_dbcan/         # CAZyme annotations
├── stage07_kg/            # Knowledge graph (373K+ triples)
├── stage08_esm2/          # ESM2 embeddings (10K+ proteins)
└── session_notes/         # Agent execution sessions
    └── [SESSION_ID]/
        ├── task_notes/           # Discovery-focused notes with references
        ├── session_data/         # Large tool results stored once
        └── synthesis_cache/      # Compressed insights for reuse
```

## Performance Summary

### Apple Silicon M4 Max Optimization
- **ESM2 Processing**: 10,102 proteins in ~2 minutes (~85 proteins/second)
- **LanceDB Queries**: Sub-millisecond similarity search
- **Neo4j Bulk Loading**: 48K nodes + 95K relationships in <10 seconds
- **Knowledge Graph**: 373,587 triples with multi-database integration
- **Agent Synthesis**: Sub-second with reference-based notes (vs 8+ minutes with full context)

## Neo4j Database Setup & Management

### **Initial Setup**
Users need to configure Neo4j with their own credentials. The system assumes:
- Neo4j installed locally (homebrew recommended: `brew install neo4j`)
- Database accessible at `bolt://localhost:7687`
- Custom password set (default examples use `your_new_password`)

### **Database Backup/Restore Commands**
```bash
# Backup current database (requires stopping Neo4j)
brew services stop neo4j
mkdir -p /tmp/neo4j_backup_$(date +%Y%m%d)
neo4j-admin database dump neo4j --to-path=/tmp/neo4j_backup_$(date +%Y%m%d)
brew services start neo4j

# Clear database for new data
cypher-shell -u neo4j -p YOUR_PASSWORD "MATCH (n) DETACH DELETE n;"

# Restore from backup
brew services stop neo4j
neo4j-admin database load --from-path=/tmp/backup_dir/neo4j.dump neo4j --overwrite-destination=true
brew services start neo4j
```

### **Configuration Requirements**
- Environment variable: `NEO4J_PASSWORD=your_actual_password`
- Update connection strings in code to match user's setup
- Consider Docker alternative for standardized deployments

---

## 📋 DEVELOPMENT ROADMAP & TODO

### **🎯 HIGH PRIORITY - Database Integration**

#### **GlobDB/Anvi'o Integration**
- **Value**: Transform Bennu from annotation+AI to full systems biology platform
- **Scope**: Metabolic modeling, comparative genomics, publication-ready visualizations
- **Challenges**: Data format reconciliation, workflow integration
- **Impact**: Massive - enables systems-level biological questions
- **Status**: Planning phase

#### **ProtT5 Embeddings Integration**  
- **Value**: Access to GlobDB's ~700GB pre-computed ProtT5 embeddings
- **Scope**: Replace/complement ESM2 with existing high-quality embeddings
- **Implementation**: Vector database integration, embedding format conversion
- **Impact**: High - leverage massive pre-existing computational resource
- **Status**: Architecture design needed

### **🔬 ANNOTATION PIPELINE ENHANCEMENTS**

#### **CRISPR Analysis Integration**
- **Implementation**: Add as Stage 6.5 (before CSV generation for Neo4j)
- **Tools**: PILER-CR vs CRISPRCasFinder evaluation needed
- **Data Products**: CRISPR arrays, cas genes, spacer sequences, repeat structures
- **Integration**: Spatial analysis perfect for CRISPR array characterization
- **Priority**: High - major functional gap in current analysis

#### **Additional HMM Database Integration**
- **TIGRFAMs**: More specific functional predictions than PFAM - straightforward integration
- **COG Database**: Phylogenetic functional categories
  - *Question*: HMM files available or requires BLAST-like annotation?
- **eggNOG**: Typically requires DIAMOND BLAST rather than HMMs
- **Specialized DBs**: TCDB (transporters), MEROPS (peptidases), others
- **Implementation**: Parallel annotation tracks in Stage 4

### **⚡ PIPELINE EFFICIENCY IMPROVEMENTS**

#### **Processing Optimization**
- **Parallelization**: Better multi-genome processing, especially Stage 4 annotations
- **Dependency Management**: Smarter task orchestration
- **Error Recovery**: Graceful handling of partial failures
- **Memory Management**: Optimize for larger datasets
- **Batch Processing**: Improve efficiency for multi-genome comparative analysis

#### **Apple Silicon Optimization**
- **Current**: ESM2 ~85 proteins/second on M4 Max
- **Target**: Further optimize Metal Performance Shaders usage
- **Scope**: All compute-intensive stages (especially embeddings)

### **🏗️ ARCHITECTURAL ENHANCEMENTS**

#### **Data Integration Layer**
- **Annotation Reconciliation**: Handle conflicts between multiple annotation sources
- **Confidence Scoring**: Weight different annotation sources appropriately  
- **Provenance Tracking**: Maintain clear data lineage
- **Quality Metrics**: Automated validation of annotation consistency

#### **Agent System Scaling**
- **Batch Queries**: Optimize comparative analysis across genomes
- **Model Allocation**: Smarter cost/performance routing for different query types
- **Caching Strategy**: Better intermediate result storage for complex analyses
- **Memory Architecture**: Reference-based storage working well, continue optimizing

#### **Comparative Genomics Framework**
- **Synteny Detection**: Cross-genome gene order analysis
- **Phylogenetic Profiling**: Gene family evolution tracking
- **HGT Identification**: Horizontal gene transfer detection
- **Pan-genome Analysis**: Core/accessory genome characterization

### **🔬 RESEARCH & VALIDATION**

#### **Tool Evaluation**
- **PILER-CR vs CRISPRCasFinder**: Efficiency and accuracy comparison
- **ProtT5 vs ESM2**: Performance benchmarking for specific use cases
- **HMM vs BLAST**: COG/eggNOG annotation strategy decision
- **Pipeline Benchmarking**: Systematic performance profiling

#### **Biological Validation**
- **CRISPR Analysis**: Validate array detection and cas gene classification
- **Cross-database Consistency**: Ensure annotation coherence across sources
- **Comparative Analysis**: Validate synteny and HGT detection accuracy

### **📅 IMPLEMENTATION PRIORITY**

1. **Phase 1** (Immediate): GlobDB/ProtT5 integration architecture design
2. **Phase 2** (Short-term): CRISPR analysis integration (Stage 6.5)
3. **Phase 3** (Medium-term): Additional HMM databases (TIGRFAMs, COG)
4. **Phase 4** (Long-term): Comprehensive comparative genomics framework
5. **Ongoing**: Pipeline efficiency improvements and optimization

### **🚧 KNOWN ISSUES TO RESOLVE**

- **Database**: Null contig fields in Neo4j gene records (high priority)
- **Pipeline**: Improve error handling and recovery mechanisms
- **Memory**: Further optimize reference-based storage for very large datasets
- **Integration**: Design clean interfaces for external database connections
## Deterministic Intent Grammar & Obligation Ledger (Option 2)

What changed:
- Replaced brittle regex-based routing with a PEG/Lark grammar that deterministically parses:
  - locus cardinality `N` with comparators (`exactly`, `at least`, `at most`)
  - flanking window (`±k` or `flanking genes = k`)
  - multi-stage obligations: `LanceDB` / `embedding` / `kNN` (including `nearest/closest` neighbors and optional count)
  - negative filters like `not annotated as <marker> by pfam/kofam`
  - optional `literature search`

Why:
- Reproducibility, lower variance, and guaranteed execution of requested stages (e.g., LanceDB) without LLM heuristics.

Models/Files:
- `src/llm/options/intent_models.py` — typed Intent, Cardinality, Obligations (Pydantic).
- `src/llm/options/intent_grammar.py` — Lark grammar + transformer that emits an Intent.
- `src/llm/options/router.py` — now delegates to the grammar (flag `USE_GRAMMAR_ROUTER`, default on); contains a minimal regex fallback.
- `src/llm/options/obligations.py` — obligation ledger used by the fast path executor to enforce requested stages.
- `requirements.txt` — add `lark==1.1.7`.

Executor behavior:
- The fast path constructs an ObligationLedger from the parsed Intent.
- Finalization is deferred until all required obligations are marked done.
- For LanceDB stages, the ledger carries `nn`, `exclude_markers`, `exclude_namespace`. The fast path must satisfy this before synthesis; otherwise it escalates to the FSM with a clear reason.

Examples parsed correctly:
- “Find five loci with integrases … then perform a LanceDB search … nearest neighbors … not annotated as integrases by pfam.”
- “Find at least 5 terminase loci with ±4 flanking genes, then kNN 2 nearest each; literature search.”

Acceptance:
- No LLM calls to parse or route.
- LanceDB obligation (when present) cannot be optimized away; it becomes part of the ledger and must be satisfied before finalization.
- Backward-compatible: toggle `USE_GRAMMAR_ROUTER=False` to use the legacy regex fallback.

Limitations:
- AUTO cardinality policies are handled separately and are not part of this change.
- Grammar currently supports number words up to 20; extend as needed.

## LanceDB kNN as a First‑Class Tool + Obligation‑Aware Scheduling

What & why
- Added `lancedb_knn` tool with typed IO and manifest parity checks.
- Fast path and FSM now honor the obligation ledger: if a query requires LanceDB, the system cannot finalize until `lancedb_knn` runs.
- A finalization gate prevents narrative synthesis with unmet obligations.
- Startup fails fast when `lancedb_knn` is required but not registered.

Execution flow (fast path)
1) Grammar → `Intent` → `ObligationLedger`.
2) Deterministic tasks:
   - `SEED_SELECTION` (compiled Cypher)
   - `LANCEDB_KNN` (one batched query; KG Pfam filter; pick top‑nn)
   - `SYNTHESIS`
3) Gate: if any required obligation is unmet, abort; else finalize.

FSM fallback
- While obligations are pending, `available_tools` is narrowed (e.g., `['lancedb_knn']`), preventing cheap loops on `database_query`.
- Finalization gate refuses synthesis if obligations remain unmet.

Logs to expect
- `FAST_PATH: tasks=[...]`
- `TOOL_INVOCATION lancedb_knn seeds=... topk=... exclude=pfam:integrase`
- `LEDGER_UPDATE lancedb_knn.done=true`
- `FSM_ALLOWED_TOOLS: ['lancedb_knn']` (only if FSM engaged)

Troubleshooting
- If synthesis happens without a LanceDB call: check the finalization gate and that `lancedb_knn` is registered.
- If neighbors are empty after filtering: we do not re‑query LanceDB; shortfalls are reported or escalated as configured.

---

Current Fix Focus (Resume Notes)

What we implemented (deterministic fast path):
- Grammar‑driven routing with obligations (LanceDB kNN, exclude “not annotated as … by pfam/kofam”), extracting: marker, N, flank, nn, filters.
- Seeds → Neighborhoods → LanceDB kNN, all via compiled templates/tools:
  - seeds_by_marker: schema‑correct; returns `seed_protein_id`; migrated to scoped CALL for Neo4j (CALL (lowers) { … }, CALL (g) { … }).
  - batched_neighborhoods_gated: contig‑ordered neighbors, one row per seed; fixed variable scope (`sp`,`sg`), and return `coalesce(s.seed_protein_id, sp.id)`.
  - LanceDB: routed through first‑class `lancedb_knn` tool; single batched call; KG Pfam filter; meta carries `knn` + `neighbors_full` for synthesis.
- Finalization parity: fast path now uses the ProgressiveSynthesizer (heavy) so the final report is FSM‑quality; raw_data includes cards + `neighbors_full`.
- Per‑genome WGR guard: skip whole_genome_reader for genomes > 20MB (via QUAST totalLength); global reader skips large genomes.

Why annotations (PFAM/KO) are missing right now:
- We intentionally stubbed neighbors with a minimal, schema‑stable payload during bring‑up and set `annotation: ''` as a placeholder to avoid hallucinations. We did not join PFAM/KO in neighborhoods yet, so the synthesizer faithfully reports none.

Next steps (to resume work):
- Enrich neighborhoods with PFAM/KO (compact):
  - In `batched_neighborhoods_gated.cypher` add OPTIONAL MATCH for `(np)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)` and `(np)-[:HASFUNCTION]->(ko:KEGGOrtholog)`, and emit per‑neighbor `pfams` (e.g., `[d.id]`) and `kos` (e.g., `[ko.id]`).
  - Drop the `annotation: ''` placeholder from neighbor maps.
  - Optionally include seed PFAM/KO in seeds_by_marker (bounded: 1–2 labels) if cheap.
- Synthesis polish:
  - Teach the finalizer to display per‑neighbor `pfams`/`kos` when present (raw_data already flows into the heavy synthesizer).
  - Ensure `neighbors_full` (from LanceDB) is summarized with IDs + distances/similarities in the final narrative.
- Planner clamp (optional):
  - Restrict FSM template/tool proposals to the registry to eliminate invented names; keep forced fallback to `proteins_with_pfam` as a safety net.
- Logging sanity:
  - Keep `MFP_RESULT` emitted after kNN so `knn_present=true` reflects reality; retain `DB_TEMPLATE_EXECUTE` traces.

Quick verification checklist:
- Fast path: seeds_by_marker → batched_neighborhoods_gated → lancedb_knn → synthesized report (no FSM).
- Final report shows real seed IDs, per‑neighbor PFAM/KO (after enrichment), and kNN neighbors with distances.
- No Neo4j CALL scope deprecation warnings from seeds_by_marker; other templates may still need scoped CALL.
