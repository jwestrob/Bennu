# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with this advanced genomic AI platform.

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
- **Intelligent model allocation** (o3/GPT-4.1-mini) for cost-optimized biological reasoning

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