# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with this advanced genomic AI platform.

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

**IMPORTANT: For testing the pipeline, just print the command and let the user run it - the output tokens will overwhelm Claude's context.**

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

## Pipeline Architecture

### 8-Stage Pipeline
0. **Input Preparation**: Validates and organizes genome assemblies
1. **Quality Assessment**: Assembly quality metrics with QUAST
2. **Taxonomic Classification**: CheckM-style completeness/contamination analysis
3. **Gene Prediction**: Protein-coding sequence prediction with Prodigal
4. **Functional Annotation**: HMM domain scanning against PFAM, KOFAM
5. **GECCO BGC Detection**: Biosynthetic gene cluster detection
6. **dbCAN CAZyme Annotation**: Carbohydrate-active enzyme annotation
7. **Knowledge Graph Construction**: RDF generation with 373K+ triples
8. **ESM2 Protein Embeddings**: 320-dimensional semantic embeddings with LanceDB

### Key Components
- **CLI Interface** (`src/cli.py`): Main entry point with `build` and `ask` commands
- **Ingest Modules** (`src/ingest/`): Stage-specific processing modules
- **Knowledge Graph** (`src/build_kg/`): RDF construction and Neo4j integration
- **LLM System** (`src/llm/`): DSPy-powered question answering with agentic capabilities

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
└── stage08_esm2/          # ESM2 embeddings (10K+ proteins)
```

## Advanced Features

### Intelligent Model Allocation
- **o3 for Complex Tasks**: Query generation, biological interpretation, synthesis
- **GPT-4.1-mini for Simple Tasks**: Classification, formatting, progress tracking
- **Cost Optimization**: Automatic model selection based on task complexity

### Agentic Task System
- **Task Graph Architecture**: DAG-based execution with dependency resolution
- **Intelligent Chunking**: Handles large datasets (>1000 items) with biological grouping
- **Error Handling**: Graceful degradation with TaskRepairAgent
- **Session Memory**: Persistent note-taking across complex analyses

### Multi-Stage Query Processing
- **Stage 1**: Neo4j finds annotated examples
- **Stage 2**: LanceDB similarity search using those as seeds
- **Result**: Combines structured annotations + sequence similarity

## Dependencies

### Core Tools
- **prodigal**: Gene prediction
- **QUAST**: Assembly quality assessment
- **PyHMMer**: Protein domain scanning
- **GECCO**: BGC detection (`mamba install -c bioconda gecco`)
- **dbCAN**: CAZyme annotation (`pip install dbcan`)

### Python Stack
- **typer**: CLI framework
- **neo4j**: Graph database client
- **lancedb**: Vector similarity search
- **dspy**: LLM structured prompting
- **torch**: ESM2 embeddings with MPS acceleration

## Development Guidelines

### **Session Notes Location**
Session notes are stored in `data/session_notes/[SESSION_ID]/` where SESSION_ID can be found in the CLI output. This directory contains:
- Individual task notes
- Detailed reports (in `detailed_reports/` subdirectory)  
- Synthesis notes and cross-task connections

### **CRITICAL: DSPy Signature Development Guidelines**
**NEVER hardcode behavior for particular query types or use dummy data directly within DSPy signatures.**

#### Prohibited Practices:
- **Hardcoded genome names**: Never use specific genome IDs like "Candidatus_Nomurabacteria_bacterium_RIFCSPHIGHO2_02_FULL_58_190_contigs" in signature examples
- **Hardcoded query patterns**: Never embed specific WHERE clauses with actual dataset values
- **Hardcoded biological IDs**: Never use specific KEGG IDs (K03406), PFAM IDs (PF00005), or other database-specific identifiers in examples
- **Dataset-specific behavior**: Never create logic that only works with the current dummy dataset

#### Required Approach:
- **Generic placeholders**: Use "[SPECIFIC_GENOME_ID_PROVIDED_BY_SYSTEM]", "[EXACT_TARGET_GENOME_VALUE]", "[KEGG_ID]", "[PFAM_ID]"
- **Pattern-based examples**: Show query patterns that work with any data, not specific to current test dataset
- **Flexible logic**: Create signatures that work with any genomic dataset, not just the current one
- **Runtime determination**: Let the system determine actual values during execution, not at signature design time

#### Rationale:
- **Maintainability**: Code breaks when datasets change
- **Reusability**: Signatures must work with different genomic datasets
- **Production readiness**: Real deployments use different data than development dummy datasets
- **Debugging clarity**: Hardcoded values mask actual system behavior and make debugging harder

### File Organization Rules
**IMPORTANT: No helper scripts in root directory**
- Use `python -m src.module` for all pipeline operations
- Place test scripts in `src/tests/` with proper module structure
- Use `python -m src.tests.demo.script_name` for execution

### Testing Requirements
- **Test-Driven Development**: Write tests first, then implement
- **100% Coverage**: All tests must pass before commit
- **Component Testing**: Unit tests with mocks, integration tests

## Performance Summary

### Apple Silicon M4 Max Optimization
- **ESM2 Processing**: 10,102 proteins in ~2 minutes (~85 proteins/second)
- **LanceDB Queries**: Sub-millisecond similarity search
- **Neo4j Bulk Loading**: 48K nodes + 95K relationships in <10 seconds
- **Knowledge Graph**: 373,587 triples with multi-database integration

### Biological Intelligence Quality
**Before**: Generic responses like "likely involved in metabolic pathways"
**After**: Sophisticated analysis with ESM2 similarity scores, genomic context, and authoritative annotations

## Recent Major Developments

### Latest: Advanced Model Allocation (January 2025)
- **Problem Solved**: GPT-4.1-mini generated naive queries with 0 results
- **Solution**: o3 generates biologically intelligent queries with flexible matching
- **Result**: 3x better query success rates with cost optimization

### Phase 1: Complete Database Integration (2025)
- **GECCO Migration**: Replaced AntiSMASH with Python-native GECCO
- **Multi-Database Support**: PFAM, KEGG, CAZyme, BGC annotations
- **Production Pipeline**: End-to-end 8-stage processing with 373K+ triples

### System Integration Completed
- **LanceDB Migration**: From FAISS to production-ready vector search
- **Functional Enrichment**: 1,145 PFAM + 813 KEGG orthologs with authoritative descriptions
- **Agentic RAG v2.0**: Task graph execution with intelligent chunking
- **Neo4j Production**: Bulk loading optimized for millions of nodes

## Known Issues & Future Work

### CRITICAL ISSUES (January 2025)

#### 1. **Spatial Genome Reading Data Loss**
- **Problem**: Aggressive context compression (67,961 tokens → 9 tokens) destroys spatial genomic data before analysis
- **Impact**: LLMs never see actual gene coordinates, hypothetical protein stretches, or spatial organization
- **Result**: No meaningful prophage/operon discovery possible
- **Fix**: Disable compression for spatial genomic data or implement smarter chunking

#### 2. **Data Truncation Limits**
- **Problem**: WholeGenomeReader limits to 1,000 genes per contig, truncating from 5,490 → 500 genes
- **Impact**: Missing 90% of genomic data needed for comprehensive spatial analysis
- **Result**: Prophage loci in truncated regions are never found
- **Fix**: Increase `max_genes_per_contig` to 10,000+ or remove arbitrary limits

#### 3. **Lost Contig Information**
- **Problem**: All genes grouped under 'unknown_contig' instead of real contig names
- **Impact**: Destroys spatial organization across actual contig boundaries
- **Result**: Cannot identify prophage insertion sites at contig junctions
- **Fix**: Ensure `g.contig` field is properly populated in Neo4j schema

#### 4. **Uninformative Note-Taking** ✅ FIXED
- **Problem**: Notes contained generic summaries instead of specific biological findings
- **Root Cause**: `_format_result_for_decision()` truncated tool results to 200 characters, destroying spatial data
- **Solution Applied**: Modified method to detect and preserve full spatial genomic content for note-taking
- **Impact**: Note-taking system now receives actual gene coordinates, hypothetical protein stretches, and spatial organization
- **Result**: LLMs can now identify and record prophage loci coordinates and biological patterns

#### 5. **Missing Code Interpreter**
- **Problem**: External code interpreter service not available
- **Impact**: All downstream operon prediction and statistical analysis fails
- **Result**: No automated pattern detection or scoring of candidate loci
- **Fix**: Either deploy code interpreter service or implement fallback analysis methods

### Available Enhancements
- **Prodigal Metadata**: Start codons, RBS motifs, quality metrics available for integration
- **Genome Quality Metrics**: QUAST metrics available but not yet integrated into knowledge graph
- **Operon Prediction**: Genomic context analysis capabilities present but not fully utilized

### Future Roadmap
- **Phase 3**: Advanced agent capabilities with knowledge gap discovery
- **Phase 4**: Large dataset optimization for metagenomes
- **Phase 5**: Production scaling with containerization and auto-scaling

## **✅ Enhancement Completed: Advanced Report Generation & Progressive Compression**

### **Issue Resolved**: Prophage Discovery Queries Now Trigger Detailed Reports
The system now automatically detects prophage discovery queries and generates comprehensive structured reports instead of compressed analysis responses.

### **Fixes Applied**:

#### **1. Multipart Report Routing Fix**
✅ **Fixed**: Added multipart report check **before** size-based routing in `_synthesize_from_raw_data()`
- Report trigger logic now checks user intent first, then falls back to size routing
- Added prophage-related keywords: `'prophage', 'phage', 'viral', 'operon', 'operons', 'spatial', 'genomic regions', 'discovery', 'find', 'explore', 'report'`

#### **2. Progressive Compression System** 
✅ **Implemented**: Replaced hardcoded 200-line limit with intelligent progressive compression
- **Automatic chunking**: Splits large contexts into 2-8 intelligent chunks based on compression ratio needed
- **Smart token allocation**: Calculates tokens per chunk (e.g., "60k tokens → 5 chunks of 12k each")
- **Priority-based content preservation**: Ultra-high priority for prophage/spatial data (100 points), high priority for functional annotations (50 points)
- **Detailed report mode**: Doubles token budget (100k → 200k) and uses minimal compression for "detailed report" requests

#### **3. Compression Bypass for Detailed Reports**
✅ **Added**: Detection system for detailed report requests
- Keywords: `'detailed report', 'full report', 'comprehensive report', 'detailed analysis', 'show me everything', 'all details', 'maximum detail', 'don't compress', 'no compression'`
- Automatically expands token budget and reduces compression for these requests

### **Technical Implementation**:
```python
# New progressive compression system
def _compress_context_for_synthesis(self, context: str, max_tokens: int = 100000, 
                                   is_detailed_report: bool = False) -> str:
    # Progressive chunking: 2-8 chunks based on compression ratio needed
    if compression_ratio > 0.8:    # Light: 2-3 large chunks
    elif compression_ratio > 0.5:  # Medium: 3-5 chunks  
    else:                          # Heavy: 5-8 chunks
    
    # Smart token allocation per chunk
    tokens_per_chunk = (max_tokens - 1000) // num_chunks
```

### **Expected Behavior**:
✅ **Working**: Queries like "Find operons containing prophage segments and give me a detailed report" now:
1. **Route correctly** to `whole_genome_reader` via agent-based tool selection
2. **Trigger multipart reports** instead of compressed analysis mode  
3. **Use progressive compression** with prophage/spatial data prioritized
4. **Expand token budgets** for detailed report requests (100k → 200k tokens)
5. **Generate structured reports** with multiple sections and comprehensive detail

### **Agent-Based Tool Selection**: ✅ **CONFIRMED WORKING**
- Binary YES/NO decisions eliminate infinite loops
- o3 provides sophisticated biological reasoning
- Prophage tasks route to `whole_genome_reader`
- Database queries route to `ATOMIC_QUERY`
- JSON parsing handles o3's detailed responses

## **🚨 CRITICAL DEVELOPMENT GUIDELINE: NO HARD-CODING**

**NEVER HARD-CODE BIOLOGICAL PATTERNS, KEYWORDS, OR BEHAVIOR UNLESS EXPLICITLY REQUESTED BY THE USER.**

### **Prohibited Hard-Coding Examples:**
- ❌ **Hard-coded gene detection**: `if gene.is_hypothetical:` or `if "hypothetical" in annotation:`
- ❌ **Hard-coded biological keywords**: `phage_keywords = ['integrase', 'capsid', 'tail']`
- ❌ **Hard-coded thresholds**: `min_hypothetical_pct = 60`, `window_size = 15`
- ❌ **Hard-coded pattern matching**: `if annotation.contains("transport")`
- ❌ **Hard-coded scoring rules**: `if len(genes) >= 3 and has_integrase:`
- ❌ **Hard-coded biological assumptions**: `if gc_content < 0.4: # likely_phage`

### **Why This Matters:**
1. **Biological diversity**: Real biological patterns are more complex than simple keywords
2. **Dataset independence**: Code should work with any genomic dataset, not just current test data
3. **LLM capabilities**: o3 can recognize biological patterns better than hard-coded rules
4. **Maintainability**: Hard-coded rules break when datasets or requirements change
5. **Scientific rigor**: Biological discoveries should come from evidence, not assumptions

### **Preferred Approach:**
- ✅ **LLM-based pattern recognition**: Let o3 analyze spatial genomic data
- ✅ **Configurable parameters**: Load thresholds from config files if needed
- ✅ **Evidence-based discovery**: Use actual sequence analysis, not keyword matching
- ✅ **Flexible queries**: Generate database queries based on user intent, not pre-defined patterns
- ✅ **Generic processing**: Write code that works with any biological annotation system

### **Exception: User-Requested Hard-Coding**
Hard-coding is acceptable ONLY when:
- User explicitly requests specific keywords or thresholds
- User provides specific biological criteria to implement
- User asks for reproduction of a specific published method

**Default behavior should always be flexible, adaptive, and driven by LLM analysis rather than pre-programmed assumptions.**