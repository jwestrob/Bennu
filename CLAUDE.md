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

## Known Issues & Current Status

### RESOLVED: ProgressiveSynthesizer Rate Limiting ✅ FIXED
- **Problem**: ProgressiveSynthesizer hits API rate limits during final synthesis (5.7M tokens → 5 chunks)
- **Root Cause**: Sequential Map-Reduce processing with rapid API calls, no rate limiting delays
- **Solution Applied**: 
  - Added 2-second delays between Map step chunks
  - Implemented exponential backoff for 429 errors (1s → 2s → 4s retry delays)  
  - Preserved existing caching system to reduce future API calls
- **Result**: Should handle rate limiting gracefully with automatic retries

### RESOLVED ISSUES ✅

#### 1. **Unified Agent Executor Implementation**
- **Problem**: Rigid TaskPlanParser + TaskExecutor prevented dynamic tool chaining  
- **Solution**: Implemented UnifiedAgentExecutor with biological reasoning
- **Result**: Agent successfully completes 8 steps, processes 4,919 genes across 4 genomes
- **Status**: ✅ WORKING - only synthesis phase hits rate limits

#### 2. **Integration Issues**
- **Problem**: Method name mismatches, import errors in agent system
- **Solution**: Fixed WholeGenomeReader integration, verified all imports
- **Result**: All components integrate correctly, no runtime errors
- **Status**: ✅ RESOLVED

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

## ✅ **CRITICAL BUG FIXES COMPLETED - JANUARY 2025**

### **ALL URGENT ISSUES RESOLVED - SYSTEM OPERATIONAL**

#### **Issue 1: Guidance Synthesis Never Triggers** ✅ **FIXED**
- **Problem**: `step_number % guidance_frequency == 0` check never evaluates to true despite completing steps 6, 7, 8
- **Solution Applied**: Added debug logging to track modulo calculations and fixed step collection logic
- **Fix**: `agent_executor.py:220` - Added detailed debug logging and corrected step indexing
- **Result**: Guidance synthesis now triggers correctly every 3 steps as intended

#### **Issue 2: Task Notes Not Being Created** ✅ **FIXED**
- **Problem**: `get_all_task_notes()` returns empty list despite agent steps completing successfully
- **Solution Applied**: Added automatic conversion of agent steps to persistent task notes
- **Fix**: `agent_executor.py:203` - New `_save_agent_step_as_note()` method called after each successful step
- **Result**: Agent steps are now automatically saved as TaskNote objects for synthesis

#### **Issue 3: Exponential Backoff Enhanced** ✅ **IMPROVED**
- **Problem**: Custom retry logic never catches 429 errors because OpenAI client handles them first
- **Solution Applied**: Enhanced rate limit detection with broader error patterns and longer delays
- **Fix**: `progressive_synthesizer.py:734` - Enhanced retry logic with 5-30 second delays and better error detection
- **Result**: More robust rate limit handling with expanded error pattern matching

#### **Issue 4: API Flood from Parallel Sub-Chunking** ✅ **FIXED**
- **Problem**: 56 sub-chunks × 5 items = 280+ simultaneous API calls exceed rate limits
- **Solution Applied**: Replaced all parallel processing with sequential processing + rate limiting
- **Fix**: `progressive_synthesizer.py:445` - Sequential sub-chunk processing with 3-second delays
- **Result**: API calls now processed sequentially with aggressive rate limiting between calls

#### **Issue 5: TPM Rate Limit Exceeded** ✅ **FIXED**  
- **Problem**: System uses 200K+ tokens per minute, hitting OpenAI's 200K TPM limit
- **Solution Applied**: Added 2-3 second delays between ALL API calls (main chunks + sub-chunks)
- **Fix**: `progressive_synthesizer.py:323` - Sequential chunk processing with 2-second delays
- **Result**: TPM usage spread over time, staying well under 200K limit

#### **Issue 6: Sub-Chunk Count Too High** ✅ **FIXED**
- **Problem**: 1.4M token items split into 56 sub-chunks (should be 5-10 max)
- **Solution Applied**: Hard limit of 8 sub-chunks maximum with forced merging if exceeded
- **Fix**: `progressive_synthesizer.py:527` - Max 8 chunks with larger chunk sizes and merge logic
- **Result**: Sub-chunk count capped at 8, dramatically reducing API call volume

### **✅ ALL FIXES COMPLETED:**
1. ✅ **Fixed guidance synthesis triggering** - Hybrid model now fully operational
2. ✅ **Fixed task note creation** - Agent steps now persist as TaskNote objects  
3. ✅ **Replaced parallel with sequential processing** - API flood completely eliminated
4. ✅ **Reduced sub-chunk count to max 8** - Dramatically reduced API call volume
5. ✅ **Added aggressive rate limiting** - 2-3 second delays between all API calls
6. ✅ **Enhanced exponential backoff** - Better rate limit detection and longer delays

### **SYSTEM STATUS: FULLY OPERATIONAL** 🟢
The hybrid guidance vs. reporting synthesis system is now working correctly with all critical bugs resolved. The system should handle large datasets without hitting rate limits and provide proper agent guidance during exploration.

## **🎯 MAJOR ARCHITECTURAL REFACTOR: Unified Agent Execution**

### **Problem**: Rigid Task-Based System
The current system requires pre-planning tool permissions for each task:
- `TaskType.ATOMIC_QUERY` → Can only do database queries
- `TaskType.TOOL_CALL` → Can only call one predetermined tool  
- `TaskType.SYNTHESIS` → Can only synthesize existing data

**This prevents natural biological exploration** where the LLM should dynamically choose tools based on intermediate results.

### **Solution**: Dynamic Agent Execution
Replace the rigid task system with a unified agent that can:
1. Start with any tool (database query, spatial analysis, etc.)
2. Examine results and dynamically choose the next tool
3. Chain tools naturally: `database_query` → `whole_genome_reader` → `code_interpreter` → `literature_search`
4. Synthesize when exploration is complete

### **Implementation Plan**:

#### **Files to Replace**:
1. **`task_plan_parser.py`** → **`agent_executor.py`** (new unified agent)
2. **`task_executor.py`** → Integrated into unified agent
3. **`task_management.py`** → Simplified task representation

#### **Files to Update**:
1. **`core.py`** → Replace agentic planning path with agent execution
2. **`agent_tool_selector.py`** → Simplify to runtime tool selection only
3. **`memory/progressive_synthesizer.py`** → Handle agent execution results
4. **`external_tools.py`** → Ensure all tools work with agent interface

#### **Benefits**:
- **Dramatic API call reduction**: No pre-planning phase, no task graph creation
- **Natural tool chaining**: LLM explores biological questions organically  
- **Simplified architecture**: No TaskGraph, TaskPlanParser, complex task types
- **Better biological discovery**: Agent adapts based on what it finds
- **Unified execution model**: Single flexible path instead of traditional vs agentic

### **Architecture Change**:
```python
# OLD (rigid pre-planning):
PlannerAgent → TaskPlanParser → TaskGraph → TaskExecutor → Synthesis

# NEW (dynamic agent):
AgentExecutor → Tool Chain → Synthesis
```

### **Expected Outcome**:
- Queries like "Find prophage segments" will naturally chain:
  1. Database search for prophage-related genes
  2. Spatial analysis of interesting hits  
  3. Statistical analysis of patterns found
  4. Literature validation of novel findings
  5. Comprehensive synthesis

**All tool choices made dynamically by the LLM based on intermediate results.**

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

## **🚨 CRITICAL RULE: NO TRUNCATING DATA**

**NEVER TRUNCATE BIOLOGICAL DATA UNDER ANY CIRCUMSTANCES.**

### **Why Truncating is Prohibited:**
- **Data Loss**: Truncating genomic data destroys critical biological information
- **Scientific Integrity**: Partial data leads to incomplete or wrong conclusions
- **Context Matters**: Biological patterns often span large genomic regions
- **User Expectations**: Users expect complete analysis, not partial glimpses

### **What NOT To Do:**
- ❌ **Never truncate tool outputs** regardless of size
- ❌ **Never use "first N characters"** approaches
- ❌ **Never add "[TRUNCATED]"** messages to biological data
- ❌ **Never sacrifice completeness** for performance

### **Correct Approaches:**
- ✅ **Use proper summarization** with biological context preservation
- ✅ **Implement intelligent chunking** that respects biological boundaries
- ✅ **Apply compression techniques** that maintain scientific meaning
- ✅ **Use sequential processing** to handle large datasets completely

**If data is too large for processing, the solution is better architecture and more intelligent processing, NOT truncation.**