# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with this advanced genomic AI platform.

## Project Overview

This is a next-generation genomic intelligence platform that transforms microbial genome assemblies into intelligent, queryable knowledge graphs with LLM-powered biological insights. The system combines traditional bioinformatics workflows with AI agents and embedding-based vector similarity search to create a comprehensive 8-stage pipeline culminating in an intelligent question-answering system.

### Key Achievements
- **373,587 RDF triples** linking genomes, proteins, domains, and functions (UPDATED)
- **1,145 PFAM families + 813 KEGG orthologs** enriched with authoritative functional descriptions
- **287 KEGG pathways** with 4,937 KO-pathway relationships integrated
- **Enhanced GECCO BGC Integration** with 17 quantitative properties per BGC including confidence scores and product-specific probabilities (NEW)
- **BGC and CAZyme annotation support** with GECCO and dbCAN integration fully tested end-to-end
- **10,102 proteins** with 320-dimensional ESM2 semantic embeddings
- **Sub-millisecond vector similarity search** with LanceDB
- **Sophisticated BGC analysis** with GECCO probability scores, product prediction, and pathway reconstruction (NEW)
- **High-confidence biological insights** using DSPy-powered RAG system
- **Apple Silicon M4 Max optimization** (~85 proteins/second processing rate)
- **Production-ready bulk Neo4j loading** with 48K nodes and 95K relationships imported in <10 seconds (NEW)

## Environment Setup

**CRITICAL: Always activate the conda environment before running any commands!**

```bash
# Activate the genome-kg conda environment (REQUIRED)
source /Users/jacob/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh && conda activate genome-kg

# Verify environment is active (should show genome-kg)
conda info --envs | grep '*'
```

**All commands below assume the `genome-kg` environment is activated.**

## Development Commands

### Testing
```bash
# Run all tests
python scripts/run_tests.py

# Quick smoke tests during development
python scripts/run_tests.py --smoke

# Run with coverage analysis
python scripts/run_tests.py --coverage

# Run tests for specific modules
python scripts/run_tests.py --module ingest
python scripts/run_tests.py --module build_kg

# Run by test category
python scripts/run_tests.py --marker unit
python scripts/run_tests.py --marker integration

# Discover all available tests
python scripts/run_tests.py --discover

# Shell wrapper (equivalent commands)
./test.sh --smoke
./test.sh --coverage
```

### Pipeline Execution
```bash
# IMPORTANT: Ensure genome-kg environment is activated first!
# source /Users/jacob/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh && conda activate genome-kg

# Build knowledge graph from genomes in data/raw/
python -m src.cli build

# Resume from specific stage
python -m src.cli build --from-stage 3

# Run complete pipeline through functional annotation
python -m src.cli build --to-stage 4

# Run complete pipeline through GECCO BGC annotation
python -m src.cli build --to-stage 5

# Run complete pipeline through CAZyme annotation
python -m src.cli build --to-stage 6

# Run complete pipeline through knowledge graph construction
python -m src.cli build --to-stage 7

# Run complete pipeline through ESM2 embeddings
python -m src.cli build --to-stage 8

# Run ESM2 embeddings optimized for Apple Silicon M4 Max (~2 minutes)
python run_esm2_m4_max.py

# Skip taxonomic classification
python -m src.cli build --skip-tax

# Load knowledge graph into Neo4j database (bulk loader - RECOMMENDED)
python -m src.build_kg.neo4j_bulk_loader --csv-dir data/stage07_kg/csv

# Alternative: Legacy loader (slower but more flexible)
python -m src.build_kg.neo4j_legacy_loader --rdf-file data/stage07_kg/knowledge_graph.ttl

# Query the knowledge graph with LLM-powered insights
python -m src.cli ask "What metabolic pathways are present in Escherichia coli?"
python -m src.cli ask "What is the function of KEGG ortholog K20469?"
python -m src.cli ask "Find proteins similar to heme transporters"
```

### Individual Stage Execution
```bash
# Stage 0: Input preparation
python -m src.ingest.00_prepare_inputs --input-dir data/raw --output-dir data/stage00_prepared

# Stage 1: Quality assessment with QUAST
python -m src.ingest.01_run_quast --input-dir data/stage00_prepared --output-dir data/stage01_quast

# Stage 3: Gene prediction with Prodigal
python -m src.ingest.03_prodigal --input-dir data/stage00_prepared --output-dir data/stage03_prodigal

# Stage 4: Functional annotation with Astra
python -m src.ingest.04_astra_scan --input-dir data/stage03_prodigal --output-dir data/stage04_astra --databases PFAM KOFAM

# Stage 5: GECCO BGC detection (NEW)
python -m src.ingest.gecco_bgc --input-dir data/stage00_prepared --output-dir data/stage05_gecco

# Stage 6: dbCAN CAZyme annotation (NEW)
python -m src.ingest.dbcan_cazyme --input-dir data/stage03_prodigal/genomes/all_protein_symlinks --output-dir data/stage06_dbcan

# Stage 7: Knowledge graph construction with extended annotations (UPDATED)
python -m src.build_kg.rdf_builder --stage03-dir data/stage03_prodigal --stage04-dir data/stage04_astra --stage05a-dir data/stage05_gecco --stage05b-dir data/stage06_dbcan --output-dir data/stage07_kg

# Stage 8: ESM2 protein embeddings (Apple Silicon optimized)
python run_esm2_m4_max.py

# Test ESM2 embeddings and similarity search
python test_esm2_similarity.py data/stage08_esm2

# Monitor ESM2 embedding progress (separate terminal)
python monitor_esm2_progress.py
```

### Nextflow Execution
```bash
# Run with standard profile (conda environment)
nextflow run main.nf -profile standard

# Run with Docker containers
nextflow run main.nf -profile docker

# Run on HPC cluster with SLURM
nextflow run main.nf -profile cluster
```

## Architecture

### Pipeline Stages
The pipeline consists of 8 main stages executed sequentially:

0. **Input Preparation** (`src.ingest.00_prepare_inputs`): Validates and organizes genome assemblies
1. **Quality Assessment** (`src.ingest.01_run_quast`): Assembly quality metrics with QUAST
2. **Taxonomic Classification** (`src.ingest.02_dfast_qc`): CheckM-style completeness/contamination analysis with ANI-based taxonomy
3. **Gene Prediction** (`src.ingest.03_prodigal`): Protein-coding sequence prediction with Prodigal, creates `all_protein_symlinks` directory
4. **Functional Annotation** (`src.ingest.04_astra_scan`): HMM domain scanning against PFAM, KOFAM using astra/PyHMMer
5. **GECCO BGC Detection** (`src.ingest.gecco_bgc`): Biosynthetic gene cluster detection using GECCO (Gene Cluster Prediction with Conditional Random Fields) - Python-native alternative avoiding Docker compatibility issues (NEW)
6. **dbCAN CAZyme Annotation** (`src.ingest.dbcan_cazyme`): Carbohydrate-active enzyme annotation using dbCAN with comprehensive CAZy family classification (NEW)
7. **Knowledge Graph Construction** (`src.build_kg`): RDF generation with 373K+ triples linking genomes, proteins, domains, functions, BGCs, and CAZymes, enriched with authoritative functional descriptions and KEGG pathway integration (ENHANCED)
8. **ESM2 Protein Embeddings** (`src.ingest.06_esm2_embeddings`): Generate 320-dimensional semantic embeddings for 10K+ proteins using ESM2 transformer with Apple Silicon MPS acceleration, complete with LanceDB similarity search indices (sub-millisecond queries)
9. **LLM Integration** (`src.llm`): Question answering with DSPy combining structured Neo4j knowledge graph and semantic LanceDB protein search

### Key Components

**CLI Interface** (`src/cli.py`): Main entry point using Typer framework with two primary commands:
- `build`: Execute pipeline stages with configurable resume/skip options
- `ask`: Natural language question answering over genomic data

**Ingest Modules** (`src/ingest/`): Each stage implemented as standalone Python modules with consistent interfaces, JSON manifests for tracking processing state, and parallel execution support.

**Data Flow**: Structured pipeline with stage-specific output directories (`data/stage0X_name/`) and standardized manifest files for inter-stage communication.

**Testing System**: Zero-maintenance test discovery with automatic detection of new tests following pytest conventions. Tests organized by markers (unit, integration, slow, external) and modules.

## Data Structure

```
data/
├── raw/                    # Input genome assemblies (.fna, .fasta, .fa)
├── stage00_prepared/       # Validated inputs with processing_manifest.json
├── stage01_quast/         # Quality metrics and reports
├── stage02_dfast_qc/      # Taxonomic classification results
├── stage03_prodigal/      # Gene predictions (.faa, .genes.fna)
├── stage04_astra/         # Functional annotations (PFAM/KOFAM HMM hits)
├── stage05_gecco/         # GECCO BGC detection results (NEW)
├── stage06_dbcan/         # dbCAN CAZyme annotation results (NEW)
├── stage07_kg/            # Knowledge graph exports (RDF triples, 373K+ triples) (UPDATED)
└── stage08_esm2/          # ESM2 protein embeddings (10K+ proteins, 320-dim, LanceDB indices) (UPDATED)
```

## Performance Summary

### Apple Silicon M4 Max Optimization
- **ESM2 Embeddings**: 10,102 proteins processed in ~2 minutes (vs estimated 21 minutes)
- **Embedding Generation Rate**: ~85 proteins/second with MPS acceleration
- **LanceDB Similarity Search**: Sub-millisecond queries with rich metadata filtering
- **Knowledge Graph**: 373,587 RDF triples linking genomes, proteins, domains, functions, BGCs, and CAZymes (UPDATED)
- **Functional Enrichment**: 1,145 PFAM families + 813 KEGG orthologs + 287 KEGG pathways with authoritative descriptions (ENHANCED)
- **Multi-Database Integration**: GECCO BGC detection + dbCAN CAZyme annotation support (NEW)
- **Memory Efficiency**: Automatic MPS cache management prevents memory overflow

### Neo4j Database Performance
- **Bulk Import Speed**: 20.08 seconds for 37,930 nodes + 85,626 relationships (15x faster than Python MERGE)
- **CSV Conversion**: 2-3 seconds for RDF → Neo4j format transformation
- **Scalability**: Production-ready for millions of nodes using neo4j-admin import
- **Data Integrity**: All relationships and properties preserved correctly
- **Default Loader**: Now uses bulk loader by default for all knowledge graph construction

### Pipeline Throughput
- **Complete Pipeline**: Stages 0-8 process 4 genomes with 10K+ proteins (UPDATED)
- **GECCO BGC Detection**: 4 genomes processed with graceful error handling and workflow continuation (NEW)
- **Knowledge Graph Construction**: 373,587 triples generated from multi-stage annotations with BGC schema integration (ENHANCED)
- **Neo4j Database**: Enhanced schema supporting BGC and CAZyme annotations for complex biological queries (UPDATED)
- **LLM Integration**: High-confidence biological insights with authoritative source citations
- **Production Ready**: Comprehensive testing suite validates all outputs including end-to-end GECCO integration (ENHANCED)

### Biological Intelligence Quality
**Before Enhancement**: Generic responses like "likely involved in metabolic pathways"
**After Multi-Stage Integration**: Sophisticated analysis like "shows distant similarity to known heme-transport systems (ESM-2 0.373), requires additional evidence for definitive annotation"

### Multi-Stage Query Performance
- **Complex Functional Searches**: "Find proteins similar to heme transporters" processes 200 Neo4j annotations + 5 LanceDB similarity results
- **Automatic Seed Selection**: Uses top 3 annotated proteins as similarity seeds with deduplication
- **Biological Context Integration**: Combines annotation-based + sequence similarity-based evidence
- **Query Classification**: Intelligent routing between semantic/structural/hybrid approaches based on query type

## Dependencies

### Core Bioinformatics Tools
- **prodigal**: Gene prediction
- **QUAST**: Assembly quality assessment
- **dfast_qc**: Taxonomic classification (install: `conda install -c bioconda dfast_qc`)
- **PyHMMer**: Protein domain scanning via Astra
- **GECCO**: Biosynthetic gene cluster detection (install: `mamba install -c bioconda gecco hmmer`) (NEW)
- **dbCAN**: Carbohydrate-active enzyme annotation (install: `pip install dbcan`) (NEW)

### Python Packages
- **typer**: CLI framework
- **pydantic**: Data validation  
- **rdflib**: RDF graph manipulation
- **neo4j**: Graph database client
- **lancedb**: Vector similarity search and metadata filtering
- **dspy**: LLM structured prompting and RAG framework
- **torch**: ESM2 transformer models with MPS acceleration
- **transformers**: Hugging Face model integration
- **rich**: Beautiful terminal UI and progress tracking

## Configuration

**Nextflow Profiles**:
- `standard`: Local execution with conda environment
- `docker`: Containerized execution
- `singularity`: HPC-compatible containers
- `cluster`: SLURM cluster execution with GPU support
- `cloud`: AWS Batch execution

**Pipeline Parameters**: Configurable through `nextflow.config` including thread counts, memory limits, container registry, and processing options.

## Development Notes

- Each pipeline stage produces standardized outputs and can be run independently
- Test system automatically discovers new tests with zero maintenance overhead
- Pipeline supports resuming from any stage and selective execution
- All stages include parallel processing capabilities where applicable
- Knowledge graph construction transforms biological annotations to RDF triples for Neo4j

### **IMPORTANT: No Helper Scripts in Root Directory**
**Always use the proper pipeline instead of creating temporary helper scripts in the root directory.** 

**File Organization Rules:**
- **NO test scripts, demos, or temporary files in the root directory**
- Use `python -m src.build_kg.rdf_builder` for knowledge graph generation
- Use `python -m src.build_kg.neo4j_legacy_loader` for database loading
- Use `python -m src.ingest.gecco_bgc` for BGC detection (NEW)
- Use `python -m src.ingest.dbcan_cazyme` for CAZyme annotation (NEW)
- Place test scripts in `src/tests/` with proper module structure
- Place demo scripts in `src/tests/demo/` or similar organized location
- Use `python -m src.tests.demo.script_name` for execution with proper imports
- Modify existing modules in `src/` as needed rather than creating one-off scripts

**Examples of Proper Organization:**
- ✅ `src/tests/demo/test_agentic_demo.py` - Demo scripts in organized test structure
- ✅ `src/tests/integration/test_full_pipeline.py` - Integration tests
- ✅ `src/tests/test_dbcan_integration.py` - New database integration tests (NEW)
- ✅ `python -m src.tests.demo.test_agentic_demo` - Proper execution with module path
- ❌ `test_something.py` in root - Clutters repository and breaks import paths
- ❌ `demo.py` in root - Poor organization and maintenance issues

This maintains workflow integrity, prevents repository fragmentation, and ensures proper Python module resolution.

## 🚧 **CURRENT WORK: Query System Fixes (January 2025)** 🚧

**Status**: Critical fixes in progress for query routing and genome scoping issues

### **Issue Identified**: Query System Cascade Failures

After implementing intelligent routing, genome scoping, and context compression improvements, several critical issues emerged that broke the query system:

#### **Problems Discovered**:
1. **❌ Genome ID Mismatch**: Overly aggressive fuzzy matching incorrectly maps user-specified genome IDs 
   - Example: User requests `PLM0_60_b1_sep16_Maxbin2_047_curated` → System detects `Acidovorax_64`
   - Impact: Queries search wrong genome, return no results

2. **❌ Query Processing Cascade**: ERROR messages being passed as executable queries
   - DSPy generates queries with comments → Query processor rejects → Creates "ERROR:" string → Tries to execute as Cypher
   - Impact: Syntax errors requiring TaskRepairAgent intervention

3. **❌ Conflicting Constraints**: Scope enforcer adds impossible query logic
   - Final query: `genome.id = 'Acidovorax_64' AND genome.id STARTS WITH 'PLM0_60_b1_sep16_Maxbin2_047_curated'`
   - Impact: No results possible due to contradictory constraints

#### **Fix Strategy In Progress**:

**Phase 1: Critical Stability Fixes** 🔧
- [x] **Conservative Genome Detection**: Remove fuzzy matching, use exact matches only
- [x] **Query Validation**: Prevent ERROR messages from being executed as queries
- [x] **Scope Logic Validation**: Check for existing constraints before adding new ones
- [x] **Fallback Logic**: Try unscoped queries when scoped ones return no results

**Phase 2: Graceful Degradation** 🛡️
- [x] **Fallback Queries**: Try unscoped version when scoped query returns no results
- [ ] **Progressive Simplification**: Simplify queries when complex ones fail
- [ ] **Better Error Messages**: Clear feedback when genome IDs not found

**Phase 3: Robust Prevention** 🔒
- [ ] **Pre-validation**: Check genome IDs against actual database before query generation
- [ ] **Query Testing**: Validate Cypher syntax before execution
- [ ] **Integration Tests**: Comprehensive testing of genome scoping edge cases

#### **Files Being Modified**:
- `src/llm/rag_system/genome_scoping.py` - Conservative genome detection
- `src/llm/query_processor.py` - Query validation and error handling
- `src/llm/rag_system/core.py` - Fallback logic and integration

#### **Expected Timeline**:
- **Phase 1 (Critical)**: 1-2 hours - System functional again
- **Phase 2 (Enhancement)**: 2-3 hours - Graceful error handling
- **Phase 3 (Prevention)**: 1-2 days - Comprehensive testing and validation

**Objective**: Restore system functionality while maintaining the benefits of intelligent routing and context compression, but with more conservative and validated approaches.

#### **✅ CRITICAL FIXES COMPLETED**:

1. **Conservative Genome Detection** 🎯
   - **Fixed**: Removed aggressive fuzzy matching that incorrectly mapped `PLM0_60_b1_sep16_Maxbin2_047_curated` → `Acidovorax_64`
   - **Solution**: Exact matching only with format normalization (. vs _)
   - **Result**: Genome IDs now correctly identified or gracefully rejected

2. **Query Validation** 🔒
   - **Fixed**: ERROR messages being passed as executable Cypher queries
   - **Solution**: Raise exceptions instead of returning ERROR strings that get executed
   - **Result**: Malformed queries now fail gracefully instead of causing syntax errors

3. **Scope Enforcement** ⚖️
   - **Fixed**: Conflicting constraints like `genome.id = 'A' AND genome.id STARTS WITH 'B'`
   - **Solution**: Check for existing genome constraints before adding new ones
   - **Result**: No more impossible query logic

4. **Fallback Logic** 🛡️
   - **Added**: Try unscoped queries when scoped ones return no results
   - **Solution**: Progressive query simplification with clear metadata
   - **Result**: Better coverage when genome scoping is too restrictive

#### **🛠️ ADDITIONAL FIXES APPLIED**:

5. **DSPy Signature Hardening** 🔧
   - **Fixed**: Added explicit NO COMMENTS instructions to ContextRetriever
   - **Solution**: Clear prohibition of comments, section headers, explanatory text
   - **Result**: DSPy now has explicit instructions to generate clean executable Cypher

6. **Genome Regex Patterns** 🎯
   - **Fixed**: Regex was matching "genome id" → capturing "id" instead of full genome name
   - **Solution**: Reordered patterns with PLM0 patterns first, added minimum length requirements
   - **Result**: Now correctly extracts `PLM0_60_b1_sep16_Maxbin2_047_curated` from query

7. **Aggressive Comment Stripping** 🧹
   - **Added**: Safety net comment stripping before query processing
   - **Solution**: Strip `//`, `--`, and `/* */` comments aggressively
   - **Result**: Even if DSPy generates comments, they're removed before execution

#### **🔧 FINAL CRITICAL FIXES**:

8. **Genome ID Mapping** 🎯
   - **Fixed**: System correctly extracts genome ID but fails to map to database ID
   - **Issue**: User `PLM0_60_b1_sep16_Maxbin2_047_curated` → Database `PLM0_60_b1_sep16_Maxbin2_047_curated_contigs`
   - **Solution**: Added suffix mapping logic to handle `_contigs`, `.contigs`, etc.
   - **Result**: Now correctly maps user input to actual database identifiers

9. **Query Constraint Replacement** ⚖️
   - **Fixed**: Query uses wrong field (`genomeId`) and wrong value (unmapped ID)
   - **Issue**: `{genomeId:'PLM0_60_b1_sep16_Maxbin2_047_curated'}` should be `{id:'PLM0_60_b1_sep16_Maxbin2_047_curated_contigs'}`
   - **Solution**: Replace existing constraints instead of skipping them
   - **Result**: Queries now use correct field (`id`) with correct mapped genome ID

**Status**: System now has **9 comprehensive fixes** covering genome detection, mapping, query generation, constraint replacement, and fallback logic. The complete pipeline should now work correctly for the original failing query.

## Recent Major Developments: Phase 1 Database Integration Complete ✅

**🎉 COMPLETED**: Phase 1 Database Integration - Complete 8-stage pipeline with GECCO BGC detection and dbCAN CAZyme annotation

### Latest Achievement: GECCO Migration & Multi-Database Integration (July 2025) ✅

#### **🎉 MAJOR MILESTONE: AntiSMASH → GECCO Migration Complete** ✅
- **✅ Complete Replacement**: Successfully replaced AntiSMASH with GECCO for BGC detection
- **✅ Docker Issues Eliminated**: Python-native GECCO avoids ARM64/AMD64 Docker compatibility problems
- **✅ End-to-End Testing**: Full validation from Stage 5 GECCO → Stage 7 Knowledge Graph
- **✅ Graceful Error Handling**: Robust fallback system handles pyrodigal compatibility issues
- **✅ Production Validation**: 4 genomes processed, 373K+ RDF triples with BGC schema integration
- **✅ Workflow Integrity**: Pipeline continues successfully despite tool compatibility challenges

#### **1. Complete dbCAN CAZyme Integration** ✅
- **✅ Parser**: `src/ingest/dbcan_cazyme.py` with full Docker/conda execution support
- **✅ RDF Integration**: Extended `src/build_kg/rdf_builder.py` with CAZyme classes and annotations
- **✅ Functional Enrichment**: Added CAZy family descriptions to enrichment pipeline
- **✅ CLI Integration**: Stage 6 "dbCAN CAZyme Annotation" fully operational
- **✅ Testing**: Comprehensive test suite validates all functionality

#### **2. Complete GECCO BGC Integration** ✅
- **✅ Parser**: `src/ingest/gecco_bgc.py` with Python-native implementation
- **✅ RDF Integration**: Extended RDF builder with BGC and BGCGene classes
- **✅ CLI Integration**: Stage 5 "GECCO BGC Detection" fully operational
- **✅ Error Handling**: Graceful handling of pyrodigal compatibility issues with fallback support

#### **3. Extended Knowledge Graph Architecture** ✅
- **✅ Enhanced Function**: `build_knowledge_graph_with_extended_annotations()` supports all annotation types
- **✅ Backward Compatibility**: Original functions preserved and working
- **✅ Pipeline Integration**: Stage 7 builds unified knowledge graph with 373,587 triples
- **✅ Namespace Support**: Proper URIs for BGC and CAZyme annotations

#### **4. Complete 8-Stage Pipeline** ✅
- **Stage 0**: Input preparation ✅
- **Stage 1**: Quality assessment ✅
- **Stage 2**: Taxonomic classification ✅
- **Stage 3**: Gene prediction ✅
- **Stage 4**: Functional annotation ✅
- **Stage 5**: GECCO BGC detection ✅ **NEW**
- **Stage 6**: dbCAN CAZyme annotation ✅ **NEW**
- **Stage 7**: Knowledge graph construction ✅ **ENHANCED**
- **Stage 8**: ESM2 protein embeddings ✅

## Previous Major Developments: Complete System Integration ✅

**🎉 COMPLETED**: Comprehensive platform transformation with functional enrichment and LLM integration

### Successfully Implemented:

#### 1. **LanceDB Migration** ✅
- **Objective**: Migrate from FAISS to LanceDB for vector similarity search
- **Benefits**: Eliminated numpy version conflicts, enabled unified environment, improved MLOps standards
- **Result**: Sub-millisecond protein similarity queries with rich metadata filtering

#### 2. **Functional Enrichment Integration** ✅  
- **Objective**: Replace hardcoded biological knowledge with authoritative reference databases
- **Implementation**: Built `src/build_kg/functional_enrichment.py` module with PFAM Stockholm and KEGG KO list parsers
- **Result**: 1,145 PFAM families + 813 KEGG orthologs enriched with authoritative descriptions
- **Knowledge Graph Growth**: Enhanced from ~242K to 276,856 RDF triples

#### 3. **Neo4j Knowledge Graph Database** ✅
- **Objective**: Production-ready graph database for complex biological queries
- **Implementation**: 44,477 nodes and 45,679 relationships with proper biological ontologies
- **Features**: Optimized Cypher queries, functional annotation integration, genomic neighborhood analysis

#### 4. **DSPy RAG System Enhancement** ✅
- **Objective**: Build intelligent question-answering system with biological expertise
- **Implementation**: Schema-aware query generation, multi-modal data source integration
- **Critical Fix**: Updated query execution logic to utilize Neo4j data for all query types (not just structural)
- **Result**: High-confidence biological insights instead of generic responses

#### 5. **Multi-Stage Query Processing** ✅
- **Objective**: Enable complex functional similarity searches via keyword→similarity expansion
- **Implementation**: Enhanced DSPy prompts with multi-stage capability, updated query processing logic
- **Capabilities**: 
  - Stage 1: Neo4j finds annotated examples (e.g., "heme transporters")
  - Stage 2: Uses those proteins as seeds for LanceDB similarity search
  - Result: Comprehensive analysis combining annotations + sequence similarity
- **Example**: "Find proteins similar to heme transporters" → 200 Neo4j annotations + 5 LanceDB similarity results
- **Performance**: Automatic seed deduplication, similarity scoring, and biological interpretation

### Biological Intelligence Transformation:

**Before Enhancement**:
```
"This protein is likely involved in a metabolic pathway and may have 
evolutionary significance across various organisms."
```

**After Multi-Stage Enhancement**:
```
"Among the sequences supplied, only OD1_41_01_41_220_01_scaffold_7 
(6208–9111 bp, – strand) shows even distant similarity to known heme-transport 
systems, based on its positional similarity score (ESM-2 0.373) to 
Acidovorax_64_scaffold_14_362, a protein from Burkholderiales where heme 
utilization systems are common. The protein should be flagged as a 'putative 
heme-transport related protein' pending additional evidence."
```

### Current System Capabilities:
- **Multi-Stage Query Processing**: Keyword→similarity expansion for comprehensive functional searches
- **Dual-Database Integration**: Neo4j structured queries + LanceDB semantic search with intelligent routing
- **Biological Knowledge**: 1,145 PFAM families + 813 KEGG orthologs with authoritative annotations
- **Advanced Context**: ESM2 similarity scoring, genomic neighborhoods, functional relationships
- **Production Performance**: Apple Silicon M4 Max optimized with sub-millisecond similarity queries
- **Comprehensive Testing**: Zero-maintenance test discovery with multi-stage query validation

### Advanced Features Now Available:
- **Multi-Stage Semantic Search**: Stage 1 (Neo4j annotations) → Stage 2 (LanceDB similarity expansion)
- **Intelligent Query Classification**: Automatic routing between semantic/structural/hybrid approaches
- **Enhanced Context Formatting**: Similarity score interpretation, biological significance analysis
- **Complex Functional Queries**: "Find proteins similar to heme transporters" → 200 annotations + 5 similar proteins
- **Functional Enrichment**: Real-time integration of reference database annotations
- **Apple Silicon Optimization**: ~85 proteins/second ESM2 processing

This represents a complete transformation from a basic bioinformatics pipeline to an intelligent genomic AI platform.

## 🧠 **Agentic Task Graph Architecture** (Comprehensive Documentation)

**Status**: Production-ready agentic execution system with intelligent task orchestration, error handling, and context persistence.

### **System Overview**

The agentic task graph system transforms complex user queries into structured, executable workflows using a Directed Acyclic Graph (DAG) of interdependent tasks. This enables sophisticated multi-step analyses while maintaining biological context and providing robust error handling.

---

## **🔧 Core Architecture Components**

### **1. Task Management System** (`src/llm/rag_system/task_management.py`)

**TaskGraph Class**: Central orchestrator for DAG-based task execution
```python
class TaskGraph:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}        # Task registry by ID
        self.execution_order: List[str] = []    # Sequential execution tracking
```

**Task Class**: Individual executable units with dependencies
```python
@dataclass
class Task:
    task_id: str                      # Unique identifier
    task_type: TaskType              # ATOMIC_QUERY or TOOL_CALL
    description: str                 # Natural language description
    dependencies: List[str]          # IDs of prerequisite tasks
    status: TaskStatus              # PENDING/RUNNING/COMPLETED/FAILED/SKIPPED
    result: Optional[Any]           # Execution results
    error: Optional[str]            # Error messages
    
    # Type-specific fields
    query: Optional[str]            # For ATOMIC_QUERY tasks
    tool_name: Optional[str]        # For TOOL_CALL tasks
    tool_args: Dict[str, Any]       # Tool execution parameters
```

**Task Status Lifecycle**:
- `PENDING` → `RUNNING` → `COMPLETED` (success path)
- `PENDING` → `RUNNING` → `FAILED` → dependent tasks marked `SKIPPED`

### **2. Task Plan Parser** (`src/llm/rag_system/task_plan_parser.py`)

**Purpose**: Bridges DSPy natural language planning with executable task structures.

**Core Functionality**:
```python
class TaskPlanParser:
    def parse_dspy_plan(self, plan_text: str) -> ParsedPlan:
        # Extract numbered steps from DSPy output
        steps = self._extract_numbered_steps(plan_text)
        
        # Convert to executable Task objects
        tasks = []
        for step_num, description in steps:
            task = self._create_task_from_description(
                step_num=step_num,
                description=description,
                previous_tasks=tasks
            )
            tasks.append(task)
```

**Pattern Recognition**:
- **Query Patterns**: `retrieve|query|find|search|database|genome|protein`
- **Tool Patterns**: 
  - `code_interpreter`: `analyz|statistical|matrix|visualiz|python|pandas`
  - `literature_search`: `literature|research|papers|pubmed|recent`

**Dependency Inference**:
- **Sequential**: Each task depends on the previous task by default
- **Future Enhancement**: Parse explicit references like "using data from step 2"

### **3. Task Executor** (`src/llm/rag_system/task_executor.py`)

**Responsibility**: Executes individual tasks based on their type with comprehensive error handling.

**Execution Flow**:
```python
async def execute_task(self, task: Task) -> ExecutionResult:
    # Enhanced logging for debugging
    if len(task.task_id) > 100:
        logger.warning("⚠️ LONG TASK ID DETECTED: recursive splitting detected")
    
    # Route based on task type
    if task.task_type == TaskType.ATOMIC_QUERY:
        result = await self._execute_query_task(task)
    elif task.task_type == TaskType.TOOL_CALL:
        result = await self._execute_tool_task(task)
    
    # Store results for dependent tasks
    self.completed_results[task.task_id] = result
    
    # Consider note-taking for session persistence
    if self.note_keeper and self.noting_decision:
        await self._consider_note_taking(task, execution_result)
```

---

## **🔄 Dynamic Graph Construction & Execution**

### **Graph Building Process**

1. **DSPy Planning**: User query → DSPy PlannerAgent → structured plan text
2. **Plan Parsing**: TaskPlanParser extracts numbered steps → Task objects
3. **Dependency Resolution**: Automatic inference of task dependencies
4. **Graph Assembly**: Tasks added to TaskGraph with dependency links
5. **Execution Planning**: DAG analysis determines parallel execution batches

### **Parallel Execution Strategy**

```python
def get_execution_plan(self) -> List[List[str]]:
    """Generate execution plan as batches of tasks that can run in parallel."""
    plan = []
    remaining_tasks = set(self.tasks.keys())
    
    while remaining_tasks:
        # Find tasks with satisfied dependencies
        current_batch = []
        for task_id in remaining_tasks:
            if all_dependencies_completed(task_id):
                current_batch.append(task_id)
        
        # Execute batch in parallel
        plan.append(current_batch)
        remaining_tasks -= set(current_batch)
```

**Advantages**:
- **Parallelization**: Independent tasks execute concurrently
- **Dependency Safety**: Never executes task before dependencies complete
- **Circular Detection**: Identifies and breaks circular dependencies

### **Intelligent Chunking Integration**

When tasks encounter large datasets (>1000 items), the system triggers intelligent chunking:

```python
# In TaskExecutor._execute_query_task()
if (context and hasattr(context, 'structured_data') and 
    not getattr(task, '_already_chunked', False) and
    not getattr(task, '_intelligent_chunked', False)):
    
    if len(raw_data) > 1000:  # Chunking threshold
        logger.info("🧠 Large dataset detected, using intelligent upfront chunking")
        
        chunking_manager = IntelligentChunkingManager(max_chunks=4, min_chunk_size=100)
        chunks = await chunking_manager.analyze_and_chunk_dataset(task, raw_data, task.description)
        
        # Execute chunks in parallel, synthesize results
        chunk_results = await chunking_manager.execute_chunked_analysis(chunks, self, task)
        synthesis = chunking_manager.synthesize_chunk_results(chunk_results, task.description, chunks)
```

---

## **⚠️ Comprehensive Error Handling**

### **Multi-Level Error Management**

#### **1. Task-Level Error Handling**
```python
async def execute_task(self, task: Task) -> ExecutionResult:
    try:
        # Task execution logic
        result = await self._execute_query_task(task)
        return ExecutionResult(task_id=task.task_id, success=True, result=result)
    except Exception as e:
        error_msg = f"Task {task.task_id} failed: {str(e)}"
        logger.error(error_msg)
        return ExecutionResult(task_id=task.task_id, success=False, error=error_msg)
```

#### **2. Dependency Cascade Handling**
```python
def mark_task_status(self, task_id: str, status: TaskStatus, error: Optional[str] = None):
    """Mark task status and handle dependency cascades."""
    self.tasks[task_id].status = status
    
    # If task failed, mark dependent tasks as skipped
    if status == TaskStatus.FAILED:
        self._mark_dependent_tasks_skipped(task_id)

def _mark_dependent_tasks_skipped(self, failed_task_id: str):
    """Automatically skip tasks dependent on failed task."""
    for task in self.tasks.values():
        if (failed_task_id in task.dependencies and 
            task.status == TaskStatus.PENDING):
            task.status = TaskStatus.SKIPPED
            logger.debug(f"Task {task.task_id} skipped due to failed dependency")
```

#### **3. TaskRepairAgent Integration**
- **Error Pattern Recognition**: Identifies common failure patterns
- **Intelligent Suggestions**: Provides context-aware repair recommendations
- **Schema-Aware Guidance**: Understands Neo4j schema limitations
- **User-Friendly Messages**: Transforms technical errors into actionable guidance

### **Graceful Degradation**
- **Partial Results**: Returns successful task results even if some tasks fail
- **Fallback Execution**: Alternative execution paths when primary approaches fail
- **Error Aggregation**: Collects and summarizes multiple error conditions

---

## **📊 Information Flow & Context Persistence**

### **Inter-Task Communication Architecture**

#### **1. Result Storage & Access**
```python
class TaskExecutor:
    def __init__(self):
        self.completed_results = {}  # Task ID → execution results
    
    async def execute_task(self, task: Task):
        # Execute task
        result = await self._execute_query_task(task)
        
        # Store for dependent tasks
        self.completed_results[task.task_id] = result
```

#### **2. Data Passing Mechanisms**

**For Tool Calls**:
```python
def _prepare_tool_arguments(self, task: Task) -> Dict[str, Any]:
    """Prepare tool arguments including dependency results."""
    args = task.tool_args.copy()
    
    # Add dependency results
    dependency_data = []
    for dep_id in task.dependencies:
        if dep_id in self.completed_results:
            dependency_data.append(self.completed_results[dep_id])
    
    if dependency_data:
        args["dependency_results"] = dependency_data
```

**For Query Tasks**:
- **Context Objects**: Rich GenomicContext with structured and semantic data
- **Metadata Preservation**: Execution metadata flows between tasks
- **Biological Focus**: Maintains biological context throughout execution

#### **3. Session-Based Note Persistence**

**NoteKeeper Architecture** (`src/llm/rag_system/memory/note_keeper.py`):
```python
class NoteKeeper:
    def record_task_notes(self, task_id: str, observations: List[str], 
                         key_findings: List[str], cross_connections: List[CrossTaskConnection]):
        """Persist task insights for session-wide analysis."""
        
        task_note = TaskNote(
            task_id=task_id,
            observations=observations,
            key_findings=key_findings,
            cross_task_connections=cross_connections,
            confidence=confidence,
            execution_time=execution_time,
            timestamp=datetime.now()
        )
        
        # Save to session-specific storage
        self._save_task_note(task_note)
```

**Cross-Task Connection Tracking**:
```python
@dataclass
class CrossTaskConnection:
    connected_task: str           # ID of related task
    connection_type: ConnectionType  # BUILDS_ON, CONTRADICTS, CONFIRMS, INFORMS
    description: str             # Nature of the relationship
    confidence: ConfidenceLevel  # HIGH, MEDIUM, LOW
```

### **Context Preservation Strategies**

#### **1. Structured Data Flow**
- **GenomicContext Objects**: Preserve structured query results
- **Semantic Data**: Maintain ESM2 embeddings and similarity scores
- **Metadata Propagation**: Execution statistics and biological insights

#### **2. Biological Context Maintenance**
```python
# Example: Chunk tasks preserve biological focus
task.biological_focus = chunk.biological_focus
task._already_chunked = True  # Prevent recursive processing
task._intelligent_chunked = True  # Mark as using advanced system
```

#### **3. Session Memory Architecture**
- **File-Based Persistence**: JSON storage with session organization
- **Intelligent Note-Taking**: DSPy-driven decisions on what to preserve
- **Progressive Synthesis**: Incremental building of session insights
- **Cross-Session Learning**: Patterns available for future sessions

---

## **🎯 Advanced Features & Optimizations**

### **Intelligent Chunking Integration**
- **Clean Task Naming**: `func_oxidation_reduction` vs recursive `_sub_1_sub_2...`
- **Biological Strategy Selection**: Functional, genomic, pathway-based chunking
- **Parallel Chunk Execution**: 3-5 chunks processed concurrently
- **Comprehensive Synthesis**: Detailed biological insights preservation

### **Model Allocation System**
- **Task-Appropriate Models**: Simple tasks → GPT-4.1-mini, Complex → o3
- **Cost Optimization**: Intelligent model selection based on task complexity
- **Premium Mode**: Switch to high-quality models for all tasks
- **Fallback Logic**: Automatic model switching on failures

### **Performance Optimizations**
- **Apple Silicon M4 Max**: ~85 proteins/second ESM2 processing
- **Sub-millisecond Queries**: LanceDB vector similarity search
- **Parallel Execution**: Independent tasks run concurrently
- **Memory Management**: Automatic cleanup and context compression

---

## **📈 Production Usage Examples**

### **Complex Functional Comparison**
```
User Query: "Generate a comprehensive functional comparison between all genomes"
↓
DSPy Planning: Multi-step plan with data retrieval → analysis → synthesis
↓
Task Graph: 4 parallel chunks (func_oxidation_reduction, func_biosynthesis, etc.)
↓
Execution: Parallel processing with intelligent chunking
↓
Synthesis: Comprehensive analysis with detailed biological insights
```

### **Cross-Genome Analysis**
```
Query: "Find proteins similar to heme transporters across all genomes"
↓
Stage 1: Neo4j finds annotated heme transport proteins
↓
Stage 2: LanceDB similarity search using those as seeds
↓
Result: 200 Neo4j annotations + 5 LanceDB similarity matches with ESM2 scores
```

### **Error Recovery Example**
```
Task 1: Retrieve genome data → SUCCESS
Task 2: Analyze protein domains → FAILED (invalid query)
Task 3: Generate visualization → SKIPPED (dependency failed)
Result: Partial results returned with clear error explanation
```

## Recent Agentic RAG v2.0 Integration (June 2025) ✅

**🎉 COMPLETED**: Full agentic capabilities integrated with multi-stage query processing and enhanced biological analysis.

### Key Achievements:

#### **Task Plan Parser & Execution** ✅
- TaskPlanParser bridges DSPy planning with TaskGraph execution (`src/llm/rag_system/task_plan_parser.py`)
- Supports ATOMIC_QUERY and TOOL_CALL task types with dependency resolution
- Agentic Task Graph System with DAG-based execution and status tracking

#### **Multi-Stage Semantic Search** ✅
- Stage 1: Neo4j finds annotated proteins → Stage 2: LanceDB similarity search
- Real embedding similarities (ESM2 cosine) instead of LLM hallucinations
- Intelligent routing between traditional vs agentic execution modes

#### **Enhanced Capabilities** ✅
- **External Tool Integration**: PubMed literature search with PFAM-aware enhancement
- **Professional Output**: PFAM domain citations (PF01594.21), KEGG pathway mapping
- **Biological Intelligence**: Operon prediction, genomic neighborhood analysis
- **Error Handling**: TaskRepairAgent transforms crashes into helpful guidance

## Key System Enhancements ✅

### **Intelligent Annotation Discovery** ✅
- **Problem Solved**: "ATP synthase problem" - system incorrectly returned energy metabolism proteins for transport queries
- **Solution**: Universal `functional_classifier` with biological exclusion logic
- **Result**: Correctly distinguishes functional categories (transport, metabolism, regulation, central_metabolism)

### **Code Interpreter Integration** ✅
- **Secure Execution**: FastAPI service with Docker containerization and session persistence
- **Large Dataset Analysis**: Handles 1,845+ CAZymes through iterative file-based analysis
- **Task Integration**: Seamless workflow coordination with dependency resolution
- **Scientific Stack**: 60+ packages for comprehensive genomic analysis

### **TaskRepairAgent System** ✅
- **Error Transformation**: Converts system crashes into helpful user guidance
- **Pattern Recognition**: 4 error patterns with 5 intelligent repair strategies
- **Biological Context**: Schema-aware suggestions with genomic database knowledge
- **Example**: "Invalid 'FakeNode'" → "Try searching for proteins, genes, or domains instead"

## Recent Development: Intelligent Task Splitting System (July 2025) 🧠

**🚀 COMPLETED**: Advanced task management system for handling unlimited dataset sizes through intelligent automatic splitting.

### **System Architecture** ✅
- **IntelligentTaskSplitter** (`src/llm/rag_system/intelligent_task_splitter.py`): Automatic detection and splitting of oversized tasks
- **TaskExecutor Integration**: Seamless integration with existing task execution pipeline
- **Notes Storage System**: Comprehensive session-based note storage in `data/session_notes/{session-uuid}/`
- **Recursive Capability**: Automatic sub-task splitting when sub-tasks exceed token limits

### **Key Features Implemented** ✅

#### **1. Automatic Oversized Task Detection** 🔍
- **Token Analysis**: Estimates task complexity using tiktoken tokenizer
- **Threshold Detection**: Automatically detects tasks >15K tokens requiring splitting
- **Data Analysis**: Analyzes both task description and associated dataset size
- **Smart Triggering**: Only splits when genuinely necessary to avoid unnecessary overhead

#### **2. Intelligent Chunking Strategies** 🧩
- **Functional Chunking**: Groups by biological function (KO descriptions, enzyme classes)
- **Genomic Chunking**: Groups by genome for comparative analysis
- **Size-Based Chunking**: Fallback chunking by logical data size
- **Context-Aware**: Analyzes query type to select optimal chunking strategy

#### **3. Parallel Sub-Task Execution** ⚡
- **Concurrent Processing**: Multiple sub-tasks executed simultaneously
- **Async Integration**: Proper async/await handling for task execution
- **Progress Tracking**: Real-time monitoring of sub-task completion
- **Error Handling**: Graceful handling of failed sub-tasks with continuation

#### **4. Comprehensive Notes System** 📝
- **Session-Based Storage**: `data/session_notes/{uuid}/task_notes/` organization
- **Hierarchical Notes**: Original tasks, sub-tasks, and sub-sub-tasks all tracked
- **Rich Metadata**: Execution time, token usage, confidence levels, cross-task connections
- **Biological Content**: Detailed functional annotations, KO/EC numbers, genomic coordinates

### **Performance Achievements** 📊

#### **Successful Capabilities**:
- ✅ **Automatic Detection**: 4,103-item dataset → 290K tokens → triggers splitting
- ✅ **Intelligent Chunking**: Functional grouping by biological categories
- ✅ **Parallel Execution**: 3 sub-tasks → multiple concurrent API calls
- ✅ **Recursive Splitting**: Sub-tasks automatically split further if needed
- ✅ **Rich Note-Taking**: Detailed biological insights captured at each level

#### **Example Processing Flow**:
```
Original Task (4,103 items, 290K tokens)
├── Sub-Task 1: "homoserine dehydrogenase functions" (1,998 items)
├── Sub-Task 2: "ATP-dependent helicase functions" (2,000 items) 
└── Sub-Task 3: "deaminase enzyme functions" (105 items)
    └── Sub-Sub-Task 1: Further split if needed
```

### **Critical Issues Discovered** ⚠️

#### **1. Recursive Description Explosion** 💥
- **Problem**: Task descriptions become unreadable with recursive splitting
- **Example**: "Analyze part 1/3 of analyze part 1/1 of analyze part 3/3..."
- **Impact**: Completely incomprehensible prompts that confuse the model
- **Root Cause**: Description concatenation in `_create_sub_task_description()`

#### **2. GPT-4.1-mini Performance Issues** 🐌
- **Problem**: Extremely poor performance on complex analytical tasks
- **Evidence**: 6,056 seconds (1.68 hours) for single sub-task, often returning no results
- **Impact**: 3+ hour total execution times for comprehensive queries
- **Conclusion**: GPT-4.1-mini unsuitable for complex biological analysis

#### **3. Excessive Recursive Depth** 🌀
- **Problem**: System creates 7+ levels of recursive splitting
- **Evidence**: Task names like `step_1_retrieve_functional__sub_1_sub_2_sub_1_sub_2_sub_2_sub_2_sub_1`
- **Impact**: Each level adds confusion and processing overhead
- **Issue**: No recursion limits implemented

### **Lessons Learned & Next Steps** 🎯

#### **Key Insights**:
1. **Recursive splitting is over-engineering** - creates more problems than it solves
2. **GPT-4.1-mini is inadequate** for complex analytical reasoning tasks
3. **Upfront intelligent chunking** is more effective than recursive splitting
4. **3-5 logical chunks** work better than unlimited recursive subdivision

#### **Recommended Evolution**: 
- **Replace recursive splitting** with intelligent upfront chunking (3-5 meaningful groups)
- **Switch to GPT-4/o3** for analytical tasks requiring complex reasoning
- **Implement chunk size limits** to prevent excessive subdivision
- **Pre-validate chunk complexity** before task creation

### **Notes Storage System** 📁
```
data/session_notes/{session-uuid}/
├── session_metadata.json              # Session stats, tokens, execution time
├── task_notes/                         # Individual task notes
│   ├── original_task_notes.json       # Main task analysis
│   ├── sub_task_notes.json            # Intelligent splitting results
│   └── sub_sub_task_notes.json        # Recursive splitting (if applicable)
└── synthesis_notes/                    # Progressive synthesis results
    └── chunk_synthesis.json
```

### **Critical Evolution: Intelligent Upfront Chunking (July 2025)** 🎯

**🚀 IMPLEMENTED**: Complete replacement of recursive splitting with intelligent upfront chunking system.

#### **System Redesign Completed** ✅

##### **1. IntelligentChunkingManager Implementation** 🧠
- **File**: `src/llm/rag_system/intelligent_chunking_manager.py`
- **Purpose**: Replace recursive splitting with smart upfront analysis and chunking
- **Strategy**: Analyze dataset once, create 3-5 meaningful biological chunks, execute in parallel

##### **2. Chunking Strategy Selection** 📊
- **Functional Analysis**: Groups by biological function (oxidation-reduction, protein synthesis, biosynthesis)
- **Genomic Comparison**: Groups by genome for cross-genome analysis
- **Metabolic Pathways**: Groups by biochemical pathway categories
- **Comprehensive Analysis**: Balanced coverage for complex queries
- **Auto-Detection**: Analyzes query type and data characteristics to select optimal strategy

##### **3. Clean Task Naming System** 📝
- **Problem Solved**: Eliminated filesystem-breaking 255+ character recursive names
- **Before**: `step_1_retrieve_functional__functional_chunk_5_functional_chunk_5_functional_chunk_5...`
- **After**: `func_oxidation_reduction`, `genome_burkholderiales`, `func_other`
- **Benefits**: Readable, filesystem-safe, biologically meaningful names

##### **4. Recursion Prevention** 🛑
- **Implementation**: `_already_chunked` flag prevents recursive splitting
- **Integration**: TaskExecutor respects flag and skips chunking for pre-chunked tasks
- **Result**: No more infinite recursive subdivision

#### **Performance Achievements** 📈

##### **Successful Chunking Examples**:
```
✅ Functional Category Analysis (4,103 items):
   - Chunk 1: Functional Analysis: Other Functions (2,092 items)
   - Chunk 2: Functional Analysis: Oxidation-Reduction (454 items)  
   - Chunk 3: Functional Analysis: Protein Synthesis (381 items)
   - Chunk 4: Functional Analysis: Biosynthesis (299 items)
   - Chunk 5: Additional Functional Categories (877 items)

✅ Strategy Selection: "Question requests functional analysis; group by biological function"
✅ Parallel Execution: 5 chunks executing simultaneously
✅ Clean Names: All task IDs under 30 characters
```

##### **Eliminated Issues**:
- ❌ **Recursive Description Explosion**: No more "analyze part 1/3 of analyze part 1/1..."
- ❌ **Excessive Depth**: Limited to 4-5 chunks maximum instead of 7+ levels
- ❌ **Filesystem Errors**: No more "File name too long" errors from 255+ character names
- ❌ **Token Waste**: No more incomprehensible prompts confusing the model

#### **Technical Architecture** 🏗️

##### **Strategy Determination Logic**:
```python
# Query Analysis → Strategy Selection
if "compare" or "comparison" in query → Genomic Comparison Strategy
if "pathway" or "metabolic" in query → Metabolic Pathway Strategy  
if "function" or "functional" in query → Functional Category Strategy
if "comprehensive" or "complete" in query → Comprehensive Analysis Strategy
else → Balanced Analysis Strategy (default)
```

##### **Chunk Creation Process**:
```python
1. Analyze dataset characteristics (genome data, function data, pathway data)
2. Select optimal chunking strategy based on query type
3. Create 3-5 logical chunks with biological meaning
4. Generate clean, short chunk IDs (func_oxidation_reduction, genome_burkholderiales)
5. Execute chunks in parallel with _already_chunked flag
6. Synthesize results into comprehensive summary
```

##### **File Organization Results**:
```
data/session_notes/{session-uuid}/task_notes/
├── func_oxidation_reduction_notes.json     # Clean biological categories
├── func_protein_synthesis_notes.json       # Readable and meaningful  
├── func_biosynthesis_notes.json           # Under 50 characters
├── func_transport_notes.json              # Filesystem-safe
└── func_other_notes.json                  # Organized by biology
```

#### **Integration Points** 🔗

##### **TaskExecutor Integration**:
- **Detection**: Automatically detects large datasets (>1000 items)
- **Chunking**: Delegates to IntelligentChunkingManager for upfront analysis
- **Prevention**: Respects `_already_chunked` flag to prevent recursion
- **Execution**: Handles chunk task execution with proper error handling

##### **Model Allocation Ready**:
- **Premium Model Support**: Prepared for GPT-4/o3 integration for complex analytical tasks
- **Cost Optimization**: Can switch models based on chunk complexity
- **Parallel Efficiency**: Multiple chunks can use different models simultaneously

#### **Biological Intelligence Preserved** 🧬

##### **Chunking Strategies Maintain Scientific Meaning**:
- **Functional Categories**: Oxidation-reduction, protein synthesis, transport, biosynthesis
- **Genomic Groupings**: Per-genome analysis or comparative genome groups
- **Metabolic Pathways**: Energy metabolism, carbohydrate metabolism, amino acid metabolism
- **Complexity Balancing**: Ensures each chunk has sufficient data for meaningful analysis

##### **Expected Insights Per Chunk**:
- **Oxidation-Reduction**: "Distribution and conservation of redox enzymes across genomes"
- **Protein Synthesis**: "Translation machinery and ribosomal protein conservation"
- **Transport**: "Membrane transport systems and nutrient uptake mechanisms"
- **Biosynthesis**: "Metabolic pathway completeness and biosynthetic capabilities"

### **Current Status**: 
**Production Ready** ✅ with **intelligent upfront chunking system**. The system now provides clean, meaningful biological analysis chunks without recursive complexity explosion, proper filesystem-safe naming, and parallel execution capabilities ready for premium model integration.

## Future Development Roadmap 🚀

### **Phase 3: Advanced Agent Capabilities** (Medium Priority)
- **KnowledgeGapAgent**: Autonomous discovery of missing biological knowledge through external database integration
- **Advanced TaskRepairAgent**: Organism-specific suggestions and pathway-informed error detection
- **Multi-Modal Integration**: Protein structure analysis, pathway diagrams, experimental data integration

### **Phase 4: Large Dataset Optimization** (High Priority)
- **Current**: Intelligent data tiering handles 1,845+ annotations through iterative analysis
- **Future**: Streaming analysis for 10K+ protein datasets with chunk-based processing
- **Target**: Real-time analysis of metagenomes and population genomics datasets

### **Phase 5: Production Scaling** (Medium Priority)
- **Performance**: Query caching, parallel execution, resource monitoring
- **Deployment**: Containerized services, load balancing, auto-scaling
- **Integration**: Galaxy/KNIME workflows, publication-quality figure generation

## Development Guidelines

### **Testing Requirements**
- **Test-Driven Development**: Write tests first, then implement
- **Component Testing**: Unit tests with mocks, integration tests, error path validation
- **Quality Gates**: 100% test coverage required, all tests must pass before commit
- **Performance**: Validate resource usage and execution times for new features

### **Implementation Standards**
- **Security**: Validate isolation and access controls (especially for code execution)
- **Backward Compatibility**: Must be maintained across all updates
- **Documentation**: Update relevant sections with each feature addition


## Known Issues & Future Work

### 🧬 Prodigal Data Integration Available

The prodigal gene prediction stage outputs rich genomic metadata that can be integrated into the knowledge graph:

**Currently Integrated:**
- Gene coordinates (start, end positions)
- Strand orientation (+1/-1)
- Amino acid length
- GC content

**Available for Future Integration:**
- Start codon type (ATG, GTG, TTG)
- Ribosome binding site (RBS) motif and spacer distance
- Partial gene indicators
- Additional quality metrics

**Prodigal Header Example:**
```
>protein_id # 76 # 171 # -1 # ID=1_1;partial=00;start_type=ATG;rbs_motif=AGGAG;rbs_spacer=5-10bp;gc_cont=0.573
```

This data enables genomic context analysis, operon prediction, and regulatory element identification for enhanced biological insights.

## TODO: Genome Quality Metrics Integration

**Future Enhancement**: Add comprehensive genome quality metrics to Genome nodes:
- **QUAST metrics**: `total_length`, `n50`, `num_contigs` - available from stage01_quast but not yet integrated
- **CheckM metrics**: `completeness`, `contamination` - requires resolving Python dependency conflicts
- **Implementation**: Extend `src/build_kg/rdf_builder.py` to parse and integrate quality metrics from stage01_quast output

