# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with this advanced genomic AI platform.

## Project Overview

This is a next-generation genomic intelligence platform that transforms microbial genome assemblies into intelligent, queryable knowledge graphs with LLM-powered biological insights. The system combines traditional bioinformatics workflows with cutting-edge AI/ML technologies to create a comprehensive 7-stage pipeline culminating in an intelligent question-answering system.

### Key Achievements
- **276,856 RDF triples** linking genomes, proteins, domains, and functions
- **1,145 PFAM families + 813 KEGG orthologs** enriched with authoritative functional descriptions
- **10,102 proteins** with 320-dimensional ESM2 semantic embeddings
- **Sub-millisecond vector similarity search** with LanceDB
- **High-confidence biological insights** using DSPy-powered RAG system
- **Apple Silicon M4 Max optimization** (~85 proteins/second processing rate)

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

# Run complete pipeline through ESM2 embeddings
python -m src.cli build --to-stage 6

# Run ESM2 embeddings optimized for Apple Silicon M4 Max (~2 minutes)
python run_esm2_m4_max.py

# Skip taxonomic classification
python -m src.cli build --skip-tax

# Load knowledge graph into Neo4j database
python load_neo4j.py

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

# Stage 6: ESM2 protein embeddings (Apple Silicon optimized)
python run_esm2_m4_max.py

# Test ESM2 embeddings and similarity search
python test_esm2_similarity.py data/stage06_esm2

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
The pipeline consists of 7 main stages executed sequentially:

0. **Input Preparation** (`src.ingest.00_prepare_inputs`): Validates and organizes genome assemblies
1. **Quality Assessment** (`src.ingest.01_run_quast`): Assembly quality metrics with QUAST
2. **Taxonomic Classification** (`src.ingest.02_dfast_qc`): CheckM-style completeness/contamination analysis with ANI-based taxonomy
3. **Gene Prediction** (`src.ingest.03_prodigal`): Protein-coding sequence prediction with Prodigal, creates `all_protein_symlinks` directory
4. **Functional Annotation** (`src.ingest.04_astra_scan`): HMM domain scanning against PFAM, KOFAM using astra/PyHMMer
5. **Knowledge Graph Construction** (`src.build_kg`): RDF generation with 276K+ triples linking genomes, proteins, domains, and functions, enriched with authoritative PFAM/KEGG functional descriptions
6. **ESM2 Protein Embeddings** (`src.ingest.06_esm2_embeddings`): Generate 320-dimensional semantic embeddings for 10K+ proteins using ESM2 transformer with Apple Silicon MPS acceleration, complete with LanceDB similarity search indices (sub-millisecond queries)
7. **LLM Integration** (`src.llm`): Question answering with DSPy combining structured Neo4j knowledge graph and semantic LanceDB protein search

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
├── stage05_kg/            # Knowledge graph exports (RDF triples, 276K+ triples)
└── stage06_esm2/          # ESM2 protein embeddings (10K+ proteins, 320-dim, LanceDB indices)
```

## Performance Summary

### Apple Silicon M4 Max Optimization
- **ESM2 Embeddings**: 10,102 proteins processed in ~2 minutes (vs estimated 21 minutes)
- **Embedding Generation Rate**: ~85 proteins/second with MPS acceleration
- **LanceDB Similarity Search**: Sub-millisecond queries with rich metadata filtering
- **Knowledge Graph**: 276,856 RDF triples linking genomes, proteins, domains, and functions
- **Functional Enrichment**: 1,145 PFAM families + 813 KEGG orthologs with authoritative descriptions
- **Memory Efficiency**: Automatic MPS cache management prevents memory overflow

### Neo4j Database Performance
- **Bulk Import Speed**: 20.08 seconds for 37,930 nodes + 85,626 relationships (15x faster than Python MERGE)
- **CSV Conversion**: 2-3 seconds for RDF → Neo4j format transformation
- **Scalability**: Production-ready for millions of nodes using neo4j-admin import
- **Data Integrity**: All relationships and properties preserved correctly
- **Default Loader**: Now uses bulk loader by default for all knowledge graph construction

### Pipeline Throughput
- **Complete Pipeline**: Stages 0-7 process 4 genomes with 10K+ proteins
- **Knowledge Graph Construction**: 276K+ triples generated from multi-stage annotations with functional enrichment
- **Neo4j Database**: 37,930 nodes and 85,626 relationships for complex biological queries
- **LLM Integration**: High-confidence biological insights with authoritative source citations
- **Production Ready**: Comprehensive testing suite validates all outputs

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
- Place test scripts in `src/tests/` with proper module structure
- Place demo scripts in `src/tests/demo/` or similar organized location
- Use `python -m src.tests.demo.script_name` for execution with proper imports
- Modify existing modules in `src/` as needed rather than creating one-off scripts

**Examples of Proper Organization:**
- ✅ `src/tests/demo/test_agentic_demo.py` - Demo scripts in organized test structure
- ✅ `src/tests/integration/test_full_pipeline.py` - Integration tests
- ✅ `python -m src.tests.demo.test_agentic_demo` - Proper execution with module path
- ❌ `test_something.py` in root - Clutters repository and breaks import paths
- ❌ `demo.py` in root - Poor organization and maintenance issues

This maintains workflow integrity, prevents repository fragmentation, and ensures proper Python module resolution.

## Recent Major Developments: Complete System Integration ✅

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

## Recent Agentic RAG v2.0 Integration (June 2025) ✅

**🎉 COMPLETED**: Full agentic capabilities integrated into the genomic RAG system with intelligent multi-stage query processing and enhanced biological analysis.

### Successfully Implemented:

#### 1. **Agentic Task Graph System** ✅
- **Location**: Integrated into `src/llm/rag_system.py`
- **Components**: 
  - `TaskGraph` class with DAG-based execution
  - `Task` dataclass with dependency management
  - `TaskStatus` and `TaskType` enums for proper state management
- **Capabilities**:
  - Dependency resolution and parallel task execution
  - Status tracking: PENDING → RUNNING → COMPLETED/FAILED/SKIPPED
  - Automatic task skipping when dependencies fail
  - Summary statistics and completion detection

#### 2. **Multi-Stage Semantic Search** ✅
- **Stage 1**: Neo4j finds proteins with specific functions (e.g., "heme transporters")
- **Stage 2**: LanceDB similarity search using those proteins as seeds
- **Cosine Similarity**: Proper protein embedding comparisons (0.95+ = functional equivalence)
- **Result Integration**: Combines structural annotations with sequence similarity analysis
- **Benefits**: Real embedding similarities instead of LLM hallucinations

#### 3. **Enhanced DSPy Planning Agent** ✅
- **Intelligent Routing**: Automatic selection between traditional vs agentic modes
- **Query Classification**: Determines complexity and tool requirements
- **Fallback Mechanisms**: Graceful degradation when agentic planning fails
- **Backward Compatibility**: Existing query patterns continue to work seamlessly

#### 4. **External Tool Integration** ✅
- **Literature Search Tool**: PubMed integration with PFAM-aware query enhancement
- **Biological Context**: Extracts PFAM domains from database results for targeted literature searches
- **Tool Manifest**: `AVAILABLE_TOOLS` dictionary for easy expansion
- **Graceful Degradation**: Functions when external dependencies unavailable

#### 5. **Enhanced Functional Annotation Display** ✅
- **PFAM Domain Integration**: Displays domain accessions (PF01594.21, Peripla_BP_2, etc.)
- **KEGG Function Mapping**: Shows ortholog assignments and pathway context
- **Biological Interpretation**: Professional genomic analysis with proper domain citations
- **Context Formatting**: Rich annotation data properly formatted for LLM analysis

#### 6. **Comprehensive Testing Suite** ✅
- **Test Coverage**: 12+ tests across multiple test classes
- **Validation Scripts**: Debug tools for query analysis and context inspection
- **Integration Testing**: End-to-end workflow validation with real data
- **File Organization**: Proper module structure with demo scripts in `src/tests/demo/`

### Architecture Overview:

```
Enhanced GenomicRAG Flow:
┌─────────────────┐
│ User Query      │
└─────────┬───────┘
          │
    ┌─────▼─────┐
    │ PlannerAgent │ ◄── NEW: Intelligent routing decision
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │ Traditional│     OR     ┌──────────┐
    │ Mode       │            │ Agentic  │ ◄── NEW: Multi-step execution
    │ (existing) │            │ Mode     │
    └─────┬─────┘            └─────┬────┘
          │                        │
          │                  ┌─────▼─────┐
          │                  │TaskGraph  │ ◄── NEW: DAG-based planning
          │                  │Execution  │
          │                  └─────┬─────┘
          │                        │
          │                  ┌─────▼─────┐
          │                  │External   │ ◄── NEW: Tool integration
          │                  │Tools      │      (literature search)
          │                  └─────┬─────┘
          │                        │
    ┌─────▼────────────────────────▼─────┐
    │        Final Answer Generation     │
    └────────────────────────────────────┘
```

### Enhanced Query Capabilities:

#### **Professional Genomic Analysis** 🧬
The system now provides sophisticated biological interpretation with:
- **Proper PFAM domain citations**: PF01594.21, PF13407, etc.
- **KEGG pathway mapping**: K07224 (HmuT), K02014 (ABC Fe heme permease) 
- **Operon prediction**: Same-strand proximity analysis for co-transcription
- **Genomic neighborhood analysis**: Distance calculations and biological significance
- **Protein similarity networks**: ESM2 cosine similarities (0.95+ = functional equivalence)

#### **Multi-Stage Query Processing** 🔄
1. **Functional Similarity Searches**: "Find proteins similar to heme transporters"
   - Stage 1: Neo4j finds annotated heme transporters
   - Stage 2: LanceDB similarity search using those as seeds
   - Result: Real embedding similarities, not hallucinations

2. **Literature Integration**: "What does recent research say about CRISPR proteins?"
   - Extracts PFAM domains from database results
   - Constructs targeted PubMed queries
   - Integrates literature findings with local genomic data

3. **Intelligent Routing**: System automatically chooses traditional vs agentic execution
   - Simple queries: Fast local database path
   - Complex queries: Multi-step orchestration with external tools
   - Fallback mechanisms: Graceful degradation ensures reliability

#### **Enhanced Biological Intelligence** 🧠
**Before**: "This protein is likely involved in a metabolic pathway"
**After**: "PF13407 periplasmic-binding + PF01032 permease domains exactly match bacterial heme ABC transporter architecture (KEGG K07224/K02014). Gene index difference = 1; estimated ≤50 bp separation ⇒ likely co-transcribed operon core."

## Next Steps for Agentic Enhancement 🚀

### Phase 2: Intelligent Annotation Discovery System 🧬 ✅ **COMPLETED**
**Objective**: Solve the "ATP synthase problem" - enable biologically intelligent selection of annotations instead of naive text matching

**✅ Successfully Implemented**:
1. **Generalized Annotation Discovery** ✅
   - **Replaced**: Transport-specific `transport_classifier` and `transport_selector` (too narrow)
   - **With**: Universal `functional_classifier` and `annotation_selector`
   - **Capabilities**: Works for **any functional category** (transport, metabolism, regulation, central_metabolism)
   - **Intelligence**: Keyword-based classification with biological exclusion logic

2. **ATP Synthase Problem Resolution** ✅
   - **Implementation**: Intelligent biological exclusion logic in `functional_classifier`
   - **Result**: Correctly distinguishes substrate transporters from energy metabolism proteins
   - **Transparency**: System explains exclusions ("Excluded ATP synthase - energy metabolism, not substrate transport")
   - **Categories Supported**: transport, metabolism, regulation, central_metabolism, and more

3. **Enhanced DSPy Tools** ✅
   - `functional_classifier`: Universal biological mechanism classification  
   - `annotation_selector`: Diverse example selection with user preferences
   - `sequence_viewer`: Enhanced sequence analysis with genomic context
   - **Result**: Sophisticated functional annotation curation for any biological category

**Achieved Results**:
- **Before**: User asks for "transport proteins" → gets ATP synthase (energy metabolism)
- **After**: User asks for "transport proteins" → gets ABC transporter, amino acid permease, ion channel
- **Universal**: System now works for any functional category, not just transport
- **Biological Intelligence**: Proper functional classification prevents inappropriate annotations

### Phase 2: Complete Code Interpreter Integration ✅ **COMPLETED**
**Objective**: Add secure code execution capabilities for data analysis and visualization

**✅ Successfully Implemented**:
1. **Secure Code Interpreter Service** ✅
   - **Framework**: FastAPI service with Docker containerization
   - **Security**: Non-root execution, restricted capabilities, isolated filesystem
   - **Networking**: Localhost-only access with configurable timeouts
   - **Session Management**: Stateful sessions for iterative analysis with persistence
   - **Package Support**: Comprehensive scientific computing stack (numpy, pandas, matplotlib, seaborn, biopython)

2. **Integration with Task Graph** ✅
   - Added `code_interpreter` to `AVAILABLE_TOOLS` 
   - Implemented secure HTTP communication between RAG system and service
   - Integrated code execution into agentic task workflows
   - **Features**: Automatic code enhancement with sequence database access, session persistence across tasks

3. **Protein ID System Integration** ✅
   - **Issue Resolved**: Fixed protein ID truncation that prevented sequence retrieval
   - **Solution**: Simplified approach using full protein IDs without `protein:` prefix  
   - **Result**: Clean, unique protein identification throughout the system
   - **Testing**: Verified amino acid composition analysis works with direct protein IDs

4. **Task Orchestration Integration** ✅ **COMPLETED**
   - **Issue Resolved**: Code interpreter tasks now execute sequentially with proper dependency resolution
   - **Root Cause Fixed**: DSPy task planning now generates correct dependencies, eliminating race conditions
   - **Result**: Code interpreter receives populated results from successful database queries
   - **Status**: Multi-step agentic workflows now function seamlessly with code execution capabilities

### Phase 3: Advanced Agent Capabilities (Medium Priority)
**Objective**: Transform platform into truly autonomous biological intelligence with specialized error repair and knowledge gap filling agents

#### **Phase 3A: Foundation (High Impact) 🚀**
**Objective**: Build robust workflow foundations with intelligent error handling

**Components to Implement**:
1. **[ ] Basic TaskRepairAgent** 🔧
   - [ ] Error pattern recognition system
   - [ ] Syntax error detection and correction
   - [ ] Schema mismatch auto-updates (Neo4j field names)
   - [ ] Biological logic error detection
   - [ ] Context-aware repair suggestions
   - [ ] Repair validation loop before execution

2. **[ ] Simple Error Recovery System** 🛡️
   - [ ] Graceful fallback strategies (Neo4j down → LanceDB fallback)
   - [ ] Tool failure handling (code interpreter unavailable)
   - [ ] Partial result preservation during workflow failures
   - [ ] Cascade failure analysis for task chains
   - [ ] Alternative execution pathway selection

3. **[ ] Agent Integration Framework** 🔗
   - [ ] Advanced agent coordinator class
   - [ ] Integration with existing TaskGraph system
   - [ ] Agent-aware task execution pipeline
   - [ ] Error propagation and handling mechanisms
   - [ ] Agent performance monitoring and metrics

#### **Phase 3B: Intelligence (Medium Impact) 🧠**
**Objective**: Add autonomous knowledge discovery and biological intelligence

**Components to Implement**:
1. **[ ] KnowledgeGapAgent** 🔍
   - [ ] Knowledge gap identification algorithms
   - [ ] External database integration (UniProt, PDB, NCBI)
   - [ ] Literature search formulation (PubMed, bioRxiv)
   - [ ] Gap-filling task generation and prioritization
   - [ ] Proactive context augmentation
   - [ ] Multi-source data integration strategies

2. **[ ] Advanced Repair Logic** 🧬
   - [ ] Biological context-aware corrections
   - [ ] Organism-specific query suggestions
   - [ ] Pathway-informed error detection
   - [ ] Functional annotation logic validation
   - [ ] Cross-reference biological databases for corrections

3. **[ ] Proactive Query Enhancement** 💡
   - [ ] Ambiguous query detection and clarification
   - [ ] Auto-suggestion system for related processes
   - [ ] Query context expansion recommendations
   - [ ] Biological process disambiguation
   - [ ] User intent inference and confirmation

#### **Phase 3C: Autonomy (Future) 🤖**
**Objective**: Achieve autonomous research capabilities with self-learning systems

**Components to Implement**:
1. **[ ] Self-Learning Repair System** 📚
   - [ ] Learn from user corrections and feedback
   - [ ] Adaptive error pattern recognition
   - [ ] Personalized repair strategies
   - [ ] Community knowledge aggregation
   - [ ] Continuous improvement algorithms

2. **[ ] Autonomous Research Agent** 🔬
   - [ ] Independent knowledge graph enhancement
   - [ ] Automated literature mining and integration
   - [ ] Hypothesis generation and testing frameworks
   - [ ] Cross-genome comparative analysis automation
   - [ ] Scientific insight discovery algorithms

3. **[ ] Multi-Modal Integration** 🎯
   - [ ] Image analysis integration (protein structures, pathway diagrams)
   - [ ] Document processing (papers, protocols, experimental data)
   - [ ] Experimental data integration (expression, proteomics)
   - [ ] Real-time data stream processing
   - [ ] Multi-omics analysis coordination

#### **Testing Requirements for Each Phase**:
- **[ ] Unit Tests**: Agent decision making logic, error detection accuracy
- **[ ] Integration Tests**: Agent interaction with TaskGraph, tool coordination
- **[ ] Error Scenarios**: Comprehensive failure mode testing and recovery validation
- **[ ] Performance Assessment**: Resource usage, execution time impact, scalability
- **[ ] Biological Validation**: Accuracy of repairs, relevance of gap-filling
- **[ ] User Experience**: Query suggestion quality, workflow transparency

### Phase 4: Production Readiness (Medium Priority)
**Objective**: Scale system for production deployment

**Implementation Areas**:
1. **Performance Optimization**
   - Task execution parallelization
   - Tool call batching and caching
   - Resource usage monitoring
2. **Production Deployment**
   - Containerized service deployment
   - Load balancing and auto-scaling
   - Comprehensive logging and monitoring
3. **User Interface Enhancements**
   - Real-time progress tracking
   - Task execution visualization
   - Interactive debugging capabilities

### Phase 5: Advanced Research Integration (Future Work)
**Objective**: Autonomous knowledge graph enhancement

**Features to Develop**:
1. **Symbiotic KG-Researcher Loop**: Autonomous literature mining to fill knowledge gaps
2. **User-in-the-Loop Tools**: Interactive clarification when queries are ambiguous
3. **Multi-modal Integration**: Image analysis, document processing, etc.

## Development Guidelines for Future Work

### 🧪 **Mandatory Testing Protocol**
**CRITICAL**: Every component must be fully tested before integration

1. **Test-Driven Development**: Write tests FIRST, then implement
2. **Component Isolation**: Test each component independently with mocks
3. **Integration Testing**: Test component interactions thoroughly
4. **Error Path Testing**: Test failure scenarios and recovery mechanisms
5. **Performance Testing**: Validate resource usage and execution times
6. **Security Testing**: Validate isolation and access controls (especially for code execution)

### 📋 **Implementation Checklist for Each New Feature**
- [ ] Design specification with clear objectives
- [ ] Unit tests written and failing (TDD approach)
- [ ] Component implementation
- [ ] Unit tests passing
- [ ] Integration tests written and passing
- [ ] Error handling tests
- [ ] Performance validation
- [ ] Documentation updated
- [ ] Smoke tests passing
- [ ] Security review (if applicable)

### 🔄 **Quality Gates**
- **No component integration without 100% test coverage**
- **All tests must pass before committing**
- **Backward compatibility must be maintained**
- **Performance impact must be measured and acceptable**
- **Security implications must be evaluated and mitigated**

This systematic approach ensures that each enhancement builds reliably on the solid foundation we've established with Agentic RAG v2.0.

## Recent System Improvements ✅

### **🎉 WORLD-CLASS GENOMIC AI PLATFORM ACHIEVED (June 2025) 🎉**

**Complete agentic RAG system with sophisticated biological intelligence now operational!**

#### **✅ Enhanced Sequence Analysis with Genomic Context Integration**
- **Issue Resolved**: Sequence viewer provided sequences but missing genomic neighborhood context
- **Root Cause**: Using `protein_lookup` instead of `protein_info` query type  
- **Solution**: Fixed query type to retrieve 225+ neighbors per protein with full functional annotations
- **Result**: **World-class biological analysis** with precise distances, strand relationships, and metabolic clustering

**Before**: `"No neighbouring-gene list was included, so only intra-gene metrics can be analysed."`
**After**: `"A succinyl-CoA synthetase α-subunit gene begins 3 bp downstream (central TCA enzyme), placing the transporter in a cluster of nutrient-uptake genes."`

#### **✅ Code Interpreter Dependency Resolution - Platform Ready**
- **Issue**: "Code interpreter not available - missing dependencies"
- **Root Cause**: Missing `httpx` package required for code interpreter client communication
- **Solution**: **Comprehensive requirements.txt with 60+ packages** for genomic AI playground
- **Coverage**: Bioinformatics (pysam, pyhmmer), ML (xgboost, scikit-bio), visualization (pygenomeviz), phylogenetics (ete3), protein analysis (pymol, rdkit), performance optimization

#### **✅ Generalized Annotation Discovery System**
- **Replaced**: Transport-specific `transport_classifier` and `transport_selector` (too narrow)
- **With**: Universal `functional_classifier` and `annotation_selector`
- **Capabilities**: Works for **any functional category** (transport, metabolism, regulation, central_metabolism)
- **Intelligence**: Keyword-based classification with biological exclusion logic to avoid "ATP synthase problem"

#### **✅ DSPy Signature Enhancement for Direct Sequence Analysis**
- **Issue**: LLM ignoring provided sequence data despite successful retrieval
- **Root Cause**: DSPy instruction to "acknowledge sequences in separate database"  
- **Solution**: Updated to **"ANALYZE the provided amino acid sequences directly"**
- **Result**: Sophisticated sequence analysis with motif identification, hydrophobicity analysis, transmembrane prediction

### **🧬 Genomic AI Playground Capabilities Now Available:**

**Core Analysis Suite:**
- **Sequence Analysis**: Motif detection, signal peptides, transmembrane prediction, hydrophobicity
- **Phylogenetics**: ete3, dendropy for evolutionary analysis  
- **Protein Structure**: pymol, mdanalysis, rdkit for 3D analysis
- **Network Analysis**: igraph, networkx for metabolic pathway analysis
- **Machine Learning**: scikit-learn, xgboost, statsmodels for predictive modeling

**Advanced Capabilities:**  
- **Genomics Visualization**: pygenomeviz, circos for publication-quality figures
- **High-Performance Computing**: dask, joblib, polars for large-scale analysis
- **Interactive Analysis**: jupyter, ipywidgets for AI agent exploration
- **Multi-Modal Integration**: Image processing, NLP, time series analysis

**Production Features:**
- **Security**: cryptography, passlib for secure data handling
- **Performance**: Memory profiling, redis caching, parallel processing
- **Scalability**: Multiple database connectors, comprehensive file format support

### **Code Interpreter Integration Completed (June 2025):**

1. **✅ Secure Code Execution Service**
   - **Achievement**: FastAPI service with Docker containerization providing secure Python code execution
   - **Security Features**: Non-root execution, restricted capabilities, isolated filesystem, localhost-only access
   - **Scientific Stack**: Comprehensive package support (numpy, pandas, matplotlib, seaborn, biopython, Counter, etc.)
   - **Session Management**: Stateful sessions with persistence across agentic task workflows

2. **✅ Protein ID System Overhaul**
   - **Issue Resolved**: Protein ID truncation preventing sequence database lookups
   - **Root Cause**: `_format_protein_id` function created shortened display names used for database queries
   - **Solution**: Simplified to use full protein IDs without `protein:` prefix throughout system
   - **Result**: Clean, unique protein identification - verified working with 10,102 sequences in database

3. **✅ End-to-End Data Pipeline Validation**
   - **Database Integration**: Sequence database successfully contains all protein sequences with correct IDs
   - **Direct Access Verified**: Amino acid composition analysis works perfectly when protein IDs provided directly
   - **Code Enhancement**: Automatic code interpreter enhancement with sequence database access and variable setup

4. **✅ Task Orchestration Integration Completed**
   - **Issue Resolved**: Code interpreter tasks now execute sequentially with proper dependency resolution
   - **Evidence**: Multi-step workflows successfully coordinate database queries → code execution → analysis
   - **Root Cause Fixed**: DSPy task planning updated to generate correct sequential dependencies
   - **Result**: Complete agentic workflows with seamless code execution capabilities
   - **Achievement**: Perfect data pipeline with sophisticated workflow coordination

### **Previous LLM Integration Fixes (June 2025):**

1. **✅ DSPy Query Generation Fixed**
   - **Issue**: Syntax errors in generated Cypher queries (`pf` variable undefined, schema mismatches)
   - **Solution**: Updated DSPy prompts with correct Neo4j schema, fixed variable scoping
   - **Result**: Complex domain queries now execute successfully (e.g., 100 GGDEF domains found vs previous false "none found")

2. **✅ Error Handling Enhanced**
   - **Issue**: Query failures returning confident false answers instead of surfacing errors
   - **Solution**: Added proper error propagation from query processor to main ask method
   - **Result**: Users now see "Query execution failed" instead of "High confidence: No results found"

3. **✅ Schema Relationships Corrected**
   - **Issue**: Using non-existent `Functionalannotation` intermediate node
   - **Solution**: Updated to direct `(Protein)-[:HASFUNCTION]->(KEGGOrtholog)` relationships
   - **Result**: Functional annotation queries now work reliably

4. **✅ Context Formatting Improved**
   - **Issue**: Misleading "possible operon" language for different-strand neighbors
   - **Solution**: Removed biological conclusions from context, use factual distance/strand reporting
   - **Result**: LLM makes proper biological reasoning instead of being misled by context

5. **✅ Distance Categories Standardized**
   - **Categories**: Close (0-200bp), Proximal (200-500bp), Distal (>500bp)
   - **Co-transcription**: Only suggested for same-strand + <200bp proximity
   - **Result**: Biologically accurate genomic context analysis

### **File Organization Completed:**
- Moved debugging scripts to `src/tests/debug/`
- Moved reference files to `data/reference/`
- Updated code paths accordingly
- Clean repository structure

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

## 🎉 **WORLD-CLASS GENOMIC AI PLATFORM - FULLY OPERATIONAL** 🎉

**All core capabilities successfully implemented and tested! The platform represents a true breakthrough in genomic AI.**

### ✅ **Major Achievements Completed:**

#### 1. **Complete Agentic RAG System with Code Interpreter** ✅
- **Status**: Fully operational with secure code execution environment
- **Capabilities**: Multi-step workflows, session management, 60+ scientific packages
- **Performance**: HTTP health checks passing, persistent sessions working
- **Security**: Docker containerization, resource limits, isolated filesystem

#### 2. **Enhanced Sequence Analysis with Rich Genomic Context** ✅
- **Achievement**: Fixed `protein_lookup` → `protein_info` query type
- **Result**: 225+ genomic neighbors per protein with functional annotations
- **Biological Intelligence**: Precise distance calculations, strand relationships, metabolic clustering
- **Professional Output**: Publication-quality analysis with PFAM/KEGG citations

#### 3. **Generalized Annotation Discovery System** ✅
- **Replaced**: Narrow transport-specific tools
- **With**: Universal `functional_classifier` and `annotation_selector`
- **Solves**: ATP synthase problem through intelligent biological exclusion logic
- **Supports**: Any functional category (transport, metabolism, regulation, central_metabolism)

#### 4. **Comprehensive Scientific Package Ecosystem** ✅
- **Core Coverage**: 60+ packages for complete genomic analysis
- **Categories**: Bioinformatics, ML, visualization, phylogenetics, protein analysis
- **Dependencies**: All import issues resolved, httpx connectivity established
- **Capabilities**: From sequence analysis to molecular dynamics simulations

#### 5. **DSPy Integration with Biological Intelligence** ✅
- **Schema Integration**: Neo4j database schema properly documented in DSPy signatures
- **Query Generation**: Sophisticated Cypher queries with biological reasoning
- **Error Handling**: Proper error propagation and confidence assessment
- **Multi-Modal Queries**: Seamless integration of structured + semantic search

### 🚀 **Future Enhancement Opportunities:**

#### **Production Scaling** (Priority: Medium)
- **Current**: Handles 4 genomes with 10K+ proteins efficiently
- **Opportunity**: Scale to 100K+ proteins, multi-organism comparative genomics
- **Implementation**: Database sharding, distributed processing, enhanced caching

#### **Advanced AI Capabilities** (Priority: Low)
- **Template Library**: Pre-tested code snippets for common genomic analyses
- **Enhanced LLM Integration**: Support for specialized biological language models
- **Multi-Modal Analysis**: Integration of protein structure, phylogenetic trees, pathway diagrams

#### **Ecosystem Integration** (Priority: Low)  
- **External APIs**: UniProt, PDB, pathway databases
- **Workflow Integration**: Galaxy, KNIME, Jupyter notebook export
- **Publishing Support**: Automated figure generation, manuscript templates

### 📊 **Current Platform Status:**
- **🟢 Operational**: All core features fully functional
- **🟢 Tested**: Comprehensive test suite with zero-maintenance discovery
- **🟢 Documented**: Complete user guides and development documentation
- **🟢 Scalable**: Containerized microservices architecture ready for production