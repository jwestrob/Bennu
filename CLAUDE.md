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

## Development Commands

### Testing
```bash
# Run all tests
python run_tests.py

# Quick smoke tests during development
python run_tests.py --smoke

# Run with coverage analysis
python run_tests.py --coverage

# Run tests for specific modules
python run_tests.py --module ingest
python run_tests.py --module build_kg

# Run by test category
python run_tests.py --marker unit
python run_tests.py --marker integration

# Discover all available tests
python run_tests.py --discover

# Shell wrapper (equivalent commands)
./test.sh --smoke
./test.sh --coverage
```

### Pipeline Execution
```bash
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

### Pipeline Throughput
- **Complete Pipeline**: Stages 0-7 process 4 genomes with 10K+ proteins
- **Knowledge Graph Construction**: 276K+ triples generated from multi-stage annotations with functional enrichment
- **Neo4j Database**: 44,477 nodes and 45,679 relationships for complex biological queries
- **LLM Integration**: High-confidence biological insights with authoritative source citations
- **Production Ready**: Comprehensive testing suite validates all outputs

### Biological Intelligence Quality
**Before Functional Enrichment**: Generic responses like "likely involved in metabolic pathways"
**After Functional Enrichment**: Specific insights like "putative heme transporter involved in oxygen transport and cytochrome electron transfer pathways"

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

### Biological Intelligence Transformation:

**Before Enhancement**:
```
"This protein is likely involved in a metabolic pathway and may have 
evolutionary significance across various organisms."
```

**After Enhancement**:
```
"Protein PLM0_60_b1_sep16_scaffold_10001_curated_6 is identified as a 
putative heme transporter (KEGG K20469), which plays a vital role in 
heme transport and metabolism. Heme is critical for oxygen transport 
in hemoglobin and electron transfer in cytochromes. The protein's 
genomic context suggests involvement in aerobic respiration pathways."
```

### Current System Capabilities:
- **Query Processing**: Neo4j structured queries + LanceDB semantic search + DSPy intelligent integration
- **Biological Knowledge**: Authoritative PFAM/KEGG annotations replace hardcoded domain knowledge  
- **Performance**: Apple Silicon M4 Max optimized pipeline with MPS acceleration
- **Testing**: Comprehensive test suite with zero-maintenance discovery (6 functional enrichment tests, all passing)
- **Production Ready**: Containerized microservices with Nextflow orchestration

### Advanced Features Now Available:
- **Semantic Protein Search**: `lancedb` vector similarity with metadata filtering
- **Knowledge Graph Queries**: Complex Cypher queries across genomic data
- **LLM Integration**: Natural language questions with biological context
- **Functional Enrichment**: Real-time integration of reference database annotations
- **Apple Silicon Optimization**: ~85 proteins/second ESM2 processing

This represents a complete transformation from a basic bioinformatics pipeline to an intelligent genomic AI platform.