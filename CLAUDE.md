# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a genome-to-LLM knowledge graph pipeline that processes microbial genome assemblies through quality assessment, taxonomic classification, gene prediction, and functional annotation to build a comprehensive knowledge graph with LLM-powered question answering.

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

# Query the knowledge graph
python -m src.cli ask "What metabolic pathways are present in Escherichia coli?"
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
5. **Knowledge Graph Construction** (`src.build_kg`): RDF generation with 241K+ triples linking genomes, proteins, domains, and functions
6. **ESM2 Protein Embeddings** (`src.ingest.06_esm2_embeddings`): Generate 320-dimensional semantic embeddings for 10K+ proteins using ESM2 transformer with Apple Silicon MPS acceleration, complete with FAISS similarity search indices (sub-millisecond queries)
7. **LLM Integration** (`src.llm`): Question answering with DSPy combining structured knowledge graph and semantic protein search

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
├── stage05_kg/            # Knowledge graph exports (RDF triples, 241K+ triples)
└── stage06_esm2/          # ESM2 protein embeddings (10K+ proteins, 320-dim, FAISS indices)
```

## Performance Summary

### Apple Silicon M4 Max Optimization
- **ESM2 Embeddings**: 10,102 proteins processed in ~2 minutes (vs estimated 21 minutes)
- **Embedding Generation Rate**: ~85 proteins/second with MPS acceleration
- **FAISS Similarity Search**: Sub-millisecond queries (0.17ms average)
- **Knowledge Graph**: 241,727 RDF triples linking genomes, proteins, domains, and functions
- **Memory Efficiency**: Automatic MPS cache management prevents memory overflow

### Pipeline Throughput
- **Complete Pipeline**: Stages 0-6 process 4 genomes with 10K+ proteins
- **Knowledge Graph Construction**: 241K+ triples generated from multi-stage annotations
- **Semantic Search Ready**: Vector embeddings enable protein similarity queries
- **Production Ready**: Comprehensive testing suite validates all outputs

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
- **faiss-cpu**: Vector similarity search
- **dsp**: LLM structured prompting

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

## Current Migration Status: FAISS → LanceDB

**🔄 IN PROGRESS**: Migrating from FAISS to LanceDB for vector similarity search

### Migration Objectives:
1. **Eliminate NumPy Compatibility Issues**: FAISS requires numpy<2.0, conflicts with DSPy
2. **Enable Unified Environment**: Single conda environment for all components
3. **Improve MLOps Standards**: LanceDB aligns better with production RAG workflows
4. **Add Rich Metadata**: Vector DB supports metadata filtering (organism, function, etc.)
5. **Simplify Architecture**: Eliminate manual index management

### Completed:
- ✅ Updated `env/environment.yml`: Removed `faiss-cpu`, added `lancedb`, enabled `numpy>=2.0`
- ✅ Modified `src/ingest/06_esm2_embeddings.py`: Replaced FAISS with LanceDB table creation
- ✅ Updated `test_esm2_similarity.py`: Changed similarity search to use LanceDB API
- ✅ Updated `run_esm2_m4_max.py`: Modified output references and test validation

### Next Steps:
1. **Reinstall Conda Environment**: `conda env create -f env/environment.yml --force`
2. **Test LanceDB Migration**: Run `python run_esm2_m4_max.py` to verify functionality
3. **Validate Vector Search**: Ensure `test_esm2_similarity.py` passes all tests
4. **Update Documentation**: Reflect LanceDB changes in performance metrics
5. **Prepare for LLM Integration**: DSPy + Neo4j + LanceDB unified RAG system

### Technical Details:
- **Vector DB**: LanceDB table with protein embeddings + metadata (genome_id, sequence_length, source_file)
- **Search API**: `table.search(query_vector).limit(k).to_pandas()` replaces FAISS index operations
- **Storage**: LanceDB handles persistence automatically vs manual HDF5 + FAISS files
- **Performance**: Expected slight latency increase but better metadata filtering capabilities

This migration enables a unified environment for the complete RAG pipeline without numpy version conflicts.