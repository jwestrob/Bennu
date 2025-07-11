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

### Available Enhancements
- **Prodigal Metadata**: Start codons, RBS motifs, quality metrics available for integration
- **Genome Quality Metrics**: QUAST metrics available but not yet integrated into knowledge graph
- **Operon Prediction**: Genomic context analysis capabilities present but not fully utilized

### Future Roadmap
- **Phase 3**: Advanced agent capabilities with knowledge gap discovery
- **Phase 4**: Large dataset optimization for metagenomes
- **Phase 5**: Production scaling with containerization and auto-scaling