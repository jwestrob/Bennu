# Microbial Genome Analysis Platform

A bioinformatics pipeline that processes microbial genome assemblies through a 7-stage workflow and provides an AI-powered query interface for biological analysis.

## Overview

This platform transforms raw genome assemblies into structured knowledge graphs and provides natural language querying capabilities. The system combines traditional bioinformatics tools with large language models to enable interactive exploration of genomic data.

**Core Components:**
- Multi-stage processing pipeline for genome annotation
- Neo4j knowledge graph for structured biological data  
- LanceDB vector database for protein similarity search
- AI agent system for natural language queries
- Code interpreter for statistical analysis and visualization

## Pipeline Stages

1. **Quality Assessment** - QUAST genome statistics
2. **Taxonomic Classification** - DFAST-based organism identification  
3. **Gene Prediction** - Prodigal open reading frame detection
4. **Functional Annotation** - KEGG and PFAM domain assignments
5. **Secondary Metabolite Detection** - GECCO biosynthetic gene cluster prediction
6. **Carbohydrate Enzyme Analysis** - dbCAN CAZyme annotation
7. **Knowledge Graph Construction** - RDF triple generation
8. **Protein Embeddings** - ESM2 semantic representations

## Installation

```bash
# Clone repository
git clone <repository-url>
cd microbial_claude_matter

# Create conda environment
conda env create -f env/environment.yml
conda activate genome-kg

# Install additional dependencies  
pip install -r requirements-llm.txt

# Verify installation
python scripts/run_tests.py --smoke
```

## Usage

### Process Genomes

Place FASTA files in `data/raw/` and run the pipeline:

```bash
python -m src.cli build
```

### Load Knowledge Graph

```bash
python load_neo4j.py
```

### Query Interface

```bash
# Basic queries
python -m src.cli ask "What metabolic pathways are present?"
python -m src.cli ask "Find transport proteins"
python -m src.cli ask "Compare CAZyme distributions across genomes"

# Complex analysis
python -m src.cli ask "Analyze the genomic neighborhood of scaffold_21_154"
python -m src.cli ask "Find operons containing hypothetical proteins"
```

## System Architecture

```
Raw Genomes → Processing Pipeline → Knowledge Graph → AI Query Interface
     ↓              ↓                    ↓              ↓
   FASTA        7 Stages           Neo4j + LanceDB    Natural Language
   Files       Annotation          Databases          Questions
```

**Data Flow:**
- Input: Microbial genome assemblies (FASTA format)
- Processing: Automated annotation pipeline  
- Storage: Graph database with biological relationships
- Interface: Natural language query system with code execution

## Key Features

**Bioinformatics Pipeline:**
- Integrates established tools (Prodigal, KEGG, PFAM, GECCO, dbCAN)
- Handles draft genomes and MAGs
- Generates comprehensive functional annotations

**Knowledge Graph:**
- Stores genes, proteins, domains, pathways, and relationships
- Enables complex queries across biological hierarchies
- Supports both structured and semantic search

**AI Interface:**
- Natural language processing for biological questions
- Multi-step reasoning with tool chaining
- Code interpreter for statistical analysis
- Literature integration via PubMed

## Data Output

The pipeline generates:
- **Structured data**: Gene predictions, functional annotations, pathway assignments
- **Knowledge graph**: 270K+ RDF triples representing biological relationships  
- **Embeddings**: 320-dimensional protein representations for similarity search
- **Reports**: Quality metrics, taxonomic classifications, annotation summaries

## Performance

**Processing Speed:**
- ~10K proteins processed in minutes (M4 Max)
- Sub-millisecond similarity queries
- Real-time natural language interface

**Data Scale:**
- Handles multiple genomes simultaneously
- 270K+ knowledge graph relationships
- 10K+ protein embeddings
- Comprehensive functional coverage

## Configuration

Key settings in `CLAUDE.md`:
- Model allocation (cost vs performance)
- Pipeline stage control
- Database connection parameters
- Analysis thresholds

## Testing

```bash
# Full test suite
python scripts/run_tests.py

# Quick validation  
python scripts/run_tests.py --smoke

# Module-specific
python scripts/run_tests.py --module llm
python scripts/run_tests.py --coverage
```

## Requirements

**Software:**
- Python 3.11+
- Conda package manager
- Docker (for code interpreter)
- Neo4j database
- LLM API access (OpenAI, Anthropic)

**Hardware:**
- 16GB+ RAM recommended
- SSD storage for databases
- Apple Silicon optimization available

## Example Analysis

**Input Query:**
"Compare CAZyme distributions across genomes"

**System Response:**
- Retrieves CAZyme annotations from knowledge graph
- Performs statistical analysis via code interpreter  
- Generates comparative visualizations
- Provides biological interpretation of differences

**Output:**
Detailed analysis showing enzyme family distributions, statistical comparisons, and functional implications across different organisms.

## Contributing

Development follows the guidelines in `CLAUDE.md`. Key areas:
- Pipeline component development
- Query interface improvements  
- Knowledge graph schema extensions
- Analysis tool integration

## License

MIT License - see LICENSE file for details.

## Dependencies

**Bioinformatics:**
- Prodigal (gene prediction)
- KEGG (functional annotation)  
- PFAM (protein domains)
- GECCO (biosynthetic clusters)
- dbCAN (carbohydrate enzymes)

**Data Management:**
- Neo4j (graph database)
- LanceDB (vector search)
- pandas/numpy (data processing)

**AI/ML:**
- ESM2 (protein embeddings)
- DSPy (structured prompting)
- OpenAI/Anthropic APIs (language models)