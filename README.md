# Genome-to-LLM Knowledge Graph

A comprehensive pipeline for extracting genomic features and building knowledge graphs from microbial genomes, with LLM-powered question answering capabilities.

## Overview

This project implements a multi-stage pipeline that processes microbial genome assemblies through quality assessment, taxonomic classification, gene prediction, and functional annotation to build a comprehensive knowledge graph. The resulting structured data enables LLM-powered semantic search and question answering about genomic features, evolutionary relationships, and functional capabilities.

## Pipeline Stages

### Stage 0: Input Preparation
- Validate and organize input genome assemblies
- Check file formats and integrity
- Create processing manifests

### Stage 1: Quality Assessment (QUAST)
- Assess assembly quality metrics
- Generate N50, L50, and contiguity statistics
- Identify potential assembly issues

### Stage 2: Taxonomic Classification (DFAST_QC)
- Evaluate genome completeness and contamination using CheckM-style analysis
- Assign taxonomic classifications using Average Nucleotide Identity (ANI)
- Generate quality metrics and taxonomic confidence scores
- Install: `conda install -c bioconda dfast_qc`
- Usage: `dfast_qc -i genome.fna -o OUT --num_threads 8`

### Stage 3: Gene Prediction (Prodigal)
- Predict protein-coding sequences
- Identify ribosomal RNA genes
- Extract genomic features and coordinates

### Stage 4: Functional Annotation (Astra/PyHMMer)
- Scan against protein domain databases
- Assign functional annotations
- Identify metabolic pathways and capabilities

### Stage 5: Knowledge Graph Construction
- Transform annotations to RDF triples
- Load structured data into Neo4j
- Create semantic relationships between entities

### Stage 6: LLM Integration
- Build FAISS vector indices for semantic search
- Implement DSPy signatures for structured queries
- Enable natural language question answering

## Installation

1. Create and activate the Conda environment:
```bash
conda env create -f env/environment.yml
conda activate genome-kg
```

2. Verify installation:
```bash
python -m src.cli --help
```

## Usage

### Build Knowledge Graph
Process genome assemblies through the complete pipeline:

```bash
# Process all genomes in data/raw/
python -m src.cli build

# Process specific genome files
python -m src.cli build --input genome1.fasta genome2.fasta

# Resume from specific stage
python -m src.cli build --from-stage 3
```

### Query Knowledge Graph
Ask natural language questions about your genomic data:

```bash
# Interactive Q&A mode
python -m src.cli ask "What metabolic pathways are present in Escherichia coli?"

# Batch query processing
python -m src.cli ask --file queries.txt --output results.json

# Advanced semantic search
python -m src.cli ask --mode semantic "Find genomes with similar CAZyme profiles"
```

## Project Structure

```
├── data/                    # Data directories (gitignored)
│   ├── raw/                # Input genome assemblies
│   ├── stage01_quast/      # Quality assessment results
│   ├── stage02_dfast_qc/   # Taxonomic classification
│   ├── stage03_prodigal/   # Gene predictions
│   └── kg/                 # Knowledge graph exports
├── src/                    # Source code
│   ├── ingest/            # Pipeline stage implementations
│   ├── build_kg/          # Knowledge graph construction
│   ├── llm/               # LLM integration and querying
│   └── cli.py             # Command-line interface
├── notebooks/             # Jupyter analysis notebooks
├── env/                   # Conda environment specifications
└── README.md              # This file
```

## Dependencies

Core bioinformatics tools:
- **prodigal**: Gene prediction
- **QUAST**: Assembly quality assessment  
- **CheckM-genome**: Genome completeness evaluation
- **GTDB-Tk**: Taxonomic classification
- **CoverM**: Coverage calculation
- **PyHMMer**: Protein domain scanning via Astra

Python packages:
- **typer**: CLI framework
- **pydantic**: Data validation
- **rdflib**: RDF graph manipulation
- **neo4j**: Graph database client
- **faiss-cpu**: Vector similarity search
- **dsp**: LLM structured prompting
- **pyarrow**: Efficient data serialization

## Development

The project follows a modular architecture with clear separation between:
- **Ingestion**: Bioinformatics pipeline stages
- **Knowledge Graph**: RDF generation and Neo4j loading  
- **LLM**: Vector search and question answering

Each stage can be run independently and produces standardized outputs for downstream processing.

## License

MIT License - see LICENSE file for details.
