# Microbial Claude Matter: Genomic AI Platform

**Transform microbial genome assemblies into intelligent, queryable knowledge graphs with LLM-powered biological insights.**

## Overview

This is a next-generation genomic intelligence platform that combines traditional bioinformatics workflows with AI agents and embedding-based vector similarity search. The system processes genome assemblies through an 8-stage pipeline culminating in an intelligent question-answering system capable of sophisticated biological analysis.

### Key Achievements

- **373,587 RDF triples** linking genomes, proteins, domains, and functions
- **1,845 CAZymes + 813 KEGG orthologs** with authoritative functional descriptions  
- **Enhanced GECCO BGC integration** with 17 quantitative properties per BGC
- **10,102 proteins** with 320-dimensional ESM2 semantic embeddings
- **Sub-millisecond vector similarity search** with LanceDB
- **Production-ready Neo4j integration** with 48K nodes and 95K relationships
- **High-confidence biological insights** using DSPy-powered RAG system

## Quick Start

```bash
# Activate environment
source /Users/jacob/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh && conda activate genome-kg

# Run complete pipeline
python -m src.cli build

# Ask questions about your genomic data
python -m src.cli ask "What transport proteins are present in the database?"
python -m src.cli ask "Show me the distribution of CAZyme types among each genome"
python -m src.cli ask "Find proteins similar to heme transporters"
```

## Example Output

```
🧬 Processing question: Show me the distribution of CAZyme types among each genome in the dataset; compare and contrast.

🤖 Answer:
Overview
Carbohydrate-active enzymes (CAZymes) fall into several functional classes: glycoside hydrolases (GH), 
glycosyltransferases (GT), carbohydrate-binding modules (CBM), carbohydrate esterases (CE), 
polysaccharide lyases (PL) and auxiliary activities (AA). Comparing four genomes reveals clear 
differences in both repertoire size and functional emphasis.

Per-genome CAZyme profiles
1. Burkholderiales_bacterium: 1,056 CAZymes
   • GH 436 (41.3%), GT 356 (33.7%), CBM 163 (15.4%), AA 56 (5.3%)
   • Interpretation: GH-rich toolkit suggests specialization for aggressive breakdown of diverse 
     plant-derived polysaccharides

2. PLM0_60_b1_sep16: 425 CAZymes  
   • GT 178 (41.9%), GH 156 (36.7%), CBM 53 (12.5%)
   • Interpretation: GT emphasis indicates investment in cell-wall/EPS biosynthesis

[...detailed analysis continues...]

Confidence: high
```

## System Architecture

The platform consists of three main components:

1. **8-Stage Bioinformatics Pipeline**: Genome assembly → Gene prediction → Functional annotation → Knowledge graph
2. **Modular RAG System**: DSPy-powered query processing with traditional and agentic execution paths
3. **Multi-Database Integration**: Neo4j structured data + LanceDB semantic search + External tools

## Navigation

- **[Architecture](architecture/overview.md)**: System design and components
- **[Pipeline](architecture/pipeline-stages.md)**: 8-stage processing workflow  
- **[Tutorials](tutorials/basic-queries.md)**: Getting started with queries
- **[API Reference](api-reference/cli-commands.md)**: Complete command documentation
- **[Examples](examples/genomic-questions.md)**: 50+ example queries with outputs

## Performance Highlights

- **Apple Silicon M4 Max Optimized**: 85 proteins/second ESM2 processing
- **Sub-millisecond Queries**: LanceDB similarity search with rich metadata
- **Production Scale**: 373K+ RDF triples, 48K Neo4j nodes processed in seconds
- **High Accuracy**: Research-grade biological insights with confidence scoring

## Next Steps

1. **Basic Usage**: Start with [Basic Queries Tutorial](tutorials/basic-queries.md)
2. **Complex Analysis**: See [CAZyme Analysis Walkthrough](tutorials/complex-analysis.md) 
3. **API Integration**: Check [Python API Reference](api-reference/python-api.md)
4. **Custom Workflows**: Explore [Agentic Workflows](tutorials/agentic-workflows.md)