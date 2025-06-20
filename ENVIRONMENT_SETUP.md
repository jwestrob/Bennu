# Environment Setup Guide

## Overview

This project uses a hybrid approach for dependency management to solve conda environment conflicts with bioinformatics tools:

- **Python packages**: Managed via conda environment (`env/environment.yml`)
- **Bioinformatics tools**: Installed separately (homebrew/system packages)

## Local Development Setup

### 1. Create Conda Environment

```bash
conda env create -f env/environment.yml
conda activate genome-kg
```

### 2. Install Bioinformatics Tools

The following tools need to be installed separately:

- **quast**: `brew install quast` or install from source
- **prodigal**: `brew install prodigal` or install from source  
- **dfast_qc**: Install from source (already done in your setup)

### 3. Verify Installation

```bash
# Activate environment
conda activate genome-kg

# Test Python packages
python -c "import pandas, numpy, rdflib, pyhmmer, Bio; print('Python packages OK')"

# Test bioinformatics tools
which quast.py
which prodigal
which dfast_qc
```

## Docker Deployment

For Docker deployment, you'll need to:

1. **Base the container on a bioinformatics-friendly image** (e.g., `continuumio/miniconda3`)

2. **Install system bioinformatics tools** in the Dockerfile:
   ```dockerfile
   # Install bioinformatics tools
   RUN apt-get update && apt-get install -y \
       build-essential \
       wget \
       curl
   
   # Install prodigal
   RUN wget https://github.com/hyattpd/Prodigal/releases/download/v2.6.3/prodigal.linux -O /usr/local/bin/prodigal && \
       chmod +x /usr/local/bin/prodigal
   
   # Install QUAST
   RUN pip install quast
   
   # Install DFAST_QC (from source)
   # ... add DFAST_QC installation steps
   ```

3. **Create conda environment** in Docker:
   ```dockerfile
   COPY env/environment.yml /tmp/environment.yml
   RUN conda env create -f /tmp/environment.yml
   ```

4. **Set activation in entrypoint**:
   ```dockerfile
   RUN echo "conda activate genome-kg" >> ~/.bashrc
   SHELL ["/bin/bash", "--login", "-c"]
   ```

## Package Rationale

### Removed from conda environment:
- `quast`, `prodigal`: Platform-specific conflicts with libgcc-ng on macOS ARM64
- `checkm-genome`, `gtdbtk`: Complex dependency conflicts with pplacer, fastani
- `eggnog-mapper`: Diamond version conflicts
- `bprom`, `coverm`, `crisprcasfinder`, `isescan`, `plasflow`, `signalp6`, `tmhmm`, `transtermhp`: Don't exist in specified channels
- `orthofinder`: Not currently used in pipeline code
- `dfast_qc`: Already installed from source

### Kept in conda environment:
- All Python data science packages (pandas, numpy, scipy, matplotlib, seaborn)
- Knowledge graph packages (rdflib, neo4j-python-driver)
- ML packages (faiss-cpu)
- Development tools (pytest, black, mypy, jupyter)
- Core bioinformatics Python packages (pyhmmer, biopython)
- dfast_qc Python dependencies (ete3, more-itertools, peewee)

## Troubleshooting

### Common Issues:

1. **"conda activate" not working**: 
   ```bash
   source $(conda info --base)/etc/profile.d/conda.sh
   conda activate genome-kg
   ```

2. **Bioinformatics tools not found**: Ensure they're installed via homebrew or from source and available on PATH

3. **Import errors**: Make sure you're in the activated conda environment

### Environment Recreation:

If you need to recreate the environment:
```bash
conda env remove -n genome-kg
conda env create -f env/environment.yml
```

## Pipeline Compatibility

This environment setup is compatible with all current pipeline stages:

- ✅ **Stage 0**: Input preparation (Python only)
- ✅ **Stage 1**: QUAST quality assessment (uses system quast.py)
- ✅ **Stage 2**: DFAST_QC taxonomy (uses system dfast_qc)
- ✅ **Stage 3**: Prodigal gene prediction (uses system prodigal)
- ✅ **Stage 4**: Astra functional annotation (uses pyhmmer from conda)
- ✅ **Knowledge graph building**: All Python packages available
- ✅ **LLM components**: All required packages available
