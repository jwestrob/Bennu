# CLI Command Reference

Complete reference for all command-line interface operations.

## Environment Setup

**Always activate the conda environment first**:
```bash
source /Users/jacob/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh && conda activate genome-kg
```

## Main Commands

### `python -m src.cli build`
**Purpose**: Execute the 8-stage genomic processing pipeline

<details>
<summary><strong>Build Command Options (Click to expand)</strong></summary>

**Basic Usage**:
```bash
# Run complete pipeline (stages 0-8)
python -m src.cli build

# Resume from specific stage
python -m src.cli build --from-stage 5

# Run only to specific stage
python -m src.cli build --to-stage 7

# Skip taxonomic classification (stage 2)
python -m src.cli build --skip-tax
```

**Stage Control Examples**:
```bash
# Run only quality assessment
python -m src.cli build --from-stage 1 --to-stage 1

# Skip BGC and CAZyme annotation
python -m src.cli build --to-stage 4

# Process only embeddings (assumes stages 0-7 complete)
python -m src.cli build --from-stage 8
```

**Pipeline Stages**:
- **Stage 0**: Input preparation and validation
- **Stage 1**: QUAST quality assessment  
- **Stage 2**: Taxonomic classification (can be skipped)
- **Stage 3**: Gene prediction with Prodigal
- **Stage 4**: Functional annotation (PFAM/KOFAM)
- **Stage 5**: GECCO BGC detection
- **Stage 6**: dbCAN CAZyme annotation
- **Stage 7**: Knowledge graph construction
- **Stage 8**: ESM2 protein embeddings

**Error Handling**: Pipeline continues on non-critical failures (stages 5-6) with empty outputs

</details>

---

### `python -m src.cli ask`
**Purpose**: Ask natural language questions about genomic data

<details>
<summary><strong>Ask Command Usage (Click to expand)</strong></summary>

**Basic Usage**:
```bash
python -m src.cli ask "How many proteins are in the database?"
python -m src.cli ask "What transport proteins are present?"
python -m src.cli ask "Show me the distribution of CAZyme types among each genome"
```

**Query Types Supported**:
- **Counting**: "How many X are there?"
- **Functional**: "Show me transport proteins"  
- **Comparative**: "Compare X across genomes"
- **Similarity**: "Find proteins similar to Y"
- **Specific**: "What is the function of protein Z?"

**Output Format**:
```
🧬 Processing question: [your question]
🤖 Agentic planning: [True/False]
💭 Planning reasoning: [explanation]
📋 Using [traditional/agentic] query path
📊 Query type: [structural/semantic/hybrid/functional/comparative]
🔍 Search strategy: [direct_query/similarity_search/hybrid_search]

🤖 Answer:
[Detailed biological analysis]

Confidence: [high/medium/low]
Sources: [data sources used]
```

**Performance Notes**:
- Simple queries: <5 seconds
- Complex analysis: 15-60 seconds  
- Uses OpenAI o3 reasoning model for sophisticated biological interpretation

</details>

## Individual Stage Commands

### Stage 0: Input Preparation
```bash
python -m src.ingest.00_prepare_inputs \
    --input-dir data/raw \
    --output-dir data/stage00_prepared
```

<details>
<summary><strong>Input Preparation Details (Click to expand)</strong></summary>

**Purpose**: Validate and organize genome assemblies

**Input Requirements**:
- Genome files in `data/raw/`
- Supported formats: `.fna`, `.fasta`, `.fa`
- FASTA format compliance required

**Validation Checks**:
- File format verification
- Sequence length distribution
- Nucleotide composition analysis
- Basic quality indicators

**Output**:
```
data/stage00_prepared/
├── processing_manifest.json
└── genomes/
    ├── genome_001.fna
    ├── genome_002.fna
    └── ...
```

</details>

---

### Stage 1: Quality Assessment
```bash
python -m src.ingest.01_run_quast \
    --input-dir data/stage00_prepared \
    --output-dir data/stage01_quast
```

<details>
<summary><strong>QUAST Analysis Details (Click to expand)</strong></summary>

**Tool**: QUAST (Quality Assessment Tool for Genome Assemblies)

**Metrics Generated**:
- Assembly size and contiguity (N50, N90)
- Contig statistics
- Gene prediction with GeneMark
- Assembly accuracy assessment

**Output Files**:
- `report.txt`: Summary statistics
- `report.html`: Interactive report
- `contigs_reports/`: Per-genome details
- `basic_stats/`: Raw metrics

**Requirements**: QUAST installed via conda

</details>

---

### Stage 2: Taxonomic Classification
```bash
python -m src.ingest.02_dfast_qc \
    --input-dir data/stage00_prepared \
    --output-dir data/stage02_dfast_qc
```

<details>
<summary><strong>Taxonomic Classification Details (Click to expand)</strong></summary>

**Tool**: dfast_qc for ANI-based classification

**Installation**:
```bash
conda install -c bioconda dfast_qc
```

**Analysis**:
- Average Nucleotide Identity (ANI) calculation
- Taxonomic assignment based on reference genomes
- Genome completeness assessment
- Contamination detection

**Skip Option**: Use `--skip-tax` in main build command to bypass this stage

</details>

---

### Stage 3: Gene Prediction
```bash
python -m src.ingest.03_prodigal \
    --input-dir data/stage00_prepared \
    --output-dir data/stage03_prodigal
```

<details>
<summary><strong>Gene Prediction Details (Click to expand)</strong></summary>

**Tool**: Prodigal (Prokaryotic gene finder)

**Output Files**:
- `.faa`: Amino acid sequences
- `.genes.fna`: Nucleotide sequences
- `.gff`: Gene coordinates and annotations
- `all_protein_symlinks/`: Symlinked proteins for downstream tools

**Features**:
- Start codon recognition (ATG, GTG, TTG)
- Ribosome binding site identification
- Partial gene handling
- Metagenomic mode for diverse input

</details>

---

### Stage 4: Functional Annotation
```bash
python -m src.ingest.04_astra_scan \
    --input-dir data/stage03_prodigal \
    --output-dir data/stage04_astra \
    --databases PFAM KOFAM
```

<details>
<summary><strong>Functional Annotation Details (Click to expand)</strong></summary>

**Tool**: PyHMMer via Astra for HMM scanning

**Databases**:
- **PFAM**: Protein family domains (1,145 families with descriptions)
- **KOFAM**: KEGG Ortholog HMM profiles (813 orthologs with descriptions)

**Parameters**:
- E-value threshold: 1e-5
- Minimum coverage: 50%
- Domain-specific gathering thresholds

**Output**:
- `astra_pfam_results.tsv`: PFAM domain hits
- `astra_kofam_results.tsv`: KEGG ortholog assignments

**Functional Enrichment**: Automatically integrates authoritative descriptions from PFAM Stockholm files and KEGG KO lists

</details>

---

### Stage 5: GECCO BGC Detection
```bash
python -m src.ingest.gecco_bgc \
    --input-dir data/stage00_prepared \
    --output-dir data/stage05_gecco
```

<details>
<summary><strong>GECCO BGC Detection Details (Click to expand)</strong></summary>

**Tool**: GECCO (Gene Cluster Prediction with Conditional Random Fields)

**Installation**:
```bash
mamba install -c bioconda gecco hmmer
```

**Features**:
- Python-native implementation (no Docker required)
- 17 quantitative BGC properties extracted
- Product-specific probability scores
- Graceful error handling with workflow continuation

**Enhanced Properties**:
- Average/max/min probability scores
- Product probabilities: terpene, NRP, polyketide, RiPP, alkaloid, saccharide
- Structural metrics: gene count, domain count, core genes

**Error Resilience**: Creates empty outputs on tool failures to maintain pipeline integrity

</details>

---

### Stage 6: dbCAN CAZyme Annotation
```bash
python -m src.ingest.dbcan_cazyme \
    --input-dir data/stage03_prodigal/genomes/all_protein_symlinks \
    --output-dir data/stage06_dbcan
```

<details>
<summary><strong>dbCAN CAZyme Details (Click to expand)</strong></summary>

**Tool**: dbCAN (Database for Carbohydrate-Active enzyme ANnotation)

**Installation**:
```bash
pip install dbcan
# Database download (~12 minutes first time)
dbcan_build --cpus 8 --db-dir data/dbcan_db
```

**CAZyme Classes**:
- **GH**: Glycoside Hydrolases (breakdown)
- **GT**: Glycosyltransferases (synthesis)
- **PL**: Polysaccharide Lyases (β-elimination)
- **CE**: Carbohydrate Esterases (ester removal)
- **AA**: Auxiliary Activities (redox support)
- **CBM**: Carbohydrate-Binding Modules (targeting)

**Analysis Methods**:
- HMMER: HMM profiles against CAZy database
- DIAMOND: Sequence similarity
- Hotpep: Conserved peptide patterns
- Consensus integration

**Output**: 1,845 CAZymes annotated across all genomes

</details>

---

### Stage 7: Knowledge Graph Construction
```bash
python -m src.build_kg.rdf_builder \
    --stage03-dir data/stage03_prodigal \
    --stage04-dir data/stage04_astra \
    --stage05a-dir data/stage05_gecco \
    --stage05b-dir data/stage06_dbcan \
    --output-dir data/stage07_kg
```

<details>
<summary><strong>Knowledge Graph Construction Details (Click to expand)</strong></summary>

**Integration**: Combines all annotation sources into unified RDF graph

**Data Sources**:
- Stage 3: Gene predictions and coordinates
- Stage 4: PFAM domains and KEGG orthologs
- Stage 5a: GECCO BGC annotations with 17 properties
- Stage 5b: dbCAN CAZyme family classifications

**Output Formats**:
- `knowledge_graph.ttl`: RDF Turtle format
- `csv/`: Neo4j bulk import format

**Statistics**:
- **373,587 RDF triples** generated
- Links genomes, genes, proteins, domains, functions, BGCs, CAZymes
- Includes functional enrichment with authoritative descriptions

**Schema Enhancement**: Extended to support BGC and CAZyme annotations

</details>

---

### Stage 8: ESM2 Protein Embeddings
```bash
python run_esm2_m4_max.py
```

<details>
<summary><strong>ESM2 Embeddings Details (Click to expand)</strong></summary>

**Model**: ESM2 (Evolutionary Scale Modeling v2) from Meta AI

**Apple Silicon Optimization**:
- MPS acceleration for M4 Max GPU
- Optimal batch sizing (32 sequences)
- Automatic memory management
- 85 proteins/second processing rate

**Output**:
- 320-dimensional embeddings for 10,102 proteins
- LanceDB vector index for sub-millisecond similarity search
- Rich metadata integration (sequence length, GC content, annotations)

**Performance**: ~2 minutes total processing time on M4 Max

**Database**: LanceDB with IVF_PQ indexing for production-scale similarity search

</details>

## Database Management

### Neo4j Bulk Loading (Recommended)
```bash
python -m src.build_kg.neo4j_bulk_loader \
    --csv-dir data/stage07_kg/csv
```

<details>
<summary><strong>Bulk Loading Details (Click to expand)</strong></summary>

**Performance**: 48K nodes + 95K relationships in <10 seconds

**Method**: CSV generation → neo4j-admin import

**Advantages**:
- 15x faster than Python MERGE operations
- Production-ready for millions of nodes
- All relationships and properties preserved

**Requirements**: Neo4j database running at bolt://localhost:7687

</details>

---

### Neo4j Legacy Loading (Alternative)
```bash
python -m src.build_kg.neo4j_legacy_loader \
    --rdf-file data/stage07_kg/knowledge_graph.ttl
```

<details>
<summary><strong>Legacy Loading Details (Click to expand)</strong></summary>

**Method**: Direct RDF → Neo4j conversion with Python MERGE

**Use Cases**:
- Incremental updates
- Development and testing
- Custom transformation requirements

**Performance**: Slower but more flexible than bulk loader

</details>

## Testing Commands

### Comprehensive Test Suite
```bash
# Run all tests
python scripts/run_tests.py

# Quick smoke tests
python scripts/run_tests.py --smoke

# Run with coverage analysis
python scripts/run_tests.py --coverage

# Test specific modules
python scripts/run_tests.py --module ingest
python scripts/run_tests.py --module build_kg

# Run by test category
python scripts/run_tests.py --marker unit
python scripts/run_tests.py --marker integration

# Discover available tests
python scripts/run_tests.py --discover
```

<details>
<summary><strong>Testing Framework Details (Click to expand)</strong></summary>

**Test Organization**:
- **Unit tests**: Individual component testing
- **Integration tests**: End-to-end workflow validation
- **Smoke tests**: Quick functionality verification
- **External tests**: Database and tool integration

**Zero-Maintenance Discovery**: Automatically finds new tests following pytest conventions

**Coverage Analysis**: Comprehensive code coverage reporting

</details>

## Monitoring and Health Checks

### System Health Check
```bash
python -c "
from src.llm.config import LLMConfig
from src.llm.rag_system import GenomicRAG

config = LLMConfig()
rag = GenomicRAG(config)
health = rag.health_check()
print('System Health:', health)
rag.close()
"
```

<details>
<summary><strong>Health Check Components (Click to expand)</strong></summary>

**Checked Components**:
- **neo4j**: Graph database connectivity
- **lancedb**: Vector database accessibility  
- **hybrid**: Combined query processor
- **dspy**: LLM framework availability

**Expected Output**:
```
System Health: {
    'neo4j': True, 
    'lancedb': True, 
    'hybrid': True, 
    'dspy': True
}
```

**Troubleshooting**: If any component shows False, check database connections and dependencies

</details>

### Configuration Validation
```bash
python -c "
from src.llm.config import LLMConfig
config = LLMConfig()
validation = config.validate_configuration()
print('Config Status:', validation)
"
```

<details>
<summary><strong>Configuration Components (Click to expand)</strong></summary>

**Validated Elements**:
- **neo4j_configured**: Database connection parameters
- **lancedb_configured**: Vector database path exists
- **llm_configured**: API key availability
- **all_paths_exist**: Required directories present

**Environment Variables** (optional):
```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j  
export NEO4J_PASSWORD=your_password
export OPENAI_API_KEY=your_key
export LANCEDB_PATH=data/stage08_esm2/lancedb
```

</details>

## Performance Optimization

### Apple Silicon M4 Max Commands
```bash
# ESM2 embeddings optimized for M4 Max
python run_esm2_m4_max.py

# Monitor ESM2 progress (separate terminal)
python monitor_esm2_progress.py

# Test ESM2 similarity search
python test_esm2_similarity.py data/stage08_esm2
```

<details>
<summary><strong>M4 Max Optimization Details (Click to expand)</strong></summary>

**Performance Gains**:
- ESM2 processing: 85 proteins/second (vs 5 proteins/second on CPU)
- Memory management: Automatic MPS cache clearing
- Batch optimization: 32 sequences per batch for optimal GPU utilization

**Memory Requirements**:
- Peak usage: ~8GB during ESM2 processing
- Resident: ~2GB for LanceDB vector search
- Recommended: 16GB+ RAM for comfortable operation

</details>

## Error Handling and Troubleshooting

### Common Issues

<details>
<summary><strong>Environment Issues (Click to expand)</strong></summary>

**Problem**: `conda: command not found`
```bash
# Solution: Initialize conda
source /Users/jacob/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh
```

**Problem**: `genome-kg environment not found`
```bash
# Solution: Create environment (if missing)
conda create -n genome-kg python=3.11
conda activate genome-kg
pip install -r requirements.txt
```

**Problem**: Database connection errors
```bash
# Check Neo4j status
docker ps | grep neo4j

# Restart Neo4j if needed
docker-compose up -d neo4j
```

</details>

<details>
<summary><strong>Pipeline Issues (Click to expand)</strong></summary>

**Problem**: Stage failures
- **Non-critical stages** (5-6): Pipeline continues with empty outputs
- **Critical stages** (0-4, 7-8): Pipeline halts with detailed error messages

**Problem**: Missing tools
```bash
# Install missing bioinformatics tools
conda install -c bioconda prodigal quast dfast_qc
mamba install -c bioconda gecco hmmer
pip install dbcan
```

**Problem**: Disk space
- **ESM2 processing**: Requires ~2GB for embeddings
- **Knowledge graph**: ~500MB for 373K triples
- **Raw data**: Depends on genome sizes

</details>

<details>
<summary><strong>Query Issues (Click to expand)</strong></summary>

**Problem**: Slow query responses
- Check LLM model configuration (o3 reasoning model takes 15-60s)
- Verify database connectivity
- Consider query complexity

**Problem**: "No results found"
- Verify pipeline completion through stage 7
- Check database loading success
- Try simpler query variants

**Problem**: Low confidence answers
- Normal for complex comparative analysis
- Try breaking complex queries into simpler parts
- Check data completeness for your specific question

</details>

## Next Steps

- **[Basic Queries Tutorial](../tutorials/basic-queries.md)**: Learn query patterns
- **[Python API Reference](python-api.md)**: Programmatic access
- **[Architecture Overview](../architecture/overview.md)**: System design details