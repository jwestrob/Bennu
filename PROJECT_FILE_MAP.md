# PROJECT FILE MAP: Microbial Claude Matter - Genomic AI Platform

**Generated**: July 7, 2025  
**Purpose**: Comprehensive file mapping for LLM context understanding  
**Project**: Next-generation genomic intelligence platform with AI-powered biological insights

## Project Overview

This is an advanced genomic AI platform that transforms microbial genome assemblies into intelligent, queryable knowledge graphs with LLM-powered biological insights. The system combines traditional bioinformatics workflows with cutting-edge AI/ML technologies to create a comprehensive 8-stage pipeline culminating in an intelligent question-answering system.

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

---

## ROOT DIRECTORY FILES

### Configuration & Documentation

#### **README.md**
- **Purpose**: Main project documentation and user guide
- **Content**: Comprehensive overview of the world-class genomic AI platform
- **Key Sections**: 
  - Platform overview with revolutionary capabilities
  - Quick start guide and installation instructions
  - Performance benchmarks and biological intelligence examples
  - Scientific package ecosystem (60+ packages)
  - Deployment options (local, Docker, HPC/cloud)
- **Target Audience**: Users, developers, and researchers

#### **CLAUDE.md** 
- **Purpose**: Comprehensive guidance for Claude Code (claude.ai/code) when working with this platform
- **Content**: Detailed development context, commands, architecture, and recent achievements
- **Key Sections**:
  - Project overview with key achievements (276K+ RDF triples, 10K+ proteins)
  - Development commands (testing, pipeline execution, individual stages)
  - Architecture overview (7-stage pipeline, key components)
  - Performance summary (Apple Silicon M4 Max optimization)
  - Recent major developments and system improvements
- **Critical Info**: Environment setup requirements, file organization rules, testing protocols

#### **requirements-llm.txt**
- **Purpose**: Python dependencies for LLM integration components
- **Content**: 32 carefully curated packages for AI/ML functionality
- **Key Dependencies**:
  - Core ML: dspy-ai, transformers, torch
  - Vector databases: lancedb, neo4j
  - Data processing: numpy, pandas, h5py, pyarrow
  - Graph processing: rdflib
  - Web/API: httpx, pydantic, typer
  - Development: pytest, pytest-asyncio

#### **pytest.ini**
- **Purpose**: Pytest configuration for comprehensive testing framework
- **Features**: 
  - Test discovery configuration (src/tests directory)
  - Markers for test categorization (slow, integration, unit, external)
  - Output formatting and warning suppression
  - Color output and verbose reporting

#### **test.sh**
- **Purpose**: Simple bash wrapper for the master test runner
- **Functionality**: Executes `python run_tests.py` with all passed arguments
- **Usage**: `./test.sh --smoke`, `./test.sh --coverage`, etc.

### Build & Deployment

#### **.gitignore**
- **Purpose**: Git ignore patterns for the project
- **Content**: Standard Python, data, and build artifact exclusions

#### **Dockerfile**
- **Purpose**: Main container definition for the genomic AI platform
- **Content**: Multi-stage build for production deployment

#### **nextflow.config** 
- **Purpose**: Nextflow workflow configuration
- **Profiles**: standard, docker, singularity, cluster, cloud
- **Parameters**: Thread counts, memory limits, container registry settings

#### **main.nf**
- **Purpose**: Nextflow workflow definition for HPC/cloud execution
- **Content**: Parallel processing pipeline with container support

---

## SOURCE CODE STRUCTURE (`src/`)

### Main Entry Point

#### **src/cli.py**
- **Purpose**: Main command-line interface using Typer framework
- **Commands**:
  - `build`: Execute 8-stage genomic processing pipeline (stages 0-8)
  - `ask`: Natural language question answering over genomic data
  - `version`: Show version information
- **Key Features**:
  - Configurable stage execution (--from-stage, --to-stage)
  - Parallel processing support (--threads)
  - Force overwrite and skip options
  - Rich console output with progress tracking
- **Pipeline Stages**:
  - Stage 0: Input preparation and validation
  - Stage 1: QUAST quality assessment
  - Stage 2: DFAST_QC taxonomic classification (optional)
  - Stage 3: Prodigal gene prediction
  - Stage 4: Astra functional annotation (PFAM/KOFAM)
  - Stage 5: GECCO BGC detection (NEW)
  - Stage 6: dbCAN CAZyme annotation (NEW)
  - Stage 7: Knowledge graph construction (RDF triples) (ENHANCED)
  - Stage 8: ESM2 protein embeddings (semantic search)

### Data Ingestion Pipeline (`src/ingest/`)

#### **src/ingest/00_prepare_inputs.py**
- **Purpose**: Stage 0 - Input preparation and validation
- **Functionality**:
  - FASTA format validation with comprehensive error checking
  - Sequence statistics calculation (count, length, IDs)
  - Duplicate ID detection and invalid character identification
  - File organization and manifest generation
  - MD5 checksum calculation for data integrity
- **Output**: Validated inputs in `data/stage00_prepared/` with processing manifest

#### **src/ingest/01_run_quast.py**
- **Purpose**: Stage 1 - Assembly quality assessment using QUAST
- **Functionality**:
  - Parallel QUAST execution across multiple genomes
  - Quality metrics extraction (N50, total length, contigs)
  - Report generation in multiple formats (TSV, TXT, TEX)
  - Summary statistics aggregation
- **Output**: Quality reports in `data/stage01_quast/` with per-genome subdirectories

#### **src/ingest/02_dfast_qc.py**
- **Purpose**: Stage 2 - Taxonomic classification using DFAST_QC
- **Functionality**:
  - ANI-based taxonomic assignment
  - CheckM-style completeness/contamination analysis
  - Optional execution (can be skipped with --skip-tax)
- **Output**: Taxonomic classifications in `data/stage02_dfast_qc/`

#### **src/ingest/03_prodigal.py**
- **Purpose**: Stage 3 - Gene prediction using Prodigal
- **Functionality**:
  - Protein-coding sequence prediction
  - Parallel processing across genomes
  - Rich genomic metadata extraction (coordinates, strand, GC content)
  - Creates `all_protein_symlinks` directory for downstream processing
- **Output**: 
  - Protein sequences (.faa files) in `data/stage03_prodigal/`
  - Gene sequences (.genes.fna files)
  - Prodigal logs with detailed prediction statistics

#### **src/ingest/04_astra_scan.py**
- **Purpose**: Stage 4 - Functional annotation using Astra/PyHMMer
- **Functionality**:
  - HMM domain scanning against PFAM and KOFAM databases
  - High-performance parallel processing
  - E-value filtering and hit quality assessment
  - Results aggregation across all proteins
- **Output**: 
  - PFAM hits in `data/stage04_astra/pfam_results/`
  - KOFAM hits in `data/stage04_astra/kofam_results/`
  - Combined annotation tables (TSV format)

#### **src/ingest/gecco_bgc.py**
- **Purpose**: Stage 5 - GECCO BGC (Biosynthetic Gene Cluster) detection (NEW)
- **Functionality**:
  - Python-native GECCO integration (replaces AntiSMASH Docker issues)
  - Biosynthetic gene cluster prediction with confidence scores
  - 17 quantitative properties per BGC including product-specific probabilities
  - Graceful error handling for tool compatibility issues
  - Workflow continuation even when pyrodigal compatibility problems occur
- **Output**: 
  - BGC predictions in `data/stage05_gecco/`
  - Cluster boundaries and gene assignments
  - Confidence scores and product predictions

#### **src/ingest/dbcan_cazyme.py**
- **Purpose**: Stage 6 - dbCAN CAZyme (Carbohydrate-Active enZyme) annotation (NEW)
- **Functionality**:
  - Carbohydrate-active enzyme annotation using dbCAN
  - Comprehensive CAZy family classification (GH, GT, PL, CE, AA, CBM)
  - Substrate specificity prediction
  - E-value and coverage quality metrics
  - Domain boundary mapping for sub-protein localization
- **Output**:
  - CAZyme annotations in `data/stage06_dbcan/`
  - Family classifications and substrate predictions
  - Quality metrics for annotation confidence

#### **src/ingest/06_esm2_embeddings.py**
- **Purpose**: Stage 8 - ESM2 protein embeddings for semantic search (UPDATED STAGE NUMBER)
- **Functionality**:
  - 320-dimensional protein embeddings using ESM2 transformer
  - Apple Silicon MPS acceleration (~85 proteins/second)
  - LanceDB index creation for sub-millisecond similarity search
  - Batch processing with memory management
  - Progress tracking and error handling
- **Output**:
  - Protein embeddings in `data/stage08_esm2/protein_embeddings.h5`
  - LanceDB indices in `data/stage08_esm2/lancedb/`
  - Embedding manifest with metadata

### Knowledge Graph Construction (`src/build_kg/`)

#### **src/build_kg/rdf_builder.py**
- **Purpose**: Stage 7 - RDF triple generation for knowledge graph construction (ENHANCED)
- **Functionality**:
  - Parses Prodigal headers to extract genomic coordinates and metadata
  - Processes functional annotations from PFAM/KOFAM results
  - Integrates GECCO BGC annotations with 17 quantitative properties per BGC
  - Integrates dbCAN CAZyme annotations with family classifications
  - Generates 373,587+ RDF triples linking genomes, proteins, domains, functions, BGCs, and CAZymes
  - Integrates functional enrichment and pathway information
  - Creates comprehensive biological ontology relationships
- **Key Features**:
  - Rich genomic metadata (coordinates, strand, GC content, start codons)
  - Functional annotation integration with confidence scores
  - BGC schema integration with probability scores and product predictions
  - CAZyme family classification and substrate specificity
  - Namespace management for biological entities
  - Provenance tracking and data lineage
- **Output**: Knowledge graph in RDF format (`data/stage07_kg/knowledge_graph.ttl`)

#### **src/build_kg/functional_enrichment.py**
- **Purpose**: Enrich knowledge graph with authoritative functional descriptions (ENHANCED)
- **Functionality**:
  - Parses PFAM Stockholm format files for family descriptions
  - Processes KEGG KO list for ortholog definitions
  - Adds CAZy family descriptions from dbCAN database
  - Adds 1,145 PFAM families + 813 KEGG orthologs + CAZyme families with descriptions
  - Replaces hardcoded biological knowledge with reference databases
- **Data Classes**:
  - `PfamEntry`: PFAM family with accession, description, type, clan
  - `KoEntry`: KEGG ortholog with definition, threshold, score type
  - `CazyEntry`: CAZy family with classification and substrate information
- **Impact**: Transforms knowledge graph from ~242K to 373,587 RDF triples (UPDATED)

#### **src/build_kg/neo4j_bulk_loader.py**
- **Purpose**: High-performance Neo4j database loading (100x faster than Python MERGE)
- **Functionality**:
  - Converts RDF triples to Neo4j CSV format
  - Uses neo4j-admin import for bulk loading
  - Auto-detects Neo4j installation (Homebrew support)
  - Manages database lifecycle (stop, import, start)
- **Performance**: 20.08 seconds for 37,930 nodes + 85,626 relationships
- **Features**:
  - Automatic Neo4j service management
  - CSV format optimization for bulk import
  - Error handling and validation
  - Production-ready scalability

#### **src/build_kg/neo4j_legacy_loader.py**
- **Purpose**: Python-based Neo4j loading using MERGE operations
- **Use Case**: Development and small datasets
- **Functionality**: Direct Python-to-Neo4j loading with relationship creation

#### **src/build_kg/annotation_processors.py**
- **Purpose**: Process functional annotation results from Astra/PyHMMer
- **Functionality**: Parse PFAM/KOFAM hit tables and extract annotation data

#### **src/build_kg/pathway_integration.py**
- **Purpose**: Integrate KEGG pathway information into knowledge graph
- **Functionality**: Add pathway relationships and metabolic context

#### **src/build_kg/sequence_db.py** & **sequence_db_builder.py**
- **Purpose**: Build and manage protein sequence database for code interpreter
- **Functionality**: Create SQLite database with protein sequences for analysis

#### **src/build_kg/schema.py**
- **Purpose**: Define Neo4j database schema and ontology
- **Content**: Node types, relationship definitions, property schemas

#### **src/build_kg/rdf_emit.py**
- **Purpose**: RDF serialization utilities
- **Functionality**: Efficient RDF triple emission and formatting

#### **src/build_kg/rdf_to_csv_converter.py**
- **Purpose**: Convert RDF triples to Neo4j CSV format
- **Functionality**: Transform RDF data for bulk import optimization

#### **src/build_kg/provenance.py**
- **Purpose**: Track data lineage and processing provenance
- **Functionality**: Record processing steps and data transformations

### LLM Integration & AI Components (`src/llm/`)

#### **src/llm/rag_system.py**
- **Purpose**: Core agentic RAG system with multi-step reasoning capabilities
- **Architecture**: DSPy-based system combining Neo4j + LanceDB + Code Interpreter
- **Key Components**:
  - `ResultStreamer`: Chunks large datasets to prevent context overflow
  - `TaskGraph`: DAG-based task execution with dependency resolution
  - `Task`: Individual task with status tracking (PENDING → RUNNING → COMPLETED)
  - `GenomicRAG`: Main RAG class with intelligent query routing
- **Capabilities**:
  - Multi-stage query processing (Neo4j → LanceDB similarity expansion)
  - Agentic task orchestration with code execution
  - Literature integration with PubMed search
  - Intelligent annotation discovery and curation
- **Advanced Features**:
  - Token-aware chunking for large result sets
  - Session management for iterative analysis
  - Automatic query classification and routing
  - Error handling with repair agent integration

#### **src/llm/task_repair_agent.py**
- **Purpose**: Autonomous error detection and repair for genomic RAG queries
- **Functionality**:
  - Detects common DSPy query generation errors
  - Provides schema-aware repair suggestions
  - Transforms technical errors into user-friendly guidance
  - Supports biological context in error messages
- **Error Patterns**: Syntax errors, relationship mapping, entity suggestions
- **Integration**: Seamlessly integrated into RAG query processing pipeline

#### **src/llm/annotation_tools.py**
- **Purpose**: Intelligent annotation discovery and biological function classification
- **Key Functions**:
  - `annotation_explorer`: Comprehensive annotation catalog exploration
  - `functional_classifier`: Universal biological mechanism classification
  - `annotation_selector`: Diverse example selection with preferences
  - `pathway_based_annotation_selector`: KEGG pathway-aware protein discovery
- **Solves**: "ATP synthase problem" through intelligent biological exclusion logic
- **Capabilities**: Works for any functional category (transport, metabolism, regulation)

#### **src/llm/config.py**
- **Purpose**: LLM configuration management
- **Content**: Database connections, API keys, model settings

#### **src/llm/cli.py**
- **Purpose**: CLI interface for LLM question answering
- **Functionality**: Process natural language questions and return biological insights

#### **src/llm/query_processor.py**
- **Purpose**: Database query processing and execution
- **Components**: Neo4j, LanceDB, and hybrid query processors

#### **src/llm/dsp_sig.py**
- **Purpose**: DSPy signature definitions and Neo4j schema documentation
- **Content**: Structured prompts and database schema for LLM integration

#### **src/llm/sequence_tools.py**
- **Purpose**: Protein sequence analysis tools
- **Functionality**: Sequence viewer, protein ID extraction, genomic context

#### **src/llm/domain_functions.py**
- **Purpose**: Protein domain analysis functions
- **Functionality**: PFAM domain queries and functional analysis

#### **src/llm/pathway_tools.py**
- **Purpose**: KEGG pathway analysis and metabolic context
- **Functionality**: Pathway-based protein discovery and classification

#### **src/llm/retrieval.py**
- **Purpose**: Information retrieval from knowledge graph and vector database
- **Functionality**: Context retrieval and relevance scoring

#### **src/llm/qa_chain.py**
- **Purpose**: Question-answering chain implementation
- **Functionality**: End-to-end QA processing with biological reasoning

#### **src/llm/error_patterns.py** & **repair_types.py**
- **Purpose**: Error pattern recognition and repair type definitions
- **Content**: Error detection patterns and repair strategy implementations

#### **src/llm/task_notes.py**
- **Purpose**: Task annotation and note-taking functionality
- **Functionality**: Track analysis progress and insights

### Code Interpreter Service (`src/code_interpreter/`)

#### **src/code_interpreter/service.py**
- **Purpose**: Secure FastAPI service for Python code execution
- **Security Features**:
  - Container isolation with read-only filesystem
  - Resource limits and timeout enforcement
  - Non-root execution with restricted capabilities
  - Localhost-only access
- **Capabilities**:
  - Stateful sessions for iterative analysis
  - 60+ scientific packages pre-installed
  - Session persistence across agentic workflows
  - Comprehensive error handling and logging

#### **src/code_interpreter/client.py**
- **Purpose**: Client interface for RAG system communication with code interpreter
- **Functionality**:
  - HTTP client for secure code execution requests
  - Session management and state persistence
  - Error handling and timeout management
  - Integration with agentic task workflows

#### **src/code_interpreter/sequence_service.py**
- **Purpose**: Sequence database service for code interpreter
- **Functionality**: Provides access to protein sequences for analysis

#### **src/code_interpreter/Dockerfile**
- **Purpose**: Container definition for secure code execution environment
- **Features**: Multi-stage build with security hardening

#### **src/code_interpreter/docker-compose.yml**
- **Purpose**: Docker Compose configuration for service deployment
- **Content**: Service definitions and networking configuration

#### **src/code_interpreter/requirements.txt**
- **Purpose**: Python dependencies for code interpreter environment
- **Content**: 60+ scientific packages for comprehensive genomic analysis

#### **src/code_interpreter/deploy.sh**
- **Purpose**: Deployment script for code interpreter service
- **Functionality**: Automated container build and deployment

#### **src/code_interpreter/README.md**
- **Purpose**: Documentation for code interpreter service
- **Content**: Setup instructions and usage examples

### Utilities (`src/utils/`)

#### **src/utils/command_runner.py**
- **Purpose**: Utility for running external commands
- **Functionality**: Subprocess management with logging and error handling

---

## TESTING FRAMEWORK (`src/tests/`)

### Test Configuration

#### **src/tests/conftest.py**
- **Purpose**: Pytest configuration and shared fixtures for comprehensive testing
- **Fixtures Provided**:
  - `temp_dir`: Temporary directory for test outputs
  - `dummy_fasta_content`: Sample FASTA content for validation testing
  - `dummy_genome_files`: Complete genome file sets for pipeline testing
  - `mock_quast_output`: QUAST quality assessment mock data
  - `mock_checkm_output`: CheckM completeness/contamination mock data
  - `mock_gtdb_output`: GTDB taxonomic classification mock data
  - `mock_prodigal_output`: Gene prediction mock data
  - `expected_pipeline_manifest`: Expected manifest format validation
  - `cli_runner`: Typer CLI test runner
  - `mock_subprocess_run`: Mock external tool execution
- **Coverage**: Supports all pipeline stages with realistic test data

### Core Test Suites

#### **src/tests/test_agentic_rag_system.py**
- **Purpose**: Comprehensive testing of agentic RAG system capabilities
- **Coverage**: Multi-step workflows, task orchestration, code interpreter integration

#### **src/tests/test_code_interpreter.py**
- **Purpose**: Code interpreter service testing
- **Coverage**: Secure execution, session management, error handling

#### **src/tests/test_genomic_questions.py**
- **Purpose**: End-to-end genomic question answering validation
- **Coverage**: Natural language queries, biological reasoning, answer quality

#### **src/tests/test_enhanced_tpr.py** & **test_tpr_context_format.py**
- **Purpose**: TPR (transport protein) analysis testing
- **Coverage**: Protein classification, context formatting, biological accuracy

#### **src/tests/test_ggdef_fix.py**
- **Purpose**: GGDEF domain query testing and validation
- **Coverage**: Domain search functionality, query generation fixes

#### **src/tests/test_multiple_questions.py**
- **Purpose**: Batch question processing testing
- **Coverage**: Multiple query handling, performance validation

### Module-Specific Tests

#### **src/tests/test_build_kg/**
- **test_annotation_processors.py**: Functional annotation processing tests
- **test_functional_enrichment.py**: PFAM/KEGG enrichment validation
- **test_integration.py**: Knowledge graph integration testing
- **test_rdf_builder.py**: RDF triple generation validation

#### **src/tests/test_ingest/**
- **test_00_prepare_inputs.py**: Input validation and preparation tests
- **test_01_quast.py**: QUAST quality assessment testing
- **test_03_prodigal.py**: Gene prediction validation
- **test_04_astra_scan.py**: Functional annotation testing

#### **src/tests/llm/**
- **test_task_repair_agent.py**: Error detection and repair testing

### Demo & Debug Tests

#### **src/tests/demo/**
- **test_agentic_demo.py**: Agentic capabilities demonstration
- **test_agentic_validation.py**: Agentic workflow validation
- **test_code_interpreter_integration.py**: Code interpreter integration demos
- **test_pathway_based_discovery.py**: Pathway-based protein discovery
- **test_protein_id_fix.py**: Protein ID system validation
- **test_template_resolution.py**: Template resolution testing
- **task_repair_agent_demo.py**: Error repair agent demonstrations
- **protein_id_fix_demo.py**: Protein ID fix demonstrations
- **final_protein_id_solution.py**: Final protein ID solution validation

#### **src/tests/debug/**
- **debug_agentic_integration.py**: Agentic system integration debugging
- **debug_domain_enhancement.py**: Domain enhancement debugging
- **debug_ggdef_query.py**: GGDEF domain query debugging
- **debug_protein_extraction.py**: Protein extraction debugging
- **debug_rag_context.py**: RAG context debugging and analysis
- **debug_sequence_db.py**: Sequence database debugging
- **debug_task_graph_issues.py**: Task graph execution debugging
- **debug_tpr_query.py**: TPR query debugging
- **fix_context_demo.py**: Context formatting fix demonstrations

### Specialized Tests

#### **src/tests/streaming/**
- **test_streamer.py**: Result streaming and chunking tests

---

## SCRIPTS & UTILITIES (`scripts/`)

### Main Test Runner

#### **scripts/run_tests.py**
- **Purpose**: Master test runner with automatic test discovery
- **Features**:
  - Zero-maintenance test discovery (follows pytest conventions)
  - Multiple execution modes (smoke, coverage, module-specific)
  - Marker-based test categorization (unit, integration, slow, external)
  - Coverage reporting with HTML output
  - Parallel execution support with pytest-xdist
  - Comprehensive reporting and statistics
- **Usage Examples**:
  - `python scripts/run_tests.py --smoke`: Quick validation tests
  - `python scripts/run_tests.py --coverage`: Full coverage analysis
  - `python scripts/run_tests.py --module llm`: LLM-specific tests
  - `python scripts/run_tests.py --marker integration`: Integration tests only

### Performance & Optimization Scripts

#### **scripts/run_esm2_m4_max.py**
- **Purpose**: Apple Silicon M4 Max optimized ESM2 embedding generation
- **Performance**: ~85 proteins/second processing rate
- **Features**: MPS acceleration, memory management, progress tracking

#### **scripts/monitor_esm2_progress.py**
- **Purpose**: Real-time monitoring of ESM2 embedding progress
- **Functionality**: Progress tracking, performance metrics, ETA calculation

#### **scripts/test_esm2_similarity.py**
- **Purpose**: ESM2 embedding similarity search validation
- **Coverage**: LanceDB integration, similarity scoring, metadata filtering

### Debugging & Analysis Scripts

#### **scripts/debug_*.py** (Multiple Files)
- **debug_context_format.py**: Context formatting debugging
- **debug_detailed_query.py**: Detailed query analysis
- **debug_genomic_context.py**: Genomic context debugging
- **debug_protein_retrieval.py**: Protein retrieval debugging
- **debug_raw_data.py**: Raw data analysis
- **debug_rdf_relationships.py**: RDF relationship debugging
- **debug_relationships.py**: General relationship debugging

### Validation & Testing Scripts

#### **scripts/test_*.py** (Multiple Files)
- **test_exact_query.py**: Exact query validation
- **test_genomic_context.py**: Genomic context testing
- **test_llm_integration.py**: LLM integration validation
- **test_neo4j.py**: Neo4j database testing
- **test_prodigal_parsing.py**: Prodigal output parsing validation
- **test_protein_query.py**: Protein query testing

### Analysis & Discovery Scripts

#### **scripts/check_*.py** (Multiple Files)
- **check_gene_coordinates.py**: Gene coordinate validation
- **check_ggdef_descriptions.py**: GGDEF domain description analysis
- **check_neo4j_schema.py**: Neo4j schema validation

#### **scripts/find_annotated_protein.py**
- **Purpose**: Find and analyze annotated proteins in the database
- **Functionality**: Protein discovery, annotation analysis, context extraction

### Coverage Reporting

#### **scripts/coverage/**
- **Purpose**: HTML coverage reports generated by pytest-cov
- **Content**: Detailed code coverage analysis with line-by-line reporting
- **Files**: HTML reports, CSS styling, JavaScript functionality

---

## DATA STRUCTURE (`data/`)

### Input Data

#### **data/raw/**
- **Purpose**: Input genome assemblies for processing
- **Content**: FASTA files (.fna, .fasta, .fa) containing microbial genome assemblies
- **Examples**:
  - `Burkholderiales_bacterium_RIFCSPHIGHO2_01_FULL_64_960.contigs.fna`
  - `Candidatus_Muproteobacteria_bacterium_RIFCSPHIGHO2_01_FULL_61_200.contigs.fna`
  - `Candidatus_Nomurabacteria_bacterium_RIFCSPLOWO2_01_FULL_41_220.contigs.fna`
  - `PLM0_60_b1_sep16_Maxbin2_047_curated.contigs.fna`

#### **data/reference/**
- **Purpose**: Reference database files for functional enrichment
- **Content**:
  - `Pfam-A.hmm.dat.stockholm`: PFAM family descriptions (1,145 families)
  - `ko_list`: KEGG ortholog definitions (813 orthologs)
  - `ko_pathway.list`: KEGG pathway mappings

### Pipeline Stage Outputs

#### **data/stage00_prepared/**
- **Purpose**: Validated and organized input files
- **Content**: Processed genome files with validation manifest

#### **data/stage01_quast/**
- **Purpose**: Assembly quality assessment results
- **Structure**:
  - `genomes/[genome_name]/`: Per-genome QUAST reports
  - `processing_manifest.json`: Processing metadata
  - `summary_stats.json`: Aggregated quality statistics
- **Reports**: TSV, TXT, TEX formats with quality metrics

#### **data/stage03_prodigal/**
- **Purpose**: Gene prediction results
- **Structure**:
  - `genomes/[genome_name]/`: Per-genome gene predictions
  - `.faa` files: Protein sequences
  - `.genes.fna` files: Gene sequences
  - `prodigal.log`: Prediction logs with statistics

#### **data/stage04_astra/**
- **Purpose**: Functional annotation results
- **Structure**:
  - `pfam_results/`: PFAM domain annotations
  - `kofam_results/`: KEGG ortholog annotations
  - `PFAM_hits_df.tsv` & `KOFAM_hits_df.tsv`: Annotation tables
  - `astra_search_log.txt`: Search logs

#### **data/stage05_gecco/**
- **Purpose**: GECCO BGC detection results (NEW)
- **Structure**:
  - BGC cluster predictions with confidence scores
  - Product-specific probability assessments
  - Cluster boundary definitions and gene assignments

#### **data/stage06_dbcan/**
- **Purpose**: dbCAN CAZyme annotation results (NEW)
- **Structure**:
  - CAZyme family classifications (GH, GT, PL, CE, AA, CBM)
  - Substrate specificity predictions
  - E-value and coverage quality metrics

#### **data/stage07_kg/**
- **Purpose**: Enhanced knowledge graph with BGC and CAZyme integration (UPDATED)
- **Structure**:
  - `knowledge_graph.ttl`: Complete RDF knowledge graph (373,587+ triples)
  - `csv/`: Neo4j bulk import format
  - BGC and CAZyme schema integration

#### **data/stage08_esm2/**
- **Purpose**: ESM2 protein embeddings and similarity search (UPDATED STAGE)
- **Structure**:
  - `lancedb/protein_embeddings.lance/`: LanceDB vector database
  - `protein_embeddings.h5`: HDF5 embedding storage
  - `embedding_manifest.json`: Embedding metadata

### Knowledge Graph & Database

#### **data/sequences.db**
- **Purpose**: SQLite database containing protein sequences for code interpreter
- **Content**: 10,102+ protein sequences with metadata

#### **data/pathway_integration/**
- **Purpose**: KEGG pathway integration results
- **Content**:
  - `pathway_integration.ttl`: Pathway RDF triples
  - `pathway_statistics.txt`: Integration statistics

### Debug & Analysis Outputs

#### **data/command_outputs/**
- **Purpose**: Stored command outputs for analysis
- **Content**: Timestamped outputs from annotation explorer and other tools

#### **data/debug_outputs/**
- **Purpose**: Debug information from agentic workflows
- **Content**: JSON files with task execution details and debugging information

#### **data/test_kg_with_pathways/**
- **Purpose**: Test knowledge graph with pathway integration
- **Content**:
  - `knowledge_graph.ttl`: Complete RDF knowledge graph
  - `build_statistics.json`: Construction statistics

---

## ROOT LEVEL DOCUMENTATION & CONFIGURATION

### Documentation Files

#### **AGENTS.md**
- **Purpose**: Documentation for agentic capabilities and AI agent architecture
- **Content**: Agent design patterns, task orchestration, multi-step reasoning

#### **AGENT_AUGMENTATION_PLAN.md**
- **Purpose**: Roadmap for agent capability enhancement
- **Content**: Future development plans, capability expansion strategies

#### **TESTING.md**
- **Purpose**: Comprehensive testing documentation
- **Content**: Testing strategies, framework usage, best practices

#### **ENVIRONMENT_SETUP.md**
- **Purpose**: Environment setup and configuration guide
- **Content**: Installation instructions, dependency management, troubleshooting

#### **COMMAND_OUTPUT_MANAGEMENT.md**
- **Purpose**: Documentation for command output handling and management
- **Content**: Output processing strategies, storage patterns, analysis workflows

### Session & Debug Files

#### **full_conversation.txt** & **session_dump.txt**
- **Purpose**: Development session logs and conversation history
- **Content**: Detailed development progress, debugging sessions, decision rationale

#### **rag_context_debug.json**
- **Purpose**: RAG system context debugging information
- **Content**: Context retrieval analysis, query processing details

#### **import.report**
- **Purpose**: Import analysis and dependency tracking
- **Content**: Module import analysis, dependency resolution

#### **opencode_session_viewer.sh**
- **Purpose**: Session viewing utility script
- **Content**: Shell script for viewing development sessions

### Workflow Files

#### **test_annotation_workflow.py** & **test_complete_workflow.py**
- **Purpose**: End-to-end workflow testing scripts
- **Content**: Complete pipeline validation, integration testing

#### **run_annotation_explorer.py**
- **Purpose**: Annotation exploration utility
- **Content**: Interactive annotation discovery and analysis

---

## SUMMARY

This genomic AI platform represents a comprehensive system with:

### **Core Capabilities**
- **8-Stage Bioinformatics Pipeline**: From raw genomes to intelligent insights (UPDATED)
- **Knowledge Graph**: 373,587 RDF triples with rich biological relationships (UPDATED)
- **BGC Analysis**: Enhanced GECCO integration with 17 quantitative properties per BGC (NEW)
- **CAZyme Annotation**: Comprehensive carbohydrate-active enzyme classification (NEW)
- **Vector Search**: 10,102+ proteins with 320-dimensional ESM2 embeddings
- **Agentic RAG System**: Multi-step reasoning with code execution capabilities
- **Secure Code Interpreter**: 60+ scientific packages in isolated environment

### **File Organization**
- **Source Code**: 50+ Python modules organized by functionality
- **Testing**: 30+ test files with comprehensive coverage
- **Scripts**: 20+ utility scripts for development and analysis
- **Documentation**: 10+ documentation files for users and developers
- **Data Pipeline**: Structured data flow through 8 processing stages (UPDATED)

### **Key Innovations**
- **Enhanced Multi-Database Integration**: GECCO BGC + dbCAN CAZyme support (NEW)
- **Docker-Free BGC Detection**: Python-native GECCO eliminates compatibility issues (NEW)
- **Intelligent Annotation Discovery**: Solves "ATP synthase problem" with biological reasoning
- **Multi-Stage Query Processing**: Neo4j → LanceDB similarity expansion
- **Task Repair Agent**: Autonomous error detection and repair
- **Apple Silicon Optimization**: ~85 proteins/second ESM2 processing
- **Production-Ready Architecture**: Containerized microservices with bulk loading

This platform transforms traditional bioinformatics into an intelligent, queryable system where AI agents can navigate complex biological data and provide world-class genomic insights.
