# LLM System Architecture Documentation

## Overview

This document provides comprehensive technical documentation for the advanced Genomic Intelligence Platform's LLM system, focusing on the `src/llm` directory and its sophisticated AI capabilities.

The LLM system transforms microbial genome assemblies into an intelligent, queryable knowledge graph with AI-powered biological insights through an 8-stage pipeline culminating in an advanced question-answering system.

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Architecture](#core-architecture)
3. [RAG System Components](#rag-system-components)
4. [Agent System & Task Management](#agent-system--task-management)
5. [Memory & Synthesis Systems](#memory--synthesis-systems)
6. [Model Allocation & Intelligence](#model-allocation--intelligence)
7. [Data Flow Architecture](#data-flow-architecture)
8. [API Reference](#api-reference)
9. [Performance & Optimization](#performance--optimization)
10. [Architectural Improvements](#architectural-improvements)

---

## System Overview

### Key Achievements
- **373,587 RDF triples** linking genomes, proteins, domains, and functions
- **Sub-millisecond vector similarity search** with LanceDB
- **Intelligent model allocation** (o3/GPT-4.1-mini) for cost-optimized biological reasoning
- **Agentic task system** with DAG-based execution and dependency resolution
- **Progressive synthesis** with Map-Reduce architecture for large datasets
- **Apple Silicon M4 Max optimization** (~85 proteins/second processing)

### Architecture Philosophy
The system follows a **modular, agent-first architecture** with:
- **LLM-first tool selection** using sophisticated biological reasoning
- **Progressive synthesis** for handling large multi-task workflows
- **Intelligent model allocation** for cost-optimization
- **Multi-stage query processing** combining structured and semantic search
- **Context compression** with biological pattern preservation

---

## Core Architecture

### Main Entry Point: GenomicRAG Class (`src/llm/rag_system/core.py`)

The `GenomicRAG` class serves as the central orchestrator, combining:
- Structured queries (Neo4j) with semantic search (LanceDB)
- Agentic task planning and execution
- Intelligent model allocation and context management
- Progressive synthesis for complex analyses

```python
class GenomicRAG(dspy.Module):
    """
    Main genomic RAG system with working implementation.
    
    Combines structured queries (Neo4j) with semantic search (LanceDB)
    and intelligent code interpreter enhancement.
    """
```

#### Key Components Initialized:
- **Query Processors**: Neo4j, LanceDB, and Hybrid processors
- **Intelligent Router**: Routes queries to appropriate processors
- **Genome Selector**: LLM-based genome selection with intent analysis
- **Context Compressor**: Intelligent compression preserving biological data
- **Memory System**: Note-taking and progressive synthesis
- **Model Allocator**: Cost-optimized model selection system

### Query Processing Architecture

The system supports two main execution paths:

#### 1. Traditional Query Path
- **Single-step execution** for direct queries
- **LLM-based classification** using o3 for biological reasoning
- **Context retrieval** with fallback logic
- **Answer synthesis** using model allocation

#### 2. Agentic Query Path
- **Multi-step task execution** with TaskGraph
- **Upfront genome selection** for workflow optimization
- **Task dependency resolution** with intelligent chunking
- **Progressive synthesis** from task results and notes

---

## RAG System Components

### 1. Query Processing (`src/llm/rag_system/`)

#### Context Processing (`context_processing.py`)
- **Context retrieval** from multiple data sources
- **Biological pattern preservation** during compression
- **Multi-database integration** (Neo4j + LanceDB)

#### Context Compression (`context_compression.py`)
- **Token-based decision making** for compression thresholds
- **Progressive compression** with priority-based content preservation
- **Biological data prioritization** (prophage/spatial data = 100 points)

#### Intelligent Routing (`intelligent_routing.py`)
- **Query type classification** using biological reasoning
- **Dynamic processor selection** based on query characteristics
- **Fallback mechanisms** for robust query handling

### 2. Database Query Processors

#### Neo4j Query Processor
- **Cypher query generation** with biological validation
- **Comparative query validation** (removes inappropriate LIMIT clauses)
- **Genome filtering** with scope enforcement
- **TaskRepairAgent integration** for error handling

#### LanceDB Query Processor
- **Semantic similarity search** with ESM2 embeddings
- **Sub-millisecond queries** across 10,102 proteins
- **320-dimensional semantic embeddings**

#### Hybrid Query Processor
- **Multi-stage processing**: Neo4j → LanceDB similarity search
- **Result integration** with biological context preservation

### 3. DSPy Signatures (`dspy_signatures.py`)

The system uses sophisticated DSPy signatures for structured LLM interactions:

#### Core Signatures:
- **PlannerAgent**: Decides between traditional vs agentic execution
- **QueryClassifier**: Biological classification of query types
- **ContextRetriever**: Intelligent database query generation
- **GenomicAnswerer**: Final answer synthesis with citations
- **BiologicalToolSelector**: LLM-first tool selection with biological reasoning

#### Schema Integration:
- **NEO4J_SCHEMA**: Comprehensive database schema with relationship rules
- **Critical query patterns** for transport proteins, CAZyme annotations, BGCs
- **Mandatory relationship directions** and query validation rules

---

## Agent System & Task Management

### 1. Task Management System (`task_management.py`)

#### Task Types:
- **ATOMIC_QUERY**: Database queries with single responsibility
- **TOOL_CALL**: External tool execution (code interpreter, whole genome reader)
- **SYNTHESIS**: Result combination and analysis

#### TaskGraph Features:
- **DAG-based execution** with dependency resolution
- **Phase tracking** for execution monitoring
- **Error handling** with graceful degradation
- **Execution logging** with detailed metadata

### 2. Task Executor (`task_executor.py`)

#### Execution Strategies:
- **Sequential processing** for dependent tasks
- **Parallel execution** where dependencies allow
- **Intelligent chunking** based on token limits rather than item counts
- **Context-aware analysis** type detection (spatial vs functional vs discovery)

#### Key Features:
```python
def _determine_analysis_type_for_task(self, task: Task, task_description: str) -> str:
    """
    Determine analysis type for a task based on original question and task description.
    
    Returns:
        Analysis type: spatial_genomic, functional_annotation, or comprehensive_discovery
    """
```

### 3. Agent Tool Selection (`agent_tool_selector.py`)

#### LLM-First Tool Selection:
- **Pure LLM-based selection** with no regex fallbacks
- **Sophisticated biological reasoning** using o3 model
- **Rich biological context** and decision criteria
- **Fail-fast approach** - LLM has complete authority

#### Tool Selection Criteria:
- **whole_genome_reader**: Global prophage/phage discovery, spatial analysis
- **database_query**: Simple annotation lookups, counting, direct searches
- **code_interpreter**: Statistical analysis, visualization, quantitative assessments
- **genome_selector**: Targeting specific organisms by name

#### Caching System:
- **Three-tier caching strategy**: Main tasks → Sub-tasks → Synthesis
- **Rule-based inheritance**: Sub-tasks inherit tool selection from parent
- **API call reduction**: Up to 80% fewer LLM calls through intelligent caching

---

## Memory & Synthesis Systems

### 1. Note-Taking System (`memory/note_keeper.py`)

#### Features:
- **Persistent session notes** stored in `data/session_notes/[SESSION_ID]/`
- **Task-based organization** with cross-task connections
- **Confidence tracking** with biological context
- **Detailed reporting** for complex discoveries

#### Note Schemas (`memory/note_schemas.py`):
```python
@dataclass
class TaskNote:
    task_id: str
    description: str
    observations: List[str]
    key_findings: List[str]
    confidence_level: ConfidenceLevel
    quantitative_data: Dict[str, Any]
    cross_task_connections: List[CrossTaskConnection]
```

### 2. Progressive Synthesis (`memory/progressive_synthesizer.py`)

#### Map-Reduce Architecture:
- **Unified entry point** processing both raw_data and task_notes
- **Token-based decision making** (not keyword or count based)
- **Direct synthesis** for data within model limits (30k tokens)
- **Map-Reduce pipeline** for larger datasets

#### Processing Strategy:
```python
def synthesize_progressive(self, 
                         task_notes: List[TaskNote],
                         question: str,
                         raw_data: List[Dict[str, Any]] = None) -> str:
    """
    Main entry point for progressive synthesis using Map-Reduce architecture.
    """
```

#### Caching System:
- **Synthesis result caching** to reduce API calls
- **Cache hit tracking** with performance metrics
- **Intelligent cache key generation** based on context hash

### 3. Model Allocation (`memory/model_allocation.py`)

#### Intelligent Model Selection:
- **Task-based allocation**: Complex biological reasoning → o3, Simple tasks → GPT-4.1-mini
- **Cost optimization** with automatic model selection
- **Context-managed calls** with proper error handling
- **Fallback mechanisms** for model availability issues

#### Allocation Strategy:
- **COMPLEX tasks** (biological_interpretation, tool_selection, final_synthesis) → o3
- **SIMPLE tasks** (classification, formatting, progress_tracking) → GPT-4.1-mini
- **Premium mode**: Override to use o3 for all tasks

---

## Data Flow Architecture

### 1. Query Processing Flow

```
User Query → PlannerAgent → [Traditional | Agentic] Path
                             ↓                    ↓
                       QueryClassifier    → TaskGraph Creation
                             ↓                    ↓
                       ContextRetriever   → Task Execution
                             ↓                    ↓
                       Database Queries   → Progressive Synthesis
                             ↓                    ↓
                       GenomicAnswerer    → Final Answer
```

### 2. Data Sources Integration

#### Neo4j Knowledge Graph:
- **373,587 RDF triples**
- **48K nodes, 95K relationships**
- **Bulk loading**: <10 seconds for full dataset

#### LanceDB Vector Store:
- **10,102 protein embeddings**
- **320-dimensional ESM2 vectors**
- **Sub-millisecond similarity search**

#### External Tools:
- **Code Interpreter**: Statistical analysis and visualization
- **Whole Genome Reader**: Spatial genomic analysis
- **Literature Search**: PubMed integration (when available)

### 3. Context Compression Pipeline

#### Progressive Compression System:
```python
def _compress_context_for_synthesis(self, context: str, max_tokens: int = 100000, 
                                   is_detailed_report: bool = False) -> str:
    # Progressive chunking: 2-8 chunks based on compression ratio needed
    if compression_ratio > 0.8:    # Light: 2-3 large chunks
    elif compression_ratio > 0.5:  # Medium: 3-5 chunks  
    else:                          # Heavy: 5-8 chunks
    
    # Smart token allocation per chunk
    tokens_per_chunk = (max_tokens - 1000) // num_chunks
```

#### Priority System:
- **Ultra-high priority**: Prophage/spatial data (100 points)
- **High priority**: Functional annotations (50 points)
- **Standard priority**: General database results (20 points)

---

## API Reference

### Primary Interface

#### `GenomicRAG.ask(question: str) -> Dict[str, Any]`

Main method to answer genomic questions with agentic planning.

**Parameters:**
- `question` (str): Natural language question about genomic data

**Returns:**
```python
{
    "question": str,
    "answer": str,
    "confidence": str,
    "citations": str,
    "query_metadata": {
        "execution_mode": str,
        "total_tasks": int,
        "completed_tasks": int,
        "synthesis_stats": dict,
        "note_taking_enabled": bool
    }
}
```

**Usage Example:**
```python
from src.llm.rag_system import GenomicRAG
from src.llm.config import LLMConfig

config = LLMConfig()
rag = GenomicRAG(config, enable_memory=True)

result = await rag.ask("Find operons containing prophage segments")
print(result["answer"])
```

### Configuration Options

#### `LLMConfig` Parameters:
- `llm_model`: Model name (default: determined by model allocation)
- `llm_provider`: "openai" or "anthropic"
- `model_mode`: "cost_effective" or "premium"
- `database.neo4j_uri`: Neo4j connection string
- `database.lancedb_path`: LanceDB database path

#### `GenomicRAG` Initialization:
```python
def __init__(self, 
             config: LLMConfig, 
             chunk_context_size: int = 4096, 
             enable_memory: bool = True, 
             enhanced_logging: bool = False):
```

### Tool Integration

#### Available Tools:
- **whole_genome_reader**: Spatial genomic analysis across all genomes
- **database_query**: Direct Neo4j/LanceDB queries
- **code_interpreter**: Statistical analysis and visualization
- **genome_selector**: Organism-specific targeting

#### Tool Selection API:
```python
from src.llm.rag_system.agent_tool_selector import get_tool_selector

selector = get_tool_selector()
result = await selector.select_tool_for_task(
    task_description="Find prophage loci",
    original_user_query="Discover prophage segments",
    previous_task_context=""
)
```

---

## Performance & Optimization

### Current Performance Metrics

#### Apple Silicon M4 Max Optimization:
- **ESM2 Processing**: 10,102 proteins in ~2 minutes (~85 proteins/second)
- **LanceDB Queries**: Sub-millisecond similarity search
- **Neo4j Bulk Loading**: 48K nodes + 95K relationships in <10 seconds
- **Knowledge Graph**: 373,587 triples with multi-database integration

#### Model Allocation Efficiency:
- **3x better query success rates** with o3 for biological reasoning
- **Cost optimization** through intelligent model selection
- **API call reduction**: Up to 80% fewer calls through caching

### Memory Management

#### Context Compression:
- **Progressive compression** preserves biological patterns
- **Token-based decisions** prevent context overflow
- **Priority-based preservation** for critical data types

#### Caching Systems:
- **Tool selection caching**: 3-tier inheritance strategy
- **Synthesis result caching**: Hash-based with performance tracking
- **Model allocation caching**: Context-managed call optimization

### Scalability Features

#### Large Dataset Handling:
- **Intelligent chunking**: Token-based rather than item-based
- **Progressive synthesis**: Map-Reduce architecture for datasets >30k tokens  
- **Task-based processing**: Handles >1000 items with biological grouping

#### Error Handling:
- **TaskRepairAgent**: Graceful degradation with user-friendly messages
- **Fallback mechanisms**: Multiple retry strategies with different models
- **Session recovery**: Persistent note-taking across interrupted sessions

---

## Architectural Improvements

### Recent Enhancements (January 2025)

#### 1. Advanced Model Allocation
- **Problem**: GPT-4.1-mini generated naive queries with 0 results
- **Solution**: o3 generates biologically intelligent queries with flexible matching
- **Result**: 3x better query success rates with cost optimization

#### 2. Progressive Compression System  
- **Fixed**: Replaced hardcoded 200-line limit with intelligent progressive compression
- **Added**: Automatic chunking splits large contexts into 2-8 intelligent chunks
- **Enhanced**: Priority-based content preservation with prophage/spatial data prioritization

#### 3. Enhanced Report Generation
- **Implemented**: Multipart report routing for prophage discovery queries
- **Added**: Detailed report mode with expanded token budgets (100k → 200k)
- **Fixed**: Query processing now preserves full spatial genomic content for analysis

### Identified Areas for Improvement

#### 1. Spatial Genome Reading Data Loss (CRITICAL)
- **Issue**: Aggressive context compression destroys spatial genomic data
- **Impact**: LLMs never see gene coordinates or spatial organization
- **Recommendation**: Disable compression for spatial data or implement smarter chunking

#### 2. Data Truncation Limits
- **Issue**: WholeGenomeReader limits to 1,000 genes per contig
- **Impact**: Missing 90% of genomic data for comprehensive analysis
- **Recommendation**: Increase `max_genes_per_contig` to 10,000+

#### 3. Lost Contig Information
- **Issue**: All genes grouped under 'unknown_contig'
- **Impact**: Destroys spatial organization across contig boundaries
- **Recommendation**: Ensure proper contig field population in Neo4j schema

### Future Roadmap

#### Phase 3: Advanced Agent Capabilities
- **Knowledge gap discovery** with automated hypothesis generation
- **Multi-modal analysis** integration (sequence + structure)
- **Interactive exploration** with follow-up question generation

#### Phase 4: Large Dataset Optimization
- **Metagenome support** with community-level analysis
- **Streaming analysis** for real-time genome processing
- **Distributed computing** integration for massive datasets

#### Phase 5: Production Scaling
- **Containerization** with Docker/Kubernetes
- **Auto-scaling** based on query complexity
- **Multi-tenant support** with resource isolation

---

## Development Guidelines

### Critical Development Rules

#### 1. No Hard-Coding of Biological Patterns
**NEVER HARD-CODE BIOLOGICAL PATTERNS, KEYWORDS, OR BEHAVIOR UNLESS EXPLICITLY REQUESTED.**

Prohibited examples:
- `if gene.is_hypothetical:` or `if "hypothetical" in annotation:`
- `phage_keywords = ['integrase', 'capsid', 'tail']`
- `min_hypothetical_pct = 60`, `window_size = 15`

Use LLM-based pattern recognition instead.

#### 2. DSPy Signature Development Guidelines  
**NEVER hardcode behavior for particular query types or use dummy data directly within DSPy signatures.**

- Use generic placeholders: `"[SPECIFIC_GENOME_ID_PROVIDED_BY_SYSTEM]"`
- Create pattern-based examples that work with any data
- Let the system determine actual values during execution

#### 3. File Organization Rules
- Use `python -m src.module` for all pipeline operations
- Place test scripts in `src/tests/` with proper module structure
- No helper scripts in root directory

### Testing Requirements
- **Test-Driven Development**: Write tests first, then implement
- **100% Coverage**: All tests must pass before commit
- **Component Testing**: Unit tests with mocks, integration tests

### Code Quality Standards
- **Type Hints**: All functions must have proper type annotations
- **Documentation**: Docstrings for all classes and methods
- **Error Handling**: Graceful degradation with informative messages
- **Logging**: Structured logging with appropriate levels

---

## Conclusion

The Genomic Intelligence Platform's LLM system represents a sophisticated integration of traditional bioinformatics workflows with cutting-edge AI agents and LLM-powered biological reasoning. 

Key strengths include:
- **Modular architecture** enabling independent component development
- **Intelligent model allocation** optimizing cost and performance
- **Progressive synthesis** handling complex multi-task workflows
- **LLM-first tool selection** with sophisticated biological reasoning
- **Context-aware compression** preserving critical biological patterns

The system successfully transforms raw genomic data into intelligent, queryable insights while maintaining biological accuracy and scientific rigor. Future development should focus on addressing spatial data preservation issues and scaling to larger datasets while maintaining the sophisticated biological reasoning capabilities that distinguish this platform.

For questions about specific components or implementation details, refer to the individual module documentation in the respective source files.