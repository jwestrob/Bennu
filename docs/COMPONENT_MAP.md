# LLM System Component Map

## Component Hierarchy & Dependencies

```
GenomicRAG (core.py)
├── Query Processors
│   ├── Neo4jQueryProcessor (query_processor.py)
│   ├── LanceDBQueryProcessor (query_processor.py)  
│   └── HybridQueryProcessor (query_processor.py)
│
├── Intelligence Layer
│   ├── IntelligentRouter (intelligent_routing.py)
│   ├── UnifiedGenomeSelector (genome_selection.py)
│   ├── ContextCompressor (context_compression.py)
│   ├── QueryValidator (query_validator.py)
│   └── IntelligentToolSelector (agent_tool_selector.py)
│
├── Agent System
│   ├── TaskGraph & Task (task_management.py)
│   ├── TaskPlanParser (task_plan_parser.py)
│   ├── TaskExecutor (task_executor.py)
│   └── CachedToolSelector (agent_tool_selector.py)
│
├── Memory System  
│   ├── NoteKeeper (memory/note_keeper.py)
│   ├── ProgressiveSynthesizer (memory/progressive_synthesizer.py)
│   ├── ModelAllocator (memory/model_allocation.py)
│   └── PolicyEngine (policy_engine.py)
│
├── External Tools
│   ├── WholeGenomeReader (whole_genome_reader.py)
│   ├── CodeInterpreter (external_tools.py)
│   ├── LiteratureSearch (external_tools.py)
│   └── TaskRepairAgent (task_repair_agent.py)
│
└── DSPy Interface
    ├── Signatures (dspy_signatures.py)
    ├── Schema (NEO4J_SCHEMA in dspy_signatures.py)
    └── Model Configuration (memory/model_config.py)
```

## Data Flow Connections

### Query Processing Flow
```
ask() method (core.py)
    ↓
PlannerAgent (dspy_signatures.py) 
    ↓
[Traditional Path]              [Agentic Path]
    ↓                               ↓
QueryClassifier                 TaskPlanParser
    ↓                               ↓
ContextRetriever               TaskGraph Creation
    ↓                               ↓
Database Queries               TaskExecutor
    ↓                               ↓
GenomicAnswerer           ProgressiveSynthesizer
    ↓                               ↓
Final Answer                   Final Answer
```

### Tool Selection Flow  
```
TaskExecutor.execute_task()
    ↓
IntelligentToolSelector (agent_tool_selector.py)
    ↓
BiologicalToolSelector (dspy_signatures.py)
    ↓
[Tool Routes]
├── whole_genome_reader → WholeGenomeReader
├── database_query → Neo4j/LanceDB processors
├── code_interpreter → CodeInterpreter  
└── genome_selector → UnifiedGenomeSelector
```

### Memory & Synthesis Flow
```
Task Results → NoteKeeper → Session Storage
                    ↓
            Task Notes + Raw Data
                    ↓
          ProgressiveSynthesizer
                    ↓
         [Direct | Map-Reduce] Synthesis
                    ↓
            ModelAllocator
                    ↓
         [o3 | GPT-4.1-mini] Selection
                    ↓
            Final Synthesis
```

## Component Responsibilities

### Core Orchestration
| Component | File | Primary Responsibility |
|-----------|------|------------------------|
| GenomicRAG | `core.py` | Main system orchestrator, execution path selection |
| IntelligentRouter | `intelligent_routing.py` | Query routing to appropriate processors |
| PolicyEngine | `policy_engine.py` | System-wide policy enforcement and limits |

### Query Processing
| Component | File | Primary Responsibility |
|-----------|------|------------------------|
| Neo4jQueryProcessor | `query_processor.py` | Cypher query generation and execution |
| LanceDBQueryProcessor | `query_processor.py` | Vector similarity search with ESM2 embeddings |
| HybridQueryProcessor | `query_processor.py` | Multi-stage Neo4j → LanceDB processing |
| QueryValidator | `query_validator.py` | Query validation and automatic fixing |

### Intelligence & Analysis  
| Component | File | Primary Responsibility |
|-----------|------|------------------------|
| UnifiedGenomeSelector | `genome_selection.py` | LLM-based genome targeting and intent analysis |
| ContextCompressor | `context_compression.py` | Intelligent context compression with biological preservation |
| IntelligentToolSelector | `agent_tool_selector.py` | LLM-first tool selection with biological reasoning |
| BiologicalToolSelector | `dspy_signatures.py` | DSPy signature for biological tool reasoning |

### Agent System
| Component | File | Primary Responsibility |
|-----------|------|------------------------|
| TaskGraph | `task_management.py` | DAG-based task dependency management |
| Task | `task_management.py` | Individual task representation with metadata |
| TaskPlanParser | `task_plan_parser.py` | Converts DSPy plans to executable Task objects |
| TaskExecutor | `task_executor.py` | Task execution with intelligent chunking |
| CachedToolSelector | `agent_tool_selector.py` | Three-tier caching for tool selection |

### Memory & Synthesis
| Component | File | Primary Responsibility |
|-----------|------|------------------------|
| NoteKeeper | `memory/note_keeper.py` | Session-based note-taking with persistence |
| ProgressiveSynthesizer | `memory/progressive_synthesizer.py` | Map-Reduce synthesis for large datasets |
| ModelAllocator | `memory/model_allocation.py` | Intelligent model selection (o3 vs GPT-4.1-mini) |
| TaskNote | `memory/note_schemas.py` | Structured note representation |

### External Tools
| Component | File | Primary Responsibility |
|-----------|------|------------------------|
| WholeGenomeReader | `whole_genome_reader.py` | Spatial genomic analysis across entire genomes |
| CodeInterpreter | `external_tools.py` | Statistical analysis and visualization |
| LiteratureSearch | `external_tools.py` | PubMed integration for recent research |
| TaskRepairAgent | `task_repair_agent.py` | Error handling with user-friendly messages |

### DSPy Interface Layer
| Component | File | Primary Responsibility |
|-----------|------|------------------------|
| PlannerAgent | `dspy_signatures.py` | Decides traditional vs agentic execution |
| QueryClassifier | `dspy_signatures.py` | Biological classification of query types |
| ContextRetriever | `dspy_signatures.py` | Intelligent database query generation |
| GenomicAnswerer | `dspy_signatures.py` | Final answer synthesis with citations |
| GenomicSummarizer | `dspy_signatures.py` | Data summarization for synthesis |
| NEO4J_SCHEMA | `dspy_signatures.py` | Comprehensive database schema and rules |

## Key Design Patterns

### 1. Factory Pattern
- **ModelAllocator**: Creates appropriate LLM instances based on task complexity
- **ToolSelector**: Creates tool instances based on biological analysis
- **ProcessorFactory**: Creates query processors based on data source

### 2. Strategy Pattern  
- **QueryProcessingStrategy**: Traditional vs Agentic execution paths
- **SynthesisStrategy**: Direct vs Map-Reduce based on data size
- **CompressionStrategy**: Priority-based vs token-based compression

### 3. Observer Pattern
- **TaskGraph**: Notifies observers of task state changes
- **NoteKeeper**: Observes task execution for note-taking
- **ProgressTracker**: Observes workflow progress for logging

### 4. Template Method Pattern
- **TaskExecutor**: Template for task execution with customizable steps
- **ProgressiveSynthesizer**: Template for synthesis with pluggable strategies
- **QueryProcessor**: Template for database interaction patterns

### 5. Decorator Pattern
- **CachedToolSelector**: Adds caching behavior to base tool selection
- **ContextManagedCall**: Adds context management to model calls
- **ValidationDecorator**: Adds validation to query generation

## Integration Points

### Database Integration
```python
# Neo4j Connection
self.neo4j_processor = Neo4jQueryProcessor(config)

# LanceDB Vector Store  
self.lancedb_processor = LanceDBQueryProcessor(config)

# Hybrid Processing
self.hybrid_processor = HybridQueryProcessor(config)
```

### LLM Integration
```python
# Model Allocation
self.model_allocator = get_model_allocator()

# DSPy Configuration
self._configure_dspy()  # Sets up o3/GPT-4.1-mini based on task

# Context-Managed Calls
result = self.model_allocator.create_context_managed_call(
    task_name="biological_interpretation",
    signature_class=GenomicAnswerer,
    module_call_func=answer_call
)
```

### External Tool Integration
```python
# Tool Availability Checking
from .external_tools import AVAILABLE_TOOLS, TOOL_CAPABILITIES

# Tool Execution
if selected_tool == "whole_genome_reader":
    reader = WholeGenomeReader(self.neo4j_processor)
    results = await reader.read_full_genomic_context(question)
```

## Configuration Dependencies

### Environment Variables
- `OPENAI_API_KEY`: Required for OpenAI models (o3, GPT-4.1-mini)
- `ANTHROPIC_API_KEY`: Required for Anthropic models (fallback)
- `NEO4J_URI`: Neo4j database connection
- `LANCEDB_PATH`: LanceDB vector store location

### File Dependencies
- **Knowledge Graph**: `data/stage07_kg/knowledge_graph.ttl`
- **Protein Embeddings**: `data/stage08_esm2/protein_embeddings.h5`
- **Session Notes**: `data/session_notes/[SESSION_ID]/`
- **Tool Databases**: `data/dbcan_db/`, HMM files

### Python Dependencies
- **DSPy**: `dspy-ai` for structured LLM interactions
- **Neo4j**: `neo4j` driver for graph database
- **LanceDB**: `lancedb` for vector similarity search
- **Rich**: `rich` for enhanced console output
- **TikToken**: `tiktoken` for token counting

## Performance Characteristics

### Memory Usage
- **Base System**: ~500MB RAM
- **With Knowledge Graph**: ~2GB RAM  
- **With Embeddings**: ~3GB RAM
- **Peak Usage**: ~5GB RAM during large queries

### Response Times
- **Simple Queries**: 2-5 seconds
- **Complex Agentic Workflows**: 30-120 seconds
- **Large Dataset Synthesis**: 2-10 minutes
- **Vector Similarity**: <1ms per query

### Scalability Limits
- **Context Window**: 30k tokens for o3, 128k for GPT-4.1-mini
- **Concurrent Users**: Limited by API rate limits
- **Dataset Size**: Tested up to 373k triples, 10k proteins
- **Session Storage**: Limited by disk space

This component map provides a comprehensive overview of the LLM system architecture, showing how all pieces fit together to create an intelligent genomic analysis platform.