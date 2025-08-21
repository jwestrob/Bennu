# LLM System Quick Reference Guide

## Key Files & Their Purposes

### Core System
- **`src/llm/rag_system/core.py`** - Main GenomicRAG class, system orchestrator
- ⚠️ `src/llm/rag_system.py` is a compatibility shim. Prefer importing from the modular paths
  (e.g., `llm.rag_system.core`, `llm.rag_system.context_processing`).

### Query Processing  
- **`src/llm/query_processor.py`** - Neo4j, LanceDB, and Hybrid query processors
- **`src/llm/rag_system/dspy_signatures.py`** - Structured LLM prompts and Neo4j schema
- **`src/llm/rag_system/context_compression.py`** - Intelligent context compression
- **`src/llm/rag_system/intelligent_routing.py`** - Query routing logic

### Agent System
- **`src/llm/rag_system/agent_tool_selector.py`** - LLM-first tool selection with biological reasoning
- **`src/llm/rag_system/task_management.py`** - TaskGraph and Task classes for agentic workflows
- **`src/llm/rag_system/task_executor.py`** - Task execution with intelligent chunking
- **`src/llm/rag_system/task_plan_parser.py`** - Converts DSPy plans to Task objects

### Memory & Synthesis
- **`src/llm/rag_system/memory/note_keeper.py`** - Session-based note-taking system
- **`src/llm/rag_system/memory/progressive_synthesizer.py`** - Map-Reduce synthesis architecture
- **`src/llm/rag_system/memory/model_allocation.py`** - Intelligent model selection (o3 vs GPT-4.1-mini)

### External Tools
- **`src/llm/rag_system/external_tools.py`** - Code interpreter, literature search, tool capabilities
- **`src/llm/rag_system/whole_genome_reader.py`** - Spatial genomic analysis tool

### Specialized Components
- **`src/llm/rag_system/genome_selection.py`** - LLM-based genome targeting
- **`src/llm/rag_system/query_validator.py`** - Query validation and fixing
- **`src/llm/task_repair_agent.py`** - Error handling with user-friendly messages

## Quick Start

### Basic Usage
```python
from src.llm.rag_system import GenomicRAG
from src.llm.config import LLMConfig

config = LLMConfig()
rag = GenomicRAG(config, enable_memory=True)

result = await rag.ask("Find prophage segments across all genomes")
print(result["answer"])
```

### Key Configuration
```bash
# Activate environment
source /Users/jacob/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh && conda activate genome-kg

# Main commands
python -m src.cli ask "Your genomic question here"
python -m src.cli build  # Build knowledge graph
```

## Architecture Overview

### Two Execution Paths
1. **Traditional**: Single-step query → Direct answer
2. **Agentic**: Multi-step planning → Task execution → Progressive synthesis

### Model Allocation Strategy
- **o3**: Complex biological reasoning, tool selection, final synthesis
- **GPT-4.1-mini**: Simple tasks, classification, formatting

### Data Flow
```
User Query → PlannerAgent → [Traditional | Agentic]
                              ↓           ↓
                         Direct Query  TaskGraph
                              ↓           ↓  
                         Answer       Progressive Synthesis
```

## Key Features

### LLM-First Tool Selection
- Pure LLM reasoning, no regex patterns
- Biological context understanding
- Tools: `whole_genome_reader`, `database_query`, `code_interpreter`, `genome_selector`

### Progressive Synthesis  
- Map-Reduce architecture for large datasets
- Token-based chunking decisions
- Intelligent caching to reduce API calls

### Context Compression
- Priority-based preservation (prophage data = highest priority)
- Progressive compression: 2-8 chunks based on size
- Detailed report mode: expanded token budgets

### Memory System
- Session-based note-taking in `data/session_notes/`
- Cross-task connections and confidence tracking
- Persistent storage across interrupted sessions

## Performance Metrics

- **373,587 RDF triples** in knowledge graph
- **Sub-millisecond** vector similarity search
- **~85 proteins/second** ESM2 processing
- **80% API call reduction** through caching
- **3x better query success** with intelligent model allocation

## Critical Development Rules

1. **No hard-coding** biological patterns or keywords
2. **LLM-first** approach for all biological reasoning
3. **Token-based** decisions over item-based counting  
4. **Generic placeholders** in DSPy signatures
5. **Proper error handling** with TaskRepairAgent

## Common Issues & Solutions

### Spatial Data Loss
- **Issue**: Context compression destroys gene coordinates  
- **Solution**: Disable compression for spatial queries or increase limits

### Query Failures
- **Issue**: Generated queries return no results
- **Solution**: Use o3 for query generation, validate comparative queries

### Memory Problems
- **Issue**: Large datasets exceed token limits
- **Solution**: Progressive synthesis with Map-Reduce architecture

### Tool Selection Problems  
- **Issue**: Wrong tools selected for tasks
- **Solution**: Enhanced biological reasoning in tool selector signatures

## File Locations

### Data Storage
- **Raw genomes**: `data/raw/`
- **Knowledge graph**: `data/stage07_kg/` (373K+ triples)
- **Protein embeddings**: `data/stage08_esm2/` (10K+ proteins)
- **Session notes**: `data/session_notes/[SESSION_ID]/`

### Configuration
- **Main config**: `src/llm/config.py`
- **Model allocation**: `src/llm/rag_system/memory/model_config.py`
- **Tool capabilities**: `src/llm/rag_system/external_tools.py`

### Testing
- **Unit tests**: `src/tests/llm/`
- **Integration tests**: `src/tests/test_agentic_rag_system.py`
- **Demo scripts**: `src/tests/demo/`