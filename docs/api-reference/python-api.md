# Python API Reference

Comprehensive reference for programmatic access to the genomic AI platform.

## Quick Start

```python
from src.llm.config import LLMConfig
from src.llm.rag_system import GenomicRAG
import asyncio

# Initialize system
config = LLMConfig()
rag = GenomicRAG(config)

# Ask questions
async def main():
    result = await rag.ask("How many proteins are in the database?")
    print(result['answer'])
    rag.close()

asyncio.run(main())
```

## Core Classes

### GenomicRAG

The main interface for genomic question answering with biological expertise.

<details>
<summary><strong>GenomicRAG Class Reference (Click to expand)</strong></summary>

```python
from src.llm.rag_system import GenomicRAG
from src.llm.config import LLMConfig

class GenomicRAG:
    """Main genomic RAG system for intelligent question answering."""
    
    def __init__(self, config: LLMConfig, chunk_context_size: int = 4096):
        """
        Initialize the genomic RAG system.
        
        Args:
            config: LLM configuration object
            chunk_context_size: Maximum context size for processing
        """
```

**Main Methods**:

```python
async def ask(self, question: str) -> Dict[str, Any]:
    """
    Ask questions about genomic data with biological intelligence.
    
    Args:
        question: Natural language question about genomic data
        
    Returns:
        Dict containing:
        - question: Original question
        - answer: Comprehensive biological analysis
        - confidence: high/medium/low
        - citations: Data sources used
        - query_metadata: Execution details
        
    Example:
        result = await rag.ask("What transport proteins are present?")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
    """

def health_check(self) -> Dict[str, bool]:
    """
    Check health of all system components.
    
    Returns:
        Dict with component status:
        - neo4j: Graph database connectivity
        - lancedb: Vector database accessibility
        - hybrid: Combined query processor
        - dspy: LLM framework availability
        
    Example:
        health = rag.health_check()
        if all(health.values()):
            print("All systems operational")
    """

def close(self):
    """
    Close all database connections and clean up resources.
    Always call this when done with the RAG system.
    """

async def ask_agentic(self, question: str, **kwargs) -> str:
    """
    Legacy method returning string instead of dict.
    Use ask() for new code.
    """
```

**Usage Examples**:

```python
# Basic usage
config = LLMConfig()
rag = GenomicRAG(config)

# Single question
result = await rag.ask("How many CAZymes are annotated?")

# Multiple questions
questions = [
    "What transport proteins are present?",
    "Show me metabolic pathway coverage",
    "Find proteins similar to heme transporters"
]

for question in questions:
    result = await rag.ask(question)
    print(f"Q: {question}")
    print(f"A: {result['answer'][:100]}...")
    print(f"Confidence: {result['confidence']}\n")

# Always clean up
rag.close()
```

</details>

### LLMConfig

Configuration management for the genomic AI platform.

<details>
<summary><strong>LLMConfig Class Reference (Click to expand)</strong></summary>

```python
from src.llm.config import LLMConfig

class LLMConfig(BaseModel):
    """Configuration for LLM integration and database connections."""
    
    # Database connections
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    
    # LLM provider settings
    llm_provider: str = Field(default="openai")
    llm_model: str = Field(default="o3")
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    
    # Embedding settings
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dim: int = Field(default=320)
    
    # RAG settings
    max_context_length: int = Field(default=8000)
    similarity_threshold: float = Field(default=0.7)
    max_results_per_query: int = Field(default=10)
```

**Main Methods**:

```python
@classmethod
def from_env(cls) -> 'LLMConfig':
    """
    Create configuration from environment variables.
    
    Environment Variables:
        NEO4J_URI: Neo4j connection URI
        NEO4J_USER: Neo4j username  
        NEO4J_PASSWORD: Neo4j password
        OPENAI_API_KEY: OpenAI API key
        ANTHROPIC_API_KEY: Anthropic API key
        LANCEDB_PATH: LanceDB database path
        
    Returns:
        LLMConfig instance with environment settings
        
    Example:
        import os
        os.environ['OPENAI_API_KEY'] = 'your-key'
        config = LLMConfig.from_env()
    """

@classmethod  
def from_file(cls, config_path: Path) -> 'LLMConfig':
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to JSON configuration file
        
    Example:
        config = LLMConfig.from_file(Path("config.json"))
    """

def get_api_key(self) -> Optional[str]:
    """
    Get appropriate API key based on provider.
    
    Returns:
        API key for configured provider, or None if not available
    """

def validate_configuration(self) -> Dict[str, bool]:
    """
    Validate all configuration components.
    
    Returns:
        Dict with validation status:
        - neo4j_configured: Database connection parameters valid
        - lancedb_configured: Vector database path exists  
        - llm_configured: API key available
        - all_paths_exist: Required directories present
        
    Example:
        config = LLMConfig()
        validation = config.validate_configuration()
        if not validation['llm_configured']:
            print("API key required")
    """
```

**Usage Examples**:

```python
# Default configuration
config = LLMConfig()

# Environment-based configuration
config = LLMConfig.from_env()

# File-based configuration
config = LLMConfig.from_file(Path("genomic_config.json"))

# Custom configuration
config = LLMConfig(
    llm_provider="openai",
    llm_model="gpt-4o-mini",  # Use different model
    max_context_length=16000,  # Larger context
    similarity_threshold=0.8   # Higher similarity threshold
)

# Validation
validation = config.validate_configuration()
print("System ready:", all(validation.values()))
```

</details>

### Task Management

Classes for building agentic workflows with multi-step execution.

<details>
<summary><strong>Task Management API (Click to expand)</strong></summary>

```python
from src.llm.rag_system import Task, TaskGraph, TaskStatus, TaskType

# Task creation
task = Task(
    task_id="analyze_cazymes",
    task_type=TaskType.ATOMIC_QUERY,
    description="Analyze CAZyme distribution",
    query="MATCH (p:Protein)-[:HAS_CAZYME]-(c:CAZyme) RETURN count(*)"
)

# Task graph
graph = TaskGraph()
task_id = graph.add_task(task)

# Execution
ready_tasks = graph.get_ready_tasks()
for task in ready_tasks:
    # Execute task
    result = await execute_task(task)
    graph.mark_task_status(task.task_id, TaskStatus.COMPLETED, result)

# Status checking
summary = graph.get_summary()
results = graph.get_completed_results()
```

**Task Types**:
- `TaskType.ATOMIC_QUERY`: Database query execution
- `TaskType.TOOL_CALL`: External tool invocation

**Task Status**:
- `TaskStatus.PENDING`: Awaiting execution
- `TaskStatus.RUNNING`: Currently executing  
- `TaskStatus.COMPLETED`: Successfully finished
- `TaskStatus.FAILED`: Execution failed
- `TaskStatus.SKIPPED`: Skipped due to dependency failure

</details>

## Pipeline Integration

### Individual Stage Execution

<details>
<summary><strong>Stage-by-Stage API (Click to expand)</strong></summary>

```python
from pathlib import Path

# Stage 0: Input preparation
from src.ingest.prepare_inputs import prepare_inputs
prepare_inputs(
    input_dir=Path("data/raw"),
    output_dir=Path("data/stage00_prepared")
)

# Stage 3: Gene prediction  
from src.ingest.prodigal import run_prodigal
run_prodigal(
    input_dir=Path("data/stage00_prepared"),
    output_dir=Path("data/stage03_prodigal"),
    threads=8
)

# Stage 4: Functional annotation
from src.ingest.astra_scan import run_astra_scan  
run_astra_scan(
    protein_dir=Path("data/stage03_prodigal/all_protein_symlinks"),
    output_dir=Path("data/stage04_astra"),
    databases=["PFAM", "KOFAM"],
    threads=8,
    evalue_threshold=1e-5
)

# Stage 7: Knowledge graph construction
from src.build_kg.rdf_builder import build_knowledge_graph_with_extended_annotations
build_knowledge_graph_with_extended_annotations(
    stage03_dir=Path("data/stage03_prodigal"),
    stage04_dir=Path("data/stage04_astra"), 
    stage05a_dir=Path("data/stage05_gecco"),
    stage05b_dir=Path("data/stage06_dbcan"),
    output_dir=Path("data/stage07_kg")
)
```

**Return Values**: Each function returns processing statistics and manifest data.

</details>

### Database Integration

<details>
<summary><strong>Database API Reference (Click to expand)</strong></summary>

**Neo4j Integration**:
```python
from src.llm.query_processor import Neo4jQueryProcessor
from src.llm.config import LLMConfig

config = LLMConfig()
processor = Neo4jQueryProcessor(config)

# Execute Cypher queries
result = await processor.process_query(
    "MATCH (p:Protein) RETURN count(p) as protein_count"
)

print(f"Found {result.results[0]['protein_count']} proteins")
processor.close()
```

**LanceDB Integration**:
```python
from src.llm.query_processor import LanceDBQueryProcessor
import numpy as np

processor = LanceDBQueryProcessor(config)

# Similarity search
query_embedding = np.random.rand(320)  # ESM2 embedding
results = await processor.similarity_search(
    query_embedding=query_embedding,
    top_k=10,
    similarity_threshold=0.7
)

for result in results.results:
    print(f"Protein: {result['protein_id']}, Similarity: {result['_distance']}")
```

**Hybrid Queries**:
```python
from src.llm.query_processor import HybridQueryProcessor

processor = HybridQueryProcessor(config)

# Combined structured + semantic search
result = await processor.process_query("Find proteins similar to transport proteins")
```

</details>

## Utility Functions

### Data Processing

<details>
<summary><strong>Utility Function Reference (Click to expand)</strong></summary>

```python
from src.llm.rag_system.utils import safe_log_data, setup_debug_logging, ResultStreamer

# Safe logging for large data structures
large_dict = {"proteins": list(range(10000))}
summary = safe_log_data(large_dict, max_length=100)
print(summary)  # "<large_dict: 1 keys, 89890 chars>"

# Debug logging setup
setup_debug_logging()  # Redirects verbose output to logs/rag_debug.log

# Result streaming for large datasets
streamer = ResultStreamer(chunk_context_size=4096)
session_dir = streamer.create_session()

# Stream large protein results
protein_iterator = get_large_protein_dataset()  # Your data source
for chunk_summary in streamer.stream_iterator(protein_iterator):
    print(f"Processed chunk: {chunk_summary}")

# Session summary
summary = streamer.get_session_summary()
print(f"Total items processed: {summary['total_items']}")
```

</details>

### Configuration Helpers

<details>
<summary><strong>Configuration Utilities (Click to expand)</strong></summary>

```python
from src.llm.config import LLMConfig, DEFAULT_CONTAINER_CONFIG

# Use default container configuration
config = DEFAULT_CONTAINER_CONFIG

# Validate and display configuration
validation = config.validate_configuration()
for component, status in validation.items():
    print(f"{component}: {'✓' if status else '✗'}")

# Create custom configuration
custom_config = LLMConfig(
    database=DatabaseConfig(
        neo4j_uri="bolt://custom-host:7687",
        neo4j_password="custom-password",
        lancedb_path="custom/path/lancedb"
    ),
    llm_provider="anthropic",
    llm_model="claude-3-haiku-20240307",
    max_context_length=12000
)

# Save configuration
custom_config.to_file(Path("custom_config.json"))
```

</details>

## Error Handling

### Exception Classes

<details>
<summary><strong>Error Handling Reference (Click to expand)</strong></summary>

```python
from src.llm.query_processor import QueryError, DatabaseError
from src.llm.task_repair_agent import RepairResult

try:
    result = await rag.ask("Invalid query with bad syntax")
except QueryError as e:
    print(f"Query failed: {e}")
    # Check for repair suggestions
    if hasattr(e, 'repair_result'):
        print(f"Suggestion: {e.repair_result.user_message}")

except DatabaseError as e:
    print(f"Database connection failed: {e}")
    # Implement retry logic or fallback

except Exception as e:
    print(f"Unexpected error: {e}")
    # General error handling
```

**Common Error Patterns**:
- `QueryError`: Cypher syntax errors, invalid relationships
- `DatabaseError`: Connection failures, timeout issues  
- `ConfigurationError`: Missing API keys, invalid paths
- `ProcessingError`: Pipeline stage failures

**Error Recovery**:
```python
def handle_query_with_repair(rag: GenomicRAG, question: str, max_retries: int = 3):
    """Execute query with automatic repair attempts."""
    for attempt in range(max_retries):
        try:
            result = await rag.ask(question)
            return result
        except QueryError as e:
            if hasattr(e, 'repair_result') and e.repair_result.success:
                print(f"Repair suggested: {e.repair_result.user_message}")
                # Optionally modify question based on suggestion
                continue
            else:
                raise
    raise Exception(f"Query failed after {max_retries} attempts")
```

</details>

## Performance Optimization

### Async Usage Patterns

<details>
<summary><strong>Async Programming Patterns (Click to expand)</strong></summary>

**Concurrent Queries**:
```python
import asyncio
from src.llm.rag_system import GenomicRAG
from src.llm.config import LLMConfig

async def concurrent_analysis():
    """Execute multiple queries concurrently."""
    config = LLMConfig()
    rag = GenomicRAG(config)
    
    questions = [
        "How many proteins are there?",
        "What transport proteins are present?", 
        "Show me metabolic pathway coverage",
        "Find CAZyme distribution across genomes"
    ]
    
    # Execute all queries concurrently
    tasks = [rag.ask(q) for q in questions]
    results = await asyncio.gather(*tasks)
    
    for question, result in zip(questions, results):
        print(f"Q: {question}")
        print(f"A: {result['answer'][:100]}...")
        print(f"Confidence: {result['confidence']}\n")
    
    rag.close()

# Run concurrent analysis
asyncio.run(concurrent_analysis())
```

**Batch Processing**:
```python
async def batch_protein_analysis(protein_ids: List[str]):
    """Analyze multiple proteins efficiently."""
    config = LLMConfig()
    rag = GenomicRAG(config)
    
    # Process in batches to avoid overwhelming the system
    batch_size = 10
    results = []
    
    for i in range(0, len(protein_ids), batch_size):
        batch = protein_ids[i:i+batch_size]
        batch_tasks = [
            rag.ask(f"What is the function of protein {pid}?") 
            for pid in batch
        ]
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)
        
        # Brief pause between batches
        await asyncio.sleep(0.1)
    
    rag.close()
    return results
```

</details>

### Caching and Optimization

<details>
<summary><strong>Performance Optimization API (Click to expand)</strong></summary>

```python
from functools import lru_cache
import time

class OptimizedGenomicRAG(GenomicRAG):
    """Extended RAG with caching and optimization features."""
    
    def __init__(self, config: LLMConfig, cache_size: int = 128):
        super().__init__(config)
        self.query_cache = {}
        self.cache_size = cache_size
    
    @lru_cache(maxsize=128)
    def _cached_health_check(self) -> Dict[str, bool]:
        """Cached health check to avoid repeated database calls."""
        return super().health_check()
    
    async def ask_with_cache(self, question: str) -> Dict[str, Any]:
        """Ask with intelligent caching."""
        # Normalize question for cache key
        cache_key = question.lower().strip()
        
        # Check cache
        if cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            # Return cached result if less than 1 hour old
            if time.time() - cached_result['timestamp'] < 3600:
                cached_result['cached'] = True
                return cached_result
        
        # Execute query
        result = await self.ask(question)
        result['timestamp'] = time.time()
        
        # Store in cache (with size limit)
        if len(self.query_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = min(self.query_cache.keys(), 
                           key=lambda k: self.query_cache[k]['timestamp'])
            del self.query_cache[oldest_key]
        
        self.query_cache[cache_key] = result
        return result

# Usage
config = LLMConfig()
optimized_rag = OptimizedGenomicRAG(config, cache_size=256)

# First call: executes query
result1 = await optimized_rag.ask_with_cache("How many proteins are there?")

# Second call: returns cached result
result2 = await optimized_rag.ask_with_cache("How many proteins are there?")
print("Cached:", result2.get('cached', False))
```

</details>

## Integration Examples

### Jupyter Notebook Integration

<details>
<summary><strong>Notebook Usage Examples (Click to expand)</strong></summary>

```python
# Cell 1: Setup
import asyncio
import pandas as pd
from src.llm.config import LLMConfig
from src.llm.rag_system import GenomicRAG

# Handle event loop in Jupyter
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    import nest_asyncio
    nest_asyncio.apply()

# Cell 2: Initialize system
config = LLMConfig()
rag = GenomicRAG(config)

# Check system health
health = rag.health_check()
print("System status:", health)

# Cell 3: Basic analysis
async def analyze_dataset():
    """Comprehensive dataset analysis."""
    analyses = {}
    
    # Basic counts
    analyses['proteins'] = await rag.ask("How many proteins are in the database?")
    analyses['genomes'] = await rag.ask("How many genomes are present?")
    analyses['cazymes'] = await rag.ask("How many CAZymes are annotated?")
    
    return analyses

results = await analyze_dataset()
for analysis, result in results.items():
    print(f"{analysis.title()}: {result['answer']}")

# Cell 4: Comparative analysis with visualization
async def cazyme_comparison():
    """Get CAZyme distribution for visualization."""
    result = await rag.ask(
        "Show me the distribution of CAZyme types among each genome; "
        "provide specific numbers for visualization"
    )
    return result

cazyme_data = await cazyme_comparison()
print("CAZyme Analysis:")
print(cazyme_data['answer'])

# Cell 5: Data extraction for plotting
# Extract numerical data from results for matplotlib/seaborn visualization
# (Implementation depends on specific query results)

# Cell 6: Cleanup
rag.close()
```

</details>

### Web Application Integration

<details>
<summary><strong>FastAPI Integration Example (Click to expand)</strong></summary>

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.llm.config import LLMConfig
from src.llm.rag_system import GenomicRAG
import asyncio

app = FastAPI(title="Genomic AI API")

# Global RAG instance
config = LLMConfig()
rag = GenomicRAG(config)

class QueryRequest(BaseModel):
    question: str
    include_metadata: bool = False

class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence: str
    execution_time: float
    metadata: dict = None

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Ask genomic questions via API."""
    try:
        import time
        start_time = time.time()
        
        result = await rag.ask(request.question)
        execution_time = time.time() - start_time
        
        response = QueryResponse(
            question=result['question'],
            answer=result['answer'],
            confidence=result['confidence'], 
            execution_time=execution_time
        )
        
        if request.include_metadata:
            response.metadata = result.get('query_metadata', {})
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """System health check endpoint."""
    try:
        health = rag.health_check()
        return {
            "status": "healthy" if all(health.values()) else "degraded",
            "components": health
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    rag.close()

# Run with: uvicorn genomic_api:app --host 0.0.0.0 --port 8000
```

</details>

## Next Steps

- **[CLI Commands](cli-commands.md)**: Command-line interface reference
- **[Basic Queries Tutorial](../tutorials/basic-queries.md)**: Learn query patterns
- **[Architecture Overview](../architecture/overview.md)**: System design details