# Context Compression System

The Context Compression System provides intelligent compression of large genomic datasets to fit within LLM token limits while preserving essential biological information.

## Overview

When processing large genomic datasets, the system can generate contexts that exceed typical LLM token limits (20K-30K tokens). The compression system solves this by:

1. **Automatic Detection**: Monitors context size and triggers compression when needed
2. **Tiered Strategy**: Uses different compression approaches based on dataset size
3. **Genomic Intelligence**: Preserves functional annotations, coordinates, and biological relationships
4. **Multi-Stage Processing**: Handles very large datasets through chunking and iterative compression

## Architecture

### Compression Levels

- **LIGHT** (10-20% reduction): Preserves most details, minor formatting cleanup
- **MEDIUM** (30-50% reduction): Balances details with size, summarizes repeated patterns
- **AGGRESSIVE** (60-80% reduction): Essential information only, heavy summarization

### Compression Strategies

1. **Direct** (<10K tokens): No compression needed
2. **Single-Pass** (10K-30K tokens): One compression cycle with appropriate level
3. **Multi-Stage** (>30K tokens): Chunking + compression + final summarization

## Key Features

### Genomic-Specific Compression

- **Protein Grouping**: Groups proteins by function to avoid repetition
- **Coordinate Summarization**: Compresses genomic coordinates to ranges
- **Domain Preservation**: Maintains PFAM and KEGG annotations
- **Context Awareness**: Preserves genomic neighborhood information

### Token Management

- **Accurate Counting**: Uses tiktoken for precise token estimation
- **Smart Truncation**: Preserves complete sentences and biological units
- **Target Enforcement**: Guarantees output fits within specified limits

### Error Handling

- **Graceful Degradation**: Falls back to truncation if compression insufficient
- **Recursion Prevention**: Avoids infinite compression loops
- **Information Preservation**: Prioritizes essential biological data

## Usage

### Automatic Integration

The compression system is automatically integrated into the RAG system:

```python
# Context compression is applied automatically when needed
result = await rag.ask("Find all transport proteins and analyze their distribution")

# Check if compression was applied
if "compression_stats" in result["query_metadata"]:
    stats = result["query_metadata"]["compression_stats"]
    print(f"Compressed {stats['original_tokens']} -> {stats['compressed_tokens']} tokens")
```

### Manual Usage

```python
from llm.context_compression import ContextCompressor

# Initialize compressor
compressor = ContextCompressor("gpt-3.5-turbo")

# Prepare context data
context_data = {
    'structured_data': protein_list,
    'semantic_data': similarity_results,
    'tool_results': analysis_results
}

# Compress context
result = await compressor.compress_context(context_data, target_tokens=20000)

print(f"Compression ratio: {result.compression_ratio:.2f}")
print(f"Preserved elements: {result.preserved_elements}")
```

### Custom Compression

```python
from llm.context_compression import GenomicDataCompressor, CompressionLevel

# Compress protein annotations with specific level
compressed_proteins, stats = GenomicDataCompressor.compress_protein_annotations(
    protein_list, 
    level=CompressionLevel.MEDIUM
)
```

## Performance

### Compression Results

- **Small datasets** (100 proteins): No compression needed
- **Medium datasets** (1K proteins): 50-70% compression 
- **Large datasets** (10K+ proteins): 85-95% compression
- **Processing time**: <1 second for most datasets

### Token Efficiency

- **Before**: 72K+ token contexts causing API errors
- **After**: 8-20K token contexts within all model limits
- **Information retention**: 90%+ of essential biological information preserved

## Configuration

### Token Limits

```python
# Default limits (adjustable)
SMALL_LIMIT = 10000   # Direct synthesis
MEDIUM_LIMIT = 30000  # Single compression 
LARGE_LIMIT = 100000  # Multi-stage compression
```

### Compression Targets

```python
# Default targets by query type
TRADITIONAL_QUERY_TARGET = 20000  # Traditional queries
AGENTIC_SYNTHESIS_TARGET = 25000  # Agentic workflows
```

## Integration Points

### Traditional Queries

```python
# Applied automatically in _execute_traditional_query()
if context_size > 25000:
    compression_result = await compressor.compress_context(context_data)
    formatted_context = compression_result.compressed_content
```

### Agentic Workflows

```python
# Applied automatically in _synthesize_agentic_results()
context_data = self._organize_results_for_compression(completed_results)
compression_result = await compressor.compress_context(context_data)
```

## Monitoring

### Compression Stats

All responses include compression statistics when applied:

```json
{
  "query_metadata": {
    "compression_stats": {
      "original_tokens": 72543,
      "compressed_tokens": 18234,
      "compression_ratio": 0.25,
      "compression_level": "aggressive", 
      "chunks_processed": 12
    }
  }
}
```

### Logging

The system provides detailed logging for troubleshooting:

```
INFO: Context compression: 72543 -> 18234 tokens (ratio: 0.25)
INFO: Multi-stage compression applied with 12 chunks
INFO: Preserved elements: ['structured_data', 'semantic_data', 'tool_results']
```

## Best Practices

1. **Monitor compression ratios** - ratios <0.1 may indicate over-compression
2. **Check preserved elements** - ensure essential data types are retained
3. **Validate outputs** - verify biological accuracy is maintained
4. **Adjust targets** - increase target tokens for complex analyses
5. **Use logging** - monitor compression performance and effectiveness

## Troubleshooting

### Common Issues

**Over-compression**: If compression ratio is too aggressive (<0.1), increase target tokens or use lighter compression levels.

**Information loss**: Check preserved_elements to ensure required data types are included.

**Performance issues**: For very large datasets (>1M tokens), consider pre-filtering data before compression.

**Recursion warnings**: System automatically handles infinite compression loops with fallback truncation.

## Technical Details

### Dependencies

- `tiktoken>=0.5.0` - Accurate token counting
- `dspy-ai>=2.6.27` - LLM-based summarization (optional)

### Files

- `src/llm/context_compression.py` - Main compression module
- `src/llm/rag_system/core.py` - Integration points
- `src/llm/rag_system/dspy_signatures.py` - Summarization signatures

### Testing

- `src/tests/test_context_compression.py` - Unit tests
- `src/tests/test_rag_compression_integration.py` - Integration tests

Run tests with:
```bash
python src/tests/test_context_compression.py
python src/tests/test_rag_compression_integration.py
```