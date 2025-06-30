# Command Output Management System

## Problem Solved

Previously, running large queries through the annotation explorer would output thousands of lines to STDOUT, polluting our conversation context and hitting API prompt length limits. This made it impossible to examine results or continue debugging effectively.

## Solution Implemented

### 1. **Automatic File Redirection System**

**Location**: `src/utils/command_runner.py`

**Key Features**:
- Automatically detects large outputs (>50 lines by default)
- Redirects full output to timestamped files in `data/command_outputs/`
- Shows only a summary (first 10 lines) in STDOUT
- Provides file paths for further examination

### 2. **Enhanced Annotation Explorer Wrapper**

**Location**: `run_annotation_explorer.py`

**Usage**:
```bash
# Basic usage
python run_annotation_explorer.py "Show me all proteins involved in central metabolism"

# Custom line limit
python run_annotation_explorer.py "What ribosomal proteins are present?" --max-lines 20

# Force file output regardless of size
python run_annotation_explorer.py "Simple query" --force-file
```

**Example Output**:
```
✅ Command completed (output redirected to file)
📁 Full output saved to: data/command_outputs/20250629_194123_annotation_explorer_Show_me_all_proteins_involved_.txt
📊 Total output lines: 5317
🔍 First few lines:
🧬 Genomic Question Answering
Question: Show me all proteins involved in central metabolism
...

📋 To examine the full results:
   cat 'data/command_outputs/20250629_194123_annotation_explorer_Show_me_all_proteins_involved_.txt'
   grep 'pattern' 'data/command_outputs/20250629_194123_annotation_explorer_Show_me_all_proteins_involved_.txt'
   tail -50 'data/command_outputs/20250629_194123_annotation_explorer_Show_me_all_proteins_involved_.txt'
```

### 3. **Enhanced Debug Scripts**

**Location**: `src/tests/debug/debug_rag_context_enhanced.py`

**Features**:
- Captures RAG context data without STDOUT pollution
- Automatically saves large contexts to files
- Provides structured analysis with file references
- Saves task results separately for large agentic workflows

## Directory Structure

```
data/
├── command_outputs/          # Large command outputs
│   ├── 20250629_194123_annotation_explorer_*.txt
│   └── 20250629_194525_annotation_explorer_*.txt
└── debug_outputs/           # Debug session outputs
    ├── 20250629_HHMMSS_rag_debug_formatted_context.txt
    ├── 20250629_HHMMSS_rag_debug_task_explore_annotations.json
    └── 20250629_HHMMSS_complete_debug.json
```

## Usage Examples

### 1. **Running Large Queries**

Instead of:
```bash
# ❌ This pollutes STDOUT with 5000+ lines
python -m src.cli ask "Show me all proteins involved in central metabolism"
```

Use:
```bash
# ✅ This redirects large output to files
python run_annotation_explorer.py "Show me all proteins involved in central metabolism"
```

### 2. **Examining Results**

```bash
# View the end of results (most important part)
tail -50 data/command_outputs/20250629_194123_annotation_explorer_*.txt

# Search for specific patterns
grep -i "ribosomal" data/command_outputs/20250629_194525_annotation_explorer_*.txt

# View specific sections
head -100 data/command_outputs/20250629_194123_annotation_explorer_*.txt
```

### 3. **Debugging RAG Context**

```bash
# Enhanced debugging with file output
python src/tests/debug/debug_rag_context_enhanced.py "Complex query here"

# Results saved to data/debug_outputs/ with structured analysis
```

## Benefits Achieved

1. **Clean Conversations**: No more STDOUT pollution in our debugging sessions
2. **Complete Data Preservation**: Full outputs saved to files for thorough analysis
3. **Efficient Examination**: Quick summaries with file references for detailed investigation
4. **Scalable**: Handles queries with 5000+ lines of output without issues
5. **Organized**: Timestamped files with descriptive names for easy tracking

## Integration with Existing Workflow

- **Backward Compatible**: Existing scripts continue to work
- **Optional**: Can be bypassed for small outputs
- **Configurable**: Line limits and output directories can be customized
- **Automatic**: No manual intervention required for file redirection

## File Naming Convention

```
YYYYMMDD_HHMMSS_description.txt
20250629_194123_annotation_explorer_Show_me_all_proteins_involved_.txt
```

This system ensures we can run complex genomic queries and examine their results without hitting conversation context limits, while maintaining full access to all output data through organized file storage.