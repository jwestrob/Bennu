# Modular RAG System Implementation Plan

## Analysis of Original rag_system.py.backup

### 🎯 **Core Classes to Implement**

#### **1. Task Management Classes** (→ `task_management.py`)
- `TaskStatus(Enum)` - Task execution states
- `TaskType(Enum)` - Types of tasks (ATOMIC_QUERY, TOOL_CALL, etc.)
- `Task` - Individual task dataclass
- `TaskGraph` - DAG-based task execution

#### **2. Main RAG Classes** (→ `core.py`)
- `GenomicContext` - Context container for query results
- `GenomicRAG(dspy.Module)` - Main RAG system class

#### **3. DSPy Signatures** (→ `dspy_signatures.py`)
- `PlannerAgent` - Determines if agentic planning needed
- `QueryClassifier` - Classifies query types 
- `ContextRetriever` - Generates retrieval strategies
- `GenomicAnswerer` - Generates final answers

#### **4. Utility Classes** (→ `utils.py`)
- `ResultStreamer` - Already implemented ✅

### 🔧 **Core Methods to Implement**

#### **Main Entry Points** (→ `core.py`)
- `ask(question: str) -> Dict[str, Any]` - **CRITICAL** Main entry point
- `_execute_traditional_query(question: str) -> Dict[str, Any]` - Single-step queries
- `_execute_agentic_plan(question: str, planning_result) -> Dict[str, Any]` - Multi-step workflows
- `health_check() -> Dict[str, bool]` - System health check

#### **Task Execution** (→ `task_management.py`)
- `_execute_task(task: Task, previous_results: Dict[str, Any]) -> Any` - Execute individual tasks
- `_combine_task_results(all_results: Dict[str, Any]) -> str` - Combine multiple task results

#### **Context Processing** (→ `context_processing.py`)
- `_retrieve_context(query_type: str, retrieval_plan) -> GenomicContext` - **CRITICAL** Context retrieval
- `_format_context(context: GenomicContext) -> str` - Format context for LLM

#### **Code Enhancement** (→ `code_enhancement.py`)
- `_enhance_code_interpreter(original_code: str, previous_results: Dict[str, Any]) -> str` - **CRITICAL** Code enhancement
- `_extract_protein_ids_from_task(task_ref: str, previous_results: Dict[str, Any]) -> List[str]` - Extract protein IDs

#### **Template Processing** (→ `template_resolution.py`)
- `_resolve_template_variables(tool_args: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]` - Template resolution
- `_enhance_literature_query(original_query: str, previous_results: Dict[str, Any]) -> str` - Literature search enhancement

### 🛠️ **External Tools Integration** (→ `external_tools.py`)
- `literature_search(query: str, email: str, **kwargs) -> str` - PubMed search
- `code_interpreter_tool(code: str, session_id: str = None, timeout: int = 30, **kwargs) -> Dict[str, Any]` - Code execution
- `AVAILABLE_TOOLS` - Tool registry dictionary

### 📊 **Configuration and Setup** (→ `core.py`)
- `_configure_dspy()` - DSPy configuration
- `__init__(config: LLMConfig, chunk_context_size: int = 4096)` - Initialization

### 🔍 **Critical Implementation Details**

#### **1. GenomicContext Class**
```python
@dataclass
class GenomicContext:
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    formatted_context: str = ""
```

#### **2. Main ask() Method Flow**
```python
async def ask(self, question: str) -> Dict[str, Any]:
    # 1. Determine if agentic planning needed
    planning_result = self.planner(user_query=question)
    
    if requires_planning:
        # Multi-step agentic execution
        return await self._execute_agentic_plan(question, planning_result)
    else:
        # Traditional single-step execution  
        return await self._execute_traditional_query(question)
```

#### **3. Traditional Query Flow**
```python
async def _execute_traditional_query(self, question: str) -> Dict[str, Any]:
    # 1. Classify query type
    classification = self.classifier(question=question)
    
    # 2. Generate retrieval strategy
    retrieval_plan = self.retriever(
        db_schema=NEO4J_SCHEMA,
        question=question, 
        query_type=classification.query_type
    )
    
    # 3. Retrieve context
    context = await self._retrieve_context(classification.query_type, retrieval_plan)
    
    # 4. Generate answer
    answer = self.answerer(
        question=question,
        context=self._format_context(context)
    )
    
    return {
        "question": question,
        "answer": answer.answer,
        "confidence": answer.confidence,
        "citations": answer.citations,
        "query_metadata": {...}
    }
```

#### **4. Context Retrieval** (Most Complex)
```python
async def _retrieve_context(self, query_type: str, retrieval_plan) -> GenomicContext:
    # Handle different query types: structural, semantic, hybrid
    # Execute Neo4j queries, LanceDB searches
    # Apply size detection and scaling strategies
    # Format results appropriately
```

### 📁 **File Organization Strategy**

#### **Priority 1: Core Functionality**
1. **`task_management.py`** - Task classes and execution
2. **`dspy_signatures.py`** - All DSPy signatures  
3. **`external_tools.py`** - Tool implementations and registry
4. **`core.py`** - Update with complete GenomicRAG implementation

#### **Priority 2: Context Processing** 
5. **`context_processing.py`** - Enhance with _retrieve_context implementation
6. **`template_resolution.py`** - Template and variable resolution

#### **Priority 3: Integration**
7. **Update `__init__.py`** - Export all necessary classes
8. **Update `utils.py`** - Add GenomicContext class

### ⚠️ **Critical Dependencies**

#### **Imports Needed**
- `dspy` - For all signatures and module base class
- `rich.console` - For progress output  
- Query processors from `query_processor.py`
- Annotation tools from `annotation_tools.py`
- Sequence tools from `sequence_tools.py`

#### **External Services**
- Neo4j database (via query_processor)
- LanceDB vector database (via query_processor)  
- Code interpreter service (FastAPI container)
- PubMed API (for literature search)

### 🎯 **Implementation Order**

#### **Phase 1: Foundation** (Essential for basic functionality)
1. Create `task_management.py` with Task classes
2. Create `dspy_signatures.py` with all signatures
3. Create `external_tools.py` with AVAILABLE_TOOLS
4. Update `core.py` with basic ask() and _execute_traditional_query()

#### **Phase 2: Context System** (Essential for working queries)
5. Implement complete `_retrieve_context()` in `context_processing.py`
6. Implement `_format_context()` and GenomicContext
7. Update core.py to use new context processing

#### **Phase 3: Advanced Features** (Agentic workflows)
8. Implement `_execute_agentic_plan()` in `core.py`
9. Implement `_execute_task()` in `task_management.py`
10. Implement template resolution system

#### **Phase 4: Code Enhancement** (Code interpreter)
11. Complete `_enhance_code_interpreter()` implementation
12. Integrate protein ID extraction and data scaling
13. Test end-to-end workflows

### 🧪 **Testing Strategy**
- Test basic ask() with simple queries first
- Test traditional query path before agentic
- Test each tool integration individually
- Test complete workflows last

### 📝 **Notes**
- The original `ask()` method returns a **Dict**, not a string
- Context retrieval is the most complex part with size scaling
- DSPy signatures are essential for LLM interactions
- Task management enables multi-step agentic workflows
- Code interpreter integration requires protein ID extraction

This plan preserves all functionality from the original while maintaining the modular architecture benefits.