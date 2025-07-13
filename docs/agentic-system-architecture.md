# Agentic RAG System: Complete Technical Architecture

## Executive Summary

This document provides comprehensive technical documentation of the genomic agentic RAG system, detailing every component, decision point, and code execution path. The system combines LLM-powered planning with graph-based task execution to answer complex biological queries about genomic data.

**Current Status**: Functional but suffering from architectural misalignments causing 1000x cost overruns and wrong biological focus for discovery queries.

**Key Issue**: Global genome discovery queries trigger expensive chunking + premium model usage instead of cheap holistic analysis.

---

## System Architecture Overview

### High-Level Flow
```
User Query → Query Classification → Genome Selection → Planning → Task Graph → Execution → Synthesis → Response
```

### Core Components
1. **GenomicRAG** (`src/llm/rag_system/core.py`) - Main orchestrator
2. **LLMGenomeSelector** (`src/llm/rag_system/llm_genome_selector.py`) - Intelligent genome targeting
3. **PlannerAgent** (DSPy) - Query decomposition and task planning
4. **TaskGraph** (`src/llm/rag_system/task_management.py`) - DAG-based task orchestration
5. **TaskExecutor** (`src/llm/rag_system/task_executor.py`) - Individual task execution
6. **IntelligentChunkingManager** - Large dataset handling
7. **ModelAllocator** - Cost-optimized model selection
8. **Query Processors** - Database interaction (Neo4j, LanceDB)

### Data Flow Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  GenomicRAG.ask │───▶│ Query Planning  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Task Execution  │◀───│   Task Graph     │◀───│ Genome Selection│
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Database Query  │───▶│ Result Synthesis │───▶│ Final Response  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## Component Deep Dive

### 1. GenomicRAG Core (`src/llm/rag_system/core.py`)

**Role**: Main system orchestrator and entry point

#### Key Methods & Decision Points

##### `async def ask(self, question: str) -> str` (Line ~102)
**Purpose**: Main query processing entry point

**Decision Flow**:
```python
# 1. Query Classification
classification = self.classifier(question=question)

# 2. Route to Agentic vs Traditional
if planning_result.requires_planning:
    return await self._execute_agentic_plan(question, planning_result, selected_genome)
else:
    return await self._execute_traditional_query(question)
```

**Critical Decision**: `requires_planning` determination
- **Logic**: DSPy PlannerAgent classifies query complexity
- **Result**: Simple queries → direct database, Complex → agentic workflow
- **Issue**: Almost all queries trigger agentic mode due to broad classification

##### Genome Selection Integration (Line ~140)
**Implementation**:
```python
# INTELLIGENT UPFRONT GENOME SELECTION
if planning_result.requires_planning:
    from .llm_genome_selector import LLMGenomeSelector
    llm_selector = LLMGenomeSelector(self.neo4j_processor)
    
    selection_result = await llm_selector.analyze_genome_intent(question)
    
    if selection_result.intent == "specific" and selection_result.target_genomes:
        selected_genome = selection_result.target_genomes[0]
    else:
        selected_genome = None  # Global analysis
```

**Design Intent**: One LLM call upfront, propagate decision to all sub-tasks
**Current Status**: ✅ Working correctly - correctly identifies global vs specific intent
**Cost**: ~$0.0001 per query (GPT-4.1-mini)

##### Agentic Execution Path (Line ~165)
**Method**: `_execute_agentic_plan(question, planning_result, selected_genome)`

**Flow**:
1. **Task Plan Generation**: DSPy PlannerAgent creates step-by-step plan
2. **Plan Parsing**: TaskPlanParser converts text plan → Task objects
3. **Graph Construction**: Tasks organized into DAG with dependencies
4. **Parallel Execution**: TaskExecutor runs tasks with dependency resolution
5. **Result Synthesis**: Progressive combination of results

**Critical Integration Point**: `selected_genome` passed to TaskExecutor
```python
executor = TaskExecutor(self, note_keeper, selected_genome)
```

### 2. LLM Genome Selector (`src/llm/rag_system/llm_genome_selector.py`)

**Role**: Replace brittle keyword-based genome selection with LLM intelligence

#### Core Logic

##### Pre-filtering (`should_use_genome_selection()` - Line 102)
**Purpose**: Avoid unnecessary LLM calls for obvious cases

**Implementation**:
```python
obvious_global_patterns = [
    'read through everything', 'analyze everything', 'scan everything',
    'across all genomes', 'all genomes', 'every genome', 'compare all',
    'global analysis', 'pan-genome', 'dataset-wide'
]

if any(pattern in query_lower for pattern in obvious_global_patterns):
    return False  # Skip LLM, use global execution
```

**Test Case**: "read through everything directly" → Pre-filter catches this, skips LLM
**Result**: ✅ Working - correctly detected global pattern

##### LLM Analysis (`analyze_genome_intent()` - Line 141)
**Purpose**: Use GPT-4.1-mini to classify user intent

**Prompt Strategy**:
```python
prompt = f"""You are a genomics expert analyzing user queries to determine genome selection intent.

Available genomes in the database:
{genomes_list}

User query: "{query}"

Analyze this query and determine:
1. Intent Classification: specific|comparative|global|ambiguous
2. Target Genomes: exact genome IDs if specific, empty otherwise
3. Reasoning: 1-2 sentence explanation
4. Confidence: 0.0 to 1.0 score

Return JSON: {
  "intent": "specific|comparative|global|ambiguous",
  "target_genomes": "comma-separated genome IDs or empty",
  "reasoning": "explanation",
  "confidence": 0.95
}"""
```

**Output Example**:
```json
{
  "intent": "global",
  "target_genomes": "",
  "reasoning": "Query uses 'read through everything' indicating comprehensive analysis across all available genomes",
  "confidence": 0.95
}
```

**Performance**: ✅ Working correctly - proper intent classification
**Cost**: ~$0.0001 per query

### 3. Task Management System (`src/llm/rag_system/task_management.py`)

**Role**: DAG-based task orchestration with dependency resolution

#### Core Components

##### Task Class (Line 22)
```python
@dataclass
class Task:
    task_id: str
    task_type: TaskType  # ATOMIC_QUERY or TOOL_CALL
    description: str
    dependencies: List[str]
    status: TaskStatus   # PENDING/RUNNING/COMPLETED/FAILED/SKIPPED
    query: Optional[str]  # For ATOMIC_QUERY tasks
    tool_name: Optional[str]  # For TOOL_CALL tasks
```

##### TaskGraph Execution Logic
**Parallel Execution Strategy**:
```python
def get_execution_plan(self) -> List[List[str]]:
    """Generate batches of tasks that can run in parallel."""
    plan = []
    remaining_tasks = set(self.tasks.keys())
    
    while remaining_tasks:
        # Find tasks with satisfied dependencies
        current_batch = []
        for task_id in remaining_tasks:
            if all_dependencies_completed(task_id):
                current_batch.append(task_id)
        
        plan.append(current_batch)
        remaining_tasks -= set(current_batch)
```

**Dependency Management**: Automatic cascade handling when tasks fail
**Error Handling**: Failed tasks mark dependents as SKIPPED

### 4. Task Executor (`src/llm/rag_system/task_executor.py`)

**Role**: Execute individual Task objects, route to appropriate processors

#### Critical Architecture Issue: Model Allocation Explosion

##### Task Execution Flow (`execute_task()` - Line 68)
**For ATOMIC_QUERY tasks**:

```python
async def _execute_query_task(self, task: Task):
    # 1. Query Classification (MEDIUM complexity = gpt-4.1-mini)
    classification = self.rag_system.model_allocator.create_context_managed_call(
        task_name="query_classification",  # Maps to MEDIUM
        signature_class=QueryClassifier,
        module_call_func=classification_call
    )
    
    # 2. Query Generation (COMPLEX = o3) ⚠️ EXPENSIVE
    retrieval_plan = self.rag_system.model_allocator.create_context_managed_call(
        task_name="context_preparation",  # Maps to COMPLEX = o3
        signature_class=ContextRetriever,
        module_call_func=retrieval_call
    )
    
    # 3. Database Execution
    context = await self.rag_system._retrieve_context(...)
    
    # 4. Chunking Decision (LINE 265 - CRITICAL ISSUE)
    if len(raw_data) > threshold:
        # Triggers intelligent chunking regardless of global intent
        chunking_manager = IntelligentChunkingManager(...)
```

**🚨 ROOT CAUSE IDENTIFIED**: Every chunk task executes this full flow
- **8 chunks** × **o3 context_preparation** = **8 expensive API calls**
- **Expected**: 1 cheap gpt-4.1-mini call for global analysis
- **Actual**: $5-10 cost instead of $0.0001

##### Genome Context Propagation (Line 158)
**Implementation**:
```python
# Use pre-selected genome if available (from agentic upfront selection)
if self.selected_genome:
    logger.info(f"🧬 Using pre-selected genome from agentic planning: {self.selected_genome}")
    genome_filter_required = True
    target_genome = self.selected_genome
    enhanced_description = f"For genome {self.selected_genome}: {transformed_description}"
else:
    logger.debug("🌐 Task does not require genome-specific targeting - using global execution")
```

**Status**: ✅ Working - genome context properly propagated from core.py
**Issue**: Chunking system ignores this context

##### Chunking Decision Logic (Line 265)
**Current Implementation**:
```python
# Check if result is too large and needs intelligent chunking
threshold = 1000 if self.selected_genome else 2000
if len(raw_data) > threshold:  # 10,102 > 2000 = always chunk
    logger.info(f"🧠 Large dataset detected ({len(raw_data)} items), using intelligent upfront chunking")
    
    chunking_manager = IntelligentChunkingManager(max_chunks=4, min_chunk_size=100)
    chunks = await chunking_manager.analyze_and_chunk_dataset(task, raw_data, task.description)
```

**🚨 ARCHITECTURAL FLAW**: 
- **Chunking based purely on dataset size (10,102 items)**
- **Ignores that global discovery should be holistic**
- **`selected_genome = None` still triggers chunking because dataset > 2000**

### 5. Intelligent Chunking Manager (`src/llm/rag_system/intelligent_chunking_manager.py`)

**Role**: Break large datasets into manageable biological chunks

#### Chunking Strategy Selection
**Implementation**:
```python
async def analyze_and_chunk_dataset(self, task, raw_data, description):
    # Analyze query to determine optimal chunking strategy
    if "compare" or "comparison" in description.lower():
        strategy = "genomic_comparison"
    elif "function" or "functional" in description.lower():
        strategy = "functional_category"
    elif "pathway" or "metabolic" in description.lower():
        strategy = "metabolic_pathway"
    else:
        strategy = "comprehensive_analysis"
```

**Chunk Creation Example** (Functional Strategy):
```python
functional_groups = {
    "oxidation_reduction": [],
    "protein_synthesis": [],
    "biosynthesis": [],
    "transport": [],
    "other": []
}

# Group proteins by biological function
for item in raw_data:
    ko_desc = item.get('ko_description', '').lower()
    if any(term in ko_desc for term in ['oxidoreductase', 'dehydrogenase']):
        functional_groups["oxidation_reduction"].append(item)
    # ... more grouping logic
```

**Result**: Creates 3-5 biologically meaningful chunks
- `func_oxidation_reduction` (454 items)
- `func_protein_synthesis` (381 items)  
- `func_biosynthesis` (299 items)
- `func_other` (2,092 items)

#### Chunk Task Creation
**Critical Issue**: Each chunk becomes a separate Task with full execution pipeline
```python
chunk_task = Task(
    task_id=chunk.chunk_id,  # e.g., "func_oxidation_reduction"
    task_type=TaskType.ATOMIC_QUERY,
    description=enhanced_description,
    biological_focus=chunk.biological_focus,
    _already_chunked=True,  # Prevents recursive chunking
    root_biological_context=original_question
)
```

**Execution**: Each chunk task goes through full TaskExecutor pipeline
- **Query classification** (gpt-4.1-mini)
- **Context preparation** (o3) ⚠️ **EXPENSIVE**
- **Database query execution**
- **Result synthesis**

**🚨 MULTIPLICATION EFFECT**: 8 chunks × expensive pipeline = massive cost explosion

### 6. Model Allocation System (`src/llm/rag_system/memory/model_allocation.py`)

**Role**: Route tasks to appropriate models for cost optimization

#### Task Complexity Mapping
**Current Implementation**:
```python
TASK_COMPLEXITY_MAP = {
    # Simple tasks - gpt-4.1-mini
    "progress_tracking": TaskComplexity.SIMPLE,
    "basic_formatting": TaskComplexity.SIMPLE,
    
    # Medium tasks - gpt-4.1-mini  
    "query_classification": TaskComplexity.MEDIUM,
    "data_overview": TaskComplexity.MEDIUM,
    
    # Complex tasks - o3
    "context_preparation": TaskComplexity.COMPLEX,  # ⚠️ PROBLEM
    "biological_interpretation": TaskComplexity.COMPLEX,
    "final_synthesis": TaskComplexity.COMPLEX,
    "agentic_planning": TaskComplexity.COMPLEX
}
```

**🚨 CORE ISSUE**: `context_preparation` hardcoded as COMPLEX
- **Every chunk** calls context_preparation → **every chunk uses o3**
- **Should be**: Global discovery → MEDIUM (gpt-4.1-mini)
- **Context unaware**: Doesn't consider query type or genome selection

#### Model Selection Logic
```python
def create_context_managed_call(self, task_name, signature_class, module_call_func):
    complexity = self.get_task_complexity(task_name)
    
    if complexity == TaskComplexity.COMPLEX:
        model = self.premium_model  # o3 - $15/1M input tokens
    else:
        model = self.default_model  # gpt-4.1-mini - $0.075/1M input tokens
```

**Cost Impact**: 200x price difference between models
- **8 chunks × o3** = **$5-10 total cost**
- **8 chunks × gpt-4.1-mini** = **$0.008 total cost**

### 7. Query Processing Layer

#### Database Integration
**Neo4j Processor**: Structured queries against knowledge graph
**LanceDB Processor**: Semantic similarity search with ESM2 embeddings
**Hybrid Processor**: Combines both approaches

#### Query Generation Process
**Current Flow**:
1. **Classification**: Determine query type (structural, semantic, hybrid)
2. **Context Preparation**: Generate Cypher query + search strategy (uses o3)
3. **Execution**: Run against appropriate database
4. **Result Formatting**: Standardize output format

**Issue with Phage Discovery**:
- **Generated Query**: 
  ```cypher
  MATCH (p:Protein)-[:HAS_DOMAIN]->(d:Domain)
  WHERE toLower(d.description) CONTAINS 'transport'
  ```
- **Should Generate**:
  ```cypher
  MATCH (p:Protein)-[:ENCODED_BY]->(g:Gene)
  WHERE p.description CONTAINS 'hypothetical' OR p.description IS NULL
  ORDER BY g.genomeId, g.start
  // Find consecutive stretches for phage detection
  ```

**Root Cause**: o3 doesn't understand spatial genomic analysis requirements

---

## Decision Flow Analysis

### Query: "Find me operons containing probable prophage segments; we don't have virus-specific annotations so read through everything directly and see what you can find."

#### Step 1: Initial Classification ✅
```python
# In GenomicRAG.ask()
planning_result = self.planner(question=question)
# Result: requires_planning = True (triggers agentic mode)
```

#### Step 2: Genome Selection ✅
```python
# In core.py line ~140
llm_selector = LLMGenomeSelector(self.neo4j_processor)
selection_result = await llm_selector.analyze_genome_intent(question)

# Pre-filter detects "read through everything" → obvious global pattern
# Result: selected_genome = None (global analysis)
```

#### Step 3: Agentic Planning ✅
```python
# TaskPlanParser creates task graph from DSPy plan
tasks = [
    Task("step_1_retrieve_functional", ATOMIC_QUERY, "Retrieve functional annotations..."),
    Task("step_2_identify_spatial", ATOMIC_QUERY, "Identify spatial patterns..."),
    # ... more tasks
]
```

#### Step 4: Task Execution ⚠️ **ISSUES BEGIN**
```python
# In TaskExecutor.execute_task() for each task
async def _execute_query_task(self, task):
    # 4a. Query Classification (gpt-4.1-mini) ✅
    classification = model_allocator.create_context_managed_call(
        task_name="query_classification"  # MEDIUM complexity
    )
    
    # 4b. Context Preparation (o3) ⚠️ EXPENSIVE
    retrieval_plan = model_allocator.create_context_managed_call(
        task_name="context_preparation"  # COMPLEX complexity
    )
    
    # 4c. Database Query ✅
    context = await self.rag_system._retrieve_context(...)
    
    # 4d. Chunking Decision ⚠️ WRONG LOGIC
    if len(raw_data) > 2000:  # 10,102 > 2000 = chunk despite global intent
        chunking_manager = IntelligentChunkingManager(...)
        chunks = await chunking_manager.analyze_and_chunk_dataset(...)
        
        # Creates 8 chunk tasks, each goes through full pipeline again
```

#### Step 5: Chunk Multiplication ⚠️ **COST EXPLOSION**
**Each of 8 chunks executes**:
- Query classification (gpt-4.1-mini): 8 × $0.0001 = $0.0008
- Context preparation (o3): 8 × $0.50 = $4.00
- Database execution: 8 × minimal cost
- **Total**: ~$4-5 instead of expected $0.0001

#### Step 6: Query Generation ⚠️ **WRONG FOCUS**
**o3 generates**:
```cypher
MATCH (p:Protein)-[:HAS_DOMAIN]->(d:Domain)-[:BELONGS_TO]->(f:PfamFamily)
WHERE toLower(f.description) CONTAINS 'transport'
```

**Should generate**:
```cypher
MATCH (p:Protein)-[:ENCODED_BY]->(g:Gene)
WHERE p.description CONTAINS 'hypothetical' OR p.description IS NULL
WITH g.genomeId, g.start, g.end, g.strand
ORDER BY g.genomeId, g.start
```

#### Step 7: Result Synthesis ⚠️ **FRAGMENT CONTEXT**
- **8 separate analyses** of transport proteins, oxidation-reduction, etc.
- **Lost**: Spatial genomic context needed for phage discovery
- **Result**: Generic functional analysis instead of phage detection

---

## Root Cause Analysis

### 1. Model Allocation System Disconnect
**Issue**: Task complexity classification ignores query context
**Code**: `context_preparation` hardcoded as COMPLEX in `model_allocation.py`
**Impact**: Every chunk uses expensive o3 instead of cheap gpt-4.1-mini
**Fix Complexity**: Medium - requires context-aware complexity scoring

### 2. Chunking System Context Blindness  
**Issue**: Chunking decision based purely on dataset size
**Code**: `task_executor.py:272` - `if len(raw_data) > threshold`
**Impact**: Global discovery queries get fragmented instead of holistic analysis
**Fix Complexity**: Medium - requires query intent integration

### 3. Query Generation Strategy Mismatch
**Issue**: LLM doesn't understand spatial genomic analysis requirements
**Code**: DSPy ContextRetriever signature and prompts
**Impact**: Generates functional queries instead of spatial/positional queries
**Fix Complexity**: High - requires domain-specific prompt engineering

### 4. Context Propagation Gaps
**Issue**: Genome selection decisions don't flow to chunking/model allocation
**Code**: Multiple integration points across components
**Impact**: Downstream decisions ignore upstream analysis
**Fix Complexity**: Low - requires standardized context object

---

## Inter-Component Dependencies

```
GenomicRAG.ask()
├── LLMGenomeSelector ✅ Working
│   ├── Pre-filtering logic
│   └── LLM intent analysis
├── DSPy PlannerAgent ✅ Working  
│   └── Task decomposition
├── TaskPlanParser ✅ Working
│   └── Text → Task objects
├── TaskExecutor ⚠️ Issues
│   ├── ModelAllocator ⚠️ Context unaware
│   ├── Query Classification ✅ Working
│   ├── Context Preparation ⚠️ Wrong model selection
│   ├── Database Query ✅ Working
│   └── Chunking Decision ⚠️ Context blind
└── IntelligentChunkingManager ⚠️ Multiplication
    ├── Strategy Selection ✅ Working
    ├── Chunk Creation ✅ Working
    └── Chunk Execution ⚠️ Full pipeline per chunk
```

### Critical Dependencies
1. **Genome Selection → Task Execution**: ✅ Working via `selected_genome` parameter
2. **Query Intent → Model Allocation**: ❌ Missing - causes cost explosion
3. **Global Analysis → Chunking**: ❌ Missing - causes fragmentation
4. **Discovery Type → Query Strategy**: ❌ Missing - causes wrong focus

---

## Performance Analysis

### Current Costs (Prophage Discovery Query)
```
Component                  | Model      | Calls | Cost/Call | Total
---------------------------|------------|-------|-----------|-------
Genome Selection          | gpt-4.1-mini|   1   | $0.0001   | $0.0001
Query Classification      | gpt-4.1-mini|   8   | $0.0001   | $0.0008  
Context Preparation       | o3         |   8   | $0.50     | $4.00
Database Execution        | N/A        |   8   | $0.0001   | $0.0008
Result Synthesis          | gpt-4.1-mini|   1   | $0.001    | $0.001
---------------------------|------------|-------|-----------|-------
TOTAL                     |            |  26   |           | $4.002
```

### Expected Costs (Fixed System)
```
Component                  | Model      | Calls | Cost/Call | Total
---------------------------|------------|-------|-----------|-------
Genome Selection          | gpt-4.1-mini|   1   | $0.0001   | $0.0001
Query Classification      | gpt-4.1-mini|   1   | $0.0001   | $0.0001
Context Preparation       | gpt-4.1-mini|   1   | $0.0001   | $0.0001
Database Execution        | N/A        |   1   | $0.0001   | $0.0001
Result Synthesis          | gpt-4.1-mini|   1   | $0.001    | $0.001
---------------------------|------------|-------|-----------|-------
TOTAL                     |            |   5   |           | $0.0023
```

**Cost Reduction**: 1,740x cheaper (99.94% reduction)

### Current Execution Time
- **8 parallel chunks** × **o3 latency** = **~45-60 seconds**
- **Plus chunking overhead** = **~60-90 seconds total**

### Expected Execution Time (Fixed)
- **1 global query** × **gpt-4.1-mini latency** = **~5-10 seconds**
- **No chunking overhead** = **~10-15 seconds total**

**Time Reduction**: 4-6x faster

---

## Code File Locations & Key Lines

### Core System Files
```
src/llm/rag_system/core.py
├── Line 102: ask() - Main entry point
├── Line 140: Genome selection integration  
├── Line 165: _execute_agentic_plan()
└── Line 200: Traditional vs agentic routing

src/llm/rag_system/task_executor.py  
├── Line 68: execute_task() - Individual task execution
├── Line 138: _execute_query_task() - Database query tasks
├── Line 184: Query classification (gpt-4.1-mini)
├── Line 211: Context preparation (o3) ⚠️ EXPENSIVE
├── Line 265: Chunking decision ⚠️ CONTEXT BLIND
└── Line 278: IntelligentChunkingManager invocation

src/llm/rag_system/llm_genome_selector.py
├── Line 102: should_use_genome_selection() - Pre-filtering
├── Line 141: analyze_genome_intent() - LLM analysis
└── Line 206: _analyze_with_direct_llm() - Fallback implementation

src/llm/rag_system/memory/model_allocation.py
├── Line 150: TASK_COMPLEXITY_MAP ⚠️ HARDCODED
├── Line 200: get_task_complexity() ⚠️ CONTEXT UNAWARE
└── Line 250: create_context_managed_call() - Model routing

src/llm/rag_system/intelligent_chunking_manager.py
├── Line 50: analyze_and_chunk_dataset() - Strategy selection
├── Line 100: create_functional_chunks() - Biological grouping
└── Line 200: execute_chunked_analysis() ⚠️ MULTIPLICATION
```

### Configuration Files
```
src/llm/rag_system/memory/model_allocation.py:295
└── use_premium_everywhere=True - Model allocation settings

src/llm/rag_system/dspy_signatures.py
├── PlannerAgent - Task decomposition
├── QueryClassifier - Query type determination  
├── ContextRetriever - Query generation ⚠️ NEEDS DOMAIN LOGIC
└── GenomicAnswerer - Final response synthesis
```

---

## Recommended Architectural Changes

### Immediate Priority (Cost Reduction)
1. **Context-Aware Model Allocation**
   - Pass query type/intent to `get_task_complexity()`
   - Map global discovery → MEDIUM (gpt-4.1-mini)
   - Reserve o3 for true biological reasoning

2. **Chunking Decision Intelligence**
   - Consider query intent, not just dataset size
   - Global discovery → higher threshold or no chunking
   - Preserve spatial context for discovery workflows

### Medium Priority (Quality Improvement)  
3. **Query Strategy Classification**
   - Add biological query types: spatial, functional, comparative
   - Route to appropriate query generation logic
   - Domain-specific prompt engineering for each type

4. **Standardized Context Propagation**
   - Create unified context object
   - Flow decisions through entire pipeline
   - Enable informed decision-making at each stage

### Long-term (System Optimization)
5. **Discovery-Optimized Execution Paths**
   - Dedicated pathways for different biological analyses
   - Bypass chunking for holistic discovery
   - Optimize for specific biological use cases

---

## Testing & Validation Strategy

### Test Cases for Architecture Changes
1. **Prophage Discovery**: "Find me operons containing probable prophage segments"
   - **Expected**: Single global query, spatial focus, $0.002 cost
   - **Current**: 8 chunks, functional focus, $4.00 cost

2. **Specific Genome Analysis**: "Find transport proteins in Nomurabacteria"
   - **Expected**: Genome-specific targeting, detailed analysis
   - **Should maintain**: Current chunking behavior for detailed analysis

3. **Comparative Study**: "Compare metabolic capabilities across all genomes"
   - **Expected**: Cross-genome comparison, moderate chunking
   - **Should optimize**: For comparative query patterns

### Success Metrics
- **Cost**: 1000x reduction for discovery queries
- **Speed**: 4-6x faster execution
- **Quality**: Biologically appropriate query focus
- **Reliability**: Maintain existing functionality for other query types

---

## Conclusion

The agentic system has a solid architectural foundation with proper task orchestration, dependency management, and result synthesis. The core issues stem from **context disconnects between components**:

1. **Model allocation** doesn't consider query intent → cost explosion
2. **Chunking decisions** ignore analysis type → context fragmentation  
3. **Query generation** lacks domain knowledge → wrong biological focus

These are **fixable architectural issues**, not fundamental design flaws. The system can achieve the intended performance with targeted fixes to context propagation and decision logic.

**Recommended Approach**: Incremental fixes starting with model allocation (immediate 99% cost reduction) followed by chunking intelligence (context preservation) and query strategy enhancement (biological accuracy).