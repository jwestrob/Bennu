# Agentic Data Flow & Note-Taking Architecture

## Overview

The agentic execution system transforms complex genomic queries into structured, multi-step workflows using a sophisticated data flow architecture that preserves context, manages dependencies, and maintains comprehensive session notes for progressive analysis.

## Table of Contents

- [Task Execution Flow](#task-execution-flow)
- [Data Passing Mechanisms](#data-passing-mechanisms)
- [Note-Taking System](#note-taking-system)
- [Data Formats](#data-formats)
- [Agent Instructions](#agent-instructions)
- [Session Management](#session-management)
- [Cross-Task Communication](#cross-task-communication)

---

## Task Execution Flow

### 1. Query Planning Phase

```
User Query → DSPy PlannerAgent → Task Plan → TaskPlanParser → TaskGraph
```

**Data Format**: Natural language task plan converted to structured Task objects

```python
# Example Task Plan (DSPy Output)
task_plan = """
1. Retrieve the complete protein catalog for the Nomurabacteria genome
2. Collect functional annotations from multiple databases  
3. Evaluate annotation completeness for each protein
4. Compile subset of proteins meeting criteria
5. Summarize statistics and prepare final report
"""

# Converted to TaskGraph
task_graph = TaskGraph([
    Task(id="step_1_retrieve", type=ATOMIC_QUERY, description="..."),
    Task(id="step_2_collect", type=ATOMIC_QUERY, dependencies=["step_1_retrieve"]),
    # ... additional tasks
])
```

### 2. Task Execution Phase

```
TaskGraph → TaskExecutor → Individual Task Execution → Result Storage
```

**Data Flow Pattern**:
- Each task receives enhanced description with genome context
- Results stored in `TaskExecutor.completed_results[task_id]`
- Dependencies resolved through result lookup

---

## Data Passing Mechanisms

### Inter-Task Data Structures

#### 1. Task Results Storage
```python
# TaskExecutor maintains results registry
class TaskExecutor:
    def __init__(self):
        self.completed_results = {}  # task_id → ExecutionResult
        
# Example stored result
completed_results["step_1_retrieve"] = {
    "context": GenomicContext(
        structured_data=[...],  # Neo4j query results
        semantic_data=[...],    # LanceDB similarity results
        metadata={...}
    ),
    "query_type": "structural",
    "search_strategy": "direct_query",
    "description": "Enhanced task description...",
    "structured_data": [...],  # Flattened for easy access
    "semantic_data": [...],
    "metadata": {...}
}
```

#### 2. Tool Argument Preparation
```python
def _prepare_tool_arguments(self, task: Task) -> Dict[str, Any]:
    """Prepare tool arguments including dependency results."""
    args = task.tool_args.copy()
    
    # Add dependency results
    dependency_data = []
    for dep_id in task.dependencies:
        if dep_id in self.completed_results:
            dependency_data.append(self.completed_results[dep_id])
    
    if dependency_data:
        args["dependency_results"] = dependency_data
        
    return args
```

### Data Flow Example

```
Step 1: Protein Retrieval
├── Input: Enhanced task description with genome context
├── Output: GenomicContext with 826 proteins
├── Storage: completed_results["step_1"] = {context, structured_data, ...}

Step 2: Annotation Collection  
├── Input: Task description + dependency_results from Step 1
├── Processing: Uses Step 1 protein list to query annotations
├── Output: GenomicContext with protein-annotation mappings
├── Storage: completed_results["step_2"] = {context, annotation_data, ...}

Step 3: Analysis
├── Input: Task description + dependency_results from Steps 1-2
├── Processing: Combines protein data + annotations for analysis
├── Output: Filtered results meeting criteria
```

---

## Note-Taking System

### Architecture Overview

```
Task Execution → Note Decision → Note Storage → Session Synthesis
```

### Note Decision Process

#### DSPy-Driven Decision Making
```python
class NotingDecision(dspy.Signature):
    """Decide whether to record notes for a completed task."""
    
    task_description = dspy.InputField(desc="Description of the completed task")
    execution_result = dspy.InputField(desc="Summary of execution results")
    existing_notes = dspy.InputField(desc="Summary of existing session notes")
    
    should_record = dspy.OutputField(desc="Whether to record notes (true/false)")
    reasoning = dspy.OutputField(desc="Explanation for the decision")
    importance_score = dspy.OutputField(desc="Importance score 1-10")
    
    # Note content fields (only filled if should_record=true)
    observations = dspy.OutputField(desc="Key observations from this task")
    key_findings = dspy.OutputField(desc="Important findings or discoveries")
    cross_connections = dspy.OutputField(desc="Connections to other tasks")
    quantitative_data = dspy.OutputField(desc="Important numbers or metrics")
```

#### Decision Criteria
The system records notes when:
- Task produces significant biological insights
- Results will be useful for future tasks
- Quantitative data worth preserving
- Novel patterns or anomalies discovered
- Cross-task connections identified

### Note Storage Structure

#### File Organization
```
data/session_notes/{session-uuid}/
├── session_metadata.json              # Session-level information
├── task_notes/                         # Individual task notes
│   ├── step_1_retrieve_notes.json     # Protein retrieval notes
│   ├── step_2_collect_notes.json      # Annotation collection notes
│   └── step_3_analysis_notes.json     # Analysis results notes
└── synthesis_notes/                    # Cross-task synthesis
    └── progressive_synthesis.json     # Combined insights
```

#### Note Schema
```python
@dataclass
class TaskNote:
    task_id: str
    task_type: str                    # "ATOMIC_QUERY" | "TOOL_CALL"
    description: str
    decision_result: NotingDecisionResult
    observations: List[str]           # Key observations
    key_findings: List[str]           # Important discoveries
    quantitative_data: Dict[str, Any] # Metrics and numbers
    cross_connections: List[CrossTaskConnection]
    confidence: ConfidenceLevel       # HIGH | MEDIUM | LOW
    execution_time: float
    tokens_used: int
    timestamp: datetime
```

### Note Content Examples

#### Protein Retrieval Task Notes
```json
{
  "task_id": "step_1_retrieve_proteins",
  "observations": [
    "Retrieved 826 proteins from Candidatus Nomurabacteria bacterium",
    "Protein lengths range from 34 to 918 amino acids",
    "Majority are small proteins (<200 aa), typical of CPR clade"
  ],
  "key_findings": [
    "Streamlined proteome consistent with symbiotic lifestyle",
    "High proportion of hypothetical proteins (~50%)"
  ],
  "quantitative_data": {
    "total_proteins": 826,
    "min_length": 34,
    "max_length": 918,
    "avg_length": 156,
    "hypothetical_percentage": 50.2
  },
  "cross_connections": [
    {
      "connected_task": "step_2_annotations",
      "connection_type": "BUILDS_ON",
      "description": "Protein list feeds into annotation analysis"
    }
  ]
}
```

#### Annotation Analysis Task Notes  
```json
{
  "task_id": "step_2_collect_annotations",
  "observations": [
    "KEGG annotations available for 412 proteins (49.9%)",
    "PFAM domains identified in 523 proteins (63.3%)",
    "302 proteins completely unannotated across all databases"
  ],
  "key_findings": [
    "Core metabolic functions well-annotated",
    "Many transport and regulatory proteins lack annotation",
    "Novel protein families potentially present"
  ],
  "quantitative_data": {
    "kegg_annotated": 412,
    "pfam_annotated": 523, 
    "completely_unannotated": 302,
    "annotation_coverage": 63.4
  }
}
```

---

## Data Formats

### 1. GenomicContext Objects
Primary data structure for query results:

```python
@dataclass
class GenomicContext:
    structured_data: List[Dict[str, Any]]  # Neo4j results
    semantic_data: List[Dict[str, Any]]    # LanceDB results  
    metadata: Dict[str, Any]               # Query metadata
    query_type: str                        # "structural"|"semantic"|"hybrid"
    
# Example structured_data entry
{
    "protein_id": "RIFCSPLOWO2_01_FULL_OD1_41_220_rifcsplowo2_01_scaffold_10964_1",
    "protein_length": 245,
    "gene_id": "gene_001",
    "start_coordinate": 1000,
    "end_coordinate": 1735,
    "strand": 1,
    "kegg_functions": ["K03406"],
    "pfam_domains": ["PF00005"]
}
```

### 2. Tool Execution Results
For code interpreter and external tool tasks:

```python
{
    "tool_result": {
        "status": "success",
        "output": "Analysis completed successfully",
        "data": {...},          # Tool-specific output
        "execution_time": 45.2
    },
    "tool_name": "code_interpreter",
    "description": "Statistical analysis of protein lengths",
    "arguments": {...}
}
```

### 3. Progressive Synthesis Format
Combines multiple task results:

```python
{
    "session_id": "uuid",
    "task_count": 5,
    "synthesis_strategy": "detailed_analysis",
    "data_sources": {
        "structured_query_results": 1245,  # Total items
        "code_interpreter_analyses": 3,
        "tool_outputs": 2
    },
    "biological_insights": [...],
    "quantitative_summary": {...},
    "recommendations": [...]
}
```

---

## Agent Instructions

### Note Recording Instructions (DSPy Prompt)

The system provides detailed instructions to the NotingDecision agent:

```python
# Embedded in NotingDecision signature
"""
RECORDING CRITERIA:
Record notes when the task:
1. Produces significant biological insights or discoveries
2. Generates quantitative data useful for session synthesis  
3. Identifies patterns, anomalies, or unexpected results
4. Creates connections between different aspects of the analysis
5. Provides foundation data that subsequent tasks will build upon

OBSERVATIONS: Record specific, factual observations about:
- Data quantities and characteristics
- Biological patterns identified
- Technical execution details
- Quality or completeness of results

KEY FINDINGS: Focus on:
- Biological significance
- Novel or unexpected discoveries  
- Patterns that support or contradict hypotheses
- Results that inform next steps

CROSS CONNECTIONS: Identify how this task:
- Builds on previous task results
- Provides input for upcoming tasks  
- Confirms or contradicts other findings
- Contributes to overall analysis narrative

QUANTITATIVE DATA: Preserve important metrics:
- Counts, percentages, statistical measures
- Threshold values, scores, confidence levels
- Performance metrics, execution times
- Data quality indicators
"""
```

### Content Guidelines

#### Observations (What was found)
- **Factual statements** about data retrieved or processed
- **Quantitative summaries** of results 
- **Quality assessments** of data completeness
- **Technical details** about execution success/failure

#### Key Findings (Why it matters)
- **Biological interpretations** of observations
- **Significance** in context of overall question
- **Novel insights** not previously apparent
- **Patterns** that support or refute hypotheses

#### Cross Connections (How it relates)
- **Dependencies**: Which tasks this builds upon
- **Contributions**: What this provides to future tasks
- **Confirmations**: Results that support other findings
- **Contradictions**: Results that challenge other findings

---

## Session Management

### Session Lifecycle

```
Session Creation → Task Execution → Note Recording → Progressive Synthesis → Final Report
```

#### 1. Session Initialization
```python
# Automatic session creation
session_id = str(uuid.uuid4())
notes_folder = f"data/session_notes/{session_id}"
note_keeper = NoteKeeper(session_id)

# Session metadata tracking
session_metadata = {
    "session_id": session_id,
    "start_time": datetime.now(),
    "original_query": user_question,
    "execution_mode": "agentic",
    "selected_genome": genome_id,
    "total_tasks": len(task_graph.tasks)
}
```

#### 2. Progressive Note Accumulation
As tasks complete, notes accumulate:
- Individual task insights stored immediately
- Cross-task connections tracked
- Quantitative data aggregated
- Session narrative builds progressively

#### 3. Final Synthesis
```python
# Combines all session data
synthesizer = ProgressiveSynthesizer(note_keeper)
final_synthesis = synthesizer.synthesize_session(
    task_results=completed_results,
    session_notes=note_keeper.get_all_notes(),
    original_query=user_question
)
```

### Data Persistence

#### File-Based Storage
- **JSON format** for easy parsing and human readability
- **Structured hierarchy** for organized access
- **Timestamped entries** for temporal analysis
- **UUID-based sessions** for unique identification

#### Access Patterns
```python
# Reading session notes
note_keeper = NoteKeeper(session_id)
all_notes = note_keeper.get_all_notes()
task_note = note_keeper.get_task_notes(task_id)
session_summary = note_keeper.get_session_summary()

# Cross-session analysis (future capability)
session_manager = SessionManager()
related_sessions = session_manager.find_similar_sessions(query_keywords)
```

---

## Cross-Task Communication

### Dependency Resolution

#### 1. Explicit Dependencies
```python
# Task A produces protein list
# Task B depends on Task A results
task_b_args = {
    "dependency_results": [completed_results["task_a"]],
    "protein_list": extract_proteins(completed_results["task_a"])
}
```

#### 2. Implicit Context Sharing
```python
# All tasks in session share:
# - Selected genome context
# - Session-level metadata  
# - Progressive synthesis state
# - Cross-task note connections
```

### Information Flow Patterns

#### 1. Sequential Building
```
Task 1: Data Collection → Task 2: Analysis → Task 3: Synthesis
```

#### 2. Parallel Processing
```
Task 1a: Protein Data ┐
                      ├→ Task 3: Combined Analysis
Task 1b: Annotation Data ┘
```

#### 3. Iterative Refinement
```
Task 1: Initial Analysis → Task 2: Refinement → Task 3: Validation
```

### Context Preservation Mechanisms

#### 1. Enhanced Task Descriptions
Original: "Collect functional annotations"
Enhanced: "For genome Candidatus_Nomurabacteria_bacterium_RIFCSPLOWO2_01_FULL_41_220_contigs: Collect functional annotations"

#### 2. Metadata Propagation
```python
# Metadata flows through execution chain
metadata = {
    "selected_genome": genome_id,
    "analysis_focus": "novelty_detection", 
    "session_context": "detailed_genome_analysis",
    "execution_strategy": "agentic_multi_step"
}
```

#### 3. Note-Based Context
Cross-task connections in notes provide narrative continuity:
```python
CrossTaskConnection(
    connected_task="step_1_proteins",
    connection_type=ConnectionType.BUILDS_ON,
    description="Uses protein list as input for annotation analysis"
)
```

---

## Performance Considerations

### Memory Management
- **Results caching** limited to session duration
- **Large datasets** handled through intelligent chunking
- **Note storage** optimized for incremental writing

### Execution Efficiency
- **Parallel task execution** where dependencies allow
- **Model allocation** based on task complexity
- **Context compression** for large data flows

### Scalability Patterns
- **Session isolation** prevents cross-contamination
- **Modular architecture** allows component replacement
- **Note indexing** enables efficient cross-session queries

---

## Future Enhancements

### Planned Improvements
1. **Cross-Session Learning**: Notes from previous sessions inform new analyses
2. **Dynamic Task Generation**: AI-driven task plan modification based on intermediate results
3. **Advanced Synthesis**: Machine learning-based pattern recognition across note collections
4. **Real-Time Collaboration**: Multiple agents contributing to shared session notes

### Extensibility Points
- **Custom note schemas** for domain-specific analyses
- **External tool integration** with standardized data formats
- **Advanced dependency tracking** with semantic relationships
- **Automated quality assessment** of note content and cross-task connections

---

This architecture provides a robust foundation for complex, multi-step genomic analyses while maintaining comprehensive audit trails and enabling sophisticated cross-task reasoning through the note-taking system.