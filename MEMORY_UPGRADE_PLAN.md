# Memory Upgrade Plan: Agentic Note-Taking System

## Overview

This document outlines the implementation plan for adding persistent note-taking capabilities to the genomic RAG system's agentic execution mode. The system will allow the agent to selectively record observations, insights, and cross-task connections during complex multi-step analyses, enabling better memory persistence and synthesis quality.

## Core Design Principles

### 1. **Selective Note-Taking**
- The agent chooses when to take notes based on information value
- Not every task result requires notes - only significant findings
- Notes are generated through LLM decision-making, not automatic recording

### 2. **Progressive Synthesis**
- Notes are processed in chunks during execution
- Cross-task connections are maintained through structured references
- Final synthesis combines all note chunks rather than raw results

### 3. **Memory Persistence**
- Notes survive token limits and context windows
- Structured storage enables efficient retrieval and processing
- Session-based organization supports complex, long-running analyses

## System Architecture

### Data Storage Structure
```
data/
├── session_notes/
│   ├── {session_id}/                      # UUID-based session identifier
│   │   ├── task_notes/                    # Individual task observations
│   │   │   ├── {task_id}_notes.json
│   │   │   ├── {task_id}_structured_data.json
│   │   │   └── {task_id}_metadata.json
│   │   ├── synthesis_notes/               # Progressive synthesis results
│   │   │   ├── chunk_001_synthesis.json
│   │   │   ├── chunk_002_synthesis.json
│   │   │   └── final_synthesis.json
│   │   └── session_metadata.json          # Session configuration and stats
```

### Note Schema Specifications

#### Task Notes Schema
```json
{
  "task_id": "step_3_analyze_transport_systems",
  "task_type": "atomic_query",
  "description": "Analyze ABC transport systems across genomes",
  "execution_timestamp": "2025-01-09T04:15:00Z",
  "note_decision": {
    "should_record": true,
    "reasoning": "Significant functional differences found between genomes",
    "importance_score": 8.5
  },
  "observations": [
    "Burkholderiales has 3x more metal transport systems than CPR genomes",
    "K24821 (Mn/Fe transporter) universally conserved across all genomes",
    "Transport system clustering suggests metabolic specialization"
  ],
  "key_findings": [
    "Transport diversity correlates with genome size and environmental niche",
    "Essential metal uptake systems highly conserved despite genome reduction"
  ],
  "quantitative_data": {
    "total_transport_systems": 47,
    "genome_distribution": {
      "Burkholderiales": 28,
      "CPR_Muproteobacteria": 12,
      "CPR_Nomurabacteria": 7
    },
    "functional_categories": {
      "metal_transport": 15,
      "sugar_transport": 12,
      "amino_acid_transport": 8,
      "other": 12
    }
  },
  "cross_task_connections": [
    {
      "connected_task": "step_1_genome_overview",
      "connection_type": "validates",
      "description": "Transport counts correlate with genome size patterns"
    },
    {
      "connected_task": "step_5_metabolic_analysis",
      "connection_type": "informs",
      "description": "Transport capabilities constrain metabolic potential"
    }
  ],
  "confidence_level": "high",
  "data_quality_notes": "Complete annotation coverage, consistent methodology",
  "follow_up_questions": [
    "Are transport systems co-located with relevant metabolic genes?",
    "Do expression patterns match transport system predictions?"
  ]
}
```

#### Synthesis Notes Schema
```json
{
  "chunk_id": "chunk_002",
  "source_tasks": ["step_3_transport_analysis", "step_4_metabolic_pathways", "step_5_comparative_genomics"],
  "synthesis_timestamp": "2025-01-09T04:20:00Z",
  "chunk_theme": "Transport-Metabolism Integration",
  "integrated_findings": [
    "Transport system diversity directly enables metabolic flexibility",
    "CPR genomes retain core transport functions but lose environmental sensors"
  ],
  "cross_task_synthesis": [
    {
      "connection": "Transport counts (step_3) explain metabolic limitations (step_4)",
      "insight": "Reduced transport = reduced metabolic substrate range",
      "confidence": "high"
    }
  ],
  "emergent_insights": [
    "Genome reduction follows clear functional hierarchy: core > auxiliary > sensing",
    "Transport system architecture reflects evolutionary constraints"
  ],
  "synthesis_confidence": "high",
  "tokens_used": 2150,
  "compression_applied": false
}
```

## Implementation Plan

### Phase 1: Core Infrastructure (2-3 hours)

#### 1.1 Create Base Classes
- **NoteKeeper**: Manages note storage and retrieval
- **NotingDecision**: DSPy signature for note-taking decisions
- **ProgressiveSynthesizer**: Handles chunked synthesis

#### 1.2 File Structure
```
src/llm/rag_system/
├── memory/
│   ├── __init__.py
│   ├── note_keeper.py          # NoteKeeper class
│   ├── note_schemas.py         # Pydantic schemas for validation
│   ├── progressive_synthesizer.py  # ProgressiveSynthesizer class
│   └── memory_utils.py         # Utility functions
├── dspy_signatures.py          # Add NotingDecision signature
└── core.py                     # Integration point
```

#### 1.3 DSPy Signature for Note Decision
```python
class NotingDecision(dspy.Signature):
    """
    Decide whether task results warrant note-taking based on information value.
    
    Only record notes for:
    - Significant biological insights or patterns
    - Unexpected or contradictory findings
    - Cross-genome comparisons with clear differences
    - Quantitative results that inform broader analysis
    - Findings that connect to other tasks
    
    Skip notes for:
    - Routine lookups with expected results
    - Single data points without broader context
    - Redundant information already captured
    - Low-confidence or unclear results
    """
    
    task_description = dspy.InputField(desc="Description of the task that was executed")
    execution_result = dspy.InputField(desc="Results from task execution")
    existing_notes = dspy.InputField(desc="Summary of notes from previous tasks")
    
    should_record = dspy.OutputField(desc="Boolean: Should we record notes for this task?")
    importance_score = dspy.OutputField(desc="Importance score 1-10 for this information")
    reasoning = dspy.OutputField(desc="Explanation of note-taking decision")
    observations = dspy.OutputField(desc="If recording: key observations to note")
    cross_connections = dspy.OutputField(desc="If recording: connections to other tasks")
```

### Phase 2: TaskExecutor Integration (1-2 hours)

#### 2.1 Modify TaskExecutor
- Add `NoteKeeper` initialization
- Integrate note-taking decision after task completion
- Implement selective note recording

#### 2.2 Enhanced Task Execution Flow
```python
async def execute_task(self, task: Task) -> ExecutionResult:
    # ... existing execution logic ...
    
    # Post-execution: Decide whether to take notes
    if self.note_keeper:
        await self._consider_note_taking(task, execution_result)
    
    return execution_result

async def _consider_note_taking(self, task: Task, result: ExecutionResult):
    # Use DSPy to decide if notes are warranted
    decision = self.noting_decision(
        task_description=task.description,
        execution_result=result.result,
        existing_notes=self.note_keeper.get_session_summary()
    )
    
    if decision.should_record:
        await self.note_keeper.record_task_notes(task, result, decision)
```

### Phase 3: Progressive Synthesis (2-3 hours)

#### 3.1 Chunked Processing Logic
- Group related tasks (5-10 tasks per chunk)
- Process notes in batches using token counting
- Store intermediate synthesis results

#### 3.2 Replace Current Synthesis
- Modify `_synthesize_agentic_results()` to use progressive approach
- Implement chunk-based processing with memory management
- Add final synthesis combining all chunks

#### 3.3 Memory Management
- Token counting for chunk size optimization
- Note compression for older chunks
- Efficient retrieval during synthesis

### Phase 4: Advanced Features (1-2 hours)

#### 4.1 Note Indexing
- Index notes by task type, genome, functional category
- Enable cross-reference lookup
- Implement relevance ranking

#### 4.2 Session Management
- Session cleanup utilities
- Note archiving for long-term storage
- Performance monitoring and optimization

## Integration Points

### Core Components to Modify

#### 1. `src/llm/rag_system/core.py`
- Add `NoteKeeper` initialization in `__init__()`
- Modify `_synthesize_agentic_results()` to use progressive synthesis
- Add session management for note-taking

#### 2. `src/llm/rag_system/task_executor.py`
- Add note-taking decision logic
- Integrate `NoteKeeper` into execution flow
- Add note recording after task completion

#### 3. `src/llm/rag_system/dspy_signatures.py`
- Add `NotingDecision` signature
- Enhance synthesis signatures for note-aware processing

### Data Flow Enhancement

#### Current Flow:
```
Task → Execute → Store Result → Aggregate → Synthesize
```

#### Enhanced Flow:
```
Task → Execute → Store Result → Note Decision → [Optional] Record Notes
     ↓
Task Group Complete → Synthesize Chunk → Store Synthesis
     ↓
All Tasks Complete → Progressive Final Synthesis → Answer
```

## Testing Strategy

### Unit Tests
- `NoteKeeper` storage and retrieval
- Note schema validation
- Progressive synthesis logic

### Integration Tests
- End-to-end note-taking workflow
- Memory persistence across sessions
- Performance with large note sets

### User Acceptance Tests
- Complex multi-genome comparisons
- Long-running analyses (>20 tasks)
- Memory-intensive queries with synthesis quality

## Performance Considerations

### Token Usage
- Additional LLM calls for note decisions (~100-200 tokens each)
- Progressive synthesis reduces peak token usage
- Note compression for long-term efficiency

### Storage Requirements
- ~1-5KB per note file
- ~10-50KB per synthesis chunk
- Automatic cleanup for old sessions

### Memory Management
- Lazy loading of notes during synthesis
- Chunked processing prevents memory overflow
- Efficient indexing for fast retrieval

## Risk Assessment

### Technical Risks
- **Note Quality**: LLM-generated observations may be inaccurate
- **Storage Bloat**: Excessive note-taking could consume significant disk space
- **Performance Impact**: Additional LLM calls may slow execution

### Mitigation Strategies
- Implement note quality validation
- Add storage limits and cleanup policies
- Optimize note decision logic to minimize unnecessary calls

### Rollback Plan
- Feature flag for note-taking system
- Graceful degradation to current synthesis method
- Preserve existing functionality as fallback

## Success Metrics

### Quantitative Metrics
- Improved synthesis quality (measured by user feedback)
- Reduced token usage in final synthesis (target: 20-30% reduction)
- Successful handling of complex queries (>20 tasks)

### Qualitative Metrics
- Better cross-task insight generation
- Improved biological reasoning in final answers
- Enhanced handling of contradictory or complex findings

## Implementation Timeline

### Day 1: Core Infrastructure
- [ ] Create memory module structure
- [ ] Implement NoteKeeper class
- [ ] Add NotingDecision DSPy signature
- [ ] Create note schemas and validation

### Day 2: TaskExecutor Integration
- [ ] Modify TaskExecutor for note-taking
- [ ] Implement selective note recording
- [ ] Add basic progressive synthesis

### Day 3: Advanced Features & Testing
- [ ] Implement full progressive synthesis
- [ ] Add session management
- [ ] Comprehensive testing
- [ ] Performance optimization

## Future Enhancements

### Phase 5: Advanced Memory Features
- Cross-session note persistence
- Note similarity search and clustering
- Automated insight discovery across sessions

### Phase 6: User Interface
- Note visualization and debugging tools
- Interactive note editing and curation
- Session replay and analysis features

## Configuration Options

### Environment Variables
```bash
# Enable/disable note-taking
GENOMIC_RAG_ENABLE_NOTES=true

# Note storage location
GENOMIC_RAG_NOTE_PATH=data/session_notes

# Synthesis chunk size (number of tasks)
GENOMIC_RAG_SYNTHESIS_CHUNK_SIZE=8

# Note retention period (days)
GENOMIC_RAG_NOTE_RETENTION=30
```

### Runtime Configuration
```python
# Configure note-taking behavior
note_config = {
    "min_importance_score": 6.0,
    "max_notes_per_session": 100,
    "enable_compression": True,
    "synthesis_chunk_size": 8
}
```

This plan provides a comprehensive roadmap for implementing intelligent, selective note-taking in the agentic system, significantly enhancing its ability to handle complex, multi-step genomic analyses while maintaining memory persistence and synthesis quality.