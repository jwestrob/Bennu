# Agent Architecture Revamp

## Objective
Transform the fixed DAG agent execution into a dynamic, early-exit capable system that prevents unnecessary expensive operations like `whole_genome_reader` when simple queries can be answered conclusively by cheaper tools.

## Core Problems Being Solved
1. **Fixed Task Graphs**: Current system creates predetermined 8-step plans that can't adapt to intermediate results
2. **No Early Exit**: System continues expensive operations even when database queries provide definitive answers
3. **Hard-coded Biology**: Biological targets baked into code instead of resolved from knowledge graph
4. **Poor Cost Management**: No eligibility gates to prevent expensive tools when unnecessary

## Architecture Changes

### Phase 1: Core Models & Settings
**Status**: Planned

**Files**:
- `src/llm/rag_system/models.py` - New Pydantic v2 models
  - Settings (BaseSettings) with environment-driven config
  - Intent enum (presence_absence, quantification, spatial_neighborhood, etc.)
  - Plan/PlanStep with guards, stop conditions, and cost tagging
  - ToolInput/ToolOutput contracts
  - EvidenceLedger with safe_summary() method

**Key Features**:
- All biological identifiers resolved at runtime (no hard-coding)
- Cost-aware planning with "cheap", "moderate", "expensive" tags
- Guard and stop condition system for dynamic control flow

### Phase 2: Schema Resolution
**Status**: Planned

**Files**:
- `src/llm/rag_system/schema_resolver.py` - Dynamic biological target resolution
  - `resolve_targets_from_query()` - Query KG for candidate biological entities
  - `has_anchor_entities()` - Check if sufficient context exists for spatial analysis
  - Data-driven matching using KG node labels/properties (no hardcoded biology)

### Phase 3: Dynamic Executor Loop
**Status**: Planned

**Files**:
- `src/llm/rag_system/agent_executor.py` - Replace fixed DAG with adaptive loop
  - While eligible steps exist: evaluate guards → execute → assess conclusiveness → early exit or replan
  - Budget enforcement (token/time limits)
  - EvidenceLedger persistence to JSON

- `src/llm/rag_system/core.py` - Update planner integration  
  - `plan_initial()` function replacing static step lists
  - Settings integration throughout

### Phase 4: Tool Eligibility & Policy
**Status**: Planned

**Files**:
- `src/llm/rag_system/agent_tool_selector.py` - Eligibility gates and cost-aware selection
  - Hard gate for `whole_genome_reader`: requires spatial intent AND anchor entities AND inconclusive cheap evidence
  - Pure functions for eligibility (no hardcoded biology)
  - Cost penalty system favoring cheaper tools

- `src/llm/rag_system/policy_engine.py` - Evidence assessment and conclusiveness
  - `evaluate_guard()` - Generic predicate evaluation
  - `assess()` - Determine conclusive_present/absent/inconclusive states
  - Generic rules based on resolver targets and tool metrics

### Phase 5: Tool Registry & Planning
**Status**: Planned

**Files**:
- `src/llm/rag_system/tool_registry.py` - Tool metadata and eligibility requirements
  - Tool descriptors with cost_tag, input/output models, default eligibility
  - `whole_genome_reader` marked as "expensive" with "requires_anchor" guard

- `src/llm/rag_system/dspy_signatures.py` - Structured planning outputs
  - JSON-only Plan conforming outputs
  - Strict parsing with error handling
  - No prose responses from planner

### Phase 6: Testing & Validation
**Status**: Planned

**Files**:
- `src/llm/rag_system/tests/test_early_exit.py` - Early exit behavior verification
  - Test: conclusive_absent verdict prevents `whole_genome_reader` invocation
  - Test: missing anchor entities makes `whole_genome_reader` ineligible
  - Mock-based testing with no real biological data

## Expected Outcomes

### Performance Improvements
- **Early Exit**: Simple presence/absence queries stop after database lookup
- **Cost Reduction**: Expensive tools only run when justified by evidence gaps
- **Budget Control**: Token and time limits prevent runaway processes

### Architectural Benefits
- **Dynamic Adaptation**: Plans adjust based on intermediate results
- **Type Safety**: Pydantic v2 models for all planning structures
- **Data-Driven Biology**: No hardcoded biological constants
- **Evidence Tracking**: Complete audit trail of tool invocations and decisions

### Behavioral Changes
- Rubisco presence query: database_query → conclusive_absent → immediate response (no whole_genome_reader)
- Complex spatial analysis: only runs when anchor entities exist and cheap evidence insufficient
- Budget exhaustion: graceful termination with best-effort summary

## Integration Analysis

### 1. Pydantic v2 Compatibility ✅
**Current Status**: `pydantic 2.11.7` with `pydantic-settings 2.10.1` installed
**Action**: Requirements satisfied (>=2.6,<3). No version conflicts expected.

### 2. Settings Integration Strategy 📋
**Current State**: `LLMConfig` (Pydantic v2 BaseModel) in `src/llm/config.py` with:
- Database configs (Neo4j, LanceDB)
- Model selection (cost_effective_model, premium_model) 
- RAG settings (similarity_threshold, max_results)
- Performance settings (timeout, caching)

**Integration Plan**:
```python
# src/llm/rag_system/models.py
class Settings(BaseSettings):
    # Absorb LLMConfig fields + new agent fields
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    # ... existing LLMConfig fields ...
    
    # New agent-specific fields
    max_budget_tokens: int = Field(default=100000)
    evidence_ledger_dir: str = Field(default="data/session_notes/{session_id}/evidence")
    expensive_tool_cost_hint: int = Field(default=1000)
    
    @classmethod
    def from_llm_config(cls, llm_config: LLMConfig, **overrides) -> 'Settings':
        """Bridge existing callers during migration"""
        
# src/llm/config.py - Add deprecation bridge
def get_unified_settings(**overrides) -> Settings:
    """Unified config - replaces LLMConfig usage"""
    warnings.warn("LLMConfig deprecated, use Settings", DeprecationWarning)
    llm_config = LLMConfig.from_env()
    return Settings.from_llm_config(llm_config, **overrides)
```

### 3. Evidence Ledger Storage ✅
**Target Location**: `data/session_notes/<SESSION_ID>/evidence/`
**Current Structure**: 
```
data/session_notes/<SESSION_ID>/
├── task_notes/           # Existing
├── synthesis_notes/      # Existing  
└── evidence/            # NEW - timestamped JSON files
    ├── query_001_20250109_143022.json
    └── query_002_20250109_143845.json
```

**Implementation**: Leverage existing session directory creation in `memory/memory_utils.py:ensure_session_directory()`

### 4. Model Allocation Independence ✅
**Current System**: Model allocation via `model_allocator` in `memory/model_allocation.py`
- Handles planning vs execution model selection
- Cost-optimized routing (NANO/MINI/PREMIUM tiers)

**New System Integration**:
- Agent executor receives **already-chosen** model from model_allocator
- New cost-aware **tool** gating operates independently 
- Tool cost tags ("cheap"/"expensive") separate from LLM model costs

### 5. Testing Strategy ✅
**Standalone Location**: `src/llm/rag_system/tests/`
```
src/llm/rag_system/tests/
├── __init__.py
├── test_early_exit.py       # Core early exit behavior
├── test_schema_resolver.py  # KG target resolution
├── test_policy_engine.py    # Evidence assessment
└── conftest.py             # Shared fixtures/mocks
```

**Integration**: Independent from `scripts/run_tests.py` (pipeline tests)

### 6. Backward Compatibility Strategy 📋
**Current Entry Points**:
- **CLI**: `src/cli.py:ask()` → calls genomic RAG system
- **Core**: `src/llm/rag_system/core.py:GenomicRAG.ask()` → main async method
- **Agentic**: `src/llm/rag_system/core.py:GenomicRAG.ask_agentic()` → agent executor

**Adapter Strategy**:
```python
# core.py - Preserve public API
async def ask(self, question: str) -> Dict[str, Any]:
    """
    Main method to answer genomic questions with agentic planning.
    
    DEPRECATED: Fixed-DAG semantics replaced with dynamic planning.
    Use ask_with_dynamic_planning() for new behavior.
    """
    # Adapter: old signature → new executor → old return format
    return await self._legacy_ask_adapter(question)

async def _legacy_ask_adapter(self, question: str) -> Dict[str, Any]:
    """Bridge old ask() calls to new plan-loop-finalize pipeline"""
    settings = Settings.from_llm_config(self.config)
    resolver = SchemaResolver(self.neo4j_processor, settings)
    
    # Run new pipeline
    plan = plan_initial(question, resolver)
    result = await execute_dynamic_loop(plan, settings)
    
    # Convert to legacy format
    return {
        "question": question,
        "answer": result.final_answer,
        "confidence": result.confidence,
        "citations": result.citations,
        "metadata": result.metadata
    }
```

### 7. File Change Impact 📋
**Zero-Change Files** (GPT-5 constraint):
- `src/cli.py` - CLI interface unchanged
- `src/llm/rag_system/memory/model_allocation.py` - Preserved 
- Any files not explicitly listed in GPT-5's task

**Minimal-Change Files**:
- `src/llm/config.py` - Add deprecation bridge only
- `src/llm/rag_system/core.py` - Add adapter, preserve signatures
- `src/llm/rag_system/agent_executor.py` - Replace DAG with loop

**New Files** (Clean slate):
- `src/llm/rag_system/models.py`
- `src/llm/rag_system/schema_resolver.py` 
- `src/llm/rag_system/tool_registry.py`
- `src/llm/rag_system/tests/test_early_exit.py`

## Implementation Phases

1. **Models & Settings** - Foundation types and configuration
2. **Schema Resolver** - Dynamic biological target resolution
3. **Loop Executor** - Replace fixed DAG with adaptive execution
4. **Eligibility & Policy** - Tool gates and evidence assessment
5. **Registry & Signatures** - Tool metadata and structured planning
6. **Testing & Validation** - Verify early exit and cost control

## Success Metrics
- [ ] Simple presence/absence queries complete without expensive tool invocation
- [ ] Spatial analyses only run with sufficient anchor entities
- [ ] All biological targets resolved dynamically from KG
- [ ] Evidence ledger tracks complete decision audit trail
- [ ] Budget limits enforced with graceful degradation
- [ ] Tests verify early exit behavior