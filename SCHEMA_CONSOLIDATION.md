# Schema Consolidation Plan

## Executive Summary

The current system has two parallel query architectures - a new "schema-locked" system and a "legacy" dynamic system. This creates unnecessary complexity, maintenance burden, and failure modes. This plan consolidates both systems into a single, unified architecture that uses schema validation throughout.

## Current Architecture Problems

### Dual System Complexity
- **Schema-locked path**: `detector_registry.py` + `query_builder.py` + `SchemaMap`
- **Legacy path**: `schema_resolver.py` + old agent logic + hardcoded queries
- **Broken handoff**: Schema-locked "preprocessing" falls back to legacy system when it fails

### Root Issues
1. **Preprocessing mindset**: Schema-locked system treated as input filter rather than core engine
2. **No shared schema**: Legacy system bypasses `SchemaMap` validation entirely  
3. **Duplicate logic**: Two different concept resolution systems
4. **Fragile fallback**: Legacy system has its own bugs (`'NoneType' object has no attribute 'hybrid_processor'`)

## Target Architecture

### Single Unified Path
```
User Query → Agent Executor → Schema-Locked Components → Neo4j/LanceDB
```

### Core Components (Schema-First)
- **`SchemaMap`**: Single source of truth for database structure
- **`DetectorRegistry`**: Concept resolution using schema validation
- **`QueryBuilder`**: Parameterized Cypher generation with schema checks
- **`AgentExecutor`**: Tool orchestration using schema-locked components only

## Implementation Plan

### Phase 1: Fix Schema-Locked System Issues (Priority: Critical)

#### 1.1 Fix DetectorRegistry Tokenization
**File**: `src/llm/rag_system/detector_registry.py`
- **Issue**: `_normalize_phrase()` removes "protein" from biological terms
- **Issue**: Tokenization corruption (`"integrase protein"` → `"integra eprotein"`)
- **Fix**: 
  - Remove "protein" from stop words list
  - Identify and fix text processing bug causing character corruption
  - Add unit tests for phrase normalization

#### 1.2 Improve Concept Matching
**File**: `src/llm/rag_system/detector_registry.py`  
- **Issue**: Exact substring matching too brittle (`"integrase"` should match `"integrase/recombinase XerC"`)
- **Fix**: 
  - Add fuzzy matching for KO/PFAM descriptions
  - Support partial word matching with biological awareness
  - Add stemming for biological terms (integrase/integrases)

#### 1.3 Test Schema-Locked Detection
**Goal**: Verify that `"integrase protein"` resolves to `K03733`, `K04763`, `K14059`
- Create integration test with actual Neo4j data
- Test complex queries like the original LanceDB use case
- Validate that schema-locked system can handle full query workflow

### Phase 2: Eliminate Legacy System (Priority: High)

#### 2.1 Remove Legacy Components
**Files to Remove**:
- `src/llm/rag_system/schema_resolver.py` 
- Legacy query generation logic in `agent_executor.py`
- Old concept extraction methods

**Files to Update**:
- `src/llm/rag_system/agent_executor.py`: Remove fallback logic, use only schema-locked path

#### 2.2 Migrate Agent Tools to Schema-First
**Tool Updates**:
- **`database_query`**: Use `QueryBuilder` for all Cypher generation
- **`whole_genome_reader`**: Use `DetectorRegistry` for concept resolution  
- **`code_interpreter`**: Maintain current functionality (no schema dependency)
- **`literature_search`**: Maintain current functionality (no schema dependency)

#### 2.3 Update Agent Execution Flow
**File**: `src/llm/rag_system/agent_executor.py`
- Remove preprocessing/fallback split
- Make `_execute_database_query()` always use `QueryBuilder`
- Ensure `SchemaMap` validation in all database operations

### Phase 3: Validation & Testing (Priority: Medium)

#### 3.1 Integration Testing
- Test original failing query: integrase proteins + LanceDB search
- Test complex spatial queries with schema validation
- Test error handling without legacy fallback

#### 3.2 Performance Validation  
- Ensure consolidated system performs as well as legacy
- Validate schema overhead is minimal
- Test memory usage with large result sets

#### 3.3 Schema Coverage Testing
- Verify all existing queries work with schema validation
- Test edge cases (missing properties, new annotations)
- Validate schema drift detection works properly

## Risk Analysis

### High Risk Items
1. **Breaking existing queries**: Legacy system may handle edge cases schema-locked doesn't
2. **Performance regression**: Schema validation overhead in critical paths  
3. **Schema completeness**: Current schema may not cover all query patterns

### Mitigation Strategies
1. **Comprehensive testing**: Test all existing query patterns before removing legacy
2. **Gradual rollout**: Keep legacy as emergency fallback during Phase 2
3. **Schema expansion**: Add missing properties/relationships as needed

## Success Criteria

### Technical Metrics
- [ ] Original integrase LanceDB query works end-to-end
- [ ] All existing query patterns work with schema-locked system
- [ ] No performance regression > 20% for common queries
- [ ] Schema validation catches all database structure issues

### Code Quality Metrics  
- [ ] Single query path through system (no fallbacks)
- [ ] All database queries use `SchemaMap` validation
- [ ] Reduced codebase size (remove duplicate logic)
- [ ] Improved error messages (schema-aware)

## Timeline

- **Phase 1**: 1-2 days (fix tokenization, test basic resolution)
- **Phase 2**: 2-3 days (remove legacy, integrate schema-locked)  
- **Phase 3**: 1-2 days (comprehensive testing)
- **Total**: 4-7 days

## Dependencies

### Critical Path
1. Fix DetectorRegistry tokenization issues
2. Verify schema-locked system handles complex queries
3. Remove legacy fallback logic
4. Test integration with actual use cases

### External Dependencies
- Neo4j database schema remains stable during migration
- LanceDB integration unaffected (no schema dependencies)

---

## Implementation Order

1. **Start with DetectorRegistry fixes** (immediate impact)
2. **Test schema-locked system in isolation** (validate approach)
3. **Remove legacy fallback** (eliminate dual paths)
4. **Full system integration testing** (ensure nothing breaks)

This consolidation will result in a simpler, more maintainable system that uses schema validation consistently throughout the query pipeline.