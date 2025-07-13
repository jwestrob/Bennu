# Agent Upgrade Tasks - Enhanced Tool Availability & Smart Routing

## 🎯 **Project Overview**
Remove artificial limitations that prevent traditional mode from using external tools. Make tools available to both execution modes while keeping the binary fork for different complexity levels.

## 📋 **Task Tracking**

### **Phase 1: Core Improvements (High Priority)**

#### **G1: Remove Artificial Tool Limitations**
- [x] **T1.1**: Remove exclusion lists from `should_use_agentic_mode()` in `intelligent_routing.py`
- [x] **T1.2**: Enable external tool access in traditional mode (`_execute_traditional_query`)
- [x] **T1.3**: Add tool decision logic: when should traditional mode use tools?
- [x] **T1.4**: Implement tool integration in traditional mode execution flow
- [x] **T1.5**: Update routing logic to be based on complexity, not artificial exclusions

#### **G2: Smart Tool Integration**
- [x] **T1.6**: Add tool availability check in traditional mode
- [x] **T1.7**: Implement tool selection logic (when is `code_interpreter` helpful?)
- [x] **T1.8**: Add tool result integration into traditional answers
- [x] **T1.9**: Ensure tool failures don't break traditional queries

#### **G3: Simple Policy Engine**
- [x] **T1.10**: Create `PolicyConfig` dataclass for user preferences
- [x] **T1.11**: Implement basic policies:
  - [x] `max_tokens_per_query: int = 50000`
  - [x] `max_latency_seconds: int = 120`
  - [x] `allow_expensive_tools: bool = True`
  - [x] `max_refinement_depth: int = 3`
- [x] **T1.12**: Add policy integration points in execution paths
- [x] **T1.13**: Make policies user-configurable via config file

### **Phase 2: Smart Token Management (Medium Priority)**

#### **G4: Better Token Prediction**
- [ ] **T2.1**: Implement output token prediction to prevent truncation
- [ ] **T2.2**: Add token budget tracking per execution path
- [ ] **T2.3**: Implement smarter context compression triggers
- [ ] **T2.4**: Add token usage reporting and optimization suggestions

#### **G5: Improved Routing Logic**
- [ ] **T2.5**: Rewrite routing heuristics based on actual query characteristics
- [ ] **T2.6**: Add user preference integration for routing decisions
- [ ] **T2.7**: Implement dynamic mode switching based on intermediate results
- [ ] **T2.8**: Add routing explanation and transparency

### **Phase 3: Advanced Features (Low Priority)**

#### **G6: Tool Output Caching**
- [ ] **T3.1**: Implement tool output caching with `(model, prompt_hash)` keys
- [ ] **T3.2**: Add cache invalidation and management
- [ ] **T3.3**: Implement cache hit rate monitoring
- [ ] **T3.4**: Add cache size limits and LRU eviction

#### **G7: Execution Persistence**
- [ ] **T3.5**: Implement `runs/<uuid>/` directory structure for debugging
- [ ] **T3.6**: Add execution state persistence for complex queries
- [ ] **T3.7**: Implement basic checkpoint/resume for long-running tasks
- [ ] **T3.8**: Add execution replay for debugging

### **Phase 4: Testing & Validation**

#### **Testing**
- [ ] **T4.1**: Test traditional mode with tools enabled
- [ ] **T4.2**: Test tool failure handling in both modes
- [ ] **T4.3**: Test policy engine with different configurations
- [ ] **T4.4**: Performance regression testing
- [ ] **T4.5**: Validate that existing queries continue working

#### **Documentation**
- [ ] **T4.6**: Update CLAUDE.md with new tool availability
- [ ] **T4.7**: Document policy configuration options
- [ ] **T4.8**: Add troubleshooting guide for tool integration
- [ ] **T4.9**: Document routing decision logic

## 🎯 **Priority Order**
1. **High**: T1.1-T1.13 (Remove limitations, enable tools)
2. **Medium**: T2.1-T2.8 (Token management, routing improvements)
3. **Low**: T3.1-T4.9 (Caching, persistence, testing)

## 📊 **Success Criteria**
- [x] **Tool Availability**: External tools work in both traditional and agentic modes
- [x] **No Artificial Limits**: Routing based on complexity, not exclusion lists
- [x] **User Control**: Policies are configurable and respected
- [ ] **Performance**: Traditional mode with tools ≤ 1.5× current latency
- [x] **Reliability**: Tool failures don't break queries
- [ ] **Backwards Compatibility**: Existing queries continue working

## 🔍 **Key Design Decisions**

### **Tool Integration in Traditional Mode**
```python
# Traditional mode execution flow:
1. Generate Cypher query
2. Execute database query
3. Check if tools would be helpful
4. If helpful: execute relevant tools
5. Integrate tool results into answer
6. Return comprehensive response
```

### **Tool Selection Logic**
```python
def should_use_tool_in_traditional(query_result, question, tool_name):
    if tool_name == "code_interpreter":
        return len(query_result) > 100 and ("analyze" in question or "distribution" in question)
    elif tool_name == "literature_search":
        return "recent" in question or "literature" in question
    return False
```

### **Simple Policy Engine**
```python
@dataclass
class PolicyConfig:
    max_tokens_per_query: int = 50000
    max_latency_seconds: int = 120
    allow_expensive_tools: bool = True
    max_refinement_depth: int = 3
    prefer_traditional_mode: bool = True
```

## 📝 **Key Changes**

### **What We're Changing**:
1. **Remove exclusion lists** that artificially limit agentic mode
2. **Enable tools in traditional mode** when they add value
3. **Add simple policy engine** for user control
4. **Improve token management** to prevent truncation
5. **Keep binary fork** but base it on actual complexity

### **What We're Not Changing**:
1. **CLI interface** - remains the same
2. **TaskGraph architecture** - extend, don't replace
3. **Model allocation** - current system works fine
4. **Core execution flow** - enhance, don't rewrite

### **What We're Skipping**:
1. **Skills registry with YAML** - over-engineering
2. **OpenTelemetry spans** - premature optimization
3. **Complex chunking strategies** - current system works
4. **Prometheus metrics** - not needed yet

---

**Last Updated**: January 11, 2025  
**Current Status**: Phase 1 Complete ✅  
**Next Action**: 
- ✅ Phase 1 Complete: All core agent improvements implemented
- 🔧 **Refined Agentic Task Planning**: Enhanced PlannerAgent to create more comprehensive task plans
- 🔧 **Improved Model Allocation**: Better distribution of tasks across gpt-4.1-mini and o3 models
- 🔧 **Increased Chunking Threshold**: Raised from 1000 to 2000 items to allow more detailed analysis
- 🔧 **Optimized Token Limits**: Set `max_tokens_per_query` to 30K to match o3's context limit for proper chunking decisions