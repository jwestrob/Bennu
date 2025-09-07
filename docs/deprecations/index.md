# DEPRECATED Components

Use the recommended modules in Architecture and Operators sections. The following modules are retained for compatibility or are no longer referenced and should be avoided in new code.

## DEPRECATED: `src/llm/rag_system/intelligent_task_splitter.py`

- Status: Deprecated. Replaced by IntelligentChunkingManager.
- Rationale: Avoids recursive naming explosions; improves biological coherence and parallel synthesis.
- Behavior: Emits warnings and falls back to direct execution.

## DEPRECATED: `src/llm/domain_functions.py`

- Status: Deprecated. Not referenced by current code paths.
- Rationale: Domain function text is not required for neighborhoods, and enrichment from reference files has been removed from Stage 07.

## Compatibility Shim: `src/llm/rag_system.py`

- Status: Backward compatibility shim (emits DeprecationWarning on import).
- Recommendation: Import directly from `src/llm/rag_system/core.py` for `GenomicRAG` and related classes.

