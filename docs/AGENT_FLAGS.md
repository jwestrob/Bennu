# Agent Flags and Strict Modes

- `AGENT_FSM_STRICT` (default: 1)
  - Controls FSM runner usage. When 1, the FSM-governed agent loop is used exclusively.

- `AGENT_DB_TEMPLATES_ONLY` (default: 1)
  - Enables strict traditional DB mode, mapping questions → named templates and blocking free-form LLM Cypher.

- `AGENT_DEFAULT_DB_LIMIT` (default: 100)
  - Default limit for list-style DB templates when no limit is supplied. Clamped 1–5000.
  - Policy engine is consulted first for database_query max results; this env var is a fallback.

- `AGENT_ENABLE_GDS` (default: 0)
  - Enables curated GDS wrappers (`src/llm/rag_system/gds_wrappers.py`).
  - CALL remains disabled for LLM; wrappers use safe Cypher expansions or precomputed indexes.

- `LANGFUSE_API_KEY`, `LANGSMITH_API_KEY` (optional)
  - If set, MultiTracer adds stub adapters (no external deps) alongside JSONL tracing.

## Defaults Summary
- FSM runner: ON
- Templates-only DB: ON
- WholeGenomeReader is unaffected by DB limit flags; use `window_bp`/`loci_limit` for WGR.

