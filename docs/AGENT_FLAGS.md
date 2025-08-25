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

- `AGENT_ENABLE_LEGACY_TASKGRAPH` (default: 0)
  - When 1, exposes legacy TaskGraph types; otherwise fully disabled.

- `AGENT_ENABLE_LEGACY_SELECTORS` (default: 0)
  - When 1, enables legacy selectors behind the new router surface.

- `FAST_PATH_ENABLED` (default: 1)
  - Enable deterministic macro options for common queries; bypass per-step LLM routing when guards pass.

- `SKEPTIC_ENABLED` (default: 1)
  - Run post-batch auditor; may request a single mini-model adjudication only on anomalies.

- `USE_GRAMMAR_ROUTER` (default: 1)
  - Enable the grammar-driven intent router (Lark) for Macro Fast Path parsing with obligations; set to 0/false to fall back to minimal regex parser.

## Defaults Summary
- FSM runner: ON
- Templates-only DB: ON
- WholeGenomeReader is unaffected by DB limit flags; use `window_bp`/`loci_limit` for WGR.
