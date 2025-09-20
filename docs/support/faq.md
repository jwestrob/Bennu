# FAQ

- How do I include degree‑0 seeds in neighborhoods?
  - Set `include_degree_zero_seeds=true` in `NeighborhoodContext` params.
- How do I run only Stage 07?
  - `python -m src.cli build -f 7 -t 7 --force`
- Where are operator tool calls logged?
  - `data/session_notes/<sid>/synthesis_notes/tool_calls.json`.
- The DB already exists; can I avoid a rebuild?
  - Yes. Use Post‑Load Tuning to apply indexes and compute `[:NEXT]` and degrees, or rebuild Stage 07 for a clean state.
- How do I connect to Neo4j with auth?
  - Set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` and run the CLI or diagnostics.

