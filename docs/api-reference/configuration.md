# Configuration

## Environment Variables

- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`: Optional. Defaults to Docker engine (`bolt://localhost:7687`, auth none).
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`: Configure LLM access as needed; see `src/llm/config.py`.
- `SYSTEM_JOBS`: Default threads used by the CLI when `--threads` is not provided.

## Notes

- Stage 07 bulk import uses `neo4j-admin` and does not require a running Bolt session with auth.
- When credentials are provided, post‑import constraints/indexes are created using them; otherwise, Docker no‑auth path applies.

