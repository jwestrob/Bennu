# Modular RAG Architecture

The LLM stack is split into clear modules for planning, execution, and synthesis. Plans are validated and tool calls are logged to enable deterministic, auditable runs.

## Planner and Validation

- Plans must provide a selection signal for `AnnotationDiscovery` via `params.keyword` (or `q`). Plans omitting this are rejected.
- Wire entity IDs through `inputs` (e.g., `inputs:{"pfam_ids":"pfam_ids"}`) instead of hard-coding.
- Keyword hygiene for gene/subunit questions: prefer direct subunit terms, ≤2 concise synonyms; avoid broad “-like” analogs unless exploring.

## Neighborhoods

- `NeighborhoodContext` requires explicit seeds via `inputs.discovered_proteins` or params (`protein_ids`, `seed_pfam_ids`, `seed_ko_ids`). No implicit fallbacks.
- Degree-aware seed filter excludes contig-isolated seeds (`Gene.nextDegree = 0`) by default; override with `include_degree_zero_seeds=true`.
- Adjacency (k-step) and flanking (±N by contig order) are APOC-free and index-aware.

## Tool Call Capture

- All tool invocations are recorded to `data/session_notes/<sid>/synthesis_notes/tool_calls.json`, including parameters and a preview of outputs.

