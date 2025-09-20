# Planner Signatures & Rules

Validation is strict to ensure deterministic, auditably correct plans.

## Mandatory Rules

- `AnnotationDiscovery` must include `params.keyword` (or `q`).
- `NeighborhoodContext` must receive seeds via `inputs.discovered_proteins` (bound rowset) or via params (`protein_ids`, `seed_pfam_ids`, `seed_ko_ids`).

## Wiring

- Chain IDs via `inputs` (e.g., `inputs:{"pfam_ids":"pfam_ids"}`), not hard-coded.
- For small seed sets (≤ 12), prefer `output_profile='rowset'` and pass all seeds to neighborhoods.

## Defaults (guidance)

- `pfam_tokens_top_n=30`, `ko_tokens_top_n=30`, `pfam_candidate_cap=200`, `pfam_top_k=20`, `ko_top_k=20`.
- For targeted rows, use small budgets (≈ 50–200) to reduce latency.

