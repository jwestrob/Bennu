# Planner Guidance

Strict validation and wiring rules ensure deterministic, auditable plans.

## Required Practices

- `AnnotationDiscovery` MUST set `params.keyword` (or `q`). Plans missing this are rejected.
- Wire IDs via `inputs` (e.g., `inputs:{"pfam_ids":"pfam_ids"}`) rather than hard‑coding.

## Neighborhoods

- `NeighborhoodContext` requires explicit seeds:
  - Bind a small rowset from `AnnotationDiscovery` and pass via `inputs.discovered_proteins`, or
  - Provide `protein_ids`, or
  - Provide `seed_pfam_ids`/`seed_ko_ids` with an explicit seed budget when needed.
- Default behavior excludes seeds with `nextDegree=0`; override via `include_degree_zero_seeds=true`.

## Keyword Hygiene

- For gene/subunit context, prefer direct subunit terms; avoid broad class analogs and “‑like” phrases unless exploring.
- Keep synonyms ≤ 2 to reduce noise.

## Numeric Defaults (guidance)

- `pfam_tokens_top_n=30`, `ko_tokens_top_n=30`, `pfam_candidate_cap=200`, `pfam_top_k=20`, `ko_top_k=20`.
- For targeted rowsets, use small budgets (≈ 50–200) to control latency.

