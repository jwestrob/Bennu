# Operator: AnnotationDiscovery

Catalog search → candidate IDs → bound rowsets for downstream operators.

## Requirements

- Plans must include `params.keyword` (or `q`). Calling with only formatting params (`output_profile`, `group_by`, `fields`) returns empty results and fails validation.

## Parameters (highlights)

- `keyword`: Required search keyword for catalog lookup.
- `output_profile`: `facet_summary` (default) or `rowset` for per‑protein rows.
- Candidate budgeting: `pfam_tokens_top_n`, `ko_tokens_top_n`, `pfam_candidate_cap`, `pfam_top_k`, `ko_top_k`.

## Wiring

- Chain IDs via `inputs` to keep plans explicit, e.g., `inputs:{"pfam_ids":"pfam_ids"}`.
- For small, focused neighborhoods, request a compact rowset and bind it (e.g., `<= 12` seeds) for `NeighborhoodContext`.

## Outputs

- `facet_summary`: KO/PFAM group counts and top‑k IDs per group (facet‑first default).
- `selection_metadata`: compact description of the selection and budgets used.
- `discovered_proteins`: optional rowset of protein anchors when `output_profile='rowset'`.
