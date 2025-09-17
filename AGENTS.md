# AGENTS.md — Project Conventions for Agents

Scope: Entire repository. These instructions constrain planner behavior, operator choices, and code changes made by agents working in this repo.

## Do Not Hard‑Code Biology

- NEVER inject biological identifiers (e.g., KO, PFAM, CAZy, gene symbols) directly into plans, prompts, code paths, or templates.
  - Examples of disallowed suggestions/changes: “Add TonB/ExbB/ExbD (K03832/K03550/K03551) to the KO list,” “Hard‑wire FeoA/B KOs into the pipeline,” “Manually append PF05031 to the query.”
- All identifier selection MUST be data‑driven:
  - Use catalog operators (e.g., SearchPfamCatalogFuzzy, SearchKoCatalogFuzzy) or explicit IDs provided by the user or dataset.
  - If curation is desired (allow/deny), it must come from data files under `data/reference/` (e.g., TSV/JSON), not from constants in code. These files are optional and should be consulted only when present.
  - Planners and operators MUST NOT invent, expand, or “helpfully” add identifiers that are not returned by a catalog search or user input.

## Planner & Composite Discipline

- Prefer composites that derive outputs deterministically from available data. Avoid heuristics that rely on implicit biological knowledge.
- When comparing features across genomes, use FeatureProfile (keyword → PFAM/KO catalogs → exact counts) or PathwayProfile (KO presence/completeness) as appropriate; do not augment identifier sets beyond catalog results.
- Avoid global queries by default. Use DatasetContext genome sampling unless the user explicitly requests full scope.

## UX / Reporting

- Favor compact matrices and labeled summaries over raw row dumps. If labels are needed, load them from `data/reference` (e.g., `pfam_id_desc.tsv`, `ko_list`). Do not embed labels in code.

## Pre‑Compaction Note — 2025‑09‑17

- Recent changes added FeatureProfile (per‑genome PFAM/KO counts), improved planner scoping, and portable Neo4j export. CRISPR arrays are integrated E2E. Future agents must adhere to the “No hard‑coded biology” rule above and keep identifier selection data‑driven.

