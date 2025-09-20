# Stage 00 — Prepare Inputs

Validates FASTA files and organizes per‑genome directories for downstream stages.

Entry: `src/ingest/00_prepare_inputs.py`

Key behaviors:
- Accepts `.fasta`, `.fa`, `.fna` by default.
- Optionally copies or symlinks into `data/stage00_prepared/`.

