# External Tools

Agent‑callable tools surfaced through the RAG system.

- whole_genome_reader: scans a genome for candidate loci with spatial markers; deterministic Stage‑A router target for spatial queries.
- genome_selector: deterministic/LLM‑assisted genome selection helper for narrowing scope.
- literature_search: retrieves and normalizes references for context.
- code_interpreter: secure sandboxed Python execution for quick data processing/plots.
- report_synthesis: convert session notes and tool results into structured report segments.
- neighborhood_extractor: convenience wrapper around neighborhood queries with templates.
- annotation_discovery: convenience wrapper around `AnnotationDiscovery` operator with safety checks.
- concept_discovery: lightweight concept extraction for Stage‑B routing hints.

All tool calls and parameters are captured in session notes. See Data → Layout for file locations.

