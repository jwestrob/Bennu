Planner Composites (Bennu)

Summary
- Consolidates 14 fine-grained operators into 5 planner-visible composites:
  - FeatureDiscovery, GeneContext, PathwayProfile, ModuleProfile, EvidenceAndNext
- Composites are macro-expanded into existing primitives before execution.
- Runtime operator registry is unchanged; only a lightweight set of Materialize* built-ins was added to package outputs into typed records.

Key Changes
- Planner overlay restricts the operator catalog shown to the LLM to 5 composites. Primitives remain available to the executor.
- Macro expansion occurs in GenomicRAG before execute_plan, producing primitives + Materialize* steps.
- Materializers added in src/llm/mfp/operators/builtin.py:
  - MaterializeFeatureDiscovery → FeatureSet, ProteinSet, FacetSummary
  - MaterializeGeneContext → NeighborhoodSet, NeighborhoodSummary
  - MaterializePathwayProfile → PresentKOsByGenome, CompletenessMatrix, CompletenessSummary
  - MaterializeModuleProfile → ModuleRows, GlobalCounts
  - MaterializeEvidenceAndNext → EvidenceMetrics, FollowupPlan

Back-compat
- Legacy plans using primitives are rewritten to composites and then expanded (soft aliasing).
- Executor input/output contracts are unchanged for existing primitives.

Operator Hints
- Intent hints are embedded in planner constraints to bias selection:
  - neighborhood|context → GeneContext
  - pathway|completeness → PathwayProfile
  - CAZy|BGC → ModuleProfile
  - evidence|follow-up → EvidenceAndNext
  - PFAM|KO|search → FeatureDiscovery

Files
- Expansion: src/llm/mfp/planning/composites.py
- Types: src/llm/mfp/types.py
- Materializers: src/llm/mfp/operators/builtin.py (registered)
- Planner integration: src/llm/rag_system/core.py (overlay + macro expansion)

