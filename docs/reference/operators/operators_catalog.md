# Operators Catalog

Summary of available MFP operators with purposes and key I/O.

- FetchPresentKOs: present KO IDs per genome (global or filtered).
  - inputs: —; outputs: `present`; params: `genome_ids`.
- LoadKoPathwayTotals: KO→pathway totals (native mode from local lists).
  - inputs: —; outputs: `totals`; params: `source`.
- ComputePathwayCompleteness: per‑genome KEGG pathway completeness.
  - inputs: `present`, `totals`; outputs: `pathway_completeness`; params: `pathways`, `min_completeness`.
- QueryBGCsByGenome: list predicted BGC clusters (global/per‑genome).
  - inputs: —; outputs: `bgcs`; params: `genome_id`, `genome_ids`.
- QueryCazymesByGenome: list CAZyme‑annotated proteins (global/per‑genome).
  - inputs: —; outputs: `cazymes`; params: `genome_id`, `genome_ids`.
- CountCazymeFamilies: global count of CAZy families across proteins.
  - inputs: —; outputs: `cazyme_family_counts`.
- AnnotationDiscovery: keyword → IDs → exact retrieval → KO/PFAM summaries.
  - inputs: optional `pfam_ids`, `ko_ids`; outputs: `facet_summary`, `selection_metadata`, `discovered_proteins`.
- NeighborhoodContext: adjacency (k-step) or flanking (±N) neighborhoods.
  - inputs: `discovered_proteins`; outputs: `neighborhoods`, `neighborhood_summary`, `neighborhood_macro_result`, `seeds_used`.
- MaterializeFeatureDiscovery: package discovery outputs into typed sets.
  - inputs: `discovered_proteins`, `pf_facet`, `ko_facet`; outputs: `FeatureSet`, `ProteinSet`, `FacetSummary`.
- MaterializeGeneContext: package neighborhoods into typed records.
  - inputs: `neighborhoods`, `neighborhood_summary`; outputs: `NeighborhoodSet`, `NeighborhoodSummary`.
- MaterializePathwayProfile: package KO presence/completeness into typed records.
  - inputs: `present`, `pathway_completeness`; outputs: `PresentKOsByGenome`, `CompletenessMatrix`, `CompletenessSummary`.
- MaterializeModuleProfile: package CAZy or BGC module rows.
  - inputs: `cazymes`, `cazyme_family_counts`, `bgcs`; outputs: `ModuleRows`, `GlobalCounts`; params: `module`.

See dedicated pages for `AnnotationDiscovery` and `NeighborhoodContext` for parameter details and examples.

