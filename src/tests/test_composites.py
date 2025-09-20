from src.llm.mfp.planning.composites import (
    expand_feature_discovery,
    expand_gene_context,
    expand_pathway_profile,
    expand_module_profile,
    expand_evidence_and_next,
)


def _ops(seq):
    return [s.get('op') for s in seq]


def test_expand_feature_discovery_keyword_basic():
    steps = expand_feature_discovery({
        "feature_selector": {"keyword": "rubisco"},
        "feature_types": ["pfam", "ko"],
        "limits": {"top_k": 10, "row_cap": 100}
    }, {"question": "q"})
    names = _ops(steps)
    assert names[:3] == ["SearchPfamCatalogFuzzy", "SearchKoCatalogFuzzy", "ExtractIdsFromCatalogHits"]
    assert "QueryProteinsByIds" in names
    assert names[-1] == "MaterializeFeatureDiscovery"


def test_expand_feature_discovery_ids_only():
    steps = expand_feature_discovery({
        "feature_selector": {"pfam_ids": ["PF00016"], "ko_ids": ["K01601"]},
        "limits": {"row_cap": 50}
    }, {"question": "q"})
    names = _ops(steps)
    # Should skip catalog searches
    assert "SearchPfamCatalogFuzzy" not in names
    assert "SearchKoCatalogFuzzy" not in names
    assert names[0] == "QueryProteinsByIds"
    assert names[-1] == "MaterializeFeatureDiscovery"


def test_expand_gene_context_from_ids():
    steps = expand_gene_context({
        "seeds": {"protein_ids": ["P1", "P2"]},
        "context": {"seeds_limit": 5, "limit": 50}
    }, {"question": "q"})
    names = _ops(steps)
    assert names[0] == "NeighborhoodContext"
    assert names[-1] == "MaterializeGeneContext"


def test_expand_pathway_profile_basic():
    steps = expand_pathway_profile({"genomes": ["G1", "G2"]}, {"question": "q"})
    names = _ops(steps)
    assert names == [
        "FetchPresentKOs",
        "LoadKoPathwayTotals",
        "ComputePathwayCompleteness",
        "MaterializePathwayProfile",
    ]


def test_expand_module_profile_cazy():
    steps = expand_module_profile({"module": "cazy", "genomes": ["G1"]}, {"question": "q"})
    names = _ops(steps)
    assert names[0] == "QueryCazymesByGenome"
    assert names[-1] == "MaterializeModuleProfile"


def test_expand_evidence_and_next_basic():
    steps = expand_evidence_and_next({"min_rows": 3, "top_n": 5}, {"question": "test"})
    names = _ops(steps)
    assert names == ["AssessEvidence", "ProposeFollowup", "MaterializeEvidenceAndNext"]

