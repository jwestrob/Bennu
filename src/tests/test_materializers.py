from src.llm.mfp.operators.base import OperatorContext, get_operator


def test_materialize_feature_discovery_shapes():
    spec = get_operator("MaterializeFeatureDiscovery")
    ctx = OperatorContext(neo4j_driver=None)
    inputs = {
        "discovered_proteins": [
            {"genome_id": "G1", "protein_id": "P1", "pfams": ["PF00001"]},
            {"genome_id": "G1", "protein_id": "P2", "kos": ["K00001"]},
        ],
        "pf_facet": {"facet_summary": {"pfams": [{"id": "PF00001", "count": 2}] }},
        "ko_facet": {"facet_summary": {"kos": [{"id": "K00001", "count": 1}] }},
    }
    out = spec.run(ctx, inputs, {"output_profile": "facet_summary"})
    assert set(out.keys()) >= {"FeatureSet", "ProteinSet", "FacetSummary"}
    assert isinstance(out["ProteinSet"].get("proteins"), list)


def test_materialize_gene_context_shapes():
    spec = get_operator("MaterializeGeneContext")
    ctx = OperatorContext(neo4j_driver=None)
    inputs = {
        "neighborhoods": [
            {"seed_protein_id": "P1", "neighbors": []}
        ],
        "neighborhood_summary": {"seeds": 1}
    }
    out = spec.run(ctx, inputs, {})
    assert "NeighborhoodSet" in out and "NeighborhoodSummary" in out


def test_materialize_pathway_profile_shapes():
    spec = get_operator("MaterializePathwayProfile")
    ctx = OperatorContext(neo4j_driver=None)
    inputs = {
        "present": {"G1": ["K00001"]},
        "pathway_completeness": [
            {"genome_id": "G1", "pathway_id": "map00710", "completeness": 0.5, "present_kos": 5, "total_kos": 10}
        ]
    }
    out = spec.run(ctx, inputs, {})
    assert set(out.keys()) >= {"PresentKOsByGenome", "CompletenessMatrix"}


def test_materialize_module_profile_shapes():
    spec = get_operator("MaterializeModuleProfile")
    ctx = OperatorContext(neo4j_driver=None)
    inputs = {"cazymes": [{"genome_id": "G1"}], "cazyme_family_counts": [{"family": "GH1", "count": 3}]}
    out = spec.run(ctx, inputs, {"module": "cazy"})
    assert set(out.keys()) == {"ModuleRows", "GlobalCounts"}


def test_materialize_evidence_and_next_shapes():
    spec = get_operator("MaterializeEvidenceAndNext")
    ctx = OperatorContext(neo4j_driver=None)
    inputs = {"evidence_metrics": {"rows": 0}, "followup_request": {"type": "followup_request"}}
    out = spec.run(ctx, inputs, {})
    assert set(out.keys()) == {"EvidenceMetrics", "FollowupPlan"}

