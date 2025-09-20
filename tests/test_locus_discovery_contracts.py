from src.llm.options.locus_discovery import LocusDiscoveryOption


class DummyDB:
    def run_template(self, name, params):
        # Do not fabricate biology; just assert the right template & params are requested.
        assert name in (
            "seeds_by_marker.cypher",
            "batched_neighborhoods_gated.cypher",
            "locus_schema_migration.cypher",
        )
        assert isinstance(params, dict)
        return []  # Contract test: caller must handle empty results gracefully


def test_contract_handles_insufficient_seeds():
    opt = LocusDiscoveryOption(db=DummyDB())
    cards, meta = opt.run(marker="integrase", N=5, k=4, nn=0)
    assert cards == []
    assert meta.get("escalate") is True

