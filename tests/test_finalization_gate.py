from src.llm.options.intent_grammar import parse_intent
from src.llm.options.obligations import ObligationLedger


def test_unmet_obligation_blocks_finalize():
    it = parse_intent("Find five loci with integrases then LanceDB nearest neighbors")
    assert it is not None
    ledger = ObligationLedger.from_intent(it)
    unmet = ledger.unmet()
    assert "lancedb_knn" in unmet

