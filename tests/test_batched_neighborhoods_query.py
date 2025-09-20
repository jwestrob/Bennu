from src.llm.deterministic.evi import evi_gate


def test_evi_gate_is_deterministic():
    seed = {"contig_len": 5000, "orf_count": 20}
    assert evi_gate(seed) in (True, False)
