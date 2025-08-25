import re
from src.llm.options.router import parse_macro_intent


def test_parse_macro_intent_basic():
    q = "Find 5 loci with terminase and ±4 flanking genes; then 2 closest relatives"
    intent = parse_macro_intent(q)
    assert intent is not None
    assert intent.option_name == "LocusDiscovery"
    assert intent.params["N"] == 5
    assert intent.params["marker"] == "terminase"
    assert intent.params["k"] == 4
    assert intent.params["nn"] == 2

