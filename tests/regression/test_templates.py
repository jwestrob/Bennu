import pytest

from src.llm.kg.cypher_templates.registry import compile_query


def test_compile_protein_by_id():
    cypher, params = compile_query("protein_by_id", {"id": "P1"})
    assert "$id" in cypher
    assert params == {"id": "P1"}


def test_count_by_label_valid():
    cypher, params = compile_query("count_by_label", {"label": "Protein"})
    # No parameters expected; label is compiled into static Cypher safely
    assert "MATCH (n:Protein)" in cypher
    assert params == {}


def test_count_by_label_invalid():
    with pytest.raises(ValueError):
        compile_query("count_by_label", {"label": "Foo"})


def test_unknown_slot_rejected():
    with pytest.raises(ValueError):
        compile_query("protein_by_id", {"id": "P1", "oops": 1})


def test_missing_required_slot_rejected():
    with pytest.raises(ValueError):
        compile_query("protein_by_id", {})

