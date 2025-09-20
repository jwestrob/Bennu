import importlib.util
import sys
from pathlib import Path
import pytest


def _load_module(rel_path: str, name: str):
    path = Path(__file__).resolve().parents[2] / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


registry = _load_module("src/llm/kg/cypher_templates/registry.py", "registry")
compile_query = registry.compile_query


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


def test_compile_proteins_by_genome():
    cypher, params = compile_query("proteins_by_genome", {"genome_id": "G1"})
    assert "$genome_id" in cypher and params == {"genome_id": "G1"}


def test_compile_genes_on_contig():
    cypher, params = compile_query("genes_on_contig", {"contig": "contig_1"})
    assert "$contig" in cypher and params == {"contig": "contig_1"}


def test_compile_proteins_with_pfam():
    cypher, params = compile_query("proteins_with_pfam", {"pfam": "PF00001"})
    assert "$pfam" in cypher and params == {"pfam": "PF00001"}


def test_compile_count_proteins_with_ko():
    cypher, params = compile_query("count_proteins_with_ko", {"ko": "K20469"})
    assert "$ko" in cypher and params == {"ko": "K20469"}



def test_limit_injection_when_provided():
    cypher, params = compile_query("proteins_with_ko", {"ko": "K20469", "limit": 50})
    assert "LIMIT $limit" in cypher
    assert params["limit"] == 50
