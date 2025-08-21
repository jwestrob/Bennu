import importlib.util
import sys
from pathlib import Path


def _load_module(rel_path: str, name: str):
    path = Path(__file__).resolve().parents[2] / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


mapper = _load_module("src/llm/rag_system/db_template_mapper.py", "db_template_mapper")


def test_map_protein_by_id():
    tpl = mapper.map_question_to_template("show protein:ABC123 context")
    assert tpl == ("protein_by_id", {"id": "protein:ABC123"})


def test_map_ko():
    tpl = mapper.map_question_to_template("list K20469 proteins")
    assert tpl == ("proteins_with_ko", {"ko": "K20469"})


def test_map_cazy():
    tpl = mapper.map_question_to_template("find GH13 family members")
    assert tpl == ("cazy_family", {"family": "GH13"})


def test_map_pathway():
    tpl = mapper.map_question_to_template("proteins in map00500")
    assert tpl == ("pathway_membership", {"pathway": "map00500"})


def test_no_mapping():
    assert mapper.map_question_to_template("how many genomes?") is None


def test_map_genome_id():
    tpl = mapper.map_question_to_template("list proteins from genome:PLM0_123")
    assert tpl == ("proteins_by_genome", {"genome_id": "genome:PLM0_123"})


def test_map_contig_id():
    tpl = mapper.map_question_to_template("genes on contig:contig_42")
    assert tpl == ("genes_on_contig", {"contig": "contig:contig_42"})
