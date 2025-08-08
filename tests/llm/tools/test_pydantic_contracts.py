import pytest
from pydantic import ValidationError

from llm.tools.io_models import (
    DatabaseQueryArgs,
    VectorSearchArgs,
    WholeGenomeReaderArgs,
    DatabaseQueryResult,
    VectorSearchResult,
    WholeGenomeReaderResult,
    Locus,
)
from llm.tools.selection_models import ToolSelection, validate_tool_params, validate_tool_output
from llm.tools.spec import build_tool_specs


class TestPydanticContracts:
    def test_valid_selection_and_param_validation(self):
        sel = ToolSelection(
            selected_tool="database_query",
            tool_parameters={"query": "MATCH (n) RETURN n LIMIT 5", "limit": 5},
            confidence=0.92,
            trace_id="abc123",
        )
        typed_params = validate_tool_params(sel)
        assert isinstance(typed_params, DatabaseQueryArgs)
        assert typed_params.limit == 5

    def test_invalid_params_raise_validation_error(self):
        sel = ToolSelection(
            selected_tool="vector_search",
            tool_parameters={"top_k": 10},  # missing required 'query'
            confidence=0.5,
        )
        with pytest.raises(ValidationError):
            _ = validate_tool_params(sel)

    def test_minimal_input_models_construct(self):
        dq = DatabaseQueryArgs(query="MATCH (n) RETURN count(n) AS c")
        vs = VectorSearchArgs(query="heme transporter", top_k=5)
        wgr = WholeGenomeReaderArgs(genome_id="GCF_00000000.1")

        assert dq.query.startswith("MATCH")
        assert vs.top_k == 5
        assert wgr.json_only is True

    def test_output_validation_smoke(self):
        # database query
        db_out = {"rows": [{"c": 10}], "summary": {"elapsed_ms": 12.3}}
        typed_db_out = validate_tool_output("database_query", db_out)
        assert isinstance(typed_db_out, DatabaseQueryResult)
        assert typed_db_out.rows[0]["c"] == 10

        # vector search
        vs_out = {
            "hits": [
                {"id": "prot:123", "score": 0.91, "metadata": {"organism": "E. coli"}},
                {"id": "prot:456", "score": 0.88, "metadata": {}},
            ]
        }
        typed_vs_out = validate_tool_output("vector_search", vs_out)
        assert isinstance(typed_vs_out, VectorSearchResult)
        assert len(typed_vs_out.hits) == 2

        # whole genome reader
        wgr_out = {
            "loci": [
                {
                    "contig_id": "contig_1",
                    "start": 100,
                    "end": 500,
                    "gene_ids": ["g1", "g2", "g3"],
                    "locus_label": "candidate prophage",
                }
            ]
        }
        typed_wgr_out = validate_tool_output("whole_genome_reader", wgr_out)
        assert isinstance(typed_wgr_out, WholeGenomeReaderResult)
        assert typed_wgr_out.loci[0].contig_id == "contig_1"

    def test_locus_coordinate_validation(self):
        with pytest.raises(ValidationError):
            _ = Locus(contig_id="c1", start=200, end=100, gene_ids=["g1"])
        ok = Locus(contig_id="c1", start=100, end=200, gene_ids=["g1"])
        assert ok.end >= ok.start

    def test_tool_specs_include_json_schemas(self):
        specs = build_tool_specs(["database_query", "vector_search", "whole_genome_reader"])
        assert len(specs) == 3
        first = specs[0]
        assert "properties" in first.io.input_schema
        assert "properties" in first.io.output_schema
