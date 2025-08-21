import os

import pytest

from src.llm.rag_system.router.two_stage import TwoStageRouter, RouterDecision
from src.llm.rag_system.agent.tools.validate import validate_toolcall


def test_stage_a_routes_spatial_to_wgr():
    r = TwoStageRouter()
    q = "find operon boundaries on contig A"
    decision = r.route(q)
    assert isinstance(decision, RouterDecision)
    assert decision.tool == "whole_genome_reader"
    # Schema validation must pass
    ok, errs = validate_toolcall({"tool": decision.tool, "params": decision.params})
    assert ok, f"schema invalid: {errs}"
    # Deterministic defaults
    assert decision.params.get("window_bp") == 20000
    assert decision.params.get("loci_limit") == 2000


def test_stage_b_stubbed_returns_schema_valid_toolcall(monkeypatch):
    r = TwoStageRouter()

    def stub_route(question, context=None):
        return RouterDecision(
            tool="database_query",
            params={"template": "protein_by_id", "slots": {"id": "P12345"}},
            reasoning="stub",
        )

    # Monkeypatch the LLM router inside TwoStageRouter
    monkeypatch.setattr(r._llm, "route", stub_route)

    decision = r.route("what is protein P12345?")
    assert decision.tool == "database_query"
    ok, errs = validate_toolcall({"tool": decision.tool, "params": decision.params})
    assert ok, f"schema invalid: {errs}"
