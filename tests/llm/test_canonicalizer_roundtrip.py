from __future__ import annotations

from src.llm.intent.models import CanonicalIntent
from src.llm.intent.dsl_renderer import render_to_dsl


def test_marker_canonicalization_roundtrip():
    ci = CanonicalIntent(
        task="FIND_LOCI_BY_MARKER",
        n=5,
        find_by_marker={"marker": "integrase", "flank_k": 2},
        actions=[],
    )
    dsl = render_to_dsl(ci)
    assert "FIND 5 LOCI WITH integrase ± 2" in dsl


def test_signature_head_rendering():
    ci = CanonicalIntent(
        task="FIND_LOCI_BY_SIGNATURE",
        n=5,
        find_by_signature={"signature_name": "PROPHAGE", "flank_k": 5},
        actions=[],
    )
    dsl = render_to_dsl(ci)
    assert "WITH PROPHAGE ± 5" in dsl

