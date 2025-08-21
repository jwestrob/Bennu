import importlib
import importlib.util
import sys
from pathlib import Path


def _import_package(module_name: str):
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module(module_name)


def test_format_context_snapshot(tmp_path):
    # Minimal stubs to import core and utils without heavy deps
    try:
        core = _import_package("src.llm.rag_system.core")
        utils = _import_package("src.llm.rag_system.utils")
    except Exception as e:
        import pytest as _pytest
        _pytest.skip(f"Skipping snapshot due to package import constraints: {e}")

    # Build a small GenomicContext and snapshot the formatted output
    ctx = utils.GenomicContext(
        structured_data=[
            {"protein_id": "protein:ABC123", "genome_id": "genome:G1", "kegg": ["K20469"]},
            {"protein_id": "protein:XYZ789", "genome_id": "genome:G2", "kegg": ["K00001"]},
        ],
        semantic_data=[],
        metadata={"analysis_type": "FUNCTIONAL_ANNOTATION", "tool_used": "database_query", "template": "protein_by_id"},
        query_time=0.01,
    )
    formatted = core.GenomicRAG._format_context(object(), ctx)

    snap = Path(__file__).parent / "snapshots/format_context_basic.txt"
    snap.parent.mkdir(parents=True, exist_ok=True)
    if not snap.exists():
        snap.write_text(formatted)
    assert formatted == snap.read_text()

