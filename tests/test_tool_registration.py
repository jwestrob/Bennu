from src.llm.rag_system.external_tools import AVAILABLE_TOOLS


def test_lancedb_tool_registered_symbol():
    assert "lancedb_knn" in AVAILABLE_TOOLS
    assert AVAILABLE_TOOLS["lancedb_knn"] is not None

