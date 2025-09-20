import importlib
import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path
import os
import asyncio


def _stub_module(name: str, module):
    sys.modules[name] = module


def _import_package(module_name: str):
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module(module_name)


def test_strict_templates_protein_id_executes_template_and_synthesizes(monkeypatch):
    # Ensure strict mode is on
    monkeypatch.setenv("AGENT_DB_TEMPLATES_ONLY", "1")

    # Stub heavy deps before importing core
    # Minimal 'rich.console'
    class _Console:
        def print(self, *a, **k):
            pass

    rich_mod = SimpleNamespace()
    rich_console_mod = SimpleNamespace(Console=_Console)
    _stub_module("rich", rich_mod)
    _stub_module("rich.console", rich_console_mod)
    # Allow dspy import to fail (handled in core), but stub lancedb/neo4j to avoid import errors downstream
    _stub_module("lancedb", SimpleNamespace(connect=lambda path: SimpleNamespace(open_table=lambda *_a, **_k: None)))
    class _DummySession:
        def run(self, *a, **k):
            return []
    class _DummyDriver:
        def session(self):
            return _DummySession()
    _stub_module("neo4j", SimpleNamespace(GraphDatabase=SimpleNamespace(driver=lambda *a, **k: _DummyDriver())))
    # Stub router import to avoid circulars in core import
    import types as _types
    mod_router = _types.ModuleType('src.llm.rag_system.router')
    def _get_router():
        return SimpleNamespace(route=lambda question, context=None: SimpleNamespace(tool='database_query', params={'template':'protein_by_id','slots':{'id':'protein:ABC123'}}))
    mod_router.get_router = _get_router
    class _TwoStageRouter: ...
    class _RouterDecision: ...
    mod_router.TwoStageRouter = _TwoStageRouter
    mod_router.RouterDecision = _RouterDecision
    sys.modules['src.llm.rag_system.router'] = mod_router

    try:
        core = _import_package("src.llm.rag_system.core")
    except Exception as e:
        import pytest as _pytest
        _pytest.skip(f"Skipping due to package import constraints: {e}")

    # Build a dummy self with the methods used inside _execute_from_templates_only
    called = {}

    class DummyNeo4j:
        async def execute_named_template(self, name, slots):
            called["template"] = name
            called["slots"] = slots
            return SimpleNamespace(results=[{"ok": 1}], execution_time=0.01)

    class DummySelf:
        neo4j_processor = DummyNeo4j()
        def _format_context(self, ctx):
            return "FMT"
        async def _synthesize_answer(self, question, formatted_context, query_type, analysis_type):
            return {"answer": "ok", "query_metadata": {"query_type": query_type, "analysis_type": analysis_type}}

    dummy = DummySelf()

    # Call the helper directly
    res = asyncio.run(core.GenomicRAG._execute_from_templates_only(dummy, "show protein:ABC123"))
    assert res["answer"] == "ok"
    assert called["template"] == "protein_by_id"
    assert called["slots"] == {"id": "protein:ABC123"}
    assert res["query_metadata"]["query_type"] == "template:protein_by_id"


def test_strict_templates_no_mapping_returns_none(monkeypatch):
    monkeypatch.setenv("AGENT_DB_TEMPLATES_ONLY", "1")
    # Stub minimal modules as above
    class _Console:
        def print(self, *a, **k):
            pass
    _stub_module("rich.console", SimpleNamespace(Console=_Console))
    _stub_module("rich", SimpleNamespace())
    _stub_module("lancedb", SimpleNamespace(connect=lambda path: SimpleNamespace(open_table=lambda *_a, **_k: None)))
    class _DummySession:
        def run(self, *a, **k):
            return []
    class _DummyDriver:
        def session(self):
            return _DummySession()
    _stub_module("neo4j", SimpleNamespace(GraphDatabase=SimpleNamespace(driver=lambda *a, **k: _DummyDriver())))
    # Stub router import to avoid circulars in core import
    import types as _types
    mod_router = _types.ModuleType('src.llm.rag_system.router')
    def _get_router():
        return SimpleNamespace(route=lambda question, context=None: SimpleNamespace(tool='database_query', params={'template':'protein_by_id','slots':{'id':'protein:ABC123'}}))
    mod_router.get_router = _get_router
    class _TwoStageRouter: ...
    class _RouterDecision: ...
    mod_router.TwoStageRouter = _TwoStageRouter
    mod_router.RouterDecision = _RouterDecision
    sys.modules['src.llm.rag_system.router'] = mod_router

    core = _import_package("src.llm.rag_system.core")

    class DummySelf:
        neo4j_processor = SimpleNamespace()
        def _format_context(self, ctx):
            return "FMT"
        async def _synthesize_answer(self, *a, **k):
            return {}

    dummy = DummySelf()
    res = asyncio.run(core.GenomicRAG._execute_from_templates_only(dummy, "how many genomes?"))
    assert res is None