import json
import os
from pathlib import Path
import importlib


def test_jsonl_tracer_emits_line(tmp_path, monkeypatch):
    out = tmp_path / "trace.jsonl"
    monkeypatch.setenv("AGENT_TRACING", f"jsonl:{out}")

    # Reload module to reset singleton and pick up env var
    import src.llm.rag_system.tracing as tracing
    importlib.reload(tracing)

    tracer = tracing.get_tracer()
    tracer.emit("test.event", {"x": 1})

    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) >= 1
    data = json.loads(lines[-1])
    assert data["event"] == "test.event"
    assert data["data"]["x"] == 1

