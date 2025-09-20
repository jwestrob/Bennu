import json
import os
from pathlib import Path
import importlib

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


def test_jsonl_tracer_emits_line(tmp_path, monkeypatch):
    out = tmp_path / "trace.jsonl"
    monkeypatch.setenv("AGENT_TRACING", f"jsonl:{out}")

    # Reload module to reset singleton and pick up env var
    tracing = _load_module("src/llm/rag_system/tracing.py", "tracing_mod")

    tracer = tracing.get_tracer()
    tracer.emit("test.event", {"x": 1})

    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) >= 1
    data = json.loads(lines[-1])
    assert data["event"] == "test.event"
    assert data["data"]["x"] == 1
