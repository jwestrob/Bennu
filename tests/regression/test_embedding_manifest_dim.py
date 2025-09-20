import json
import os
import importlib.util
import sys
from pathlib import Path


def _load_module(rel_path: str, name: str):
    root = Path(__file__).resolve().parents[2]
    path = root / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_embedding_manifest_present_and_well_formed(monkeypatch):
    # Optional parity check; skip unless LANCEDB_PATH set
    lancedb_path = os.getenv("LANCEDB_PATH")
    if not lancedb_path:
        import pytest
        pytest.skip("LANCEDB_PATH not set; skipping manifest checks")

    mod = _load_module("src/llm/embedding/runtime_embedder.py", "runtime_embedder_mod")
    manifest_path = mod.find_embedding_manifest(lancedb_path)
    if not manifest_path:
        import pytest
        pytest.skip("embedding_manifest.json not found; skipping")

    data = json.loads(Path(manifest_path).read_text())
    assert isinstance(data, dict)
    assert isinstance(data.get("embedding_dim"), int) and data["embedding_dim"] > 0


def test_embedding_dimension_parity_optional(monkeypatch):
    # Opt-in heavier check; requires transformers+torch and RUN_EMBED_DIM_CHECK=1
    if os.getenv("RUN_EMBED_DIM_CHECK") != "1":
        import pytest
        pytest.skip("RUN_EMBED_DIM_CHECK!=1; skipping parity check")

    lancedb_path = os.getenv("LANCEDB_PATH")
    if not lancedb_path:
        import pytest
        pytest.skip("LANCEDB_PATH not set; skipping")

    mod = _load_module("src/llm/embedding/runtime_embedder.py", "runtime_embedder_mod")
    manifest_path = mod.find_embedding_manifest(lancedb_path)
    if not manifest_path:
        import pytest
        pytest.skip("embedding_manifest.json not found; skipping")

    cfg = mod.ESM2RuntimeEmbedder.load_manifest(manifest_path)
    try:
        embedder = mod.ESM2RuntimeEmbedder(cfg)
    except Exception as e:  # missing deps or model
        import pytest
        pytest.skip(f"embedder init skipped due to deps: {e}")

    data = json.loads(Path(manifest_path).read_text())
    expected = int(data.get("embedding_dim", 0))
    if expected:
        assert embedder.embedding_dim == expected

