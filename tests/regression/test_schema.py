import importlib.util
import sys
from pathlib import Path
import jsonschema


def _load_module(rel_path: str, name: str):
    path = Path(__file__).resolve().parents[2] / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


schemas = _load_module("src/llm/rag_system/agent/tools/schemas.py", "schemas_mod")


def test_schema_valid_database_query():
    obj = {
        "tool": "database_query",
        "params": {
            "template": "protein_by_id",
            "slots": {"id": "P12345"}
        }
    }
    # Validate against JSON Schema directly to avoid package import complexity
    jsonschema.validate(instance=obj, schema=schemas.TOOLCALL_JSON_SCHEMA)


def test_schema_rejects_unknown_fields():
    obj = {
        "tool": "database_query",
        "params": {
            "template": "protein_by_id",
            "slots": {"id": "P12345"},
            "extra": 1
        }
    }
    try:
        jsonschema.validate(instance=obj, schema=schemas.TOOLCALL_JSON_SCHEMA)
        assert False, "expected validation failure for unknown field"
    except jsonschema.ValidationError:
        pass
