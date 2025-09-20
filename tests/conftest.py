import importlib.util
from pathlib import Path


def load_module(rel_path: str, name: str):
    """Load a module from a relative file path without importing package __init__ files."""
    path = Path(rel_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

