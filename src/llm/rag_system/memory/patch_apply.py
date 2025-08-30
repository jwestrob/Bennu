from __future__ import annotations
import json
from typing import Any, List, Dict
from .doc_ast import Document


def _pointer_parts(path: str) -> List[str]:
    if not path or path == "/":
        return []
    if not path.startswith("/"):
        raise ValueError(f"Invalid JSON pointer: {path}")
    parts = path.split("/")[1:]
    # unescape
    return [p.replace("~1", "/").replace("~0", "~") for p in parts]


def _get_parent_and_key(obj: Any, parts: List[str]):
    cur = obj
    for p in parts[:-1]:
        if isinstance(cur, list):
            idx = int(p) if p != "-" else len(cur)
            if idx >= len(cur):
                raise IndexError("Pointer out of range")
            cur = cur[idx]
        else:
            if p not in cur:
                # create dict path if missing
                cur[p] = {}
            cur = cur[p]
    return cur, parts[-1] if parts else None


def _test(obj: Any, path: str, value: Any) -> bool:
    parts = _pointer_parts(path)
    cur = obj
    for p in parts:
        if isinstance(cur, list):
            idx = int(p) if p != "-" else len(cur)
            if idx >= len(cur):
                return False
            cur = cur[idx]
        else:
            if p not in cur:
                return False
            cur = cur[p]
    # JSON equality
    return json.dumps(cur, sort_keys=True) == json.dumps(value, sort_keys=True)


def apply_patch(doc: Document, env: Dict[str, Any]) -> Document:
    # Serialize to plain JSON/dict
    # Pydantic v1/v2 compatible serialization
    try:
        ast_json = json.loads(doc.json())
    except Exception:
        ast_json = json.loads(doc.model_dump_json())
    patch = env.get("patch") or []

    # Enforce at least one test op
    if not any((op.get("op") == "test") for op in patch):
        raise AssertionError("Missing test op in patch")

    for op in patch:
        t = op.get("op")
        path = op.get("path")
        val = op.get("value") if "value" in op else None
        parts = _pointer_parts(path)

        if t == "test":
            if not _test(ast_json, path, val):
                raise AssertionError(f"Test op failed at {path}")
            continue

        parent, key = _get_parent_and_key(ast_json, parts)
        if isinstance(parent, list):
            if t == "add":
                if key == "-":
                    parent.append(val)
                else:
                    idx = int(key)
                    parent.insert(idx, val)
            elif t == "replace":
                idx = int(key)
                parent[idx] = val
            elif t == "remove":
                idx = int(key)
                del parent[idx]
            else:
                raise ValueError(f"Unsupported op: {t}")
        else:
            if t == "add" or t == "replace":
                parent[key] = val
            elif t == "remove":
                if key in parent:
                    del parent[key]
                else:
                    # removing non-existent is a no-op for our purposes
                    pass
            else:
                raise ValueError(f"Unsupported op: {t}")

    try:
        return Document.parse_obj(ast_json)
    except Exception:
        return Document.model_validate(ast_json)
