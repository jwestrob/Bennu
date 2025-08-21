from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


class Tracer:
    def emit(self, event: str, data: Dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class NoopTracer(Tracer):
    def emit(self, event: str, data: Dict[str, Any]) -> None:
        return


class JsonlTracer(Tracer):
    def __init__(self, path: Optional[str] = None) -> None:
        default_path = Path("logs/agent_traces.jsonl")
        self.path = Path(path) if path else default_path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, data: Dict[str, Any]) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event,
            "data": data,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


_TRACER: Optional[Tracer] = None


def get_tracer() -> Tracer:
    global _TRACER
    if _TRACER is not None:
        return _TRACER

    spec = os.getenv("AGENT_TRACING", "").strip()
    if not spec:
        _TRACER = NoopTracer()
        return _TRACER

    kind, _, arg = spec.partition(":")
    kind = kind.lower()
    if kind == "jsonl":
        _TRACER = JsonlTracer(arg or None)
    else:
        _TRACER = NoopTracer()
    return _TRACER

