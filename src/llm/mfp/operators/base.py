from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Callable, List, Optional


@dataclass
class OperatorContext:
    neo4j_driver: Any
    project_root: Optional[str] = None
    dataset_context: Optional[Dict[str, Any]] = None


@dataclass
class OperatorSpec:
    name: str
    inputs: List[str]
    outputs: List[str]
    params: Dict[str, str]  # name -> type description
    run: Callable[[OperatorContext, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
    description: str = ""


_REGISTRY: Dict[str, OperatorSpec] = {}


def register_operator(spec: OperatorSpec) -> None:
    _REGISTRY[spec.name] = spec


def get_operator(name: str) -> OperatorSpec:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown operator: {name}")
    return _REGISTRY[name]


def operator_catalog() -> Dict[str, Any]:
    items = []
    for spec in _REGISTRY.values():
        items.append({
            "name": spec.name,
            "inputs": spec.inputs,
            "outputs": spec.outputs,
            "params": spec.params,
            "description": spec.description or "",
        })
    return {"operators": items}
