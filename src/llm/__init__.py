"""
LLM Integration Module for Genomic Knowledge Graph

This module provides natural language question answering over genomic data
by combining structured queries (Neo4j) with semantic search (LanceDB).
"""

<<<<<<< HEAD
from .config import LLMConfig
from .query_processor import Neo4jQueryProcessor, LanceDBQueryProcessor, HybridQueryProcessor
from .rag_system import GenomicRAG
from .cli import ask_question
from .task_repair_agent import TaskRepairAgent
from .repair_types import RepairResult, RepairStrategy, SchemaInfo
from .error_patterns import ErrorPatternRegistry

__all__ = [
    "LLMConfig", 
    "Neo4jQueryProcessor", 
    "LanceDBQueryProcessor", 
    "HybridQueryProcessor", 
    "GenomicRAG", 
    "ask_question",
    "TaskRepairAgent",
    "RepairResult",
    "RepairStrategy", 
    "SchemaInfo",
    "ErrorPatternRegistry"
]
=======
# Avoid importing heavy dependencies at package import time. Expose a lazy loader
# via __getattr__ to preserve public APIs while keeping submodule imports optional.

__all__ = [
    "LLMConfig",
    "Neo4jQueryProcessor",
    "LanceDBQueryProcessor",
    "HybridQueryProcessor",
    "GenomicRAG",
    "ask_question",
    "TaskRepairAgent",
    "RepairResult",
    "RepairStrategy",
    "SchemaInfo",
    "ErrorPatternRegistry",
]


def __getattr__(name: str):
    if name == "LLMConfig":
        from .config import LLMConfig
        return LLMConfig
    if name in ("Neo4jQueryProcessor", "LanceDBQueryProcessor", "HybridQueryProcessor"):
        from .query_processor import (
            Neo4jQueryProcessor,
            LanceDBQueryProcessor,
            HybridQueryProcessor,
        )
        return {
            "Neo4jQueryProcessor": Neo4jQueryProcessor,
            "LanceDBQueryProcessor": LanceDBQueryProcessor,
            "HybridQueryProcessor": HybridQueryProcessor,
        }[name]
    if name == "GenomicRAG":
        from .rag_system import GenomicRAG
        return GenomicRAG
    if name == "ask_question":
        from .cli import ask_question
        return ask_question
    if name in ("TaskRepairAgent",):
        from .task_repair_agent import TaskRepairAgent
        return TaskRepairAgent
    if name in ("RepairResult", "RepairStrategy", "SchemaInfo"):
        from .repair_types import RepairResult, RepairStrategy, SchemaInfo
        return {"RepairResult": RepairResult, "RepairStrategy": RepairStrategy, "SchemaInfo": SchemaInfo}[name]
    if name == "ErrorPatternRegistry":
        from .error_patterns import ErrorPatternRegistry
        return ErrorPatternRegistry
    raise AttributeError(name)
>>>>>>> feat/agent-router-typed
