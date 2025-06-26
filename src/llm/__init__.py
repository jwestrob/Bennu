"""
LLM Integration Module for Genomic Knowledge Graph

This module provides natural language question answering over genomic data
by combining structured queries (Neo4j) with semantic search (LanceDB).
"""

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