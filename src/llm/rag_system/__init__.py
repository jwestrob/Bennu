"""
Modular RAG system for genomic knowledge graph.
Clean interface with separated concerns for maintainability.
"""

import os
from .core import GenomicRAG
from .utils import EXAMPLE_GENOMIC_QUESTIONS, ResultStreamer, safe_log_data, setup_debug_logging, GenomicContext
from .agent_executor import UnifiedAgentExecutor
from .external_tools import AVAILABLE_TOOLS, literature_search, code_interpreter_tool
from .dspy_signatures import PlannerAgent, QueryClassifier, ContextRetriever, GenomicAnswerer
from .router import get_router, TwoStageRouter, RouterDecision

# Gate legacy TaskGraph exports behind a feature flag to aid quarantine.
_LEGACY_TASKGRAPH = os.getenv("AGENT_ENABLE_LEGACY_TASKGRAPH", "0") == "1"
if _LEGACY_TASKGRAPH:
    try:
        from .task_management import Task, TaskGraph, TaskStatus, TaskType  # type: ignore
    except Exception:  # pragma: no cover - keep package import stable if missing
        Task = TaskGraph = TaskStatus = TaskType = None  # type: ignore
else:
    Task = TaskGraph = TaskStatus = TaskType = None  # type: ignore

__all__ = [
    'GenomicRAG',
    'UnifiedAgentExecutor',
    'GenomicContext',
    'EXAMPLE_GENOMIC_QUESTIONS', 
    'Task',
    'TaskGraph',
    'TaskStatus',
    'TaskType',
    'AVAILABLE_TOOLS',
    'literature_search',
    'code_interpreter_tool',
    'PlannerAgent',
    'QueryClassifier', 
    'ContextRetriever',
    'GenomicAnswerer',
    'get_router',
    'TwoStageRouter',
    'RouterDecision',
    'ResultStreamer',
    'safe_log_data',
    'setup_debug_logging'
]
