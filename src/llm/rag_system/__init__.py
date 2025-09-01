"""
Modular RAG system for genomic knowledge graph.
Clean interface with separated concerns for maintainability.
"""

import os
from .core import GenomicRAG
from .utils import EXAMPLE_GENOMIC_QUESTIONS, ResultStreamer, safe_log_data, setup_debug_logging, GenomicContext
# Heavy modules imported lazily to avoid optional dependency issues
try:
    from .agent_executor import UnifiedAgentExecutor  # type: ignore
except Exception:  # pragma: no cover
    UnifiedAgentExecutor = None  # type: ignore
try:
    from .external_tools import AVAILABLE_TOOLS, literature_search, code_interpreter_tool  # type: ignore
except Exception:  # pragma: no cover
    AVAILABLE_TOOLS = []  # type: ignore
    literature_search = None  # type: ignore
    code_interpreter_tool = None  # type: ignore
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
