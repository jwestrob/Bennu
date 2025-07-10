"""
Memory system for agentic RAG workflows.

This module provides persistent note-taking capabilities for multi-step genomic analyses,
enabling the agent to selectively record observations, insights, and cross-task connections
while maintaining memory persistence across token limits.
"""

from .note_keeper import NoteKeeper
from .note_schemas import (
    TaskNote,
    SynthesisNote,
    NotingDecisionResult,
    CrossTaskConnection,
    SessionMetadata,
    ConfidenceLevel
)
from .progressive_synthesizer import ProgressiveSynthesizer
from .multipart_synthesizer import MultiPartReportSynthesizer
from .task_based_synthesizer import TaskBasedSynthesizer
from .parallel_config import (
    set_parallel_profile,
    set_custom_parallel_config,
    get_parallel_config,
    print_parallel_status,
    print_parallel_profiles,
    estimate_parallel_speedup,
    set_conservative_parallel,
    set_balanced_parallel,
    set_aggressive_parallel,
    set_ultra_parallel
)
from .report_manager import ReportPlanner, ReportPlan, ReportChunk, ReportType, ChunkingStrategy
from .model_allocation import ModelAllocation, get_model_allocator, switch_to_premium_everywhere, switch_to_optimized_mode
from .model_config import (
    ModelConfigManager, 
    set_optimized_mode, 
    set_premium_mode, 
    set_testing_mode,
    get_current_mode,
    print_model_status,
    quick_switch_to_o3,
    quick_switch_to_optimized,
    quick_switch_to_testing
)
from .memory_utils import (
    generate_session_id,
    cleanup_old_sessions,
    get_session_stats,
    validate_note_structure
)

__all__ = [
    "NoteKeeper",
    "TaskNote",
    "SynthesisNote", 
    "NotingDecisionResult",
    "CrossTaskConnection",
    "SessionMetadata",
    "ConfidenceLevel",
    "ProgressiveSynthesizer",
    "MultiPartReportSynthesizer",
    "TaskBasedSynthesizer",
    "set_parallel_profile",
    "set_custom_parallel_config",
    "get_parallel_config",
    "print_parallel_status",
    "print_parallel_profiles",
    "estimate_parallel_speedup",
    "set_conservative_parallel",
    "set_balanced_parallel",
    "set_aggressive_parallel",
    "set_ultra_parallel",
    "ReportPlanner",
    "ReportPlan",
    "ReportChunk",
    "ReportType",
    "ChunkingStrategy",
    "ModelAllocation",
    "get_model_allocator",
    "switch_to_premium_everywhere",
    "switch_to_optimized_mode",
    "ModelConfigManager",
    "set_optimized_mode",
    "set_premium_mode", 
    "set_testing_mode",
    "get_current_mode",
    "print_model_status",
    "quick_switch_to_o3",
    "quick_switch_to_optimized",
    "quick_switch_to_testing",
    "generate_session_id",
    "cleanup_old_sessions",
    "get_session_stats",
    "validate_note_structure"
]