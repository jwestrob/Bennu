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
    "generate_session_id",
    "cleanup_old_sessions",
    "get_session_stats",
    "validate_note_structure"
]