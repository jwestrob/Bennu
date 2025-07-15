"""
NoteKeeper class for managing persistent notes in agentic workflows.

Handles storage, retrieval, and management of task notes and synthesis notes
with session-based organization and intelligent note-taking decisions.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .note_schemas import (
    TaskNote,
    SynthesisNote,
    SessionMetadata,
    NotingDecisionResult,
    CrossTaskConnection,
    ConfidenceLevel,
    SessionStats,
    NoteSearchResult
)
from .memory_utils import (
    generate_session_id,
    ensure_session_directory,
    validate_note_structure,
    save_note_to_file,
    load_note_from_file,
    get_session_stats,
    search_notes,
    get_cross_task_connections,
    estimate_storage_usage
)

logger = logging.getLogger(__name__)


class NoteKeeper:
    """
    Manages persistent notes for agentic RAG workflows.
    
    Provides session-based storage of task observations, cross-task connections,
    and progressive synthesis results with intelligent note-taking decisions.
    """
    
    def __init__(self, session_id: Optional[str] = None, base_path: str = "data/session_notes"):
        """
        Initialize NoteKeeper.
        
        Args:
            session_id: Optional session ID. If None, generates new ID.
            base_path: Base directory for note storage.
        """
        self.session_id = session_id or generate_session_id()
        self.base_path = Path(base_path)
        self.session_path = self.base_path / self.session_id
        self.task_notes_path = self.session_path / "task_notes"
        self.synthesis_notes_path = self.session_path / "synthesis_notes"
        
        # Ensure directory structure exists
        ensure_session_directory(self.session_path)
        
        # Initialize session metadata
        self.metadata = self._load_or_create_metadata()
        
        # Initialize discovery results accumulator  
        from .session_results_accumulator import SessionResultsAccumulator
        self.results_accumulator = SessionResultsAccumulator(self.session_id, self.session_path)
        
        # Cache for frequently accessed notes
        self._note_cache = {}
        
        logger.info(f"NoteKeeper initialized for session: {self.session_id}")
    
    def _load_or_create_metadata(self) -> SessionMetadata:
        """Load existing metadata or create new session metadata."""
        metadata_path = self.session_path / "session_metadata.json"
        
        if metadata_path.exists():
            try:
                metadata_data = load_note_from_file(metadata_path)
                if metadata_data:
                    return SessionMetadata(**metadata_data)
            except Exception as e:
                logger.warning(f"Failed to load existing metadata: {e}")
        
        # Create new metadata
        metadata = SessionMetadata(
            session_id=self.session_id,
            original_question="",
            execution_mode="agentic"
        )
        
        # Save metadata with temporary assignment
        self.metadata = metadata
        self._save_metadata()
        return metadata
    
    def _save_metadata(self) -> None:
        """Save session metadata to file."""
        metadata_path = self.session_path / "session_metadata.json"
        metadata_dict = self.metadata.dict()
        metadata_dict["last_updated"] = datetime.utcnow().isoformat()
        
        save_note_to_file(metadata_dict, metadata_path)
    
    def set_session_context(self, question: str, execution_mode: str = "agentic") -> None:
        """
        Set session context information.
        
        Args:
            question: Original user question
            execution_mode: Execution mode (agentic/traditional)
        """
        self.metadata.original_question = question
        self.metadata.execution_mode = execution_mode
        self._save_metadata()
    
    def record_task_notes(self, 
                         task_id: str,
                         task_type: str,
                         description: str,
                         decision_result: NotingDecisionResult,
                         observations: List[str],
                         key_findings: List[str],
                         quantitative_data: Dict[str, Any] = None,
                         cross_connections: List[CrossTaskConnection] = None,
                         confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
                         execution_time: float = 0.0,
                         tokens_used: int = 0) -> bool:
        """
        Record notes for a completed task.
        
        Args:
            task_id: Unique task identifier
            task_type: Type of task
            description: Task description
            decision_result: Result of note-taking decision
            observations: Key observations
            key_findings: Important findings
            quantitative_data: Numerical data
            cross_connections: Connections to other tasks
            confidence: Confidence level
            execution_time: Task execution time
            tokens_used: Tokens used for task
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create task note
            task_note = TaskNote(
                task_id=task_id,
                task_type=task_type,
                description=description,
                note_decision=decision_result,
                observations=observations,
                key_findings=key_findings,
                quantitative_data=quantitative_data or {},
                cross_task_connections=cross_connections or [],
                confidence_level=confidence,
                execution_time=execution_time,
                tokens_used=tokens_used
            )
            
            # Save to file
            note_path = self.task_notes_path / f"{task_id}_notes.json"
            success = save_note_to_file(task_note.dict(), note_path)
            
            if success:
                # Update session metadata
                self.metadata.tasks_with_notes += 1
                self.metadata.total_tokens_used += tokens_used
                self.metadata.note_generation_time += execution_time
                self._save_metadata()
                
                # Cache the note
                self._note_cache[task_id] = task_note
                
                logger.info(f"Recorded notes for task {task_id}")
                return True
            else:
                logger.error(f"Failed to save notes for task {task_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error recording task notes: {e}")
            return False
    
    def get_task_notes(self, task_id: str) -> Optional[TaskNote]:
        """
        Retrieve notes for a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            TaskNote if found, None otherwise
        """
        # Check cache first
        if task_id in self._note_cache:
            return self._note_cache[task_id]
        
        # Load from file
        note_path = self.task_notes_path / f"{task_id}_notes.json"
        note_data = load_note_from_file(note_path)
        
        if note_data:
            try:
                task_note = TaskNote(**note_data)
                self._note_cache[task_id] = task_note
                return task_note
            except Exception as e:
                logger.error(f"Failed to parse task note {task_id}: {e}")
        
        return None
    
    def get_all_task_notes(self) -> List[TaskNote]:
        """
        Get all task notes for the session.
        
        Returns:
            List of TaskNote objects
        """
        notes = []
        
        if not self.task_notes_path.exists():
            return notes
        
        for note_file in self.task_notes_path.glob("*_notes.json"):
            note_data = load_note_from_file(note_file)
            if note_data:
                try:
                    notes.append(TaskNote(**note_data))
                except Exception as e:
                    logger.error(f"Failed to parse task note {note_file.name}: {e}")
        
        return notes
    
    def record_synthesis_notes(self,
                              chunk_id: str,
                              source_tasks: List[str],
                              chunk_theme: str,
                              integrated_findings: List[str],
                              cross_task_synthesis: List[Dict[str, Any]],
                              emergent_insights: List[str],
                              confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
                              tokens_used: int = 0,
                              compression_applied: bool = False) -> bool:
        """
        Record notes from synthesis process.
        
        Args:
            chunk_id: Unique chunk identifier
            source_tasks: Task IDs that contributed to synthesis
            chunk_theme: Main theme of the synthesis
            integrated_findings: Cross-task findings
            cross_task_synthesis: Synthesis of connections
            emergent_insights: New insights from synthesis
            confidence: Confidence level
            tokens_used: Tokens used for synthesis
            compression_applied: Whether compression was used
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create synthesis note
            synthesis_note = SynthesisNote(
                chunk_id=chunk_id,
                source_tasks=source_tasks,
                chunk_theme=chunk_theme,
                integrated_findings=integrated_findings,
                cross_task_synthesis=cross_task_synthesis,
                emergent_insights=emergent_insights,
                synthesis_confidence=confidence,
                tokens_used=tokens_used,
                compression_applied=compression_applied
            )
            
            # Save to file
            note_path = self.synthesis_notes_path / f"{chunk_id}_synthesis.json"
            success = save_note_to_file(synthesis_note.dict(), note_path)
            
            if success:
                # Update session metadata
                self.metadata.synthesis_chunks += 1
                self.metadata.total_tokens_used += tokens_used
                self._save_metadata()
                
                logger.info(f"Recorded synthesis notes for chunk {chunk_id}")
                return True
            else:
                logger.error(f"Failed to save synthesis notes for chunk {chunk_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error recording synthesis notes: {e}")
            return False
    
    def get_synthesis_notes(self, chunk_id: str) -> Optional[SynthesisNote]:
        """
        Retrieve synthesis notes for a specific chunk.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            SynthesisNote if found, None otherwise
        """
        note_path = self.synthesis_notes_path / f"{chunk_id}_synthesis.json"
        note_data = load_note_from_file(note_path)
        
        if note_data:
            try:
                return SynthesisNote(**note_data)
            except Exception as e:
                logger.error(f"Failed to parse synthesis note {chunk_id}: {e}")
        
        return None
    
    def get_all_synthesis_notes(self) -> List[SynthesisNote]:
        """
        Get all synthesis notes for the session.
        
        Returns:
            List of SynthesisNote objects
        """
        notes = []
        
        if not self.synthesis_notes_path.exists():
            return notes
        
        for note_file in self.synthesis_notes_path.glob("*_synthesis.json"):
            note_data = load_note_from_file(note_file)
            if note_data:
                try:
                    notes.append(SynthesisNote(**note_data))
                except Exception as e:
                    logger.error(f"Failed to parse synthesis note {note_file.name}: {e}")
        
        return notes
    
    def get_session_summary(self) -> str:
        """
        Get a summary of the current session for context.
        
        Returns:
            String summary of session progress
        """
        task_notes = self.get_all_task_notes()
        synthesis_notes = self.get_all_synthesis_notes()
        
        summary_parts = []
        
        # Session overview
        summary_parts.append(f"Session {self.session_id}")
        summary_parts.append(f"Question: {self.metadata.original_question}")
        summary_parts.append(f"Tasks with notes: {len(task_notes)}")
        summary_parts.append(f"Synthesis chunks: {len(synthesis_notes)}")
        
        # Recent key findings
        if task_notes:
            recent_findings = []
            for note in task_notes[-3:]:  # Last 3 tasks
                if note.key_findings:
                    recent_findings.extend(note.key_findings[:2])  # Top 2 findings per task
            
            if recent_findings:
                summary_parts.append(f"Recent key findings: {'; '.join(recent_findings)}")
        
        # Cross-task connections
        total_connections = sum(len(note.cross_task_connections) for note in task_notes)
        if total_connections > 0:
            summary_parts.append(f"Cross-task connections: {total_connections}")
        
        return " | ".join(summary_parts)
    
    def search_session_notes(self, query: str, note_type: str = "all") -> List[NoteSearchResult]:
        """
        Search notes in the current session.
        
        Args:
            query: Search query
            note_type: Type of notes to search ('task', 'synthesis', or 'all')
            
        Returns:
            List of search results
        """
        return search_notes(self.session_path, query, note_type)
    
    def get_related_notes(self, task_id: str) -> List[TaskNote]:
        """
        Get notes from tasks related to the specified task.
        
        Args:
            task_id: Task ID to find related notes for
            
        Returns:
            List of related TaskNote objects
        """
        related_notes = []
        all_notes = self.get_all_task_notes()
        
        for note in all_notes:
            # Check if this note connects to the specified task
            for connection in note.cross_task_connections:
                if connection.connected_task == task_id:
                    related_notes.append(note)
                    break
        
        return related_notes
    
    def get_session_statistics(self) -> SessionStats:
        """
        Get comprehensive statistics for the current session.
        
        Returns:
            SessionStats object with calculated metrics
        """
        return get_session_stats(self.session_path)
    
    def cleanup_session(self) -> bool:
        """
        Clean up the current session (remove all files).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import shutil
            shutil.rmtree(self.session_path)
            logger.info(f"Cleaned up session: {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup session: {e}")
            return False
    
    def export_session_data(self, export_path: Path) -> bool:
        """
        Export session data to a single JSON file.
        
        Args:
            export_path: Path to export file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            export_data = {
                "session_metadata": self.metadata.dict(),
                "task_notes": [note.dict() for note in self.get_all_task_notes()],
                "synthesis_notes": [note.dict() for note in self.get_all_synthesis_notes()],
                "session_stats": self.get_session_statistics().dict()
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported session data to: {export_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export session data: {e}")
            return False
    
    def get_storage_usage(self) -> Dict[str, Any]:
        """
        Get storage usage information for the session.
        
        Returns:
            Dictionary with storage statistics
        """
        return estimate_storage_usage(self.session_path)