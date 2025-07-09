"""
Utility functions for the memory system.

Provides helper functions for session management, note validation,
storage operations, and performance monitoring.
"""

import json
import uuid
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

from .note_schemas import (
    TaskNote, 
    SynthesisNote, 
    SessionMetadata, 
    SessionStats,
    NoteSearchResult
)

logger = logging.getLogger(__name__)


def generate_session_id() -> str:
    """Generate a unique session identifier."""
    return str(uuid.uuid4())


def ensure_session_directory(session_path: Path) -> None:
    """Ensure session directory structure exists."""
    session_path.mkdir(parents=True, exist_ok=True)
    (session_path / "task_notes").mkdir(exist_ok=True)
    (session_path / "synthesis_notes").mkdir(exist_ok=True)


def validate_note_structure(note_data: Dict[str, Any], note_type: str) -> bool:
    """
    Validate note structure against schema.
    
    Args:
        note_data: Note data to validate
        note_type: Type of note ('task' or 'synthesis')
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if note_type == "task":
            TaskNote(**note_data)
        elif note_type == "synthesis":
            SynthesisNote(**note_data)
        else:
            logger.error(f"Unknown note type: {note_type}")
            return False
        return True
    except Exception as e:
        logger.error(f"Note validation failed: {e}")
        return False


def save_note_to_file(note: Dict[str, Any], file_path: Path) -> bool:
    """
    Save note to JSON file with error handling.
    
    Args:
        note: Note data to save
        file_path: Path to save file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(note, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"Failed to save note to {file_path}: {e}")
        return False


def load_note_from_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load note from JSON file with error handling.
    
    Args:
        file_path: Path to load file
        
    Returns:
        Note data if successful, None otherwise
    """
    try:
        if not file_path.exists():
            return None
        
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load note from {file_path}: {e}")
        return None


def get_session_stats(session_path: Path) -> SessionStats:
    """
    Calculate statistics for a session.
    
    Args:
        session_path: Path to session directory
        
    Returns:
        SessionStats object with calculated metrics
    """
    task_notes_path = session_path / "task_notes"
    synthesis_notes_path = session_path / "synthesis_notes"
    
    # Count task notes
    task_notes = list(task_notes_path.glob("*_notes.json")) if task_notes_path.exists() else []
    total_tasks = len(task_notes)
    
    # Count synthesis notes
    synthesis_notes = list(synthesis_notes_path.glob("*.json")) if synthesis_notes_path.exists() else []
    synthesis_chunks = len(synthesis_notes)
    
    # Calculate aggregated metrics
    total_observations = 0
    total_findings = 0
    total_connections = 0
    total_tokens = 0
    total_execution_time = 0.0
    
    for note_file in task_notes:
        note_data = load_note_from_file(note_file)
        if note_data:
            total_observations += len(note_data.get("observations", []))
            total_findings += len(note_data.get("key_findings", []))
            total_connections += len(note_data.get("cross_task_connections", []))
            total_tokens += note_data.get("tokens_used", 0)
            total_execution_time += note_data.get("execution_time", 0.0)
    
    # Add synthesis tokens
    for synthesis_file in synthesis_notes:
        synthesis_data = load_note_from_file(synthesis_file)
        if synthesis_data:
            total_tokens += synthesis_data.get("tokens_used", 0)
    
    # Load session metadata for session_id
    metadata_path = session_path / "session_metadata.json"
    metadata = load_note_from_file(metadata_path)
    session_id = metadata.get("session_id", "unknown") if metadata else "unknown"
    
    return SessionStats(
        session_id=session_id,
        total_tasks=total_tasks,
        tasks_with_notes=total_tasks,  # All task notes represent tasks with notes
        total_observations=total_observations,
        total_findings=total_findings,
        total_connections=total_connections,
        synthesis_chunks=synthesis_chunks,
        total_tokens=total_tokens,
        execution_time=total_execution_time,
        note_efficiency=1.0 if total_tasks > 0 else 0.0
    )


def cleanup_old_sessions(base_path: Path, retention_days: int = 30) -> int:
    """
    Clean up old session directories.
    
    Args:
        base_path: Base path for session storage
        retention_days: Number of days to retain sessions
        
    Returns:
        Number of sessions cleaned up
    """
    if not base_path.exists():
        return 0
    
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    cleaned_count = 0
    
    for session_dir in base_path.iterdir():
        if not session_dir.is_dir():
            continue
            
        # Check session metadata for creation date
        metadata_path = session_dir / "session_metadata.json"
        metadata = load_note_from_file(metadata_path)
        
        if metadata:
            created_str = metadata.get("created_timestamp")
            if created_str:
                try:
                    created_date = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                    if created_date.replace(tzinfo=None) < cutoff_date:
                        logger.info(f"Cleaning up old session: {session_dir.name}")
                        shutil.rmtree(session_dir)
                        cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to parse date for session {session_dir.name}: {e}")
    
    return cleaned_count


def search_notes(session_path: Path, query: str, note_type: str = "all") -> List[NoteSearchResult]:
    """
    Search notes for relevant content.
    
    Args:
        session_path: Path to session directory
        query: Search query
        note_type: Type of notes to search ('task', 'synthesis', or 'all')
        
    Returns:
        List of search results sorted by relevance
    """
    results = []
    query_lower = query.lower()
    
    # Search task notes
    if note_type in ["task", "all"]:
        task_notes_path = session_path / "task_notes"
        if task_notes_path.exists():
            for note_file in task_notes_path.glob("*_notes.json"):
                note_data = load_note_from_file(note_file)
                if note_data:
                    relevance = _calculate_relevance(note_data, query_lower)
                    if relevance > 0:
                        results.append(NoteSearchResult(
                            note_id=note_file.stem,
                            note_type="task",
                            relevance_score=relevance,
                            content_summary=_generate_content_summary(note_data)
                        ))
    
    # Search synthesis notes
    if note_type in ["synthesis", "all"]:
        synthesis_notes_path = session_path / "synthesis_notes"
        if synthesis_notes_path.exists():
            for note_file in synthesis_notes_path.glob("*.json"):
                note_data = load_note_from_file(note_file)
                if note_data:
                    relevance = _calculate_relevance(note_data, query_lower)
                    if relevance > 0:
                        results.append(NoteSearchResult(
                            note_id=note_file.stem,
                            note_type="synthesis",
                            relevance_score=relevance,
                            content_summary=_generate_content_summary(note_data)
                        ))
    
    # Sort by relevance score
    results.sort(key=lambda x: x.relevance_score, reverse=True)
    return results


def _calculate_relevance(note_data: Dict[str, Any], query: str) -> float:
    """
    Calculate relevance score for a note based on query.
    
    Args:
        note_data: Note data to evaluate
        query: Search query (lowercase)
        
    Returns:
        Relevance score between 0.0 and 1.0
    """
    score = 0.0
    
    # Search in different fields with different weights
    fields_to_search = [
        ("observations", 1.0),
        ("key_findings", 1.2),
        ("integrated_findings", 1.2),
        ("emergent_insights", 1.5),
        ("description", 0.8),
        ("chunk_theme", 1.0)
    ]
    
    total_weight = 0.0
    for field, weight in fields_to_search:
        if field in note_data:
            field_data = note_data[field]
            if isinstance(field_data, list):
                text = " ".join(field_data).lower()
            else:
                text = str(field_data).lower()
            
            if query in text:
                score += weight
                total_weight += weight
    
    # Normalize score
    if total_weight > 0:
        score = min(score / total_weight, 1.0)
    
    return score


def _generate_content_summary(note_data: Dict[str, Any]) -> str:
    """Generate a brief summary of note content."""
    if "key_findings" in note_data and note_data["key_findings"]:
        return note_data["key_findings"][0][:100] + "..."
    elif "observations" in note_data and note_data["observations"]:
        return note_data["observations"][0][:100] + "..."
    elif "integrated_findings" in note_data and note_data["integrated_findings"]:
        return note_data["integrated_findings"][0][:100] + "..."
    elif "description" in note_data:
        return note_data["description"][:100] + "..."
    else:
        return "No summary available"


def get_cross_task_connections(session_path: Path, task_id: str) -> List[Dict[str, Any]]:
    """
    Get all cross-task connections for a specific task.
    
    Args:
        session_path: Path to session directory
        task_id: ID of the task to find connections for
        
    Returns:
        List of connection dictionaries
    """
    connections = []
    task_notes_path = session_path / "task_notes"
    
    if not task_notes_path.exists():
        return connections
    
    for note_file in task_notes_path.glob("*_notes.json"):
        note_data = load_note_from_file(note_file)
        if note_data and "cross_task_connections" in note_data:
            for connection in note_data["cross_task_connections"]:
                if connection.get("connected_task") == task_id:
                    connections.append({
                        "source_task": note_data["task_id"],
                        "connection": connection
                    })
    
    return connections


def estimate_storage_usage(session_path: Path) -> Dict[str, Any]:
    """
    Estimate storage usage for a session.
    
    Args:
        session_path: Path to session directory
        
    Returns:
        Dictionary with storage statistics
    """
    if not session_path.exists():
        return {"total_size": 0, "file_count": 0, "breakdown": {}}
    
    total_size = 0
    file_count = 0
    breakdown = {}
    
    for category in ["task_notes", "synthesis_notes"]:
        category_path = session_path / category
        if category_path.exists():
            category_size = 0
            category_files = 0
            
            for file_path in category_path.glob("*.json"):
                file_size = file_path.stat().st_size
                category_size += file_size
                category_files += 1
            
            breakdown[category] = {
                "size": category_size,
                "files": category_files
            }
            total_size += category_size
            file_count += category_files
    
    # Add session metadata
    metadata_path = session_path / "session_metadata.json"
    if metadata_path.exists():
        metadata_size = metadata_path.stat().st_size
        breakdown["metadata"] = {
            "size": metadata_size,
            "files": 1
        }
        total_size += metadata_size
        file_count += 1
    
    return {
        "total_size": total_size,
        "file_count": file_count,
        "breakdown": breakdown
    }