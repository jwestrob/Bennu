"""
Pydantic schemas for note-taking system data structures.

Defines structured schemas for task notes, synthesis notes, and related metadata
with validation, serialization, and type safety.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class ConnectionType(str, Enum):
    """Types of cross-task connections."""
    VALIDATES = "validates"
    INFORMS = "informs"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    DEPENDS_ON = "depends_on"
    SUPPORTS = "supports"


class ConfidenceLevel(str, Enum):
    """Confidence levels for findings."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CrossTaskConnection(BaseModel):
    """Represents a connection between tasks."""
    connected_task: str = Field(description="ID of the connected task")
    connection_type: ConnectionType = Field(description="Type of connection")
    description: str = Field(description="Description of the connection")
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)


class NotingDecisionResult(BaseModel):
    """Result of the note-taking decision process."""
    should_record: bool = Field(description="Whether to record notes for this task")
    reasoning: str = Field(description="Explanation of the decision")
    importance_score: float = Field(ge=1.0, le=10.0, description="Importance score 1-10")
    
    @validator('importance_score')
    def validate_importance_score(cls, v):
        if not 1.0 <= v <= 10.0:
            raise ValueError('Importance score must be between 1.0 and 10.0')
        return v


class TaskNote(BaseModel):
    """Schema for individual task notes."""
    task_id: str = Field(description="Unique identifier for the task")
    task_type: str = Field(description="Type of task (e.g., 'atomic_query', 'tool_call')")
    description: str = Field(description="Task description")
    execution_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Note decision metadata
    note_decision: NotingDecisionResult = Field(description="Decision result for note-taking")
    
    # Core note content
    observations: List[str] = Field(default_factory=list, description="Key observations from task")
    key_findings: List[str] = Field(default_factory=list, description="Important findings")
    quantitative_data: Dict[str, Any] = Field(default_factory=dict, description="Numerical data")
    
    # Cross-task relationships
    cross_task_connections: List[CrossTaskConnection] = Field(
        default_factory=list, 
        description="Connections to other tasks"
    )
    
    # Quality metadata
    confidence_level: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)
    data_quality_notes: str = Field(default="", description="Notes on data quality")
    follow_up_questions: List[str] = Field(default_factory=list, description="Questions for future investigation")
    
    # Execution metadata
    execution_time: float = Field(default=0.0, description="Task execution time in seconds")
    tokens_used: int = Field(default=0, description="Tokens used for this task")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SynthesisNote(BaseModel):
    """Schema for synthesis chunk notes."""
    chunk_id: str = Field(description="Unique identifier for the synthesis chunk")
    source_tasks: List[str] = Field(description="Task IDs that contributed to this synthesis")
    synthesis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Synthesis content
    chunk_theme: str = Field(description="Main theme or topic of this synthesis chunk")
    integrated_findings: List[str] = Field(description="Findings that integrate multiple tasks")
    cross_task_synthesis: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Synthesis of cross-task connections"
    )
    emergent_insights: List[str] = Field(
        default_factory=list,
        description="New insights that emerged from synthesis"
    )
    
    # Quality metadata
    synthesis_confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)
    tokens_used: int = Field(description="Tokens used for this synthesis")
    compression_applied: bool = Field(default=False, description="Whether compression was applied")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SessionMetadata(BaseModel):
    """Metadata for a note-taking session."""
    session_id: str = Field(description="Unique session identifier")
    created_timestamp: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    # Session configuration
    original_question: str = Field(description="Original user question")
    execution_mode: str = Field(description="Execution mode (agentic/traditional)")
    note_taking_enabled: bool = Field(default=True)
    
    # Session statistics
    total_tasks: int = Field(default=0, description="Total number of tasks")
    tasks_with_notes: int = Field(default=0, description="Number of tasks with notes")
    synthesis_chunks: int = Field(default=0, description="Number of synthesis chunks")
    total_tokens_used: int = Field(default=0, description="Total tokens used in session")
    
    # Performance metrics
    execution_time: float = Field(default=0.0, description="Total execution time")
    note_generation_time: float = Field(default=0.0, description="Time spent on note generation")
    synthesis_time: float = Field(default=0.0, description="Time spent on synthesis")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class NoteSearchResult(BaseModel):
    """Result from note search operations."""
    note_id: str = Field(description="ID of the found note")
    note_type: str = Field(description="Type of note (task/synthesis)")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance score 0-1")
    content_summary: str = Field(description="Summary of relevant content")
    
    @validator('relevance_score')
    def validate_relevance_score(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Relevance score must be between 0.0 and 1.0')
        return v


class SessionStats(BaseModel):
    """Statistics for a note-taking session."""
    session_id: str
    total_tasks: int
    tasks_with_notes: int
    total_observations: int
    total_findings: int
    total_connections: int
    synthesis_chunks: int
    total_tokens: int
    execution_time: float
    note_efficiency: float = Field(description="Ratio of noted tasks to total tasks")
    
    @validator('note_efficiency')
    def calculate_note_efficiency(cls, v, values):
        if 'total_tasks' in values and values['total_tasks'] > 0:
            return values['tasks_with_notes'] / values['total_tasks']
        return 0.0