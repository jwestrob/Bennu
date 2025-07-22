#!/usr/bin/env python3
"""
Task management system for agentic workflows.
Handles DAG-based task execution with dependencies.
"""

import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime

if TYPE_CHECKING:
    from .agent_tool_selector import ToolSelectionResult

logger = logging.getLogger(__name__)

class TaskGraphLogger:
    """Enhanced logging for task graph execution with visual hierarchy."""
    
    def __init__(self, user_query: str = "Unknown Query"):
        self.user_query = user_query
        self.session_start = datetime.now()
        self.task_logs: List[Dict[str, Any]] = []
        self.current_phase = "Initialization"
    
    def log_task_created(self, task, source: str = "manual"):
        """Log task creation with hierarchy information."""
        indent = "  " if not task.is_main_task else ""
        task_type_icon = "🔍" if task.task_type == TaskType.ATOMIC_QUERY else "🛠️" if task.task_type == TaskType.TOOL_CALL else "⚙️"
        logger.info(f"{indent}TASK CREATED: {task_type_icon} {task.task_id} ({task.task_type.value}) - {task.description}")
    
    def log_task_execution(self, task, status: str):
        """Log task execution progress."""
        indent = "  " if not task.is_main_task else ""
        status_icon = "🔄" if status == "EXECUTING" else "✅" if status == "COMPLETED" else "❌"
        logger.info(f"{indent}EXECUTING: {status_icon} {task.task_id} - {status}")
    
    def log_phase_change(self, phase: str):
        """Log phase transitions."""
        self.current_phase = phase
        logger.info(f"PHASE: 🚀 {phase}")
    
    def log_phase_start(self, phase_name: str):
        """Log the start of a new execution phase."""
        self.current_phase = phase_name
        logger.info(f"PHASE: 🚀 Starting {phase_name}")
    
    def log_task_completed(self, task, result_summary: str = ""):
        """Log task completion with result summary."""
        indent = "  " if not task.is_main_task else ""
        status_icon = "✅"
        summary_text = f" - {result_summary}" if result_summary else ""
        logger.info(f"{indent}COMPLETED: {status_icon} {task.task_id}{summary_text}")
    
    def log_task_started(self, task):
        """Log task start."""
        indent = "  " if not task.is_main_task else ""
        status_icon = "🔄"
        logger.info(f"{indent}EXECUTING: {status_icon} {task.task_id} - STARTED")
    
    def log_task_graph_summary(self, tasks: Dict[str, Any]):
        """Log a summary of the task graph execution."""
        total_tasks = len(tasks)
        completed = sum(1 for t in tasks.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in tasks.values() if t.status == TaskStatus.FAILED)
        skipped = sum(1 for t in tasks.values() if t.status == TaskStatus.SKIPPED)
        
        logger.info(f"PHASE: 📊 Task Graph Summary")
        logger.info(f"  Total Tasks: {total_tasks}")
        logger.info(f"  ✅ Completed: {completed}")
        logger.info(f"  ❌ Failed: {failed}")
        logger.info(f"  ⏭️ Skipped: {skipped}")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary for export."""
        return {
            "user_query": self.user_query,
            "session_start": self.session_start.isoformat(),
            "current_phase": self.current_phase,
            "task_count": len(self.task_logs)
        }

class TaskStatus(Enum):
    """Status of task execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class TaskType(Enum):
    """Type of task to execute."""
    ATOMIC_QUERY = "atomic_query"
    TOOL_CALL = "tool_call"
    SYNTHESIS = "synthesis"

@dataclass
class Task:
    """Individual task with dependencies and metadata."""
    task_id: str
    task_type: TaskType
    description: str
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # Task-specific fields
    query: Optional[str] = None  # For ATOMIC_QUERY
    tool_name: Optional[str] = None  # For TOOL_CALL
    tool_args: Dict[str, Any] = field(default_factory=dict)  # For TOOL_CALL
    
    # Tool selection hierarchy fields
    is_main_task: bool = True  # Main tasks get full LLM tool selection
    parent_task_id: Optional[str] = None  # For inheritance chain
    tool_selection_result: Optional[Any] = None  # Cache tool selection result
    tool_selection_source: str = "planned"  # "planned", "inherited", "synthesized"
    
    # Execution timing and metadata
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[float] = None

class TaskGraph:
    """DAG-based task execution system with enhanced logging."""
    
    def __init__(self, user_query: str = "Unknown Query"):
        self.tasks: Dict[str, Task] = {}
        self.execution_order: List[str] = []
        self.logger = TaskGraphLogger(user_query)
        self.user_query = user_query
    
    def add_task(self, task: Task, source: str = "manual") -> str:
        """Add a task to the graph with enhanced logging."""
        if not task.task_id:
            task.task_id = str(uuid.uuid4())[:8]
        self.tasks[task.task_id] = task
        
        # Log task creation
        self.logger.log_task_created(task, source)
        
        return task.task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute (all dependencies completed)."""
        ready_tasks = []
        
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
                
            # Check if all dependencies are completed
            dependencies_completed = True
            for dep_id in task.dependencies:
                dep_task = self.tasks.get(dep_id)
                if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                    dependencies_completed = False
                    break
            
            if dependencies_completed:
                ready_tasks.append(task)
        
        return ready_tasks
    
    def get_executable_tasks(self) -> List[str]:
        """Get list of task IDs that are ready to execute."""
        ready_tasks = self.get_ready_tasks()
        return [task.task_id for task in ready_tasks]
    
    def mark_task_completed(self, task_id: str, result: Optional[Any] = None, result_summary: str = ""):
        """Mark a task as completed with optional result and logging."""
        self.mark_task_status(task_id, TaskStatus.COMPLETED, result=result)
        
        # Enhanced logging for task completion
        if hasattr(self, 'logger') and task_id in self.tasks:
            self.logger.log_task_completed(self.tasks[task_id], result_summary)
    
    def mark_task_failed(self, task_id: str, error: str):
        """Mark a task as failed with error message and logging."""
        self.mark_task_status(task_id, TaskStatus.FAILED, error=error)
        
        # Enhanced logging for task failure
        if hasattr(self, 'logger') and task_id in self.tasks:
            self.logger.log_task_completed(self.tasks[task_id], f"FAILED: {error}")
    
    def mark_task_status(self, task_id: str, status: TaskStatus, result: Optional[Any] = None, error: Optional[str] = None):
        """Mark a task with new status and optional result/error."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = status
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
            
            # Enhanced logging for status changes
            if status == TaskStatus.RUNNING and hasattr(self, 'logger'):
                self.logger.log_task_started(task)
            
            logger.debug(f"Task {task_id} marked as {status.value}")
            
            # If task failed, mark dependent tasks as skipped
            if status == TaskStatus.FAILED:
                self._mark_dependent_tasks_skipped(task_id)
    
    def _mark_dependent_tasks_skipped(self, failed_task_id: str):
        """Mark all tasks dependent on a failed task as skipped."""
        for task in self.tasks.values():
            if (failed_task_id in task.dependencies and 
                task.status == TaskStatus.PENDING):
                task.status = TaskStatus.SKIPPED
                logger.debug(f"Task {task.task_id} skipped due to failed dependency {failed_task_id}")
    
    def get_failed_dependencies(self, task_id: str) -> List[str]:
        """Get list of failed dependencies for a task."""
        task = self.tasks.get(task_id)
        if not task:
            return []
        
        failed_deps = []
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if dep_task and dep_task.status == TaskStatus.FAILED:
                failed_deps.append(dep_id)
        
        return failed_deps
    
    def mark_skipped_tasks(self):
        """Mark tasks with failed dependencies as skipped."""
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                failed_deps = self.get_failed_dependencies(task.task_id)
                if failed_deps:
                    task.status = TaskStatus.SKIPPED
                    logger.debug(f"Task {task.task_id} skipped due to failed dependencies: {failed_deps}")
    
    def set_phase(self, phase_name: str):
        """Set the current execution phase for logging."""
        self.logger.log_phase_start(phase_name)
    
    def get_execution_summary(self):
        """Get and log execution summary."""
        self.logger.log_task_graph_summary(self.tasks)
        return self.logger.task_logs
    
    def export_log(self, filepath: str = None) -> str:
        """Export execution log to file."""
        return self.logger.export_execution_log(filepath)
    
    def is_complete(self) -> bool:
        """Check if all tasks are in terminal states."""
        for task in self.tasks.values():
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                return False
        return True
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary of task statuses."""
        summary = {status.value: 0 for status in TaskStatus}
        for task in self.tasks.values():
            summary[task.status.value] += 1
        return summary
    
    def get_completed_results(self) -> Dict[str, Any]:
        """Get results from all completed tasks."""
        results = {}
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.COMPLETED and task.result is not None:
                results[task_id] = task.result
        return results
    
    def get_execution_plan(self) -> List[List[str]]:
        """Get execution plan as list of task batches that can run in parallel."""
        plan = []
        remaining_tasks = set(self.tasks.keys())
        
        while remaining_tasks:
            # Find tasks with no dependencies in remaining set
            current_batch = []
            for task_id in list(remaining_tasks):
                task = self.tasks[task_id]
                deps_satisfied = all(dep_id not in remaining_tasks for dep_id in task.dependencies)
                if deps_satisfied:
                    current_batch.append(task_id)
            
            if not current_batch:
                # Circular dependency detected
                logger.warning(f"Circular dependency detected in remaining tasks: {remaining_tasks}")
                break
            
            plan.append(current_batch)
            remaining_tasks -= set(current_batch)
        
        return plan