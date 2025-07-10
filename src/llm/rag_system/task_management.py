#!/usr/bin/env python3
"""
Task management system for agentic workflows.
Handles DAG-based task execution with dependencies.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

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

class TaskGraph:
    """DAG-based task execution system."""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.execution_order: List[str] = []
    
    def add_task(self, task: Task) -> str:
        """Add a task to the graph."""
        if not task.task_id:
            task.task_id = str(uuid.uuid4())[:8]
        self.tasks[task.task_id] = task
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
    
    def mark_task_completed(self, task_id: str, result: Optional[Any] = None):
        """Mark a task as completed with optional result."""
        self.mark_task_status(task_id, TaskStatus.COMPLETED, result=result)
    
    def mark_task_failed(self, task_id: str, error: str):
        """Mark a task as failed with error message."""
        self.mark_task_status(task_id, TaskStatus.FAILED, error=error)
    
    def mark_task_status(self, task_id: str, status: TaskStatus, result: Optional[Any] = None, error: Optional[str] = None):
        """Mark a task with new status and optional result/error."""
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            if result is not None:
                self.tasks[task_id].result = result
            if error is not None:
                self.tasks[task_id].error = error
            
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