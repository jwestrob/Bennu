"""
Parallel task executor for concurrent API calls.

Enables hundreds of simultaneous analysis tasks to dramatically reduce execution time.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass

from ..task_management import TaskGraph, TaskStatus
from ..task_executor import TaskExecutor, ExecutionResult

logger = logging.getLogger(__name__)

@dataclass
class ParallelExecutionConfig:
    """Configuration for parallel task execution."""
    max_concurrent_tasks: int = 10  # Start conservative, can increase
    batch_size: int = 5  # Process in batches to avoid overwhelming API
    timeout_per_task: float = 120.0  # 2 minutes per task
    retry_failed_tasks: bool = True
    max_retries: int = 2

class ParallelTaskExecutor:
    """
    Executes analysis tasks in parallel for massive speed improvements.
    
    Features:
    - Concurrent API calls using ThreadPoolExecutor
    - Configurable concurrency limits
    - Batch processing to avoid API rate limits
    - Automatic retry for failed tasks
    - Progress tracking and statistics
    """
    
    def __init__(self, config: ParallelExecutionConfig = None):
        """
        Initialize parallel executor.
        
        Args:
            config: Configuration for parallel execution
        """
        self.config = config or ParallelExecutionConfig()
        self.execution_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'retried_tasks': 0,
            'start_time': None,
            'end_time': None
        }
        
    def execute_analysis_tasks_parallel(self, 
                                      task_graph: TaskGraph,
                                      task_executor: TaskExecutor) -> List[Dict[str, Any]]:
        """
        Execute analysis tasks in parallel for massive speed improvement.
        
        Args:
            task_graph: Task graph with analysis tasks
            task_executor: Task executor instance
            
        Returns:
            List of execution results
        """
        self.execution_stats['start_time'] = time.time()
        executable_tasks = task_graph.get_executable_tasks()
        self.execution_stats['total_tasks'] = len(executable_tasks)
        
        logger.info(f"🚀 Starting parallel execution of {len(executable_tasks)} tasks")
        logger.info(f"📊 Config: {self.config.max_concurrent_tasks} concurrent, batch size {self.config.batch_size}")
        
        # Execute tasks in parallel using ThreadPoolExecutor
        results = self._execute_tasks_concurrent(executable_tasks, task_graph, task_executor)
        
        # Handle retries if enabled
        if self.config.retry_failed_tasks:
            results.extend(self._retry_failed_tasks(task_graph, task_executor))
        
        self.execution_stats['end_time'] = time.time()
        self._log_execution_statistics()
        
        return results
    
    def _execute_tasks_concurrent(self, 
                                 executable_tasks: List[str],
                                 task_graph: TaskGraph,
                                 task_executor: TaskExecutor) -> List[Dict[str, Any]]:
        """Execute tasks concurrently using ThreadPoolExecutor."""
        results = []
        
        # Process tasks in batches to manage API load
        for i in range(0, len(executable_tasks), self.config.batch_size):
            batch = executable_tasks[i:i + self.config.batch_size]
            logger.info(f"🔄 Processing batch {i//self.config.batch_size + 1} ({len(batch)} tasks)")
            
            batch_results = self._execute_batch_parallel(batch, task_graph, task_executor)
            results.extend(batch_results)
            
            # Small delay between batches to be gentle on API
            if i + self.config.batch_size < len(executable_tasks):
                time.sleep(0.5)
        
        return results
    
    def _execute_batch_parallel(self, 
                               batch_tasks: List[str],
                               task_graph: TaskGraph,
                               task_executor: TaskExecutor) -> List[Dict[str, Any]]:
        """Execute a batch of tasks in parallel."""
        results = []
        
        # Use ThreadPoolExecutor for true parallelism
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_tasks) as executor:
            # Submit all tasks in the batch
            future_to_task = {}
            for task_id in batch_tasks:
                task = task_graph.tasks[task_id]
                future = executor.submit(self._execute_single_task, task, task_executor)
                future_to_task[future] = task_id
            
            # Collect results as they complete
            for future in as_completed(future_to_task, timeout=self.config.timeout_per_task * len(batch_tasks)):
                task_id = future_to_task[future]
                try:
                    execution_result = future.result()
                    
                    if execution_result.success:
                        results.append({
                            'task_id': task_id,
                            'result': execution_result.result,
                            'metadata': execution_result.metadata or {}
                        })
                        task_graph.mark_task_completed(task_id)
                        self.execution_stats['completed_tasks'] += 1
                        logger.debug(f"✅ Task {task_id} completed successfully")
                    else:
                        logger.warning(f"⚠️ Task {task_id} failed: {execution_result.error}")
                        task_graph.mark_task_failed(task_id, execution_result.error)
                        self.execution_stats['failed_tasks'] += 1
                        
                except Exception as e:
                    logger.error(f"❌ Exception in task {task_id}: {e}")
                    task_graph.mark_task_failed(task_id, str(e))
                    self.execution_stats['failed_tasks'] += 1
        
        return results
    
    def _execute_single_task(self, task, task_executor: TaskExecutor) -> ExecutionResult:
        """Execute a single task with error handling (sync wrapper for async task)."""
        try:
            start_time = time.time()
            
            # Use asyncio.run to handle async task execution in thread
            result = asyncio.run(task_executor.execute_task(task))
            
            execution_time = time.time() - start_time
            
            logger.debug(f"Task {task.task_id} executed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")
            return ExecutionResult(
                task_id=task.task_id,
                success=False,
                result=None,
                error=str(e)
            )
    
    def _retry_failed_tasks(self, 
                           task_graph: TaskGraph,
                           task_executor: TaskExecutor) -> List[Dict[str, Any]]:
        """Retry failed tasks with exponential backoff."""
        failed_tasks = [task_id for task_id, task in task_graph.tasks.items() 
                       if task.status == TaskStatus.FAILED]
        
        if not failed_tasks:
            return []
        
        logger.info(f"🔄 Retrying {len(failed_tasks)} failed tasks")
        retry_results = []
        
        for retry_attempt in range(self.config.max_retries):
            if not failed_tasks:
                break
                
            logger.info(f"🔄 Retry attempt {retry_attempt + 1}/{self.config.max_retries}")
            
            # Reset failed tasks to pending for retry
            for task_id in failed_tasks:
                task_graph.tasks[task_id].status = TaskStatus.PENDING
                task_graph.tasks[task_id].error = None
            
            # Retry with longer timeout and lower concurrency
            retry_config = ParallelExecutionConfig(
                max_concurrent_tasks=max(1, self.config.max_concurrent_tasks // 2),
                batch_size=max(1, self.config.batch_size // 2),
                timeout_per_task=self.config.timeout_per_task * 2,
                retry_failed_tasks=False  # Avoid infinite recursion
            )
            
            retry_executor = ParallelTaskExecutor(retry_config)
            batch_results = retry_executor._execute_tasks_concurrent(
                failed_tasks, task_graph, task_executor
            )
            retry_results.extend(batch_results)
            
            # Update failed tasks list
            failed_tasks = [task_id for task_id in failed_tasks 
                           if task_graph.tasks[task_id].status == TaskStatus.FAILED]
            
            self.execution_stats['retried_tasks'] += len(batch_results)
            
            # Exponential backoff between retries
            if failed_tasks and retry_attempt < self.config.max_retries - 1:
                delay = 2 ** retry_attempt
                logger.info(f"⏳ Waiting {delay}s before next retry...")
                time.sleep(delay)
        
        if failed_tasks:
            logger.warning(f"⚠️ {len(failed_tasks)} tasks failed after all retry attempts")
        
        return retry_results
    
    def _log_execution_statistics(self):
        """Log detailed execution statistics."""
        total_time = self.execution_stats['end_time'] - self.execution_stats['start_time']
        success_rate = (self.execution_stats['completed_tasks'] / 
                       self.execution_stats['total_tasks'] * 100) if self.execution_stats['total_tasks'] > 0 else 0
        
        logger.info("📊 Parallel Execution Statistics:")
        logger.info(f"   ⏱️  Total time: {total_time:.1f}s")
        logger.info(f"   📋 Total tasks: {self.execution_stats['total_tasks']}")
        logger.info(f"   ✅ Completed: {self.execution_stats['completed_tasks']}")
        logger.info(f"   ❌ Failed: {self.execution_stats['failed_tasks']}")
        logger.info(f"   🔄 Retried: {self.execution_stats['retried_tasks']}")
        logger.info(f"   📈 Success rate: {success_rate:.1f}%")
        logger.info(f"   ⚡ Tasks/second: {self.execution_stats['completed_tasks']/total_time:.1f}")
        
        # Calculate speedup estimate
        estimated_sequential_time = self.execution_stats['total_tasks'] * 10  # Assume 10s per task
        speedup = estimated_sequential_time / total_time if total_time > 0 else 1
        logger.info(f"   🚀 Estimated speedup: {speedup:.1f}x faster than sequential")

class AsyncTaskExecutor:
    """
    Alternative async implementation for even better performance.
    
    Note: This requires async-compatible task executors and API clients.
    """
    
    def __init__(self, max_concurrent: int = 15):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_tasks_async(self, 
                                 tasks: List[str],
                                 task_graph: TaskGraph,
                                 task_executor: TaskExecutor) -> List[Dict[str, Any]]:
        """Execute tasks asynchronously with semaphore for concurrency control."""
        logger.info(f"🚀 Starting async execution of {len(tasks)} tasks")
        
        # Create coroutines for all tasks
        coroutines = [
            self._execute_task_async(task_id, task_graph, task_executor)
            for task_id in tasks
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Filter successful results
        successful_results = [
            result for result in results 
            if isinstance(result, dict) and 'task_id' in result
        ]
        
        logger.info(f"✅ Completed {len(successful_results)}/{len(tasks)} tasks")
        return successful_results
    
    async def _execute_task_async(self, 
                                 task_id: str,
                                 task_graph: TaskGraph,
                                 task_executor: TaskExecutor) -> Dict[str, Any]:
        """Execute a single task asynchronously with semaphore."""
        async with self.semaphore:
            try:
                task = task_graph.tasks[task_id]
                
                # Run in thread pool since task_executor is sync
                loop = asyncio.get_event_loop()
                execution_result = await loop.run_in_executor(
                    None, task_executor.execute_task, task
                )
                
                if execution_result.success:
                    task_graph.mark_task_completed(task_id)
                    return {
                        'task_id': task_id,
                        'result': execution_result.result,
                        'metadata': execution_result.metadata or {}
                    }
                else:
                    task_graph.mark_task_failed(task_id, execution_result.error)
                    raise Exception(f"Task failed: {execution_result.error}")
                    
            except Exception as e:
                task_graph.mark_task_failed(task_id, str(e))
                logger.error(f"❌ Async task {task_id} failed: {e}")
                raise