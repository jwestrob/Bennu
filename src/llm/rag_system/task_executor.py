"""
Task Executor for executing individual tasks in agentic workflows.

This module handles the execution of Task objects created by TaskPlanParser,
routing them to appropriate processors (database queries, external tools) and
managing results and error handling.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .task_management import Task, TaskGraph, TaskStatus, TaskType
from .external_tools import AVAILABLE_TOOLS
from .utils import safe_log_data

logger = logging.getLogger(__name__)

@dataclass
class ExecutionResult:
    """Result of task execution with metadata."""
    task_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None

class TaskExecutor:
    """
    Executes individual Task objects based on their type.
    
    Handles:
    - ATOMIC_QUERY: Routes to appropriate database processors
    - TOOL_CALL: Routes to external tools with proper arguments
    - Error handling and result formatting
    - Task result aggregation for final analysis
    """
    
    def __init__(self, rag_system):
        """
        Initialize executor with access to RAG system components.
        
        Args:
            rag_system: GenomicRAG instance for access to processors and DSPy modules
        """
        self.rag_system = rag_system
        self.completed_results = {}  # Store results for inter-task dependencies
        
    async def execute_task(self, task: Task) -> ExecutionResult:
        """
        Execute an individual task based on its type.
        
        Args:
            task: Task object to execute
            
        Returns:
            ExecutionResult with success status and result data
        """
        logger.info(f"Executing task {task.task_id}: {task.task_type.value}")
        
        import time
        start_time = time.time()
        
        try:
            if task.task_type == TaskType.ATOMIC_QUERY:
                result = await self._execute_query_task(task)
            elif task.task_type == TaskType.TOOL_CALL:
                result = await self._execute_tool_task(task)
            else:
                raise ValueError(f"Unsupported task type: {task.task_type}")
            
            execution_time = time.time() - start_time
            
            # Store result for future tasks
            self.completed_results[task.task_id] = result
            
            logger.info(f"Task {task.task_id} completed successfully in {execution_time:.2f}s")
            
            return ExecutionResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={
                    "task_type": task.task_type.value,
                    "description": task.description
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Task {task.task_id} failed: {str(e)}"
            logger.error(error_msg)
            
            return ExecutionResult(
                task_id=task.task_id,
                success=False,
                result=None,
                error=error_msg,
                execution_time=execution_time,
                metadata={
                    "task_type": task.task_type.value,
                    "description": task.description
                }
            )
    
    async def _execute_query_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute an ATOMIC_QUERY task using DSPy and database processors.
        
        Args:
            task: Task with query description
            
        Returns:
            Query results with context and metadata
        """
        logger.debug(f"Executing query task: {task.description}")
        
        # Transform "for each" patterns to comparative language for better DSPy understanding
        transformed_description = self._transform_for_each_patterns(task.description)
        
        # Use DSPy to classify the query and generate appropriate strategy
        classification = self.rag_system.classifier(question=transformed_description)
        
        # Import schema
        from ..dsp_sig import NEO4J_SCHEMA
        
        # Generate retrieval strategy
        retrieval_plan = self.rag_system.retriever(
            db_schema=NEO4J_SCHEMA,
            question=transformed_description,
            query_type=classification.query_type
        )
        
        # Validate query for comparative questions (same as in core.py)
        cypher_query = retrieval_plan.cypher_query
        validated_query = self.rag_system._validate_comparative_query(task.description, cypher_query)
        if validated_query != cypher_query:
            logger.info("Fixed comparative query in task execution - removed inappropriate LIMIT")
            retrieval_plan.cypher_query = validated_query
        
        # Execute the query
        context = await self.rag_system._retrieve_context(
            classification.query_type, 
            retrieval_plan,
            task.description
        )
        
        # Format results for consumption by downstream tasks
        return {
            "context": context,
            "query_type": classification.query_type,
            "search_strategy": retrieval_plan.search_strategy,
            "description": task.description,
            "structured_data": context.structured_data if context else [],
            "semantic_data": context.semantic_data if context else [],
            "metadata": context.metadata if context else {}
        }
    
    async def _execute_tool_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a TOOL_CALL task using external tools.
        
        Args:
            task: Task with tool name and arguments
            
        Returns:
            Tool execution results
        """
        logger.debug(f"Executing tool task: {task.tool_name} - {task.description}")
        
        if task.tool_name not in AVAILABLE_TOOLS:
            raise ValueError(f"Tool '{task.tool_name}' not available. Available tools: {list(AVAILABLE_TOOLS.keys())}")
        
        tool_function = AVAILABLE_TOOLS[task.tool_name]
        
        # Prepare arguments for tool execution
        tool_args = self._prepare_tool_arguments(task)
        
        # Execute the tool
        if asyncio.iscoroutinefunction(tool_function):
            result = await tool_function(**tool_args)
        else:
            result = tool_function(**tool_args)
        
        return {
            "tool_result": result,
            "tool_name": task.tool_name,
            "description": task.description,
            "arguments": tool_args
        }
    
    def _prepare_tool_arguments(self, task: Task) -> Dict[str, Any]:
        """
        Prepare arguments for tool execution, including dependency results.
        
        Args:
            task: Task with tool arguments and dependencies
            
        Returns:
            Dictionary of arguments for tool execution
        """
        args = task.tool_args.copy()
        
        # Add dependency results to tool arguments
        dependency_data = []
        for dep_id in task.dependencies:
            if dep_id in self.completed_results:
                dependency_data.append(self.completed_results[dep_id])
        
        if dependency_data:
            args["dependency_results"] = dependency_data
            
            # For code interpreter, prepare data in a convenient format
            if task.tool_name == "code_interpreter":
                args.update(self._prepare_code_interpreter_args(dependency_data, task))
        
        return args
    
    def _transform_for_each_patterns(self, description: str) -> str:
        """
        Transform 'for each' patterns into comparative query language.
        
        This helps DSPy understand that these should be comparative queries
        showing all results, not single-item queries.
        
        Args:
            description: Original task description
            
        Returns:
            Transformed description that's clearer for DSPy
        """
        import re
        
        # Transform "for each genome" patterns
        patterns = [
            (r'\bfor\s+each\s+genome,?\s+', 'compare across all genomes to '),
            (r'\bfor\s+each\s+protein,?\s+', 'compare across all proteins to '),
            (r'\bfor\s+each\s+gene,?\s+', 'compare across all genes to '),
            (r'\bfor\s+each\s+domain,?\s+', 'compare across all domains to '),
        ]
        
        transformed = description
        for pattern, replacement in patterns:
            transformed = re.sub(pattern, replacement, transformed, flags=re.IGNORECASE)
        
        # If we transformed anything, log it
        if transformed != description:
            logger.debug(f"Transformed task description: '{description}' -> '{transformed}'")
        
        return transformed
    
    def _prepare_code_interpreter_args(self, dependency_data: List[Dict], task: Task) -> Dict[str, Any]:
        """
        Prepare specialized arguments for code interpreter execution.
        
        Args:
            dependency_data: Results from dependency tasks
            task: Current task being executed
            
        Returns:
            Enhanced arguments for code interpreter
        """
        # Extract structured data from dependencies
        all_structured_data = []
        for data in dependency_data:
            if "structured_data" in data:
                all_structured_data.extend(data["structured_data"])
        
        # Generate Python code based on task description and available data
        code = self._generate_analysis_code(task, all_structured_data)
        
        return {
            "code": code,
            "session_id": f"task_{task.task_id}",
            "timeout": 60,  # 1 minute timeout for analysis tasks
            "data_summary": f"Available data: {len(all_structured_data)} records"
        }
    
    def _generate_analysis_code(self, task: Task, structured_data: List[Dict]) -> str:
        """
        Generate Python code for analysis tasks based on description and data.
        
        This is a simplified approach - in production, this could use
        more sophisticated code generation techniques.
        
        Args:
            task: Task with description
            structured_data: Available data from previous tasks
            
        Returns:
            Python code string for execution
        """
        # Basic code template based on task type
        if "matrix" in task.description.lower():
            code = '''
import pandas as pd
import numpy as np

# Create data analysis from available results
print("Creating analysis matrix...")

# This is a placeholder - in production, this would be generated
# based on the actual structured_data content and task requirements
data_summary = f"Analyzed {len(structured_data) if 'structured_data' in globals() else 0} records"
print(f"Analysis result: {data_summary}")

result = {"analysis_type": "matrix", "summary": data_summary}
print("Matrix analysis completed")
'''
        
        elif "visualiz" in task.description.lower():
            code = '''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print("Generating visualization...")

# Placeholder visualization code
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(['Genome A', 'Genome B', 'Genome C'], [10, 15, 8])
ax.set_title('DUF Domain Distribution')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('duf_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

print("Visualization saved as duf_distribution.png")
result = {"visualization": "duf_distribution.png", "type": "bar_chart"}
'''
        
        else:
            # Generic analysis code
            code = f'''
import pandas as pd
import numpy as np

print("Executing analysis task: {task.description}")

# Generic analysis placeholder
data_available = len(structured_data) if 'structured_data' in globals() else 0
print(f"Processing {{data_available}} data points")

result = {{
    "task": "{task.description}",
    "data_processed": data_available,
    "status": "completed"
}}

print("Analysis completed")
'''
        
        return code.strip()
    
    async def execute_graph(self, graph: TaskGraph) -> Dict[str, Any]:
        """
        Execute a complete TaskGraph with dependency resolution.
        
        Args:
            graph: TaskGraph with tasks and dependencies
            
        Returns:
            Aggregated results from all completed tasks
        """
        logger.info(f"Executing task graph with {len(graph.tasks)} tasks")
        
        execution_results = []
        
        while not graph.is_complete():
            ready_tasks = graph.get_ready_tasks()
            
            if not ready_tasks:
                # Check if we're stuck (circular dependencies or all tasks failed)
                remaining_tasks = [t for t in graph.tasks.values() if t.status == TaskStatus.PENDING]
                if remaining_tasks:
                    logger.error(f"Execution stuck: {len(remaining_tasks)} tasks remaining but none ready")
                    break
                
            # Execute ready tasks in parallel
            if ready_tasks:
                logger.info(f"Executing {len(ready_tasks)} ready tasks")
                
                # Execute tasks concurrently
                execution_tasks = [self.execute_task(task) for task in ready_tasks]
                results = await asyncio.gather(*execution_tasks, return_exceptions=True)
                
                # Process results
                for task, result in zip(ready_tasks, results):
                    if isinstance(result, Exception):
                        graph.mark_task_status(task.task_id, TaskStatus.FAILED, error=str(result))
                        execution_results.append(ExecutionResult(
                            task_id=task.task_id,
                            success=False,
                            result=None,
                            error=str(result)
                        ))
                    else:
                        status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
                        graph.mark_task_status(
                            task.task_id, 
                            status, 
                            result=result.result, 
                            error=result.error
                        )
                        execution_results.append(result)
        
        # Generate summary
        summary = graph.get_summary()
        completed_results = graph.get_completed_results()
        
        logger.info(f"Graph execution completed: {summary}")
        
        return {
            "execution_summary": summary,
            "completed_results": completed_results,
            "all_results": execution_results,
            "success": summary.get("completed", 0) > 0
        }

def test_task_executor():
    """Test function for TaskExecutor."""
    from .task_plan_parser import TaskPlanParser
    from ..config import LLMConfig
    from .. import GenomicRAG
    
    # Create test setup
    config = LLMConfig()
    rag = GenomicRAG(config)
    executor = TaskExecutor(rag)
    
    # Create a simple test task
    test_task = Task(
        task_id="test_query",
        task_type=TaskType.ATOMIC_QUERY,
        description="Find all genomes in the dataset",
        query="MATCH (g:Genome) RETURN g.genome_id"
    )
    
    print("=== Task Executor Test ===")
    print(f"Test task: {test_task.description}")
    
    # Test task execution (async)
    import asyncio
    
    async def run_test():
        try:
            result = await executor.execute_task(test_task)
            print(f"Execution success: {result.success}")
            print(f"Execution time: {result.execution_time:.2f}s")
            if result.success:
                print(f"Result summary: {safe_log_data(result.result, 200)}")
            else:
                print(f"Error: {result.error}")
        except Exception as e:
            print(f"Test failed: {e}")
        finally:
            rag.close()
    
    asyncio.run(run_test())

if __name__ == "__main__":
    test_task_executor()