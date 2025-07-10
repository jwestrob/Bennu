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
from .memory import NoteKeeper, NotingDecisionResult, CrossTaskConnection, ConfidenceLevel
from .dspy_signatures import NotingDecision

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
    
    def __init__(self, rag_system, note_keeper: Optional[NoteKeeper] = None):
        """
        Initialize executor with access to RAG system components.
        
        Args:
            rag_system: GenomicRAG instance for access to processors and DSPy modules
            note_keeper: Optional NoteKeeper for persistent note-taking
        """
        self.rag_system = rag_system
        self.completed_results = {}  # Store results for inter-task dependencies
        self.note_keeper = note_keeper
        
        # Initialize note-taking decision module if DSPy is available
        if hasattr(self.rag_system, 'dspy_available') and self.rag_system.dspy_available:
            try:
                import dspy
                self.noting_decision = dspy.Predict(NotingDecision)
            except ImportError:
                logger.warning("DSPy not available - note-taking disabled")
                self.noting_decision = None
        else:
            self.noting_decision = None
        
    async def execute_task(self, task: Task) -> ExecutionResult:
        """
        Execute an individual task based on its type.
        
        Args:
            task: Task object to execute
            
        Returns:
            ExecutionResult with success status and result data
        """
        # Add enhanced logging to track task naming issues
        if len(task.task_id) > 100:
            logger.warning(f"⚠️ LONG TASK ID DETECTED: {task.task_id[:100]}...")
            logger.warning("📋 This indicates old recursive splitting system may still be active")
        
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
            
            # Create execution result
            execution_result = ExecutionResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={
                    "task_type": task.task_type.value,
                    "description": task.description
                }
            )
            
            # Consider note-taking after successful execution
            if self.note_keeper and self.noting_decision:
                await self._consider_note_taking(task, execution_result)
            
            logger.info(f"Task {task.task_id} completed successfully in {execution_time:.2f}s")
            
            return execution_result
            
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
        
        # Check if result is too large and needs intelligent chunking
        # BUT only if this task hasn't already been chunked (prevent recursive chunking)
        if (context and hasattr(context, 'structured_data') and 
            not getattr(task, '_already_chunked', False) and
            not getattr(task, '_intelligent_chunked', False)):  # Extra protection
            raw_data = context.structured_data
            
            # Use intelligent upfront chunking for large datasets
            if len(raw_data) > 1000:  # Threshold for intelligent chunking
                logger.info(f"🧠 Large dataset detected ({len(raw_data)} items), using intelligent upfront chunking")
                logger.info(f"✅ Using NEW IntelligentChunkingManager (not old recursive splitter)")
                
                try:
                    from .intelligent_chunking_manager import IntelligentChunkingManager
                    chunking_manager = IntelligentChunkingManager(max_chunks=4, min_chunk_size=100)
                    
                    # Create chunks upfront based on biological meaning
                    chunks = await chunking_manager.analyze_and_chunk_dataset(task, raw_data, task.description)
                    
                    if len(chunks) > 1:
                        logger.info(f"🔀 Created {len(chunks)} intelligent chunks, executing in parallel")
                        
                        # Execute chunked analysis
                        chunk_results = await chunking_manager.execute_chunked_analysis(chunks, self, task)
                        
                        # Synthesize results
                        synthesis = chunking_manager.synthesize_chunk_results(
                            chunk_results, task.description, chunks
                        )
                        
                        # Return the chunked analysis result
                        return {
                            "context": context,
                            "query_type": classification.query_type,
                            "search_strategy": "intelligent_chunking",
                            "description": task.description,
                            "structured_data": [{"summary": synthesis, "chunked_execution": True}],
                            "semantic_data": [],
                            "metadata": {
                                "chunks_created": len(chunks),
                                "chunks_completed": len(chunk_results),
                                "chunking_strategy": "intelligent_upfront",
                                "total_items_processed": len(raw_data)
                            },
                            "chunked_execution_result": {
                                "summary": synthesis,
                                "chunk_count": len(chunks),
                                "successful_chunks": len(chunk_results),
                                "total_items": len(raw_data)
                            }
                        }
                        
                except ImportError as e:
                    logger.warning(f"Intelligent chunking manager not available: {e}")
                except Exception as e:
                    logger.error(f"Intelligent chunking failed: {e}")
                    # Fall through to normal execution
        
        # Format results for consumption by downstream tasks (normal execution)
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
    
    async def _consider_note_taking(self, task: Task, execution_result: ExecutionResult) -> None:
        """
        Consider whether to take notes for a completed task.
        
        Args:
            task: Task that was executed
            execution_result: Result of task execution
        """
        try:
            # Get session summary for context
            session_summary = self.note_keeper.get_session_summary()
            
            # Format execution result for decision
            result_summary = self._format_result_for_decision(execution_result)
            
            # Use DSPy to decide whether to take notes
            decision = self.noting_decision(
                task_description=task.description,
                execution_result=result_summary,
                existing_notes=session_summary
            )
            
            # Parse decision result
            should_record = getattr(decision, 'should_record', False)
            
            if should_record:
                logger.info(f"📝 Recording notes for task {task.task_id}: {decision.reasoning}")
                
                # Extract note content from decision
                observations = self._parse_list_output(getattr(decision, 'observations', []))
                key_findings = self._parse_list_output(getattr(decision, 'key_findings', []))
                cross_connections = self._parse_cross_connections(getattr(decision, 'cross_connections', []))
                quantitative_data = self._parse_quantitative_data(getattr(decision, 'quantitative_data', {}))
                
                # Create decision result
                decision_result = NotingDecisionResult(
                    should_record=should_record,
                    reasoning=getattr(decision, 'reasoning', ''),
                    importance_score=float(getattr(decision, 'importance_score', 5.0))
                )
                
                # Record the notes
                success = self.note_keeper.record_task_notes(
                    task_id=task.task_id,
                    task_type=task.task_type.value,
                    description=task.description,
                    decision_result=decision_result,
                    observations=observations,
                    key_findings=key_findings,
                    quantitative_data=quantitative_data,
                    cross_connections=cross_connections,
                    confidence=ConfidenceLevel.MEDIUM,
                    execution_time=execution_result.execution_time,
                    tokens_used=self._estimate_tokens_used(execution_result)
                )
                
                if success:
                    logger.info(f"✅ Notes recorded for task {task.task_id}")
                else:
                    logger.warning(f"❌ Failed to record notes for task {task.task_id}")
            else:
                logger.debug(f"⏭️ Skipping notes for task {task.task_id}: {decision.reasoning}")
                
        except Exception as e:
            logger.error(f"Error in note-taking consideration: {e}")
    
    def _format_result_for_decision(self, execution_result: ExecutionResult) -> str:
        """Format execution result for note-taking decision."""
        result_parts = []
        
        # Basic execution info
        result_parts.append(f"Execution time: {execution_result.execution_time:.2f}s")
        result_parts.append(f"Success: {execution_result.success}")
        
        # Format result content
        if execution_result.result:
            result_data = execution_result.result
            
            # Handle different result types
            if isinstance(result_data, dict):
                if 'context' in result_data:
                    context = result_data['context']
                    structured_count = len(context.structured_data) if hasattr(context, 'structured_data') else 0
                    semantic_count = len(context.semantic_data) if hasattr(context, 'semantic_data') else 0
                    result_parts.append(f"Retrieved {structured_count} structured + {semantic_count} semantic results")
                    
                    # Sample some results
                    if structured_count > 0:
                        sample_results = context.structured_data[:3]
                        result_parts.append(f"Sample results: {str(sample_results)}")
                
                if 'tool_result' in result_data:
                    result_parts.append(f"Tool result: {str(result_data['tool_result'])[:200]}...")
            else:
                result_parts.append(f"Result: {str(result_data)[:200]}...")
        
        return " | ".join(result_parts)
    
    def _parse_list_output(self, output: Any) -> List[str]:
        """Parse list output from DSPy, handling various formats."""
        if isinstance(output, list):
            return [str(item) for item in output if item]
        elif isinstance(output, str):
            # Try to parse as list-like string
            if output.startswith('[') and output.endswith(']'):
                try:
                    import ast
                    parsed = ast.literal_eval(output)
                    return [str(item) for item in parsed if item]
                except:
                    pass
            # Split by common delimiters
            items = output.split(';') if ';' in output else output.split(',')
            return [item.strip() for item in items if item.strip()]
        else:
            return [str(output)] if output else []
    
    def _parse_cross_connections(self, connections: Any) -> List[CrossTaskConnection]:
        """Parse cross-task connections from DSPy output."""
        parsed_connections = []
        
        if isinstance(connections, list):
            items = connections
        elif isinstance(connections, str):
            items = connections.split(';') if ';' in connections else [connections]
        else:
            return parsed_connections
        
        for item in items:
            if not item or not str(item).strip():
                continue
                
            try:
                # Expected format: "task_id:connection_type:description"
                parts = str(item).split(':')
                if len(parts) >= 3:
                    task_id = parts[0].strip()
                    connection_type = parts[1].strip()
                    description = ':'.join(parts[2:]).strip()
                    
                    # Map connection type to enum
                    from .memory.note_schemas import ConnectionType
                    connection_enum = ConnectionType.INFORMS  # Default
                    
                    for conn_type in ConnectionType:
                        if connection_type.lower() == conn_type.value.lower():
                            connection_enum = conn_type
                            break
                    
                    parsed_connections.append(CrossTaskConnection(
                        connected_task=task_id,
                        connection_type=connection_enum,
                        description=description
                    ))
                    
            except Exception as e:
                logger.warning(f"Failed to parse connection '{item}': {e}")
        
        return parsed_connections
    
    def _parse_quantitative_data(self, data: Any) -> Dict[str, Any]:
        """Parse quantitative data from DSPy output."""
        if isinstance(data, dict):
            return data
        elif isinstance(data, str) and data.strip():
            try:
                import json
                return json.loads(data)
            except:
                # Simple key:value parsing
                result = {}
                pairs = data.split(',')
                for pair in pairs:
                    if ':' in pair:
                        key, value = pair.split(':', 1)
                        result[key.strip()] = value.strip()
                return result
        else:
            return {}
    
    def _estimate_tokens_used(self, execution_result: ExecutionResult) -> int:
        """Estimate tokens used for this task execution."""
        # Simple estimation based on result size
        result_str = str(execution_result.result)
        return len(result_str) // 4  # Rough estimate: 4 chars per token
    
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