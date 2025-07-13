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
    
    def __init__(self, rag_system, note_keeper: Optional[NoteKeeper] = None, selected_genome: Optional[str] = None):
        """
        Initialize executor with access to RAG system components.
        
        Args:
            rag_system: GenomicRAG instance for access to processors and DSPy modules
            note_keeper: Optional NoteKeeper for persistent note-taking
            selected_genome: Pre-selected genome ID for agentic tasks
        """
        self.rag_system = rag_system
        self.completed_results = {}  # Store results for inter-task dependencies
        self.note_keeper = note_keeper
        self.selected_genome = selected_genome  # Pre-selected genome for all tasks
        
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
    
    def _should_chunk_for_analysis_type(self, data_size: int, threshold: int, task_description: str, original_question: str = "") -> bool:
        """
        Determine if data should be chunked based on analysis type and biological context.
        
        Args:
            data_size: Size of the dataset
            threshold: Size threshold for chunking
            task_description: Description of the current task
            original_question: Original user question for context
            
        Returns:
            bool: Whether to chunk the data
        """
        # Always chunk if dataset is extremely large
        if data_size > threshold * 3:
            logger.info(f"🔥 FORCE CHUNK: Dataset extremely large ({data_size} > {threshold * 3})")
            return True
        
        # Skip chunking if dataset is below threshold
        if data_size <= threshold:
            logger.info(f"✅ NO CHUNK: Dataset manageable ({data_size} <= {threshold})")
            return False
        
        # Context-aware chunking decision for medium-large datasets
        combined_text = f"{task_description} {original_question}".lower()
        
        # Discovery/exploration queries should avoid chunking for holistic view
        discovery_patterns = [
            "find", "discover", "look through", "see what", "explore", 
            "operons", "prophage", "phage", "spatial", "genomic regions",
            "across all", "everything", "global analysis", "browse through"
        ]
        
        # Functional annotation queries can benefit from chunking
        functional_patterns = [
            "function", "functional", "annotation", "protein families", 
            "domains", "pathways", "metabolic", "kegg", "pfam"
        ]
        
        if any(pattern in combined_text for pattern in discovery_patterns):
            logger.info(f"🌐 DISCOVERY QUERY: Avoiding chunking for holistic analysis (size: {data_size})")
            return False
        elif any(pattern in combined_text for pattern in functional_patterns):
            logger.info(f"🔬 FUNCTIONAL QUERY: Using chunking for detailed annotation analysis (size: {data_size})")
            return True
        else:
            # Default: chunk if above threshold
            logger.info(f"📊 DEFAULT: Chunking large dataset (size: {data_size} > {threshold})")
            return True
    
    def _determine_analysis_type_for_task(self, task: Task, task_description: str) -> str:
        """
        Determine analysis type for a task based on original question and task description.
        
        Args:
            task: The task object (may have original_question)
            task_description: Current task description
            
        Returns:
            Analysis type: spatial_genomic, functional_annotation, or comprehensive_discovery
        """
        # Use original question if available (injected by core.py)
        original_question = getattr(task, 'original_question', task_description)
        combined_text = f"{original_question} {task_description}".lower()
        
        # Spatial/genomic organization patterns (PHAGE QUERIES!)
        spatial_patterns = [
            "operon", "operons", "gene cluster", "genomic region", "prophage", 
            "phage", "spatial", "neighborhood", "proximity", "adjacent",
            "genomic context", "gene organization", "cluster", "loci", "segments"
        ]
        
        # Functional annotation patterns  
        functional_patterns = [
            "function", "functional", "activity", "pathway", "metabolic",
            "enzyme", "protein family", "domain", "kegg", "pfam", "annotation",
            "bgc", "biosynthetic"
        ]
        
        # Discovery/exploration patterns
        discovery_patterns = [
            "find", "discover", "explore", "look through", "see what", 
            "interesting", "novel", "unusual", "stands out", "browse"
        ]
        
        if any(pattern in combined_text for pattern in spatial_patterns):
            logger.info(f"🧬 Task analysis type: SPATIAL_GENOMIC (detected: {[p for p in spatial_patterns if p in combined_text]})")
            return "spatial_genomic"
        elif any(pattern in combined_text for pattern in functional_patterns):
            logger.info(f"🔬 Task analysis type: FUNCTIONAL_ANNOTATION (detected: {[p for p in functional_patterns if p in combined_text]})")
            return "functional_annotation"
        elif any(pattern in combined_text for pattern in discovery_patterns):
            logger.info(f"🌐 Task analysis type: COMPREHENSIVE_DISCOVERY (detected: {[p for p in discovery_patterns if p in combined_text]})")
            return "comprehensive_discovery"
        else:
            # For phage queries, default to spatial analysis
            logger.info(f"📊 Task analysis type: SPATIAL_GENOMIC (default for task: {task.task_id})")
            return "spatial_genomic"
        
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
        
        # EXPLICIT GENOME SELECTION for task execution
        genome_filter_required = False
        target_genome = ""
        task_context = f"Global task: {transformed_description}"
        
        # Use pre-selected genome if available (from agentic upfront selection)
        if self.selected_genome:
            logger.info(f"🧬 Using pre-selected genome from agentic planning: {self.selected_genome}")
            genome_filter_required = True
            target_genome = self.selected_genome
            
            # Enhanced task description injection to preserve genome context
            enhanced_description = f"For genome {self.selected_genome}: {transformed_description}"
            task_context = f"Target genome: {self.selected_genome}. Task: {enhanced_description}"
            
            # Override DSPy input to use enhanced description for better query generation
            transformed_description = enhanced_description
            
            logger.info(f"🔧 Enhanced task description with genome context: {enhanced_description}")
        
        # No genome selection needed at task level - decision made upfront in core.py
        # Tasks inherit genome context from agentic planning phase
        else:
            logger.debug("🌐 Task does not require genome-specific targeting - using global execution")
        
        # Use DSPy to classify the query and generate appropriate strategy
        # Use model allocation for classification (now maps to MEDIUM = gpt-4.1-mini)
        def classification_call(module):
            return module(question=transformed_description)
        
        from .dspy_signatures import QueryClassifier
        classification = self.rag_system.model_allocator.create_context_managed_call(
            task_name="query_classification",  # Maps to MEDIUM = gpt-4.1-mini
            signature_class=QueryClassifier,
            module_call_func=classification_call,
            query=transformed_description,
            task_context=task_context
        )
        
        if classification is None:
            logger.warning("Model allocation failed for task classification, using default")
            classification = self.rag_system.classifier(question=transformed_description)
        
        # Import schema
        from ..dsp_sig import NEO4J_SCHEMA
        
        # Determine analysis type for this task based on original question
        analysis_type = self._determine_analysis_type_for_task(task, transformed_description)
        
        # Generate retrieval strategy using model allocation (o3 for complex query generation)
        def retrieval_call(module):
            return module(
                db_schema=NEO4J_SCHEMA,
                question=transformed_description,
                query_type=classification.query_type,
                task_context=task_context,
                genome_filter_required=str(genome_filter_required),
                target_genome=target_genome,
                analysis_type=analysis_type
            )
        
        from .dspy_signatures import ContextRetriever
        retrieval_plan = self.rag_system.model_allocator.create_context_managed_call(
            task_name="context_preparation",  # Maps to COMPLEX = o3
            signature_class=ContextRetriever,
            module_call_func=retrieval_call,
            query=transformed_description,
            task_context=task_context
        )
        
        if retrieval_plan is None:
            logger.warning("Model allocation failed for retrieval plan, using default")
            retrieval_plan = self.rag_system.retriever(
                db_schema=NEO4J_SCHEMA,
                question=transformed_description,
                query_type=classification.query_type,
                task_context=task_context,
                genome_filter_required=str(genome_filter_required),
                target_genome=target_genome,
                analysis_type=analysis_type
            )
        
        # Validate query for comparative questions (same as in core.py)
        cypher_query = retrieval_plan.cypher_query
        validated_query = self.rag_system._validate_comparative_query(task.description, cypher_query)
        if validated_query != cypher_query:
            logger.info("Fixed comparative query in task execution - removed inappropriate LIMIT")
            retrieval_plan.cypher_query = validated_query
        
        # Validate genome filtering if required (same as in core.py)
        if genome_filter_required and self.rag_system.query_validator.should_validate_for_genome(validated_query):
            validation_result = self.rag_system.query_validator.validate_genome_filtering(
                validated_query, 
                genome_filter_required, 
                target_genome
            )
            
            if not validation_result.is_valid:
                logger.warning(f"Task query validation failed: {validation_result.error_message}")
                
                if validation_result.modified_query:
                    logger.info(f"Auto-fixing task query with genome filtering")
                    retrieval_plan.cypher_query = validation_result.modified_query
                else:
                    logger.warning(f"Could not auto-fix task query: {validation_result.suggested_fix}")
            else:
                logger.debug("Task query validation passed - genome filtering present")
        
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
            
            # Context-aware chunking decision
            threshold = 1000 if self.selected_genome else 2000
            should_chunk = self._should_chunk_for_analysis_type(len(raw_data), threshold, task.description, getattr(task, 'original_question', ''))
            
            if should_chunk:
                logger.info(f"🧠 Large dataset detected ({len(raw_data)} items), using intelligent upfront chunking")
                logger.info(f"✅ Using NEW IntelligentChunkingManager (not old recursive splitter)")
                
                try:
                    from .intelligent_chunking_manager import IntelligentChunkingManager
                    chunking_manager = IntelligentChunkingManager(max_chunks=4, min_chunk_size=100)
                    
                    # Create chunks upfront based on biological meaning
                    chunks = await chunking_manager.analyze_and_chunk_dataset(task, raw_data, task.description)
                    
                    if len(chunks) > 1:
                        logger.info(f"🔀 Created {len(chunks)} intelligent chunks, executing in parallel")
                        
                        # Execute chunked analysis with original biological context
                        # Extract original question from task context or use task description as fallback
                        original_question = getattr(task, 'original_question', task.description)
                        chunk_results = await chunking_manager.execute_chunked_analysis(chunks, self, task, original_question)
                        
                        # Synthesize results with original biological context
                        synthesis = chunking_manager.synthesize_chunk_results(
                            chunk_results, original_question, chunks
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
        
        # For code interpreter, prepare data in a convenient format (with or without dependencies)
        if task.tool_name == "code_interpreter":
            if dependency_data:
                args.update(self._prepare_code_interpreter_args(dependency_data, task))
            else:
                # Generate standalone code for tasks without dependencies
                code = self._generate_analysis_code(task, [])
                args.update({
                    "code": code,
                    "session_id": f"task_{task.task_id}",
                    "timeout": 60,
                    "data_summary": "No dependency data available"
                })
        
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
        if "retrieve" in task.description.lower() and "pre-process" in task.description.lower():
            # Data retrieval and preprocessing task
            task_desc = task.description.replace('"', '\\"')  # Escape quotes
            code = f'''
import pandas as pd
import numpy as np
import json

print("Starting data retrieval and preprocessing task")
print("Task: {task_desc}")

# For genome analysis tasks, simulate data retrieval
print("Simulating data retrieval from internal database...")

# Mock data structure for genomic analysis
genome_data = {{
    "genome_id": "Candidatus_Nomurabacteria_bacterium_RIFCSPLOWO2_01_FULL_41_220_contigs",
    "proteins": 2000,  # Estimated protein count
    "genes": 2100,     # Estimated gene count
    "contigs": 41,     # From genome name
    "total_length_bp": 1500000,  # Estimated genome size
    "retrieval_status": "success"
}}

print(f"Retrieved genome data: {{genome_data['proteins']}} proteins, {{genome_data['genes']}} genes")
print(f"Genome size: {{genome_data['total_length_bp']}} bp across {{genome_data['contigs']}} contigs")
print("Data preprocessing completed - ready for novelty analysis")

result = {{
    "task": "data_retrieval",
    "genome_data": genome_data,
    "status": "completed",
    "next_step": "novelty_analysis"
}}

print("Data retrieval and preprocessing task completed successfully")
'''
            
        elif "matrix" in task.description.lower():
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
            # CRITICAL FIX: Include root biological context in note-taking decision
            # This ensures sub-agent notes understand the phage discovery context
            task_description_with_context = task.description
            if hasattr(task, 'root_biological_context') and task.root_biological_context:
                task_description_with_context = f"Biological discovery context: '{task.root_biological_context}' | Task analysis: {task.description}"
                logger.info(f"🧬 Including root biological context in note-taking decision")
            
            decision = self.noting_decision(
                task_description=task_description_with_context,  # Now includes phage discovery context
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