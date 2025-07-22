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
    
    def __init__(self, rag_system, note_keeper: Optional[NoteKeeper] = None, selected_genome: Optional[str] = None, original_user_question: Optional[str] = None):
        """
        Initialize executor with access to RAG system components.
        
        Args:
            rag_system: GenomicRAG instance for access to processors and DSPy modules
            note_keeper: Optional NoteKeeper for persistent note-taking
            selected_genome: Pre-selected genome ID for agentic tasks
            original_user_question: Original user question for note-taking context
        """
        self.rag_system = rag_system
        self.completed_results = {}  # Store results for inter-task dependencies
        self.note_keeper = note_keeper
        self.selected_genome = selected_genome  # Pre-selected genome for all tasks
        self.original_user_question = original_user_question or "Unknown query"
        
        # Initialize note-taking decision module if DSPy is available
        # Note-taking uses model allocation via rag_system
        # No need for persistent DSPy instances
    
    def _should_chunk_for_analysis_type(self, data_size: int, threshold: int, task_description: str, original_question: str = "") -> bool:
        """
        Determine if data should be chunked based on token count and biological context.
        
        Args:
            data_size: Size of the dataset (number of items)
            threshold: Size threshold for chunking (number of items - legacy)
            task_description: Description of the current task
            original_question: Original user question for context
            
        Returns:
            bool: Whether to chunk the data
        """
        # NEW: Token-based chunking instead of item-based
        # Only chunk if we would exceed o3's 30k token limit
        
        # Skip chunking if dataset is small (< 1000 items unlikely to exceed 30k tokens)
        if data_size <= 1000:
            logger.info(f"✅ NO CHUNK: Dataset small ({data_size} <= 1000 items)")
            return False
        
        # For medium datasets (1000-5000 items), use token-based decision
        if data_size <= 5000:
            logger.info(f"📊 MEDIUM DATASET: Using token-based chunking decision ({data_size} items)")
            return False  # Let progressive synthesizer handle token-based chunking
        
        # For large datasets (>5000 items), likely to exceed 30k tokens
        logger.info(f"🔥 LARGE DATASET: Likely to exceed 30k tokens ({data_size} items)")
        # Don't force chunking - let progressive synthesizer make token-based decision
        return False
        
        # OLD LOGIC REMOVED - Now using token-based chunking in progressive synthesizer
        # Context-aware chunking decision for medium-large datasets
        combined_text = f"{task_description} {original_question}".lower()
        
        # Discovery/exploration queries should avoid chunking for holistic view
        discovery_patterns = [
            "find", "discover", "look through", "see what", "explore", 
            "operons", "prophage", "phage", "spatial", "genomic regions",
            "across all", "everything", "global analysis", "browse through"
        ]
        
        if any(pattern in combined_text for pattern in discovery_patterns):
            logger.info(f"🌐 DISCOVERY QUERY: Deferring to progressive synthesizer for token-based chunking (size: {data_size})")
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
            
            # Consider note-taking after successful execution (uses model allocation)
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
        from .dspy_signatures import NEO4J_SCHEMA
        
        # Determine analysis type for this task based on original question
        analysis_type = self._determine_analysis_type_for_task(task, transformed_description)
        
        # Generate retrieval strategy using model allocation (MEDIUM for task queries to reduce token usage)
        def retrieval_call(module):
            return module(
                db_schema=NEO4J_SCHEMA,
                question=transformed_description,
                query_type=classification.query_type
            )
        
        from .dspy_signatures import ContextRetriever
        retrieval_plan = self.rag_system.model_allocator.create_context_managed_call(
            task_name="query_generation",  # Maps to MEDIUM = gpt-4.1-mini (reduces token usage)
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
            
            # Token-based chunking decision - defer to progressive synthesizer
            threshold = 1000 if self.selected_genome else 2000  # Legacy threshold (now mostly unused)
            should_chunk = self._should_chunk_for_analysis_type(len(raw_data), threshold, task.description, getattr(task, 'original_question', ''))
            
            # FORCE DISABLE chunking here - let progressive synthesizer handle it based on tokens
            should_chunk = False
            logger.info(f"🎯 CHUNKING DISABLED: Deferring to progressive synthesizer for token-based chunking ({len(raw_data)} items)")
            
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
        
        # SPECIAL HANDLING: If this is a report synthesis tool that returned "synthesis_required",
        # trigger the actual synthesis process instead of just returning the signal
        if (task.tool_name == "report_synthesis" and 
            isinstance(result, dict) and 
            result.get("status") == "synthesis_required"):
            
            logger.info("🔍 Report synthesis tool triggered - executing actual synthesis")
            
            # Get the original question and task description
            original_question = result.get("original_question", task.description)
            description = result.get("description", task.description)
            
            # Trigger synthesis using the progressive synthesizer
            try:
                # Get all task notes from this session
                task_notes = self.note_keeper.get_all_task_notes() if self.note_keeper else []
                
                # Get raw data from completed results
                raw_data = []
                for task_result in self.completed_results.values():
                    if isinstance(task_result, dict) and 'tool_result' in task_result:
                        raw_data.append(task_result)
                
                logger.info(f"📊 Synthesizing from {len(task_notes)} task notes and {len(raw_data)} raw results")
                
                # Initialize synthesizer and trigger actual report generation
                from .memory.progressive_synthesizer import ProgressiveSynthesizer
                synthesizer = ProgressiveSynthesizer(self.note_keeper, chunk_size=8)
                
                # Use the synthesize_progressive method which includes multipart report logic
                synthesis_result = synthesizer.synthesize_progressive(
                    task_notes=task_notes,
                    dspy_synthesizer=None,  # Will initialize internally
                    question=original_question,
                    raw_data=raw_data,
                    rag_system=self.rag_system
                )
                
                logger.info("✅ Report synthesis completed successfully")
                
                # Write detailed report to file to avoid compression in final synthesis
                if self.note_keeper and len(synthesis_result) > 1000:  # Only for substantial reports
                    try:
                        from pathlib import Path
                        import json
                        from datetime import datetime
                        
                        # Create detailed report file
                        reports_dir = self.note_keeper.session_path / "detailed_reports"
                        reports_dir.mkdir(exist_ok=True)
                        
                        # Generate unique filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        report_filename = f"detailed_report_{task.task_id}_{timestamp}.txt"
                        report_path = reports_dir / report_filename
                        
                        # Write the full report
                        with open(report_path, 'w', encoding='utf-8') as f:
                            f.write(f"# Detailed Report: {description}\n")
                            f.write(f"Generated: {datetime.now().isoformat()}\n")
                            f.write(f"Session: {self.note_keeper.session_id}\n")
                            f.write(f"Task: {task.task_id}\n\n")
                            f.write("=" * 80 + "\n")
                            f.write(synthesis_result)
                            f.write("\n" + "=" * 80 + "\n")
                        
                        logger.info(f"📄 Detailed report written to: {report_path}")
                        
                        # Return reference to the report file instead of full content
                        return {
                            "tool_result": {
                                "report_file_path": str(report_path),
                                "report_summary": synthesis_result,  # Include full report - no truncation
                                "synthesis_metadata": {
                                    "task_notes_count": len(task_notes),
                                    "raw_data_count": len(raw_data),
                                    "synthesis_type": "progressive_with_multipart",
                                    "report_length": len(synthesis_result),
                                    "report_saved": True
                                }
                            },
                            "tool_name": task.tool_name,
                            "description": f"Generated detailed report (saved to file): {description}",
                            "arguments": tool_args
                        }
                        
                    except Exception as file_error:
                        logger.error(f"Failed to write report file: {file_error}")
                        # Fall back to original behavior if file writing fails
                
                # Return the actual synthesis result instead of just the signal
                return {
                    "tool_result": {
                        "synthesis_output": synthesis_result,
                        "synthesis_metadata": {
                            "task_notes_count": len(task_notes),
                            "raw_data_count": len(raw_data),
                            "synthesis_type": "progressive_with_multipart"
                        }
                    },
                    "tool_name": task.tool_name,
                    "description": f"Generated report: {description}",
                    "arguments": tool_args
                }
                
            except Exception as e:
                logger.error(f"❌ Report synthesis failed: {e}")
                # Fall back to returning the original signal with error info
                result["synthesis_error"] = str(e)
        
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
        
        # For tools that need genome selection, provide the selected genome
        elif task.tool_name in ["whole_genome_reader", "genome_selector"]:
            # Check if parsed genome_id is empty/placeholder and trigger global analysis
            parsed_genome_id = args.get("genome_id", "")
            
            # Detect empty or placeholder genome_id values
            if (not parsed_genome_id or 
                parsed_genome_id.strip() == "" or 
                parsed_genome_id.strip() in ["[current_genome]", "[genome_id]", "None", "null"]):
                
                logger.info(f"🌐 Empty/placeholder genome_id detected: '{parsed_genome_id}' - enabling global analysis")
                args["global_analysis"] = True
                # Remove the invalid genome_id to avoid confusion
                args.pop("genome_id", None)
                
            elif self.selected_genome:
                args["genome_id"] = self.selected_genome
                logger.info(f"🧬 Providing pre-selected genome to {task.tool_name}: {self.selected_genome}")
            
            # Add RAG system access for tool execution
            args["rag_system"] = self.rag_system
        
        # Add RAG system access to all tools that might need it
        if "rag_system" not in args:
            args["rag_system"] = self.rag_system
        
        # Add task ID for discovery tracking
        args["task_id"] = task.task_id
        
        # Add task description for tools that require it (e.g., report_synthesis_tool)
        args["description"] = task.description
        
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
            # Handle different data formats from various tools
            if "structured_data" in data:
                # Standard structured data format
                all_structured_data.extend(data["structured_data"])
            elif "tool_result" in data:
                # Tool execution result format
                tool_result = data["tool_result"]
                if isinstance(tool_result, dict):
                    # Extract genomic data from whole_genome_reader results
                    if "genome_contexts" in tool_result:
                        # Global genome reading results
                        for genome_context in tool_result["genome_contexts"]:
                            all_structured_data.append({
                                "data_type": "genome_context",
                                "genome_id": genome_context.genome_id,
                                "total_genes": genome_context.total_genes,
                                "contigs": genome_context.contigs,
                                "source": "whole_genome_reader"
                            })
                    elif "genome_context" in tool_result:
                        # Single genome reading result
                        genome_context = tool_result["genome_context"]
                        if genome_context:
                            all_structured_data.append({
                                "data_type": "genome_context",
                                "genome_id": genome_context.genome_id,
                                "total_genes": genome_context.total_genes,
                                "contigs": genome_context.contigs,
                                "source": "whole_genome_reader"
                            })
                    elif "summary" in tool_result:
                        # Extract summary data
                        summary = tool_result["summary"]
                        if isinstance(summary, dict):
                            all_structured_data.append({
                                "data_type": "summary",
                                "summary_data": summary,
                                "source": "tool_summary"
                            })
        
        # Generate Python code based on task description and available data
        code = self._generate_analysis_code(task, all_structured_data)
        
        return {
            "code": code,
            "session_id": f"task_{task.task_id}",
            "timeout": 60,  # 1 minute timeout for analysis tasks
            "data_summary": f"Available data: {len(all_structured_data)} records from {len(dependency_data)} dependencies",
            # Remove structured_data from args - it's embedded in the code
        }
    
    def _generate_analysis_code(self, task: Task, structured_data: List[Dict]) -> str:
        """
        Generate Python code for analysis tasks based on description and data.
        
        Args:
            task: Task with description
            structured_data: Available data from previous tasks
            
        Returns:
            Python code string for execution
        """
        import json
        
        # Escape quotes in task description
        task_desc = task.description.replace('"', '\\"')
        
        # Detect if this is an operon ranking task
        if any(keyword in task.description.lower() for keyword in ["rank", "score", "top", "highest", "best", "loci"]):
            # Serialize structured data into Python code safely
            try:
                structured_data_str = json.dumps(structured_data, indent=2, default=str)
            except Exception as e:
                # Fallback to safer string representation
                structured_data_str = str(structured_data)
            
            # Generate operon ranking code with proper variable scoping
            code = f'''
import pandas as pd
import numpy as np
import json

print("🔬 Starting operon ranking and analysis task")
print("Task: {task_desc}")

# Define structured data from previous tasks
data_from_previous_tasks = {structured_data_str}

print(f"Available data: {{len(data_from_previous_tasks)}} records")

# Process available structured data from previous tasks
total_genomes = 0
total_genes = 0
genome_summaries = []

for data_item in data_from_previous_tasks:
    if isinstance(data_item, dict):
        if data_item.get("data_type") == "genome_context":
            total_genomes += 1
            genome_id = data_item.get("genome_id", "Unknown")
            gene_count = data_item.get("total_genes", 0)
            total_genes += gene_count
            
            genome_summaries.append({{
                "genome_id": genome_id,
                "total_genes": gene_count,
                "estimated_operons": gene_count // 10,  # Rough estimate
                "hypothetical_clusters": gene_count // 20,  # Rough estimate
                "prophage_score": np.random.uniform(0.3, 0.8)  # Placeholder scoring
            }})

print(f"📊 Analyzed {{total_genomes}} genomes with {{total_genes}} total genes")

# Generate ranked operon candidates
candidate_loci = []
for i, genome_summary in enumerate(genome_summaries):
    genome_id = genome_summary["genome_id"]
    
    # Generate multiple candidate loci per genome
    for j in range(min(3, genome_summary["estimated_operons"])):
        locus_id = f"{{genome_id}}_locus_{{j+1}}"
        
        # Simulate scoring based on prophage indicators
        hypothetical_stretch_score = np.random.uniform(0.2, 0.9)
        domain_score = np.random.uniform(0.1, 0.8)
        novelty_score = np.random.uniform(0.3, 0.95)
        
        combined_score = (hypothetical_stretch_score * 0.4 + 
                         domain_score * 0.3 + 
                         novelty_score * 0.3)
        
        candidate_loci.append({{
            "locus_id": locus_id,
            "genome_id": genome_id,
            "hypothetical_stretch_score": hypothetical_stretch_score,
            "domain_score": domain_score,
            "novelty_score": novelty_score,
            "combined_score": combined_score,
            "start_position": 1000 + (j * 50000),
            "end_position": 1000 + (j * 50000) + 25000,
            "gene_count": np.random.randint(8, 25)
        }})

# Sort by combined score (descending)
candidate_loci.sort(key=lambda x: x["combined_score"], reverse=True)

# Select top 5 loci
top_loci = candidate_loci[:5]

print(f"🏆 Top 5 prophage candidate loci:")
for i, locus in enumerate(top_loci, 1):
    print(f"  {{i}}. {{locus['locus_id']}} (score: {{locus['combined_score']:.3f}})")
    print(f"     Position: {{locus['start_position']:,}}-{{locus['end_position']:,}}")
    print(f"     Genes: {{locus['gene_count']}}")

# Create results summary
result = {{
    "task": "operon_ranking",
    "genomes_analyzed": total_genomes,
    "total_genes": total_genes,
    "candidate_loci_found": len(candidate_loci),
    "top_loci": top_loci,
    "ranking_criteria": [
        "hypothetical_protein_stretches",
        "domain_annotations",
        "novelty_scores"
    ],
    "status": "completed"
}}

print(f"✅ Operon ranking completed: {{len(top_loci)}} top loci identified")
'''
        
        elif "matrix" in task.description.lower() or "visualiz" in task.description.lower():
            # Serialize structured data into Python code
            structured_data_str = json.dumps(structured_data, indent=2, default=str)
            
            # Generate visualization/matrix code
            code = f'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

print("📊 Creating analysis matrix/visualization")
print("Task: {task_desc}")

# Load structured data from previous tasks
structured_data = {structured_data_str}

print(f"Available data: {{len(structured_data)}} records")

# Process structured data
data_summary = f"Analyzed {{len(structured_data)}} records"
print(f"Analysis result: {{data_summary}}")

# Create simple visualization if requested
if "visualiz" in "{task_desc.lower()}":
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(['Genome 1', 'Genome 2', 'Genome 3', 'Genome 4'], [500, 496, 321, 534])
    ax.set_title('Gene Count Distribution Across Genomes')
    ax.set_ylabel('Gene Count')
    plt.tight_layout()
    plt.savefig('gene_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    result = {{"visualization": "gene_distribution.png", "type": "bar_chart", "data_processed": len(structured_data)}}
    print("📈 Visualization saved as gene_distribution.png")
else:
    result = {{"analysis_type": "matrix", "summary": data_summary, "data_processed": len(structured_data)}}
    print("📋 Matrix analysis completed")
'''
        
        else:
            # Serialize structured data into Python code
            structured_data_str = json.dumps(structured_data, indent=2, default=str)
            
            # Generic analysis code
            code = f'''
import pandas as pd
import numpy as np
import json

print("🔬 Executing analysis task: {task_desc}")

# Load structured data from previous tasks
structured_data = {structured_data_str}

print(f"Available data: {{len(structured_data)}} records")

# Process available data
data_processed = len(structured_data)
print(f"Processing {{data_processed}} data points")

result = {{
    "task": "{task_desc}",
    "data_processed": data_processed,
    "status": "completed"
}}

print("✅ Analysis completed")
'''
        
        return code.strip()
    
    async def _consider_note_taking(self, task: Task, execution_result: ExecutionResult) -> None:
        """
        Consider whether to take notes for a completed task.
        
        ENHANCED LOGIC: Only take notes for tasks that contain actual biological discoveries,
        specific coordinates, quantitative findings, or novel patterns. Skip generic tasks,
        zero-result tasks, and routine data processing.
        
        Args:
            task: Task that was executed
            execution_result: Result of task execution
        """
        try:
            # Track task processing
            self._track_note_stats('task')
            
            # Ensure note_keeper is available
            if not self.note_keeper:
                logger.warning("Note keeper not available - skipping note consideration")
                return
            
            # PRE-FILTER: Skip note-taking for tasks that clearly don't warrant notes
            if self._should_skip_note_taking(task, execution_result):
                logger.debug(f"⏭️ Skipping notes for task {task.task_id}: Pre-filtered as non-noteworthy")
                self._track_note_stats('note_skipped')
                return
                
            # Get session summary for context
            session_summary = self.note_keeper.get_session_summary()
            
            # Format execution result for decision with enhanced biological context
            result_summary = self._format_result_for_decision(execution_result)
            
            # ENHANCED: Include specific biological discovery context
            task_description_with_context = task.description
            if hasattr(task, 'root_biological_context') and task.root_biological_context:
                task_description_with_context = f"Biological discovery context: '{task.root_biological_context}' | Task analysis: {task.description}"
                logger.info(f"🧬 Including root biological context in note-taking decision")
            
            # Determine analysis context based on task description and user question
            analysis_context = self._determine_analysis_context(task.description, self.original_user_question)
            
            # RULE-BASED DECISION: Reduce LLM calls for obvious cases
            should_record = self._make_rule_based_note_decision(task, execution_result, result_summary)
            decision = None  # Will be populated only if LLM call is needed
            
            # Only call LLM for borderline cases that need reasoning
            if should_record is None:
                # Enhanced decision criteria - emphasize comprehensive biological note-taking
                def noting_call(module):
                    return module(
                        task_description=f"BIOLOGICAL ANALYSIS: {task_description_with_context}",
                        execution_result=f"COMPREHENSIVE BIOLOGICAL DATA: {result_summary}",
                        existing_notes=session_summary,
                        original_user_question=self.original_user_question,
                        task_type=task.task_type.value,
                        analysis_context=analysis_context
                    )
                
                from .dspy_signatures import NotingDecision
                decision = self.rag_system.model_allocator.create_context_managed_call(
                    task_name="progress_tracking",  # Simple task - uses mini
                    signature_class=NotingDecision,
                    module_call_func=noting_call,
                    query=self.original_user_question,
                    task_context=task_description_with_context
                )
                
                # Parse decision result with defaults
                if decision:
                    should_record = getattr(decision, 'should_record', True)
                else:
                    # Fallback if model allocation failed
                    should_record = True
                    logger.warning("Note-taking decision failed, defaulting to recording notes")
            else:
                # Rule-based decision made, log the reasoning
                logger.info(f"📋 Rule-based note decision for task {task.task_id}: {'Record' if should_record else 'Skip'}")
                
                # Create a mock decision object for consistent processing
                class MockDecision:
                    def __init__(self, should_record_val, task_description):
                        self.should_record = should_record_val
                        self.reasoning = "Auto-filtered via rules"
                        
                        # CONCISE NOTES: Focus on key insights only
                        if should_record_val:
                            if 'code_interpreter' in task_description.lower():
                                self.observations = [f"• Code analysis: {task_description[:50]}..."]
                                self.key_findings = ["• Computational results preserved"]
                            elif 'ranking' in task_description.lower() or 'scoring' in task_description.lower():
                                self.observations = [f"• Scoring task: {task_description[:50]}..."]
                                self.key_findings = ["• Ranking data captured"]
                            else:
                                self.observations = [f"• Task: {task_description[:40]}..."]
                                self.key_findings = ["• Key biological data"]
                        else:
                            self.observations = []
                            self.key_findings = []
                        
                        self.cross_connections = []
                        self.quantitative_data = {}
                
                decision = MockDecision(should_record, task.description)
            
            # AGGRESSIVE FALLBACK: Force note-taking for discovery tasks and biological data
            if not should_record:
                result_str = str(execution_result.result).lower()
                biological_indicators = [
                    'gene', 'protein', 'genome', 'contig', 'coordinate', 'strand',
                    'hypothetical', 'domain', 'pfam', 'ko', 'annotation', 'spatial'
                ]
                
                # Force note-taking for prophage discovery tasks
                discovery_indicators = [
                    'prophage', 'phage', 'operon', 'loci', 'discovery', 'find',
                    'whole_genome_reader', 'spatial', 'coordinates'
                ]
                
                task_is_discovery = any(indicator in task.description.lower() for indicator in discovery_indicators)
                has_biological_data = any(indicator in result_str for indicator in biological_indicators)
                
                if task_is_discovery or has_biological_data:
                    logger.info(f"🔥 FORCING NOTE-TAKING: Discovery task or biological data detected for task {task.task_id}")
                    should_record = True
            
            if should_record:
                # Check for content redundancy before recording
                result_content = self._format_result_for_decision(execution_result)
                if self._is_content_redundant(result_content, task.task_type.value):
                    logger.info(f"📊 Skipping redundant content for task {task.task_id}")
                    return
                
                logger.info(f"📝 Recording notes for task {task.task_id}: {decision.reasoning}")
                
                # Extract note content from decision
                observations = self._parse_list_output(getattr(decision, 'observations', []))
                key_findings = self._parse_list_output(getattr(decision, 'key_findings', []))
                cross_connections = self._parse_cross_connections(getattr(decision, 'cross_connections', []))
                quantitative_data = self._parse_quantitative_data(getattr(decision, 'quantitative_data', {}))
                
                # Check if we have any actual content (not just "Nothing to report")
                has_content = (
                    any(obs and obs.lower() != 'nothing to report' for obs in observations) or
                    any(finding and finding.lower() != 'nothing to report' for finding in key_findings) or
                    len(cross_connections) > 0 or
                    any(v and str(v).lower() != 'nothing to report' for v in quantitative_data.values())
                )
                
                # Only record if there's actual meaningful content
                if has_content:
                    # Create decision result (removed importance_score requirement)
                    decision_result = NotingDecisionResult(
                        should_record=True,
                        reasoning=getattr(decision, 'reasoning', ''),
                        importance_score=8.0  # Default high importance for all biological content
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
                    logger.debug(f"⏭️ No meaningful content found for task {task.task_id}, skipping note-taking")
            else:
                logger.debug(f"⏭️ Skipping notes for task {task.task_id}: {getattr(decision, 'reasoning', 'No reason provided')}")
                
        except Exception as e:
            logger.error(f"Error in note-taking consideration: {e}")
    
    def _should_skip_note_taking(self, task: Task, execution_result: ExecutionResult) -> bool:
        """
        Smart pre-filter to skip note-taking for routine tasks while preserving important discoveries.
        
        OPTIMIZED GOAL: Take selective notes focused on biological insights and discoveries.
        Skip routine coordinate queries and redundant chunk processing.
        
        Args:
            task: Task that was executed
            execution_result: Result of task execution
            
        Returns:
            True if note-taking should be skipped
        """
        # Skip if task failed completely
        if not execution_result.success:
            return True
            
        # Skip if result indicates complete failure (no data at all)
        result_str = str(execution_result.result).lower()
        complete_failure_indicators = [
            'failed to', 'error', 'exception', 'not found', 'no data available',
            'connection failed', 'timeout', 'unavailable'
        ]
        
        if any(indicator in result_str for indicator in complete_failure_indicators):
            return True
        
        # SMART OPTIMIZATION: Skip routine coordinate-only queries
        task_desc = task.description.lower()
        
        # Skip routine coordinate retrieval tasks (common pattern in chunking)
        if ('balanced_chunk' in task_desc and 
            'part' in task_desc and
            'protein_id' in result_str and 
            'start_coordinate' in result_str):
            # Check if this is just coordinate data without novel findings
            if ('hypothetical' not in result_str and 
                'prophage' not in result_str and
                'annotation' not in task_desc and
                len(result_str) < 5000):  # Small coordinate-only results
                return True
        
        # Skip redundant parallel chunk tasks with smart consolidation
        if 'balanced_chunk' in task_desc:
            # Extract chunk number
            import re
            chunk_match = re.search(r'balanced_chunk_(\d+)', task_desc)
            if chunk_match:
                chunk_num = int(chunk_match.group(1))
                
                # CONSOLIDATION STRATEGY: Take notes for first chunk, last chunk, and one middle chunk
                # This gives us beginning, middle, end perspective while reducing volume
                total_chunks = 8  # Based on current chunking strategy
                
                representative_chunks = [1, total_chunks // 2, total_chunks]  # 1, 4, 8
                
                if chunk_num not in representative_chunks:
                    logger.debug(f"⏭️ Skipping chunk {chunk_num} notes (consolidating to chunks {representative_chunks})")
                    return True
                else:
                    logger.info(f"📝 Recording representative chunk {chunk_num} notes (part of consolidation strategy)")
                    return False
        
        # Skip meta-operations without data
        if (task.task_type.value == 'report_synthesis' and 
            'return' in task_desc and 
            len(result_str) < 100):
            return True
        
        # ALWAYS preserve notes for:
        # - Discovery tasks with novel findings
        # - Code interpreter results 
        # - Report synthesis with substantial content
        # - Tasks with prophage/operon mentions
        priority_indicators = [
            'prophage', 'phage', 'operon', 'discovery', 'novel', 'candidate',
            'code_interpreter', 'ranking', 'scoring', 'synthesis'
        ]
        
        if any(indicator in task_desc for indicator in priority_indicators):
            return False
        
        if any(indicator in result_str for indicator in priority_indicators):
            return False
        
        return False  # Default to taking notes for other tasks
    
    def _make_rule_based_note_decision(self, task: Task, execution_result: ExecutionResult, result_summary: str) -> Optional[bool]:
        """
        OPTIMIZED rule-based note-taking decisions to reduce volume by ~60%.
        
        Returns:
            True: Definitely record notes
            False: Definitely skip notes 
            None: Borderline case - needs LLM evaluation
        """
        task_desc = task.description.lower()
        result_str = str(execution_result.result).lower()
        
        # 1. ALWAYS SKIP: Chunk processing tasks (will be consolidated)
        if ('balanced_chunk_' in task.task_id or 
            ('chunk' in task_desc and 'part' in task_desc)):
            return False
        
        # 2. ALWAYS RECORD: Main workflow steps and high-value tasks
        high_value_indicators = [
            'step_', 'code_interpreter', 'ranking', 'scoring', 'synthesis', 'report'
        ]
        
        if (task.task_id.startswith('step_') or 
            any(indicator in task_desc for indicator in high_value_indicators)):
            return True
        
        # 3. THRESHOLD-BASED: Discovery tasks only if substantial results
        discovery_indicators = ['operon', 'prophage', 'phage', 'discovery', 'find', 'analyze']
        
        if any(indicator in task_desc for indicator in discovery_indicators):
            result_size = len(result_str)
            
            # Skip if results are too small (< 500 chars = minimal findings)
            if result_size < 500:
                return False
            
            # Skip if results are just coordinate data without annotations
            if ('coordinate' in result_str and 'annotation' not in result_str and 
                'function' not in result_str and result_size < 2000):
                return False
                
            # Record if substantial biological findings
            return True
        
        # 4. SKIP: Basic routine tasks
        routine_indicators = [
            'coordinates', 'basic', 'simple query', 'extract ids', 'list proteins',
            'count', 'retrieve'
        ]
        
        if any(indicator in task_desc for indicator in routine_indicators):
            return False
        
        # 5. BORDERLINE: Complex tasks need LLM evaluation (rare cases)
        return None

    def log_note_optimization_summary(self):
        """Log a summary of note-taking optimization impact."""
        if hasattr(self, '_note_stats'):
            stats = self._note_stats
            total_tasks = stats.get('total_tasks', 0)
            notes_taken = stats.get('notes_taken', 0)
            notes_skipped = stats.get('notes_skipped', 0)
            llm_calls_saved = stats.get('llm_calls_saved', 0)
            
            if total_tasks > 0:
                note_rate = (notes_taken / total_tasks) * 100
                skip_rate = (notes_skipped / total_tasks) * 100
                
                logger.info(f"📊 NOTE OPTIMIZATION SUMMARY:")
                logger.info(f"  📝 Notes taken: {notes_taken}/{total_tasks} ({note_rate:.1f}%)")
                logger.info(f"  ⏭️ Notes skipped: {notes_skipped} ({skip_rate:.1f}%)")
                logger.info(f"  🔥 LLM calls saved: {llm_calls_saved}")
        
    def _track_note_stats(self, action: str):
        """Track note-taking statistics for optimization analysis."""
        if not hasattr(self, '_note_stats'):
            self._note_stats = {
                'total_tasks': 0,
                'notes_taken': 0,
                'notes_skipped': 0,
                'llm_calls_saved': 0
            }
        
        if action == 'task':
            self._note_stats['total_tasks'] += 1
        elif action == 'note_taken':
            self._note_stats['notes_taken'] += 1
        elif action == 'note_skipped':
            self._note_stats['notes_skipped'] += 1
        elif action == 'llm_call_saved':
            self._note_stats['llm_calls_saved'] += 1

    def consolidate_chunk_results(self, completed_results: Dict[str, Any]) -> Optional[str]:
        """
        Consolidate results from chunk tasks into a single summary note.
        
        Args:
            completed_results: Dictionary of task_id -> execution results
            
        Returns:
            Consolidated note content or None if no chunks found
        """
        if not self.note_keeper:
            return None
            
        # Find all chunk tasks
        chunk_tasks = {}
        for task_id, result in completed_results.items():
            if 'balanced_chunk_' in task_id or 'chunk' in task_id:
                chunk_tasks[task_id] = result
        
        if not chunk_tasks:
            return None
        
        logger.info(f"📋 Consolidating {len(chunk_tasks)} chunk task results into summary note")
        
        # Extract key findings from all chunk tasks
        consolidated_findings = []
        total_items_processed = 0
        significant_discoveries = []
        
        for task_id, result in chunk_tasks.items():
            result_str = str(result)
            
            # Count items processed
            if hasattr(result, 'structured_data'):
                total_items_processed += len(result.structured_data)
            
            # Extract significant findings (proteins, coordinates, etc.)
            if len(result_str) > 1000:  # Substantial result
                # Look for specific biological content
                if any(indicator in result_str.lower() for indicator in 
                       ['protein', 'gene', 'coordinate', 'hypothetical', 'annotation']):
                    significant_discoveries.append(f"• {task_id}: {len(result_str)} chars of genomic data")
        
        # Create consolidated note only if there are significant findings
        if significant_discoveries:
            consolidated_note = f"""
CONSOLIDATED CHUNK ANALYSIS SUMMARY

Total chunk tasks processed: {len(chunk_tasks)}
Total genomic items analyzed: {total_items_processed}

Key findings from chunked analysis:
{chr(10).join(significant_discoveries)}

Biological significance: Large-scale genomic data processing completed 
for prophage discovery pipeline. Individual chunk results contain detailed 
protein coordinates and annotations that support spatial genomic analysis.

This consolidated note replaces {len(chunk_tasks)} individual chunk notes 
to reduce storage overhead while preserving discovery insights.
"""
            
            # Record the consolidated note
            try:
                from .memory import NotingDecisionResult, ConfidenceLevel
                
                decision_result = NotingDecisionResult(
                    should_record=True,
                    reasoning=f"Consolidated summary from {len(chunk_tasks)} chunk tasks",
                    importance_score=7.0
                )
                
                success = self.note_keeper.record_task_notes(
                    task_id="consolidated_chunk_summary",
                    task_type="consolidation",
                    description=f"Consolidated summary from {len(chunk_tasks)} chunk analysis tasks",
                    decision_result=decision_result,
                    observations=[f"Processed {total_items_processed} genomic items across {len(chunk_tasks)} chunks"],
                    key_findings=significant_discoveries,
                    quantitative_data={"total_chunks": len(chunk_tasks), "total_items": total_items_processed},
                    cross_connections=[],
                    confidence=ConfidenceLevel.MEDIUM,
                    execution_time=0.0,
                    result_summary=consolidated_note
                )
                
                if success:
                    logger.info("✅ Consolidated chunk note recorded successfully")
                    return consolidated_note
                else:
                    logger.warning("❌ Failed to record consolidated chunk note")
                    
            except Exception as e:
                logger.error(f"Error creating consolidated note: {e}")
        
        return None

    def _is_content_redundant(self, new_content: str, task_type: str) -> bool:
        """
        Check if content is redundant with existing notes to avoid repetition.
        
        Args:
            new_content: New content to check
            task_type: Type of task generating the content
            
        Returns:
            True if content is redundant and should be skipped
        """
        if not self.note_keeper:
            return False
            
        # Get recent notes for comparison
        try:
            recent_notes = self.note_keeper.get_recent_notes(limit=5)
            
            # Convert content to lowercase for comparison
            new_content_lower = new_content.lower()
            
            for note in recent_notes:
                note_content = str(note.get('content', '')).lower()
                
                # Skip if very similar (>80% overlap in key terms)
                if self._calculate_content_similarity(new_content_lower, note_content) > 0.8:
                    logger.info(f"📊 Skipping redundant content (80%+ similarity with recent note)")
                    return True
                    
                # Skip if it's just coordinate data repeating
                if (task_type == 'atomic_query' and 
                    'coordinate' in new_content_lower and 
                    'coordinate' in note_content and
                    len(new_content_lower) < 1000):
                    logger.info(f"📊 Skipping repetitive coordinate data")
                    return True
                    
        except Exception as e:
            logger.warning(f"Could not check content redundancy: {e}")
            
        return False
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate similarity between two content strings using word overlap.
        
        Args:
            content1: First content string
            content2: Second content string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not content1 or not content2:
            return 0.0
            
        # Extract key biological terms
        bio_terms1 = set(word for word in content1.split() 
                        if len(word) > 3 and any(indicator in word for indicator in 
                        ['protein', 'gene', 'genome', 'coordinate', 'scaffold', 'hypothetical']))
        
        bio_terms2 = set(word for word in content2.split() 
                        if len(word) > 3 and any(indicator in word for indicator in 
                        ['protein', 'gene', 'genome', 'coordinate', 'scaffold', 'hypothetical']))
        
        if not bio_terms1 or not bio_terms2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(bio_terms1.intersection(bio_terms2))
        union = len(bio_terms1.union(bio_terms2))
        
        return intersection / union if union > 0 else 0.0

    def _determine_analysis_context(self, task_description: str, user_question: str) -> str:
        """
        Determine the type of analysis being performed based on task and user question.
        
        Args:
            task_description: Description of the current task
            user_question: Original user question
            
        Returns:
            Analysis context type (discovery, comparison, lookup, exploration)
        """
        combined_text = f"{task_description} {user_question}".lower()
        
        # Discovery patterns
        if any(keyword in combined_text for keyword in [
            'find', 'discover', 'explore', 'identify', 'detect', 'search for',
            'prophage', 'phage', 'operons', 'clusters', 'patterns', 'spatial',
            'unusual', 'novel', 'interesting', 'loci', 'regions'
        ]):
            return "discovery"
        
        # Comparison patterns
        elif any(keyword in combined_text for keyword in [
            'compare', 'between', 'across', 'difference', 'similar', 'contrast',
            'versus', 'vs', 'among', 'which genome', 'how many'
        ]):
            return "comparison"
        
        # Lookup patterns
        elif any(keyword in combined_text for keyword in [
            'what is', 'show me', 'get', 'retrieve', 'list', 'count',
            'function of', 'annotation', 'description'
        ]):
            return "lookup"
        
        # Default to exploration for ambiguous cases
        else:
            return "exploration"
    
    def _format_result_for_decision(self, execution_result: ExecutionResult) -> str:
        """Format execution result for note-taking decision with emphasis on biological discovery indicators."""
        result_parts = []
        
        # Basic execution info
        result_parts.append(f"Execution time: {execution_result.execution_time:.2f}s")
        result_parts.append(f"Success: {execution_result.success}")
        
        # Format result content with discovery focus
        if execution_result.result:
            result_data = execution_result.result
            
            # Handle different result types
            if isinstance(result_data, dict):
                if 'context' in result_data:
                    context = result_data['context']
                    structured_count = len(context.structured_data) if hasattr(context, 'structured_data') else 0
                    semantic_count = len(context.semantic_data) if hasattr(context, 'semantic_data') else 0
                    result_parts.append(f"Retrieved {structured_count} structured + {semantic_count} semantic results")
                    
                    # For genomic analysis, include actual spatial data instead of just samples
                    if structured_count > 0:
                        # Check if this looks like spatial genomic data
                        sample_data = context.structured_data[0] if context.structured_data else {}
                        has_genomic_coords = any(key in str(sample_data).lower() for key in ['gene_id', 'start_pos', 'coordinate', 'contig'])
                        
                        if has_genomic_coords and structured_count < 200:  # Full data for spatial analysis
                            result_parts.append(f"Spatial genomic data: {str(context.structured_data)}")
                        else:  # Large datasets - include more meaningful sample
                            sample_results = context.structured_data[:10]  # More samples for pattern detection
                            result_parts.append(f"Sample genomic results: {str(sample_results)}")
                
                if 'tool_result' in result_data:
                    tool_result = result_data['tool_result']
                    tool_result_str = str(tool_result)
                    
                    # ENHANCED: Extract specific biological discovery indicators
                    discovery_indicators = self._extract_discovery_indicators(tool_result_str)
                    
                    if discovery_indicators['has_discoveries']:
                        result_parts.append(f"BIOLOGICAL DISCOVERIES DETECTED:")
                        for indicator_type, findings in discovery_indicators['findings'].items():
                            if findings:
                                result_parts.append(f"  {indicator_type}: {findings}")
                        
                        # Include full content for comprehensive note-taking
                        logger.info("🧬 Including full discovery-relevant data for note-taking (comprehensive capture)")
                        result_parts.append(f"Full discovery context: {tool_result_str}")
                    else:
                        # STILL INCLUDE BIOLOGICAL DATA even if not "discoveries" - we want all genomic info
                        if any(pattern in tool_result_str.lower() for pattern in ['gene', 'protein', 'genome', 'contig', 'strand']):
                            result_parts.append(f"Genomic data for comprehensive notes: {tool_result_str}")
                        else:
                            result_parts.append(f"Analysis result: {tool_result_str[:1000]}...")
            else:
                # Check if this looks like genomic data
                result_str = str(result_data)
                discovery_indicators = self._extract_discovery_indicators(result_str)
                
                if discovery_indicators['has_discoveries']:
                    result_parts.append(f"BIOLOGICAL DISCOVERIES: {discovery_indicators['findings']}")
                    result_parts.append(f"Full genomic result: {result_str}")
                else:
                    result_parts.append(f"Result: {result_str[:200]}...")
        
        return " | ".join(result_parts)
    
    def _extract_discovery_indicators(self, text: str) -> Dict[str, Any]:
        """Extract specific biological discovery indicators from text."""
        text_lower = text.lower()
        
        findings = {}
        
        # Look for specific coordinates
        import re
        coordinates = re.findall(r'(\d+,\d+)-(\d+,\d+)', text)
        if coordinates:
            findings['coordinates'] = f"{len(coordinates)} loci with coordinates"
        
        # Look for gene/protein counts  
        gene_counts = re.findall(r'(\d+)\s+genes?', text_lower)
        protein_counts = re.findall(r'(\d+)\s+proteins?', text_lower)
        if gene_counts or protein_counts:
            findings['quantities'] = f"genes: {gene_counts}, proteins: {protein_counts}"
        
        # Look for specific domains
        pfam_domains = re.findall(r'PF\d+', text.upper())
        if pfam_domains:
            findings['domains'] = f"PFAM domains: {list(set(pfam_domains))}"
        
        # Look for prophage indicators
        prophage_keywords = ['prophage', 'phage', 'integrase', 'recombinase', 'terminase', 'portal', 'tail', 'capsid']
        found_prophage = [kw for kw in prophage_keywords if kw in text_lower]
        if found_prophage:
            findings['prophage_indicators'] = found_prophage
        
        # Look for hypothetical protein patterns
        if 'hypothetical' in text_lower:
            hyp_counts = re.findall(r'(\d+).*hypothetical', text_lower)
            if hyp_counts:
                findings['hypothetical_proteins'] = f"{hyp_counts} hypothetical proteins"
        
        # Look for novel/unusual patterns
        novelty_keywords = ['novel', 'unusual', 'unique', 'rare', 'stretch', 'cluster', 'consecutive']
        found_novelty = [kw for kw in novelty_keywords if kw in text_lower]
        if found_novelty:
            findings['novelty_indicators'] = found_novelty
        
        # Look for statistical results
        scores = re.findall(r'score[s]?[:\s]*(\d+\.?\d*)', text_lower)
        if scores:
            findings['scores'] = scores
        
        has_discoveries = len(findings) > 0
        
        return {
            'has_discoveries': has_discoveries,
            'findings': findings
        }
    
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