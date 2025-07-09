#!/usr/bin/env python3
"""
Core GenomicRAG class with working implementation.
Restored from backup with modular organization.
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio

try:
    import dspy
    from rich.console import Console
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    logging.warning("DSPy not available - install dsp-ml package")

from ..config import LLMConfig
from ..query_processor import Neo4jQueryProcessor, LanceDBQueryProcessor, HybridQueryProcessor
from ..dsp_sig import NEO4J_SCHEMA
from .utils import setup_debug_logging, GenomicContext
from .dspy_signatures import PlannerAgent, QueryClassifier, ContextRetriever, GenomicAnswerer
from .task_management import TaskGraph, Task, TaskType, TaskStatus
from .external_tools import AVAILABLE_TOOLS
from .intelligent_routing import IntelligentRouter
from .genome_scoping import QueryScopeEnforcer
from .context_compression import ContextCompressor

logger = logging.getLogger(__name__)
console = Console()

class GenomicRAG(dspy.Module if DSPY_AVAILABLE else object):
    """
    Main genomic RAG system with working implementation.
    
    Combines structured queries (Neo4j) with semantic search (LanceDB)
    and intelligent code interpreter enhancement.
    """
    
    def __init__(self, config: LLMConfig, chunk_context_size: int = 4096):
        """Initialize the genomic RAG system."""
        if DSPY_AVAILABLE:
            super().__init__()
        
        self.config = config
        self.chunk_context_size = chunk_context_size
        
        # Initialize processors
        self.neo4j_processor = Neo4jQueryProcessor(config)
        self.lancedb_processor = LanceDBQueryProcessor(config)
        self.hybrid_processor = HybridQueryProcessor(config)
        
        # Initialize new intelligent components
        self.intelligent_router = IntelligentRouter()
        self.scope_enforcer = QueryScopeEnforcer()
        self.context_compressor = ContextCompressor()
        
        # Configure DSPy
        self._configure_dspy()
        
        # Initialize DSPy components
        if DSPY_AVAILABLE:
            self.planner = dspy.Predict(PlannerAgent)
            self.classifier = dspy.Predict(QueryClassifier)
            self.retriever = dspy.Predict(ContextRetriever)
            self.answerer = dspy.Predict(GenomicAnswerer)
        
        # Setup debug logging
        setup_debug_logging()
        
        logger.info("🧬 GenomicRAG initialized with working implementation")
    
    def _configure_dspy(self):
        """Configure DSPy with LLM backend."""
        if not DSPY_AVAILABLE:
            return
            
        try:
            # Configure based on available API keys
            api_key = self.config.get_api_key()
            
            if self.config.llm_provider == "openai" and api_key:
                import os
                os.environ['OPENAI_API_KEY'] = api_key
                
                # Use the model from config, fallback to gpt-3.5-turbo
                model_name = getattr(self.config, 'llm_model', 'gpt-3.5-turbo')
                
                # DSPy 2.6+ uses LM with provider/model format
                model_string = f"openai/{model_name}"
                
                # Special handling for OpenAI reasoning models (o1, o3)
                if model_name.startswith(('o1', 'o3')):
                    lm = dspy.LM(model=model_string, temperature=1.0, max_tokens=20000)
                    logger.info(f"DSPy configured with OpenAI reasoning model: {model_string} (temp=1.0, max_tokens=20000)")
                else:
                    lm = dspy.LM(model=model_string, temperature=0.0, max_tokens=2000)
                    logger.info(f"DSPy configured with OpenAI model: {model_string}")
                
                dspy.settings.configure(lm=lm)
                
            elif self.config.llm_provider == "anthropic" and api_key:
                # Anthropic configuration would go here
                import os
                os.environ['ANTHROPIC_API_KEY'] = api_key
                model_name = getattr(self.config, 'llm_model', 'claude-3-haiku-20240307')
                model_string = f"anthropic/{model_name}"
                lm = dspy.LM(model=model_string, max_tokens=1000)
                dspy.settings.configure(lm=lm)
                logger.info(f"DSPy configured with Anthropic model: {model_string}")
                
            else:
                logger.warning("No LLM API key configured for DSPy")
                
        except Exception as e:
            logger.error(f"Failed to configure DSPy: {e}")
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all system components."""
        try:
            health_status = {}
            
            # Check processors
            health_status['neo4j'] = self.neo4j_processor.health_check() if hasattr(self.neo4j_processor, 'health_check') else False
            health_status['lancedb'] = self.lancedb_processor.health_check() if hasattr(self.lancedb_processor, 'health_check') else False
            health_status['hybrid'] = self.hybrid_processor.health_check() if hasattr(self.hybrid_processor, 'health_check') else False
            
            # Check DSPy
            health_status['dspy'] = DSPY_AVAILABLE
            
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                'neo4j': False,
                'lancedb': False, 
                'hybrid': False,
                'dspy': False
            }
    
    async def ask(self, question: str) -> Dict[str, Any]:
        """
        Main method to answer genomic questions with agentic planning.
        
        Args:
            question: Natural language question about genomic data
            
        Returns:
            Dict containing answer, confidence, sources, and metadata
        """
        try:
            console.print(f"🧬 [bold blue]Processing question:[/bold blue] {question}")
            
            if not DSPY_AVAILABLE:
                return {
                    "question": question,
                    "answer": "DSPy not available - install dsp-ml package for full functionality",
                    "confidence": "low",
                    "citations": "",
                    "error": "Missing dependencies"
                }
            
            # STEP 1: Use intelligent router to determine execution strategy
            routing_recommendation = self.intelligent_router.get_routing_recommendation(question)
            console.print(f"🤖 Routing recommendation: {routing_recommendation['recommendation']}")
            console.print(f"💭 Reasoning: {routing_recommendation['reasoning']}")
            console.print(f"🎯 Confidence: {routing_recommendation['confidence']:.2f}")
            
            if routing_recommendation['use_agentic_mode']:
                # AGENTIC PATH: Multi-step task execution
                # Still use DSPy planner for detailed task planning
                planning_result = self.planner(user_query=question)
                
                # Check if we actually have a valid task plan
                task_plan = planning_result.task_plan
                if task_plan == "N/A" or not task_plan or task_plan.strip() == "":
                    console.print("⚠️ [yellow]Agentic mode recommended but no task plan provided, falling back to traditional mode[/yellow]")
                    return await self._execute_traditional_query(question, routing_recommendation)
                return await self._execute_agentic_plan(question, planning_result, routing_recommendation)
            else:
                # TRADITIONAL PATH: Direct query execution
                return await self._execute_traditional_query(question, routing_recommendation)
                
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            
            # Check if this is a repairable error from query processor
            repair_message = None
            if hasattr(self.hybrid_processor, 'neo4j_processor') and hasattr(self.hybrid_processor.neo4j_processor, 'get_last_repair_result'):
                repair_result = self.hybrid_processor.neo4j_processor.get_last_repair_result()
                if repair_result and repair_result.success and repair_result.user_message:
                    repair_message = repair_result.user_message
                    logger.info(f"Using TaskRepairAgent message: {repair_message[:100]}...")
            
            if repair_message:
                return {
                    "question": question,
                    "answer": repair_message,
                    "confidence": "medium - error handled gracefully",
                    "citations": "",
                    "repair_info": "TaskRepairAgent provided helpful guidance"
                }
            else:
                return {
                    "question": question,
                    "answer": f"I encountered an error while processing your question: {str(e)}",
                    "confidence": "low",
                    "citations": "",
                    "error": str(e)
                }
    
    async def _execute_traditional_query(self, question: str, routing_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute traditional single-step query with enhanced genome scoping and compression."""
        console.print("📋 [dim]Using traditional query path[/dim]")
        
        # Step 1: Classify the query type
        classification = self.classifier(question=question)
        console.print(f"📊 Query type: {classification.query_type}")
        console.print(f"💭 Reasoning: {classification.reasoning}")
        
        # Step 2: Generate retrieval strategy
        retrieval_plan = self.retriever(
            db_schema=NEO4J_SCHEMA,
            question=question,
            query_type=classification.query_type
        )
        console.print(f"🔍 Search strategy: {retrieval_plan.search_strategy}")
        
        # Step 2.5: Validate query for comparative questions
        cypher_query = retrieval_plan.cypher_query
        validated_query = self._validate_comparative_query(question, cypher_query)
        if validated_query != cypher_query:
            logger.info("Fixed comparative query - removed inappropriate LIMIT")
            retrieval_plan.cypher_query = validated_query
        
        # Step 3: Enforce genome scoping in generated query
        scoped_query, scope_metadata = self.scope_enforcer.enforce_genome_scope(question, validated_query)
        
        if scope_metadata['scope_applied']:
            console.print(f"🎯 Applied genome scoping: {scope_metadata['scope_reasoning']}")
            retrieval_plan.cypher_query = scoped_query
        
        # Step 4: Execute database queries with fallback logic
        context = await self._retrieve_context_with_fallback(question, classification.query_type, retrieval_plan, scoped_query, cypher_query)
        
        # Check for TaskRepairAgent messages first
        if 'repair_message' in context.metadata:
            logger.info("TaskRepairAgent provided helpful guidance - returning repair message")
            return {
                "question": question,
                "answer": context.metadata['repair_message'],
                "confidence": "medium - error handled gracefully by TaskRepairAgent",
                "citations": "",
                "repair_info": "TaskRepairAgent provided helpful error guidance"
            }
        
        # Check for retrieval errors
        if 'retrieval_error' in context.metadata:
            error_msg = context.metadata['retrieval_error']
            logger.error(f"Context retrieval failed: {error_msg}")
            return {
                "question": question,
                "answer": f"I couldn't retrieve information to answer your question: {error_msg}",
                "confidence": "low",
                "citations": "",
                "error": error_msg
            }
        
        # Step 4: Use compressed context if available, otherwise format normally
        if hasattr(context, 'compressed_context') and context.compressed_context:
            formatted_context = context.compressed_context
            compression_stats = context.metadata.get('compression_stats')
            console.print(f"🗜️ Using compressed context: {compression_stats.original_count} → {compression_stats.compressed_count} results")
        else:
            formatted_context = self._format_context(context)
            compression_stats = None
        
        # Step 5: Generate answer
        answer_result = self.answerer(
            question=question,
            context=formatted_context
        )
        
        # Return structured response
        metadata = {
            "query_type": classification.query_type,
            "search_strategy": retrieval_plan.search_strategy,
            "context_size": len(formatted_context),
            "retrieval_time": context.query_time,
            "total_results": len(context.structured_data) + len(context.semantic_data)
        }
        
        if compression_stats:
            metadata["compression_stats"] = compression_stats
        
        return {
            "question": question,
            "answer": answer_result.answer,
            "confidence": answer_result.confidence,
            "citations": answer_result.citations,
            "query_metadata": metadata
        }
    
    async def _execute_agentic_plan(self, question: str, planning_result, routing_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute multi-step agentic plan using TaskGraph."""
        console.print("🤖 [bold]Using agentic execution path[/bold]")
        console.print(f"📋 Task plan: {planning_result.task_plan}")
        
        try:
            # Import parser and executor
            from .task_plan_parser import TaskPlanParser
            from .task_executor import TaskExecutor
            
            # Step 1: Parse DSPy plan into Task objects
            parser = TaskPlanParser()
            parsed_plan = parser.parse_dspy_plan(planning_result.task_plan)
            
            if not parsed_plan.parsing_success:
                console.print(f"⚠️ [yellow]Plan parsing failed: {parsed_plan.errors}[/yellow]")
                console.print("🔄 [dim]Falling back to traditional mode[/dim]")
                return await self._execute_traditional_query(question)
            
            console.print(f"✅ [green]Successfully parsed {len(parsed_plan.tasks)} tasks[/green]")
            
            # Step 2: Create TaskGraph and add tasks
            graph = TaskGraph()
            for task in parsed_plan.tasks:
                graph.add_task(task)
            
            # Step 3: Execute TaskGraph with dependency resolution
            executor = TaskExecutor(self)
            execution_results = await executor.execute_graph(graph)
            
            # Check execution success
            if not execution_results["success"]:
                console.print("⚠️ [yellow]Task execution failed[/yellow]")
                console.print("🔄 [dim]Falling back to traditional mode[/dim]")
                return await self._execute_traditional_query(question)
            
            console.print(f"✅ [green]Task graph executed successfully[/green]")
            console.print(f"📊 Execution summary: {execution_results['execution_summary']}")
            
            # Step 4: Synthesize final answer from all task results
            return await self._synthesize_agentic_results(question, execution_results)
            
        except Exception as e:
            logger.error(f"Agentic execution failed: {str(e)}")
            console.print(f"⚠️ [yellow]Agentic execution error: {str(e)}[/yellow]")
            console.print("🔄 [dim]Falling back to traditional mode[/dim]")
            return await self._execute_traditional_query(question)
    
    async def _synthesize_agentic_results(self, question: str, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize final answer from agentic task execution results with context compression.
        
        Args:
            question: Original user question
            execution_results: Results from TaskGraph execution
            
        Returns:
            Formatted response dict with answer, confidence, and citations
        """
        logger.info("Synthesizing agentic results into final answer")
        
        try:
            # Import context compression
            from ..context_compression import ContextCompressor
            
            # Collect all completed task results
            completed_results = execution_results.get("completed_results", {})
            execution_summary = execution_results.get("execution_summary", {})
            
            # Organize context data for compression
            context_data = self._organize_results_for_compression(completed_results)
            
            # Initialize context compressor
            compressor = ContextCompressor(self.config.llm_model if hasattr(self.config, 'llm_model') else 'gpt-3.5-turbo')
            
            # Compress context to fit within limits
            compression_result = await compressor.compress_context(context_data, target_tokens=25000)
            
            logger.info(f"Context compression: {compression_result.original_size} -> {compression_result.compressed_size} tokens (ratio: {compression_result.compression_ratio:.2f})")
            
            # Use compressed context for synthesis
            combined_context = compression_result.compressed_content
            
            # Use GenomicAnswerer to synthesize final response
            if combined_context.strip():
                answer_result = self.answerer(
                    question=question,
                    context=combined_context
                )
                
                confidence = answer_result.confidence
                answer = answer_result.answer
                citations = answer_result.citations
            else:
                # Fallback if no context available
                answer = f"I completed a {len(completed_results)}-step analysis for your question about '{question}', but couldn't retrieve specific results to provide a detailed answer."
                confidence = "low"
                citations = "Agentic workflow execution"
            
            # Add execution metadata including compression stats
            total_tasks = execution_summary.get("total", 0)
            completed_tasks = execution_summary.get("completed", 0)
            
            return {
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "citations": citations,
                "query_metadata": {
                    "execution_mode": "agentic",
                    "total_tasks": total_tasks,
                    "completed_tasks": completed_tasks,
                    "execution_summary": execution_summary,
                    "compression_stats": {
                        "original_tokens": compression_result.original_size,
                        "compressed_tokens": compression_result.compressed_size,
                        "compression_ratio": compression_result.compression_ratio,
                        "compression_level": compression_result.compression_level.value,
                        "chunks_processed": compression_result.chunks_processed
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to synthesize agentic results: {str(e)}")
            return {
                "question": question,
                "answer": f"I completed a multi-step analysis but encountered an error while synthesizing the final answer: {str(e)}",
                "confidence": "low",
                "citations": "Agentic workflow with synthesis error",
                "error": str(e)
            }
    
    def _organize_results_for_compression(self, completed_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Organize task execution results into structured format for compression.
        
        Args:
            completed_results: Dictionary of task results from execution
            
        Returns:
            Organized context data dictionary
        """
        organized_data = {
            'structured_data': [],
            'semantic_data': [],
            'genomic_context': [],
            'tool_results': [],
            'metadata': {}
        }
        
        for task_id, result in completed_results.items():
            if isinstance(result, dict):
                # Add structured database results
                if "structured_data" in result and result["structured_data"]:
                    if isinstance(result["structured_data"], list):
                        organized_data['structured_data'].extend(result["structured_data"])
                    else:
                        organized_data['structured_data'].append(result["structured_data"])
                
                # Add semantic similarity data
                if "semantic_data" in result and result["semantic_data"]:
                    if isinstance(result["semantic_data"], list):
                        organized_data['semantic_data'].extend(result["semantic_data"])
                    else:
                        organized_data['semantic_data'].append(result["semantic_data"])
                
                # Add genomic context information
                if "context" in result and hasattr(result["context"], 'structured_data'):
                    context_obj = result["context"]
                    if context_obj.structured_data:
                        organized_data['structured_data'].extend(context_obj.structured_data)
                    if context_obj.semantic_data:
                        organized_data['semantic_data'].extend(context_obj.semantic_data)
                
                # Add tool execution results
                if "tool_result" in result:
                    organized_data['tool_results'].append({
                        'task_id': task_id,
                        'tool_name': result.get('tool_name', 'unknown'),
                        'result': result["tool_result"]
                    })
                
                # Add metadata
                if "metadata" in result:
                    organized_data['metadata'][task_id] = result["metadata"]
        
        return organized_data
    
    async def _retrieve_context_with_fallback(self, question: str, query_type: str, retrieval_plan, 
                                            scoped_query: str, original_query: str) -> GenomicContext:
        """
        Retrieve context with fallback logic - try scoped query first, fallback to original if no results.
        """
        # CRITICAL: Validate comparative queries BEFORE execution
        retrieval_plan.cypher_query = self._validate_comparative_query(question, retrieval_plan.cypher_query)
        
        # First try the scoped query
        logger.info("🎯 Trying scoped query first")
        context = await self._retrieve_context(query_type, retrieval_plan)
        
        # If we got results, return them
        if context.structured_data or context.semantic_data:
            logger.info(f"✅ Scoped query successful: {len(context.structured_data)} results")
            return context
        
        # If scoped query returned no results and we applied scoping, try original query
        if scoped_query != original_query:
            logger.info("⚠️ Scoped query returned no results, trying original unscoped query")
            
            # Restore original query and retry (also validate it)
            retrieval_plan.cypher_query = self._validate_comparative_query(question, original_query)
            fallback_context = await self._retrieve_context(query_type, retrieval_plan)
            
            if fallback_context.structured_data or fallback_context.semantic_data:
                logger.info(f"✅ Fallback unscoped query successful: {len(fallback_context.structured_data)} results")
                # Add metadata about fallback
                fallback_context.metadata['used_fallback'] = True
                fallback_context.metadata['fallback_reason'] = "Scoped query returned no results"
                return fallback_context
        
        logger.warning("❌ Both scoped and unscoped queries returned no results")
        return context
    
    async def _retrieve_context(self, query_type: str, retrieval_plan) -> GenomicContext:
        """
        Retrieve context based on query type and plan.
        This is a simplified version - the full implementation is complex.
        """
        import time
        start_time = time.time()
        
        try:
            if query_type in ["structural", "general"]:
                # Use Neo4j for structured queries
                cypher_query = retrieval_plan.cypher_query
                result = await self.neo4j_processor.process_query(cypher_query, query_type="cypher")
                
                # Check for repair messages
                repair_message = None
                if hasattr(self.neo4j_processor, 'last_repair_result') and self.neo4j_processor.last_repair_result:
                    repair_result = self.neo4j_processor.last_repair_result
                    if repair_result.success and repair_result.user_message:
                        repair_message = repair_result.user_message
                
                if result.results:
                    # Apply context compression if needed
                    compressed_context, compression_stats = self.context_compressor.compress_context(
                        result.results, 
                        target_size=50, 
                        preserve_diversity=True
                    )
                    
                    metadata = result.metadata.copy()
                    metadata['compression_stats'] = compression_stats
                    
                    return GenomicContext(
                        structured_data=result.results,
                        semantic_data=[],
                        metadata=metadata,
                        query_time=time.time() - start_time,
                        compressed_context=compressed_context
                    )
                elif repair_message:
                    return GenomicContext(
                        structured_data=[],
                        semantic_data=[],
                        metadata={"repair_message": repair_message},
                        query_time=time.time() - start_time
                    )
                else:
                    return GenomicContext(
                        structured_data=[],
                        semantic_data=[],
                        metadata={"retrieval_error": "No results found"},
                        query_time=time.time() - start_time
                    )
            
            else:
                # For semantic/hybrid queries, use hybrid processor
                result = await self.hybrid_processor.process_query(retrieval_plan.cypher_query)
                
                if result.results:
                    combined_data = result.results[0] if result.results else {}
                    return GenomicContext(
                        structured_data=combined_data.get("structured_data", []),
                        semantic_data=combined_data.get("semantic_data", []),
                        metadata=result.metadata,
                        query_time=time.time() - start_time
                    )
                else:
                    return GenomicContext(
                        structured_data=[],
                        semantic_data=[],
                        metadata={"retrieval_error": "No results found"},
                        query_time=time.time() - start_time
                    )
                    
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return GenomicContext(
                structured_data=[],
                semantic_data=[],
                metadata={"retrieval_error": str(e)},
                query_time=time.time() - start_time
            )
    
    def _validate_comparative_query(self, question: str, cypher_query: str) -> str:
        """
        Validate and fix comparative queries that incorrectly use LIMIT 1.
        
        Args:
            question: Original user question
            cypher_query: Generated Cypher query to validate
            
        Returns:
            Validated (and potentially fixed) Cypher query
        """
        import re
        
        # Define patterns that indicate comparative questions requiring ALL results
        comparative_patterns = [
            r"which\s+(?:of\s+the\s+)?genomes?\s+(?:have|has|contain)",  # "which (of the) genomes have"
            r"for\s+each\s+genome",  # "for each genome"
            r"compare\s+.*?\s+(?:across\s+)?(?:all\s+)?genomes?",  # "compare X across genomes"
            r"(?:most|least|highest|lowest|best|worst)\s+(?:among|across|between)\s+genomes?",  # "most among genomes"
            r"which\s+.*?genomes?\s+.*?(?:has|have)\s+.*?(?:most|least|highest|lowest)",  # "which ... genomes ... has ... most"
            r"how\s+(?:many|much).+(?:across|between|among)\s+genomes?",  # "how many across genomes"
            r"distribution\s+(?:across|among|between)\s+genomes?",  # "distribution across genomes"
            r"all\s+genomes?.+(?:count|number|amount)",  # "all genomes count"
            r"rank\s+genomes?\s+by",  # "rank genomes by"
            r"(?:count|number|total).+per\s+genome"  # "count per genome"
        ]
        
        # Check if question contains comparative patterns
        question_lower = question.lower()
        is_comparative = any(re.search(pattern, question_lower) for pattern in comparative_patterns)
        
        if not is_comparative:
            return cypher_query
        
        # Check if query has LIMIT 1 (problematic for comparative queries)
        if re.search(r'\bLIMIT\s+1\b', cypher_query, re.IGNORECASE):
            logger.warning(f"Detected LIMIT 1 in comparative query: {question}")
            
            # Remove LIMIT 1 but keep other LIMIT values
            fixed_query = re.sub(r'\bLIMIT\s+1\b', '', cypher_query, flags=re.IGNORECASE)
            
            # Clean up any trailing whitespace or newlines
            fixed_query = fixed_query.strip()
            
            logger.info(f"Fixed comparative query by removing LIMIT 1")
            return fixed_query
        
        return cypher_query

    def _format_context(self, context: GenomicContext) -> str:
        """Format genomic context for LLM processing."""
        formatted_parts = []
        
        # Add structured data
        if context.structured_data:
            formatted_parts.append(f"=== STRUCTURED DATA ({len(context.structured_data)} results) ===")
            for i, item in enumerate(context.structured_data[:50]):  # Limit for context size
                formatted_parts.append(f"Result {i+1}: {item}")
            
            if len(context.structured_data) > 50:
                formatted_parts.append(f"... and {len(context.structured_data) - 50} more results")
        
        # Add semantic data
        if context.semantic_data:
            formatted_parts.append(f"\\n=== SEMANTIC DATA ({len(context.semantic_data)} results) ===")
            for i, item in enumerate(context.semantic_data[:20]):  # Limit for context size
                formatted_parts.append(f"Similar {i+1}: {item}")
            
            if len(context.semantic_data) > 20:
                formatted_parts.append(f"... and {len(context.semantic_data) - 20} more results")
        
        # Add metadata
        if context.metadata:
            formatted_parts.append(f"\\n=== METADATA ===")
            for key, value in context.metadata.items():
                if key not in ['retrieval_error', 'repair_message']:  # Skip error fields
                    formatted_parts.append(f"{key}: {value}")
        
        return "\\n".join(formatted_parts)
    
    def close(self):
        """Close all processor connections."""
        try:
            if hasattr(self.neo4j_processor, 'close'):
                self.neo4j_processor.close()
            if hasattr(self.lancedb_processor, 'close'):
                self.lancedb_processor.close()
            if hasattr(self.hybrid_processor, 'close'):
                self.hybrid_processor.close()
            logger.info("🔌 GenomicRAG connections closed")
        except Exception as e:
            logger.error(f"❌ Error closing connections: {e}")

    # Legacy methods for backward compatibility
    async def ask_agentic(self, question: str, **kwargs) -> str:
        """Legacy method that returns string instead of dict."""
        result = await self.ask(question)
        return result.get('answer', 'No answer generated')
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all system components."""
        health = self.health_check()
        return {
            f"{component}_processor": "available" if status else "unavailable" 
            for component, status in health.items()
        }