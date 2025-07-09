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
from .memory import NoteKeeper, ProgressiveSynthesizer

logger = logging.getLogger(__name__)
console = Console()

class GenomicRAG(dspy.Module if DSPY_AVAILABLE else object):
    """
    Main genomic RAG system with working implementation.
    
    Combines structured queries (Neo4j) with semantic search (LanceDB)
    and intelligent code interpreter enhancement.
    """
    
    def __init__(self, config: LLMConfig, chunk_context_size: int = 4096, enable_memory: bool = True):
        """Initialize the genomic RAG system."""
        if DSPY_AVAILABLE:
            super().__init__()
        
        self.config = config
        self.chunk_context_size = chunk_context_size
        self.enable_memory = enable_memory
        
        # Initialize processors
        self.neo4j_processor = Neo4jQueryProcessor(config)
        self.lancedb_processor = LanceDBQueryProcessor(config)
        self.hybrid_processor = HybridQueryProcessor(config)
        
        # Initialize new intelligent components
        self.intelligent_router = IntelligentRouter()
        self.scope_enforcer = QueryScopeEnforcer()
        self.context_compressor = ContextCompressor()
        
        # Initialize memory system
        self.note_keeper = NoteKeeper() if enable_memory else None
        self.progressive_synthesizer = None  # Will be initialized when needed
        
        # Configure DSPy
        self._configure_dspy()
        
        # Initialize DSPy components
        if DSPY_AVAILABLE:
            self.planner = dspy.Predict(PlannerAgent)
            self.classifier = dspy.Predict(QueryClassifier)
            self.retriever = dspy.Predict(ContextRetriever)
            self.answerer = dspy.Predict(GenomicAnswerer)
            
            # Initialize synthesizer for progressive synthesis
            from .dspy_signatures import GenomicSummarizer
            self.synthesizer = dspy.Predict(GenomicSummarizer)
        
        # Store DSPy availability for task executor
        self.dspy_available = DSPY_AVAILABLE
        
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
        
        # Step 4: Format context and apply compression if needed
        formatted_context = self._format_context(context)
        compression_stats = None
        
        # Check if context is too large and apply compression
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(self.config.llm_model if hasattr(self.config, 'llm_model') else 'gpt-3.5-turbo')
            token_count = len(encoding.encode(formatted_context))
            
            if token_count > 30000:
                logger.info(f"🗜️ Context too large ({token_count} tokens), applying compression")
                # Initialize context compressor only when needed
                compressor = ContextCompressor()
                
                # Get raw results for compression
                all_results = context.structured_data + context.semantic_data
                compressed_context, compression_stats = compressor.compress_context(all_results, target_size=25)
                
                logger.info(f"Context compression: {compression_stats.original_count} -> {compression_stats.compressed_count} results")
                formatted_context = compressed_context
                console.print(f"🗜️ Applied compression: {compression_stats.original_count} → {compression_stats.compressed_count} results")
            else:
                logger.info(f"✅ Context size acceptable ({token_count} tokens), using full context")
                
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using full context")
        
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
            executor = TaskExecutor(self, note_keeper=self.note_keeper)
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
        Synthesize final answer from agentic task execution results using progressive synthesis.
        
        Args:
            question: Original user question
            execution_results: Results from TaskGraph execution
            
        Returns:
            Formatted response dict with answer, confidence, and citations
        """
        logger.info("Synthesizing agentic results into final answer")
        
        try:
            # Set session context for note-taking
            if self.note_keeper:
                self.note_keeper.set_session_context(question, "agentic")
            
            # Check if we have notes to use for progressive synthesis
            if self.note_keeper:
                task_notes = self.note_keeper.get_all_task_notes()
                
                if task_notes:
                    logger.info(f"🧠 Using progressive synthesis with {len(task_notes)} task notes")
                    
                    # Initialize progressive synthesizer
                    if not self.progressive_synthesizer:
                        self.progressive_synthesizer = ProgressiveSynthesizer(self.note_keeper)
                    
                    # Use progressive synthesis
                    answer = self.progressive_synthesizer.synthesize_progressive(
                        task_notes=task_notes,
                        dspy_synthesizer=self.synthesizer,
                        question=question
                    )
                    
                    # Get synthesis statistics
                    synthesis_stats = self.progressive_synthesizer.get_synthesis_statistics()
                    
                    return {
                        "question": question,
                        "answer": answer,
                        "confidence": "high",
                        "citations": f"Progressive synthesis from {len(task_notes)} task notes",
                        "query_metadata": {
                            "execution_mode": "agentic_with_memory",
                            "total_tasks": execution_results.get("execution_summary", {}).get("total", 0),
                            "completed_tasks": execution_results.get("execution_summary", {}).get("completed", 0),
                            "task_notes": len(task_notes),
                            "synthesis_stats": synthesis_stats,
                            "note_taking_enabled": True
                        }
                    }
            
            # Fallback to traditional synthesis if no notes available
            logger.info("📝 No task notes available, using traditional synthesis")
            
            # Collect all completed task results
            completed_results = execution_results.get("completed_results", {})
            execution_summary = execution_results.get("execution_summary", {})
            
            # Organize context data for traditional synthesis
            context_data = self._organize_results_for_compression(completed_results)
            organized_context = self._organize_context_for_synthesis(context_data)
            
            # Apply compression if needed (same as before)
            import tiktoken
            compression_result = None
            try:
                encoding = tiktoken.encoding_for_model(self.config.llm_model if hasattr(self.config, 'llm_model') else 'gpt-3.5-turbo')
                token_count = len(encoding.encode(organized_context))
                
                if token_count > 30000:
                    logger.info(f"🗜️ Context too large ({token_count} tokens), applying compression")
                    # Use context compression as fallback
                    compressor = ContextCompressor()
                    combined_context, compression_stats = compressor.compress_context(
                        [{"context": organized_context}], target_size=25
                    )
                    logger.info(f"Context compression: {compression_stats.original_count} -> {compression_stats.compressed_count} results")
                else:
                    logger.info(f"✅ Context size acceptable ({token_count} tokens), using full context")
                    combined_context = organized_context
                    
            except Exception as e:
                logger.warning(f"Token counting failed: {e}, using full context")
                combined_context = organized_context
            
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
            
            # Add execution metadata
            total_tasks = execution_summary.get("total", 0)
            completed_tasks = execution_summary.get("completed", 0)
            
            metadata = {
                "execution_mode": "agentic_traditional_synthesis",
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "execution_summary": execution_summary,
                "note_taking_enabled": self.note_keeper is not None
            }
            
            return {
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "citations": citations,
                "query_metadata": metadata
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
    
    def _organize_context_for_synthesis(self, context_data: Dict[str, Any]) -> str:
        """
        Organize context data into readable format for synthesis without compression.
        
        Args:
            context_data: Organized context data from task results
            
        Returns:
            Formatted context string for LLM synthesis
        """
        context_parts = []
        
        # Add structured data results
        if context_data.get('structured_data'):
            context_parts.append("=== STRUCTURED DATA RESULTS ===")
            for i, item in enumerate(context_data['structured_data'], 1):
                context_parts.append(f"Result {i}: {item}")
            context_parts.append("")
        
        # Add semantic data results
        if context_data.get('semantic_data'):
            context_parts.append("=== SEMANTIC SIMILARITY RESULTS ===")
            for i, item in enumerate(context_data['semantic_data'], 1):
                context_parts.append(f"Similar {i}: {item}")
            context_parts.append("")
        
        # Add tool execution results
        if context_data.get('tool_results'):
            context_parts.append("=== TOOL EXECUTION RESULTS ===")
            for tool_result in context_data['tool_results']:
                context_parts.append(f"Task {tool_result['task_id']} ({tool_result['tool_name']}):")
                context_parts.append(f"  {tool_result['result']}")
            context_parts.append("")
        
        # Add metadata
        if context_data.get('metadata'):
            context_parts.append("=== EXECUTION METADATA ===")
            for task_id, metadata in context_data['metadata'].items():
                context_parts.append(f"Task {task_id}: {metadata}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _format_context_for_token_check(self, results: List[Dict[str, Any]]) -> str:
        """Format results into a string for token counting without full formatting."""
        if not results:
            return ""
        
        # Create a representative sample for token counting
        formatted_parts = []
        for i, result in enumerate(results[:10]):  # Sample first 10 for token estimation
            formatted_parts.append(f"Result {i+1}: {str(result)}")
        
        # Estimate total size based on sample
        sample_size = len("\n".join(formatted_parts))
        estimated_total = sample_size * (len(results) / min(len(results), 10))
        
        # Return either sample or indication of size
        if len(results) <= 10:
            return "\n".join(formatted_parts)
        else:
            return "\n".join(formatted_parts) + f"\n... (estimated {estimated_total} characters for {len(results)} total results)"
    
    def _get_compression_target_size(self, retrieval_plan, results: List[Dict[str, Any]], question: str = "") -> int:
        """
        Determine appropriate compression target size based on query type and data characteristics.
        
        Args:
            retrieval_plan: DSPy retrieval plan with query information
            results: Raw results from database query
            question: Original user question for context
            
        Returns:
            Target size for compression
        """
        # Get query details
        cypher_query = getattr(retrieval_plan, 'cypher_query', '')
        search_strategy = getattr(retrieval_plan, 'search_strategy', '')
        
        # Check question and query for CAZyme-related queries (need more comprehensive data)
        cazyme_terms = ['cazyme', 'carbohydrate', 'glycoside', 'hydrolase', 'transferase']
        if any(cazyme_term in cypher_query.lower() for cazyme_term in cazyme_terms) or \
           any(cazyme_term in question.lower() for cazyme_term in cazyme_terms):
            logger.info("🧬 CAZyme query detected - using expanded target size")
            return min(len(results), 300)  # Allow up to 300 CAZymes for full analysis
        
        # Check for comparative queries that need to show distributions
        comp_terms = ['compare', 'distribution', 'across genomes', 'contrast', 'each genome']
        if any(comp_term in cypher_query.lower() for comp_term in comp_terms) or \
           any(comp_term in question.lower() for comp_term in comp_terms):
            logger.info("📊 Comparative query detected - using expanded target size")
            return min(len(results), 200)  # Allow up to 200 for comparison
        
        # Check for large result sets that might need more space
        if len(results) > 100:
            logger.info(f"📈 Large result set detected ({len(results)} results) - using expanded target size")
            return min(len(results), 100)  # Allow up to 100 for large datasets
        
        # Default compression for smaller queries
        return min(len(results), 50)
    
    async def _retrieve_context_with_fallback(self, question: str, query_type: str, retrieval_plan, 
                                            scoped_query: str, original_query: str) -> GenomicContext:
        """
        Retrieve context with fallback logic - try scoped query first, fallback to original if no results.
        """
        # CRITICAL: Validate comparative queries BEFORE execution
        retrieval_plan.cypher_query = self._validate_comparative_query(question, retrieval_plan.cypher_query)
        
        # First try the scoped query
        logger.info("🎯 Trying scoped query first")
        context = await self._retrieve_context(query_type, retrieval_plan, question)
        
        # If we got results, return them
        if context.structured_data or context.semantic_data:
            logger.info(f"✅ Scoped query successful: {len(context.structured_data)} results")
            return context
        
        # If scoped query returned no results and we applied scoping, try original query
        if scoped_query != original_query:
            logger.info("⚠️ Scoped query returned no results, trying original unscoped query")
            
            # Restore original query and retry (also validate it)
            retrieval_plan.cypher_query = self._validate_comparative_query(question, original_query)
            fallback_context = await self._retrieve_context(query_type, retrieval_plan, question)
            
            if fallback_context.structured_data or fallback_context.semantic_data:
                logger.info(f"✅ Fallback unscoped query successful: {len(fallback_context.structured_data)} results")
                # Add metadata about fallback
                fallback_context.metadata['used_fallback'] = True
                fallback_context.metadata['fallback_reason'] = "Scoped query returned no results"
                return fallback_context
        
        logger.warning("❌ Both scoped and unscoped queries returned no results")
        return context
    
    async def _retrieve_context(self, query_type: str, retrieval_plan, question: str = "") -> GenomicContext:
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
                    # Check if compression is needed based on context size
                    formatted_context = self._format_context_for_token_check(result.results)
                    
                    import tiktoken
                    try:
                        encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')  # Default for token counting
                        token_count = len(encoding.encode(formatted_context))
                        
                        if token_count > 30000:
                            logger.info(f"🗜️ Context too large ({token_count} tokens), applying compression")
                            # Apply context compression with smart target sizing
                            target_size = self._get_compression_target_size(retrieval_plan, result.results, question)
                            compressed_context, compression_stats = self.context_compressor.compress_context(
                                result.results, 
                                target_size=target_size, 
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
                        else:
                            logger.info(f"✅ Context size acceptable ({token_count} tokens), using full results")
                            return GenomicContext(
                                structured_data=result.results,
                                semantic_data=[],
                                metadata=result.metadata,
                                query_time=time.time() - start_time
                            )
                    except Exception as e:
                        logger.warning(f"Token counting failed: {e}, using full results")
                        return GenomicContext(
                            structured_data=result.results,
                            semantic_data=[],
                            metadata=result.metadata,
                            query_time=time.time() - start_time
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