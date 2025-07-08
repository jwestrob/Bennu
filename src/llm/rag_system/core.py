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
            
            # STEP 1: Determine if agentic planning is needed
            planning_result = self.planner(user_query=question)
            console.print(f"🤖 Agentic planning: {planning_result.requires_planning}")
            console.print(f"💭 Planning reasoning: {planning_result.reasoning}")
            
            # Convert string boolean to actual boolean if needed
            requires_planning = planning_result.requires_planning
            if isinstance(requires_planning, str):
                requires_planning = requires_planning.lower() == 'true'
            
            if requires_planning:
                # AGENTIC PATH: Multi-step task execution
                # Check if we actually have a valid task plan
                task_plan = planning_result.task_plan
                if task_plan == "N/A" or not task_plan or task_plan.strip() == "":
                    console.print("⚠️ [yellow]Agentic planning requested but no task plan provided, falling back to traditional mode[/yellow]")
                    return await self._execute_traditional_query(question)
                return await self._execute_agentic_plan(question, planning_result)
            else:
                # TRADITIONAL PATH: Direct query execution
                return await self._execute_traditional_query(question)
                
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
    
    async def _execute_traditional_query(self, question: str) -> Dict[str, Any]:
        """Execute traditional single-step query (backward compatibility)."""
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
        
        # Step 3: Execute database queries
        context = await self._retrieve_context(classification.query_type, retrieval_plan)
        
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
        
        # Step 4: Generate answer
        formatted_context = self._format_context(context)
        answer_result = self.answerer(
            question=question,
            context=formatted_context
        )
        
        # Return structured response
        return {
            "question": question,
            "answer": answer_result.answer,
            "confidence": answer_result.confidence,
            "citations": answer_result.citations,
            "query_metadata": {
                "query_type": classification.query_type,
                "search_strategy": retrieval_plan.search_strategy,
                "context_size": len(formatted_context),
                "retrieval_time": context.query_time,
                "total_results": len(context.structured_data) + len(context.semantic_data)
            }
        }
    
    async def _execute_agentic_plan(self, question: str, planning_result) -> Dict[str, Any]:
        """Execute multi-step agentic plan."""
        console.print("🤖 [bold]Using agentic execution path[/bold]")
        console.print(f"📋 Task plan: {planning_result.task_plan}")
        
        # For now, fall back to traditional query
        # TODO: Implement full agentic task execution
        console.print("⚠️ [yellow]Agentic execution not fully implemented, falling back to traditional mode[/yellow]")
        return await self._execute_traditional_query(question)
    
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