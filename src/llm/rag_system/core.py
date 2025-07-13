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
    console = Console()
except ImportError:
    DSPY_AVAILABLE = False
    # Create a fallback console that prints to stdout
    class FallbackConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = FallbackConsole()
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
from .memory import NoteKeeper, ProgressiveSynthesizer, get_model_allocator
from .policy_engine import get_policy_engine
from .genome_context_extractor import GenomeContextExtractor
from .query_validator import QueryValidator
from .genome_selector import GenomeSelector

logger = logging.getLogger(__name__)

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
        self.genome_context_extractor = GenomeContextExtractor()
        self.query_validator = QueryValidator()
        self.genome_selector = GenomeSelector(self.neo4j_processor)
        
        # Initialize memory system
        self.note_keeper = NoteKeeper() if enable_memory else None
        self.progressive_synthesizer = None  # Will be initialized when needed
        
        # Initialize model allocation system
        self.model_allocator = get_model_allocator()
        
        # Initialize policy engine
        self.policy_engine = get_policy_engine()
        
        # Configure DSPy with model allocation
        self._configure_dspy()
        
        # Initialize DSPy components (using global DSPy configuration)
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
        """Configure DSPy with model allocation system."""
        if not DSPY_AVAILABLE:
            return
            
        try:
            # Configure based on available API keys
            api_key = self.config.get_api_key()
            
            if self.config.llm_provider == "openai" and api_key:
                import os
                os.environ['OPENAI_API_KEY'] = api_key
                
                # Use model allocation system for intelligent model selection
                if self.model_allocator.use_premium_everywhere:
                    # Premium mode: use o3 for all tasks
                    model_name, model_config = self.model_allocator.get_model_for_task("final_synthesis")  # Gets o3
                    model_string = f"openai/{model_name}"
                    
                    if model_name.startswith(('o1', 'o3')):
                        lm = dspy.LM(model=model_string, temperature=1.0, max_tokens=20000)
                        logger.info(f"🎯 DSPy configured with premium reasoning model: {model_string} (temp=1.0, max_tokens=20000)")
                    else:
                        lm = dspy.LM(model=model_string, temperature=0.0, max_tokens=8000)
                        logger.info(f"🎯 DSPy configured with premium model: {model_string}")
                else:
                    # Cost-effective mode: use mini for default, but allow allocation for complex tasks
                    model_name, model_config = self.model_allocator.get_model_for_task("query_classification")  # Gets o3 for complex tasks
                    model_string = f"openai/{model_name}"
                    
                    # Handle reasoning models (o3) properly even in cost-effective mode
                    if model_name.startswith(('o1', 'o3')):
                        lm = dspy.LM(model=model_string, temperature=1.0, max_tokens=20000)
                        logger.info(f"🎯 DSPy configured with reasoning model: {model_string} (temp=1.0, max_tokens=20000)")
                    else:
                        lm = dspy.LM(model=model_string, temperature=0.0, max_tokens=8000)
                        logger.info(f"🎯 DSPy configured with cost-effective model: {model_string}")
                
                dspy.settings.configure(lm=lm)
                
                # Log model allocation configuration
                allocation_summary = self.model_allocator.get_allocation_summary()
                logger.info(f"💰 Model allocation mode: {allocation_summary['mode']}")
                if allocation_summary['mode'] == 'premium_everywhere':
                    logger.info(f"🔥 Using {allocation_summary['primary_model']} for all tasks")
                else:
                    logger.info(f"💡 Using task-specific model allocation for cost optimization")
                
                # Log available models
                logger.info(f"💡 Cost-effective option: gpt-4.1-mini")
                logger.info(f"🔥 Premium option: o3")
                
            elif self.config.llm_provider == "anthropic" and api_key:
                # Anthropic configuration
                import os
                os.environ['ANTHROPIC_API_KEY'] = api_key
                
                current_model = self.config.get_current_model()
                # Map to Anthropic models if needed
                if current_model.startswith(('gpt', 'o1', 'o3')):
                    # Use Anthropic equivalent
                    anthropic_model = "claude-3-haiku-20240307" if self.config.model_mode == "cost_effective" else "claude-3-opus-20240229"
                else:
                    anthropic_model = current_model
                
                model_string = f"anthropic/{anthropic_model}"
                lm = dspy.LM(model=model_string, max_tokens=1000)
                dspy.settings.configure(lm=lm)
                logger.info(f"🎯 DSPy configured with Anthropic model: {model_string}")
                
            else:
                logger.warning("No LLM API key configured for DSPy")
                
        except Exception as e:
            logger.error(f"Failed to configure DSPy: {e}")
            
            # Fallback to original configuration
            try:
                api_key = self.config.get_api_key()
                if self.config.llm_provider == "openai" and api_key:
                    import os
                    os.environ['OPENAI_API_KEY'] = api_key
                    model_name = getattr(self.config, 'llm_model', 'gpt-4o-mini')
                    model_string = f"openai/{model_name}"
                    lm = dspy.LM(model=model_string, temperature=0.0, max_tokens=2000)
                    dspy.settings.configure(lm=lm)
                    logger.info(f"🔄 DSPy configured with fallback model: {model_string}")
            except Exception as fallback_error:
                logger.error(f"Fallback DSPy configuration also failed: {fallback_error}")
    
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
            
            # STEP 1: Let the LLM decide execution strategy directly
            console.print("🤖 [bold]Using LLM-based execution planning[/bold]")
            
            # Use model allocation for planning (o3 for complex planning tasks)
            logger.info("🧠 Using model allocation for intelligent planning")
            
            def planning_call(module):
                return module(user_query=question)
            
            planning_result = self.model_allocator.create_context_managed_call(
                task_name="agentic_planning",  # Maps to COMPLEX = o3
                signature_class=PlannerAgent,
                module_call_func=planning_call,
                query=question,
                task_context="Agentic planning for user query"
            )
            
            if planning_result is None:
                logger.warning("Model allocation failed for planning, falling back to default")
                planning_result = self.planner(user_query=question)
            
            console.print(f"🎯 Planning decision: {'agentic' if planning_result.requires_planning else 'traditional'}")
            console.print(f"💭 Reasoning: {planning_result.reasoning}")
            
            # Execute based on LLM's decision
            if planning_result.requires_planning:
                # AGENTIC PATH: Multi-step task execution with upfront genome selection
                task_plan = planning_result.task_plan
                if task_plan == "N/A" or not task_plan or task_plan.strip() == "":
                    console.print("⚠️ [yellow]Agentic mode chosen but no task plan provided, falling back to traditional mode[/yellow]")
                    return await self._execute_traditional_query(question, None)
                
                # INTELLIGENT UPFRONT GENOME SELECTION - One LLM call for the entire agentic workflow
                console.print("🧠 [bold blue]Analyzing genome selection intent for agentic workflow[/bold blue]")
                
                try:
                    from .llm_genome_selector import LLMGenomeSelector
                    llm_selector = LLMGenomeSelector(self.neo4j_processor)
                    
                    selection_result = await llm_selector.analyze_genome_intent(question)
                    
                    if selection_result.success:
                        console.print(f"🧬 [bold green]LLM genome analysis:[/bold green] intent={selection_result.intent}, confidence={selection_result.confidence:.2f}")
                        console.print(f"💭 [dim]Reasoning: {selection_result.reasoning}[/dim]")
                        
                        if selection_result.intent == "specific" and selection_result.target_genomes:
                            selected_genome = selection_result.target_genomes[0]  # Use first genome for now
                            console.print(f"🎯 [bold cyan]All agentic tasks will target genome:[/bold cyan] {selected_genome}")
                        else:
                            selected_genome = None
                            console.print(f"🌐 [bold cyan]All agentic tasks will analyze across all genomes[/bold cyan] (intent: {selection_result.intent})")
                    else:
                        logger.warning(f"LLM genome analysis failed: {selection_result.error_message}")
                        selected_genome = None
                        console.print("🌐 [bold cyan]Falling back to global analysis across all genomes[/bold cyan]")
                        
                except Exception as e:
                    logger.error(f"LLM genome selection failed: {e}")
                    selected_genome = None
                    console.print("⚠️ [yellow]Genome selection error, using global analysis[/yellow]")
                
                return await self._execute_agentic_plan(question, planning_result, selected_genome)
            else:
                # TRADITIONAL PATH: Direct query execution
                return await self._execute_traditional_query(question, None)
                
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
        
        # Step 1: Classify the query type using model allocation (o3 for biological reasoning)
        def classification_call(module):
            return module(question=question)
        
        from .dspy_signatures import QueryClassifier
        classification = self.model_allocator.create_context_managed_call(
            task_name="query_classification",  # Now maps to COMPLEX = o3
            signature_class=QueryClassifier,
            module_call_func=classification_call
        )
        
        # Step 1.5: Determine analysis type for biological context
        analysis_type = self._determine_analysis_type(question)
        
        if classification is None:
            logger.warning("Model allocation failed for classification, falling back to default")
            # Ensure there's a default LM configured for fallback
            if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
                logger.warning("No default LM configured, setting up fallback")
                fallback_lm = dspy.LM(model="openai/gpt-4.1-mini", temperature=0.0, max_tokens=8000)
                dspy.settings.configure(lm=fallback_lm)
            classification = self.classifier(question=question)
        
        console.print(f"📊 Query type: {classification.query_type}")
        console.print(f"💭 Reasoning: {classification.reasoning}")
        
        # Step 2: INTELLIGENT GENOME SELECTION - Use LLM to analyze genome selection intent
        genome_filter_required = False
        target_genome = ""
        task_context = "Global query across all genomes"
        
        try:
            from .llm_genome_selector import LLMGenomeSelector
            llm_selector = LLMGenomeSelector(self.neo4j_processor)
            
            # Check if this query needs genome selection analysis  
            if llm_selector.should_use_genome_selection(question):
                console.print("🔍 [bold yellow]Analyzing genome selection intent[/bold yellow]")
                
                selection_result = await llm_selector.analyze_genome_intent(question)
                
                if selection_result.success:
                    console.print(f"🧬 [bold green]LLM analysis:[/bold green] intent={selection_result.intent}, confidence={selection_result.confidence:.2f}")
                    console.print(f"💭 [dim]Reasoning: {selection_result.reasoning}[/dim]")
                    
                    if selection_result.intent == "specific" and selection_result.target_genomes:
                        genome_filter_required = True
                        target_genome = selection_result.target_genomes[0]  # Use first genome
                        task_context = f"Target genome: {target_genome}. LLM confidence: {selection_result.confidence:.2f}"
                        console.print(f"🎯 [bold cyan]Query will target genome:[/bold cyan] {target_genome}")
                    else:
                        console.print(f"🌐 [bold cyan]Query will analyze across all genomes[/bold cyan] (intent: {selection_result.intent})")
                else:
                    console.print(f"❌ [red]LLM genome analysis failed:[/red] {selection_result.error_message}")
                    console.print("🌐 [dim]Continuing with global analysis[/dim]")
            else:
                console.print("🌐 [dim]Using global analysis across all genomes[/dim]")
                
        except Exception as e:
            logger.error(f"LLM genome selection failed: {e}")
            console.print("⚠️ [yellow]Genome selection error, using global analysis[/yellow]")
        
        def retrieval_call(module):
            return module(
                db_schema=NEO4J_SCHEMA,
                question=question,
                query_type=classification.query_type,
                task_context=task_context,
                genome_filter_required=str(genome_filter_required),
                target_genome=target_genome,
                analysis_type=analysis_type
            )
        
        from .dspy_signatures import ContextRetriever
        retrieval_plan = self.model_allocator.create_context_managed_call(
            task_name="context_preparation",  # Now maps to COMPLEX = o3
            signature_class=ContextRetriever,
            module_call_func=retrieval_call
        )
        
        if retrieval_plan is None:
            logger.warning("Model allocation failed for retrieval, falling back to default")
            # Ensure there's a default LM configured for fallback
            if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
                logger.warning("No default LM configured, setting up fallback")
                fallback_lm = dspy.LM(model="openai/gpt-4.1-mini", temperature=0.0, max_tokens=8000)
                dspy.settings.configure(lm=fallback_lm)
            retrieval_plan = self.retriever(
                db_schema=NEO4J_SCHEMA,
                question=question,
                query_type=classification.query_type,
                task_context=task_context,
                genome_filter_required=str(genome_filter_required),
                target_genome=target_genome,
                analysis_type=analysis_type
            )
        
        console.print(f"🔍 Search strategy: {retrieval_plan.search_strategy}")
        
        # Step 2.5: Validate query for comparative questions
        cypher_query = retrieval_plan.cypher_query
        validated_query = self._validate_comparative_query(question, cypher_query)
        if validated_query != cypher_query:
            logger.info("Fixed comparative query - removed inappropriate LIMIT")
            retrieval_plan.cypher_query = validated_query
        
        # Step 2.6: Validate genome filtering if required
        if genome_filter_required and self.query_validator.should_validate_for_genome(validated_query):
            validation_result = self.query_validator.validate_genome_filtering(
                validated_query, 
                genome_filter_required, 
                target_genome
            )
            
            if not validation_result.is_valid:
                console.print(f"⚠️ [yellow]Query validation failed:[/yellow] {validation_result.error_message}")
                
                if validation_result.modified_query:
                    console.print(f"🔧 [cyan]Auto-fixing query with genome filtering[/cyan]")
                    retrieval_plan.cypher_query = validation_result.modified_query
                    logger.info(f"Applied genome filtering fix: {validation_result.suggested_fix}")
                else:
                    console.print(f"💡 [blue]Suggestion:[/blue] {validation_result.suggested_fix}")
                    logger.warning(f"Could not auto-fix query: {validation_result.suggested_fix}")
            else:
                console.print(f"✅ [green]Query validation passed - genome filtering present[/green]")
        
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
            
            if self.policy_engine.should_compress_context(token_count):
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
        
        # Step 5: Check if external tools would be helpful and execute if so
        tool_results = await self._check_and_execute_tools(question, context, classification.query_type)
        
        # Step 6: Generate answer using model allocation (integrate tool results if available)
        final_context = formatted_context
        if tool_results:
            final_context = self._integrate_tool_results(formatted_context, tool_results)
        
        def answer_call(module):
            return module(
                question=question,
                context=final_context
            )
        
        from .dspy_signatures import GenomicAnswerer
        answer_result = self.model_allocator.create_context_managed_call(
            task_name="biological_interpretation",  # Maps to COMPLEX = o3
            signature_class=GenomicAnswerer,
            module_call_func=answer_call
        )
        
        if answer_result is None:
            logger.warning("Model allocation failed for answer generation, falling back to default")
            # Ensure there's a default LM configured for fallback
            if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
                logger.warning("No default LM configured, setting up fallback")
                fallback_lm = dspy.LM(model="openai/gpt-4.1-mini", temperature=0.0, max_tokens=8000)
                dspy.settings.configure(lm=fallback_lm)
            
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
    
    async def _check_and_execute_tools(self, question: str, context, query_type: str) -> Optional[Dict[str, Any]]:
        """Check if external tools would be helpful and execute them if so."""
        tool_results = {}
        
        # Check if literature search would be helpful and is available
        if (self._should_use_literature_search(question, query_type) and 
            self.policy_engine.should_use_tool("literature_search")):
            if await self._check_literature_search_availability():
                console.print("🔍 [dim]Literature search would be helpful, executing...[/dim]")
                literature_result = await self._execute_literature_search(question)
                if literature_result:
                    tool_results["literature_search"] = literature_result
            else:
                console.print("⚠️ [dim]Literature search unavailable (missing dependencies)[/dim]")
        
        # Check if code interpreter would be helpful and is available
        if (self._should_use_code_interpreter(question, context, query_type) and 
            self.policy_engine.should_use_tool("code_interpreter")):
            if await self._check_code_interpreter_availability():
                console.print("🧮 [dim]Code interpreter would be helpful, executing...[/dim]")
                code_result = await self._execute_code_interpreter(question, context)
                if code_result:
                    tool_results["code_interpreter"] = code_result
            else:
                console.print("⚠️ [dim]Code interpreter unavailable (service not running)[/dim]")
        
        return tool_results if tool_results else None
    
    async def _check_literature_search_availability(self) -> bool:
        """Check if literature search dependencies are available."""
        try:
            from Bio import Entrez
            return True
        except ImportError:
            logger.warning("Biopython not available for literature search")
            return False
    
    async def _check_code_interpreter_availability(self) -> bool:
        """Check if code interpreter service is available."""
        try:
            from .external_tools import check_code_interpreter_health
            return await check_code_interpreter_health()
        except Exception as e:
            logger.warning(f"Code interpreter health check failed: {e}")
            return False
    
    def _should_use_literature_search(self, question: str, query_type: str) -> bool:
        """Determine if literature search would be helpful."""
        question_lower = question.lower()
        
        # Look for explicit literature requests
        literature_keywords = ["recent", "literature", "research", "papers", "pubmed", "studies", "publications"]
        if any(keyword in question_lower for keyword in literature_keywords):
            return True
        
        # Look for functional questions that might benefit from literature
        functional_keywords = ["function", "role", "mechanism", "pathway", "regulation"]
        if any(keyword in question_lower for keyword in functional_keywords):
            return True
        
        return False
    
    def _should_use_code_interpreter(self, question: str, context, query_type: str) -> bool:
        """Determine if code interpreter would be helpful."""
        question_lower = question.lower()
        
        # Look for analysis/computation keywords
        analysis_keywords = ["analyze", "analysis", "distribution", "statistics", "statistical", 
                           "compare", "comparison", "pattern", "trend", "visualization", "plot", "chart"]
        if any(keyword in question_lower for keyword in analysis_keywords):
            return True
        
        # Check if we have large datasets that could benefit from analysis
        total_results = len(context.structured_data) + len(context.semantic_data)
        if total_results > 50:  # Arbitrary threshold for "large" datasets
            return True
        
        return False
    
    async def _execute_literature_search(self, question: str) -> Optional[str]:
        """Execute literature search tool."""
        try:
            from .external_tools import literature_search
            
            # Configure search parameters from policy engine
            email = self.config.get("email", "user@example.com")  # Should be configured
            max_results = self.policy_engine.get_max_results("literature_search")
            
            # Execute search
            result = literature_search(question, email, max_results=max_results)
            logger.info(f"Literature search completed: {len(result)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Literature search failed: {e}")
            return None
    
    async def _execute_code_interpreter(self, question: str, context) -> Optional[str]:
        """Execute code interpreter tool."""
        try:
            from .external_tools import code_interpreter_tool
            
            # Prepare data for analysis
            data_summary = self._prepare_data_for_analysis(context)
            
            # Generate analysis code based on question
            analysis_code = self._generate_analysis_code(question, data_summary)
            
            # Execute code
            result = await code_interpreter_tool(analysis_code)
            
            if result.get("success"):
                logger.info("Code interpreter execution completed successfully")
                return result.get("output", "")
            else:
                logger.warning(f"Code interpreter execution failed: {result.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            logger.error(f"Code interpreter execution failed: {e}")
            return None
    
    def _prepare_data_for_analysis(self, context) -> str:
        """Prepare a summary of available data for code analysis."""
        summary = []
        
        if context.structured_data:
            summary.append(f"Structured data: {len(context.structured_data)} records")
            # Add sample of data structure
            if context.structured_data:
                sample = context.structured_data[0]
                if isinstance(sample, dict):
                    summary.append(f"Sample keys: {list(sample.keys())[:5]}")
        
        if context.semantic_data:
            summary.append(f"Semantic data: {len(context.semantic_data)} records")
        
        return "; ".join(summary)
    
    def _generate_analysis_code(self, question: str, data_summary: str) -> str:
        """Generate Python code for analysis based on question."""
        # This is a simple heuristic approach - in practice, this could be more sophisticated
        question_lower = question.lower()
        
        if "distribution" in question_lower:
            return """
import pandas as pd
import matplotlib.pyplot as plt

# Create sample analysis for distribution
print("Distribution analysis would go here")
print("Data summary:", data_summary)
"""
        elif "compare" in question_lower or "comparison" in question_lower:
            return """
import pandas as pd
import numpy as np

# Create sample comparison analysis
print("Comparison analysis would go here")
print("Data summary:", data_summary)
"""
        else:
            return f"""
# General analysis
print("General analysis for question: {question[:50]}...")
print("Data available: {data_summary}")
"""
    
    def _integrate_tool_results(self, original_context: str, tool_results: Dict[str, Any]) -> str:
        """Integrate tool results into the context."""
        integrated_context = original_context
        
        # Add tool results section
        if tool_results:
            integrated_context += "\n\n=== EXTERNAL TOOL RESULTS ===\n"
            
            if "literature_search" in tool_results:
                integrated_context += f"\n--- Literature Search Results ---\n{tool_results['literature_search']}\n"
            
            if "code_interpreter" in tool_results:
                integrated_context += f"\n--- Code Analysis Results ---\n{tool_results['code_interpreter']}\n"
        
        return integrated_context
    
    async def _execute_agentic_plan(self, question: str, planning_result, selected_genome: Optional[str] = None) -> Dict[str, Any]:
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
                # CRITICAL FIX: Inject original question into each task for biological context preservation
                task.original_question = question
                logger.info(f"🧬 Injected original question into task {task.task_id}: '{question[:50]}...'")
                graph.add_task(task)
            
            # Step 3: Execute TaskGraph with dependency resolution and pre-selected genome
            executor = TaskExecutor(self, note_keeper=self.note_keeper, selected_genome=selected_genome)
            if selected_genome:
                console.print(f"🧬 [cyan]All tasks will target genome:[/cyan] {selected_genome}")
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
                    
                    # Organize raw data for multi-part reports - ENHANCED data extraction
                    completed_results = execution_results.get("completed_results", {})
                    raw_data = self._extract_raw_data_for_multipart(completed_results)
                    
                    # ENHANCEMENT: Include raw task results in synthesis for richer data flow
                    # This ensures detailed analysis from chunking and code interpreter reaches final answer
                    logger.info(f"📊 Including {len(completed_results)} task results alongside {len(raw_data)} raw data items")
                    
                    # Use progressive synthesis (now with task-based capability for large datasets)
                    answer = self.progressive_synthesizer.synthesize_progressive(
                        task_notes=task_notes,
                        dspy_synthesizer=self.synthesizer,
                        question=question,
                        raw_data=raw_data,
                        rag_system=self  # Pass self for task-based processing
                    )
                    
                    # ENHANCEMENT: If progressive synthesis seems sparse, supplement with task results
                    if len(answer) < 500 and completed_results:
                        logger.warning("🔄 Progressive synthesis seems sparse, enriching with task results")
                        supplemented_answer = self._supplement_synthesis_with_task_results(
                            answer, completed_results, question
                        )
                        if len(supplemented_answer) > len(answer):
                            answer = supplemented_answer
                    
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
    
    def _extract_raw_data_for_multipart(self, completed_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        ENHANCED: Extract comprehensive raw data from completed task results.
        
        Args:
            completed_results: Dictionary of task results from execution
            
        Returns:
            List of raw data items for multi-part report synthesis
        """
        raw_data = []
        logger.info(f"🔍 ENHANCED EXTRACTION: Processing {len(completed_results)} task results")
        
        for task_id, result in completed_results.items():
            logger.info(f"📋 Processing task: {task_id}")
            
            if isinstance(result, dict):
                
                # PRIORITY 1: Extract GenomicContext objects (main data source)
                if "context" in result and hasattr(result["context"], 'structured_data'):
                    context_obj = result["context"]
                    logger.info(f"✅ Found GenomicContext: {len(context_obj.structured_data)} structured items")
                    
                    # Add structured data with task metadata
                    if context_obj.structured_data:
                        for item in context_obj.structured_data:
                            enriched_item = dict(item) if isinstance(item, dict) else {"data": item}
                            enriched_item["_source_task"] = task_id
                            enriched_item["_data_type"] = "structured_query_result"
                            raw_data.append(enriched_item)
                    
                    # Add semantic data with task metadata  
                    if context_obj.semantic_data:
                        for item in context_obj.semantic_data:
                            enriched_item = dict(item) if isinstance(item, dict) else {"data": item}
                            enriched_item["_source_task"] = task_id
                            enriched_item["_data_type"] = "semantic_similarity_result"
                            raw_data.append(enriched_item)
                
                # PRIORITY 2: Extract tool execution results (code interpreter, etc.)
                elif result.get("tool_name") and result.get("tool_result"):
                    logger.info(f"🔧 Found tool result: {result['tool_name']}")
                    
                    # Parse tool result content
                    tool_content = result["tool_result"]
                    
                    # For code interpreter results, try to extract data analysis
                    if result["tool_name"] == "code_interpreter":
                        # Add the full tool result as a rich data item
                        raw_data.append({
                            "_source_task": task_id,
                            "_data_type": "code_interpreter_analysis", 
                            "tool_name": result["tool_name"],
                            "analysis_content": tool_content,
                            "summary": tool_content[:500] + "..." if len(tool_content) > 500 else tool_content
                        })
                    else:
                        # Other tool results
                        raw_data.append({
                            "_source_task": task_id,
                            "_data_type": "external_tool_result",
                            "tool_name": result["tool_name"], 
                            "result_content": tool_content
                        })
                
                # PRIORITY 3: Extract direct data fields (legacy support)
                else:
                    
                    # Extract structured data from database queries
                    if "structured_data" in result and result["structured_data"]:
                        if isinstance(result["structured_data"], list):
                            for item in result["structured_data"]:
                                enriched_item = dict(item) if isinstance(item, dict) else {"data": item}
                                enriched_item["_source_task"] = task_id
                                enriched_item["_data_type"] = "direct_structured_data"
                                raw_data.append(enriched_item)
                        else:
                            raw_data.append({
                                "_source_task": task_id,
                                "_data_type": "direct_structured_data",
                                "data": result["structured_data"]
                            })
                    
                    # Extract semantic similarity data
                    if "semantic_data" in result and result["semantic_data"]:
                        if isinstance(result["semantic_data"], list):
                            for item in result["semantic_data"]:
                                enriched_item = dict(item) if isinstance(item, dict) else {"data": item}
                                enriched_item["_source_task"] = task_id
                                enriched_item["_data_type"] = "direct_semantic_data"
                                raw_data.append(enriched_item)
                        else:
                            raw_data.append({
                                "_source_task": task_id,
                                "_data_type": "direct_semantic_data",
                                "data": result["semantic_data"]
                            })
                    
                    # Extract any other results
                    if "results" in result and result["results"]:
                        if isinstance(result["results"], list):
                            for item in result["results"]:
                                enriched_item = dict(item) if isinstance(item, dict) else {"data": item}
                                enriched_item["_source_task"] = task_id
                                enriched_item["_data_type"] = "generic_results"
                                raw_data.append(enriched_item)
                        else:
                            raw_data.append({
                                "_source_task": task_id,
                                "_data_type": "generic_results",
                                "data": result["results"]
                            })
                    
                    # Handle string context (try to parse as JSON)
                    if "context" in result and isinstance(result["context"], str):
                        try:
                            import json
                            context_data = json.loads(result["context"])
                            if isinstance(context_data, list):
                                for item in context_data:
                                    enriched_item = dict(item) if isinstance(item, dict) else {"data": item}
                                    enriched_item["_source_task"] = task_id
                                    enriched_item["_data_type"] = "parsed_context_data"
                                    raw_data.append(enriched_item)
                            else:
                                raw_data.append({
                                    "_source_task": task_id,
                                    "_data_type": "parsed_context_data",
                                    "data": context_data
                                })
                        except:
                            # If not JSON, add as text data
                            raw_data.append({
                                "_source_task": task_id,
                                "_data_type": "text_context",
                                "text_content": result["context"]
                            })
        
        # Log extraction summary
        data_types = {}
        for item in raw_data:
            dtype = item.get("_data_type", "unknown")
            data_types[dtype] = data_types.get(dtype, 0) + 1
        
        logger.info(f"📊 EXTRACTION SUMMARY: {len(raw_data)} total items")
        for dtype, count in data_types.items():
            logger.info(f"  - {dtype}: {count} items")
        
        return raw_data
    
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
    
    def _supplement_synthesis_with_task_results(self, 
                                              sparse_answer: str, 
                                              completed_results: Dict[str, Any], 
                                              question: str) -> str:
        """
        Supplement sparse progressive synthesis with rich task execution results.
        
        Args:
            sparse_answer: Initial answer from progressive synthesis
            completed_results: Dictionary of completed task results
            question: Original user question
            
        Returns:
            Enhanced answer incorporating task results
        """
        logger.info("🔧 Supplementing sparse synthesis with detailed task results")
        
        # Extract the most information-rich task results
        detailed_sections = []
        code_interpreter_results = []
        chunked_analysis_results = []
        
        for task_id, result in completed_results.items():
            if isinstance(result, dict):
                # Extract code interpreter results (usually very detailed)
                if result.get("tool_name") == "code_interpreter" and result.get("tool_result"):
                    code_interpreter_results.append({
                        "task": task_id,
                        "result": result["tool_result"]
                    })
                
                # Extract chunked analysis results (rich functional analysis)
                elif "func_" in task_id or "chunk" in task_id.lower():
                    if result.get("context") and hasattr(result["context"], "structured_data"):
                        if len(result["context"].structured_data) > 50:  # Rich dataset
                            chunked_analysis_results.append({
                                "task": task_id,
                                "data_count": len(result["context"].structured_data),
                                "summary": self._summarize_chunked_data(result["context"].structured_data)
                            })
        
        # Build supplemented answer
        enhanced_parts = [sparse_answer, ""]
        
        # Add code interpreter insights
        if code_interpreter_results:
            enhanced_parts.append("## Detailed Analysis Results")
            for ci_result in code_interpreter_results:
                enhanced_parts.append(f"**{ci_result['task']} Analysis:**")
                enhanced_parts.append(ci_result['result'][:2000])  # Include substantial detail
                enhanced_parts.append("")
        
        # Add chunked analysis summaries
        if chunked_analysis_results:
            enhanced_parts.append("## Functional Analysis Summary")
            for chunk_result in chunked_analysis_results:
                enhanced_parts.append(f"**{chunk_result['task']}** ({chunk_result['data_count']} proteins):")
                enhanced_parts.append(chunk_result['summary'])
                enhanced_parts.append("")
        
        # Add task execution summary
        enhanced_parts.append("## Execution Summary")
        enhanced_parts.append(f"Analysis completed through {len(completed_results)} comprehensive tasks, including:")
        
        task_summaries = []
        for task_id, result in completed_results.items():
            if isinstance(result, dict):
                if result.get("tool_name"):
                    task_summaries.append(f"- {task_id}: {result['tool_name']} analysis")
                elif "func_" in task_id:
                    task_summaries.append(f"- {task_id}: Functional classification analysis")
                else:
                    task_summaries.append(f"- {task_id}: Database query and analysis")
        
        enhanced_parts.extend(task_summaries)
        
        supplemented_answer = "\\n".join(enhanced_parts)
        logger.info(f"✅ Enhanced answer length: {len(sparse_answer)} → {len(supplemented_answer)} characters")
        
        return supplemented_answer
    
    def _summarize_chunked_data(self, structured_data: List[Dict[str, Any]]) -> str:
        """
        Create a concise summary of chunked data analysis.
        
        Args:
            structured_data: List of data items from chunked analysis
            
        Returns:
            Concise summary of the data
        """
        if not structured_data:
            return "No data available"
        
        # Count different types of functions/categories
        function_counts = {}
        protein_counts = 0
        
        for item in structured_data:
            protein_counts += 1
            
            # Count by KO description if available
            if "ko_description" in item:
                desc = item["ko_description"]
                if desc:
                    # Extract main function type
                    if "transport" in desc.lower():
                        function_counts["transport"] = function_counts.get("transport", 0) + 1
                    elif "metabolism" in desc.lower() or "synthase" in desc.lower():
                        function_counts["metabolism"] = function_counts.get("metabolism", 0) + 1
                    elif "regulation" in desc.lower() or "regulatory" in desc.lower():
                        function_counts["regulation"] = function_counts.get("regulation", 0) + 1
                    else:
                        function_counts["other"] = function_counts.get("other", 0) + 1
        
        # Build summary
        summary_parts = [f"{protein_counts} proteins analyzed"]
        
        if function_counts:
            sorted_functions = sorted(function_counts.items(), key=lambda x: x[1], reverse=True)
            func_summary = ", ".join([f"{count} {func}" for func, count in sorted_functions[:3]])
            summary_parts.append(f"Functions: {func_summary}")
        
        return "; ".join(summary_parts)
    
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
    
    def _determine_analysis_type(self, question: str) -> str:
        """
        Determine the analysis type based on question content for biological context.
        
        Args:
            question: User's question
            
        Returns:
            Analysis type: spatial_genomic, functional_annotation, or comprehensive_discovery
        """
        question_lower = question.lower()
        
        # Spatial/genomic organization patterns
        spatial_patterns = [
            "operon", "operons", "gene cluster", "genomic region", "prophage", 
            "phage", "spatial", "neighborhood", "proximity", "adjacent",
            "genomic context", "gene organization", "cluster", "loci"
        ]
        
        # Functional annotation patterns  
        functional_patterns = [
            "function", "functional", "activity", "pathway", "metabolic",
            "enzyme", "protein family", "domain", "kegg", "pfam", "annotation"
        ]
        
        # Discovery/exploration patterns
        discovery_patterns = [
            "find", "discover", "explore", "look through", "see what", 
            "interesting", "novel", "unusual", "stands out", "browse"
        ]
        
        if any(pattern in question_lower for pattern in spatial_patterns):
            logger.info(f"🧬 Analysis type: SPATIAL_GENOMIC (detected patterns for spatial organization)")
            return "spatial_genomic"
        elif any(pattern in question_lower for pattern in functional_patterns):
            logger.info(f"🔬 Analysis type: FUNCTIONAL_ANNOTATION (detected patterns for functional analysis)")
            return "functional_annotation"
        elif any(pattern in question_lower for pattern in discovery_patterns):
            logger.info(f"🌐 Analysis type: COMPREHENSIVE_DISCOVERY (detected patterns for exploration)")
            return "comprehensive_discovery"
        else:
            # Default to functional annotation for general queries
            logger.info(f"📊 Analysis type: FUNCTIONAL_ANNOTATION (default for general queries)")
            return "functional_annotation"