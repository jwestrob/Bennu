#!/usr/bin/env python3
"""
Core GenomicRAG class with working implementation.
Restored from backup with modular organization.
"""

import logging
from typing import List, Dict, Any, Optional
import os
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
from .dspy_signatures import NEO4J_SCHEMA
from .utils import setup_debug_logging, GenomicContext
from .log_formatter import setup_enhanced_logging
from .dspy_signatures import PlannerAgent, QueryClassifier, ContextRetriever, GenomicAnswerer
# Legacy TaskGraph types gated behind flag to enable quarantine without breakage
if os.getenv("AGENT_ENABLE_LEGACY_TASKGRAPH", "1") == "1":
    try:
        from .task_management import TaskGraph, Task, TaskType, TaskStatus  # type: ignore
    except Exception:  # pragma: no cover
        TaskGraph = Task = TaskType = TaskStatus = None  # type: ignore
else:
    TaskGraph = Task = TaskType = TaskStatus = None  # type: ignore
from .external_tools import AVAILABLE_TOOLS
from .intelligent_routing import IntelligentRouter
from .genome_selection import UnifiedGenomeSelector
from ..context_compression import ContextCompressor
from .memory import NoteKeeper, ProgressiveSynthesizer, get_model_allocator
from .policy_engine import get_policy_engine
from .genome_context_extractor import GenomeContextExtractor
from .query_validator import QueryValidator
# Old genome_selector.py replaced by unified genome_selection.py
from .router import get_router
from .agent.tools.validate import validate_toolcall
from .tracing import get_tracer

logger = logging.getLogger(__name__)

class GenomicRAG(dspy.Module if DSPY_AVAILABLE else object):
    """
    Main genomic RAG system with working implementation.
    
    Combines structured queries (Neo4j) with semantic search (LanceDB)
    and intelligent code interpreter enhancement.
    """
    
    def __init__(self, config: LLMConfig, chunk_context_size: int = 4096, enable_memory: bool = True, enhanced_logging: bool = False):
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
        self.router = get_router()
        self.tracer = get_tracer()
        self.genome_selector = UnifiedGenomeSelector(self.neo4j_processor)
        self.context_compressor = ContextCompressor()
        self.genome_context_extractor = GenomeContextExtractor()
        self.query_validator = QueryValidator()
# Unified genome selector initialized above
        
        # Initialize memory system
        self.note_keeper = NoteKeeper() if enable_memory else None
        self.progressive_synthesizer = None  # Will be initialized when needed
        
        # Initialize model allocation system
        self.model_allocator = get_model_allocator()
        
        # Initialize policy engine
        self.policy_engine = get_policy_engine()
        
        # Configure DSPy with model allocation
        self._configure_dspy()
        
        # DSPy components are now instantiated on-demand via _run() method
        # No need for persistent Predict attributes
        
        # Store DSPy availability for task executor
        self.dspy_available = DSPY_AVAILABLE
        
        # Set up enhanced logging if requested, otherwise use debug logging
        if enhanced_logging:
            try:
                setup_enhanced_logging(
                    log_level="INFO",
                    filter_noise=True,
                    show_timestamps=True,
                    export_to_file=False
                )
                logger.info("🎯 GenomicRAG initialized with enhanced logging")
            except Exception as e:
                logger.warning(f"Enhanced logging setup failed: {e}, using default logging")
                setup_debug_logging()
        else:
            setup_debug_logging()
        
        logger.info("✅ GenomicRAG system initialized successfully")
        logger.info(f"🔧 Configuration: Neo4j={bool(config.database.neo4j_uri)}, LanceDB={bool(config.database.lancedb_path)}")
        logger.info(f"🧠 Memory: {'Enabled' if enable_memory else 'Disabled'}, Chunk Size: {chunk_context_size}")
        logger.info(f"🔥 Model: {config.llm_model} ({config.model_mode} mode)")
    
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
                    # Cost-effective mode: use ultra-cheap fallback as global default
                    # Individual tasks will use model allocation for intelligent selection
                    fallback_model = "gpt-4.1-nano"
                    model_string = f"openai/{fallback_model}"
                    lm = dspy.LM(model=model_string, temperature=0.0, max_tokens=8000)
                    logger.info(f"🎯 DSPy configured with ultra-cheap fallback: {model_string} (temp=0.0, max_tokens=8000)")
                    logger.info(f"💡 Model allocation will override this fallback for complex tasks")
                
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
    
    def _run(self, task_name: str, signature_cls, **kwargs):
        """
        Centralized Predict wrapper. Allocates the appropriate model via ModelAllocator,
        instantiates the module, executes it, and returns the result.
        """
        def _call(module):
            return module(**kwargs)
        
        return self.model_allocator.create_context_managed_call(
            task_name=task_name,
            signature_class=signature_cls,
            module_call_func=_call,
            query=kwargs.get("question", "") or kwargs.get("user_query", ""),
            task_context=kwargs.get("task_context", "")
        )
    
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
            try:
                self.tracer.emit("pipeline.start", {"question": question})
            except Exception:
                pass
            
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
                planning_result = self._run("agentic_planning", PlannerAgent, user_query=question)
            
            console.print(f"🎯 Planning decision: {'agentic' if planning_result.requires_planning else 'traditional'}")
            try:
                self.tracer.emit("pipeline.plan_decision", {
                    "requires_planning": bool(planning_result.requires_planning),
                    "reasoning": getattr(planning_result, 'reasoning', None),
                })
            except Exception:
                pass
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
                    # Use unified genome selector for LLM-based analysis
                    llm_selector = self.genome_selector
                    
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

        # Stage A deterministic guardrail via unified router
        try:
            router_decision = self.router.route(question)
            if router_decision.tool == "whole_genome_reader":
                toolcall = {"tool": router_decision.tool, "params": router_decision.params}
                ok, errs = validate_toolcall(toolcall)
                if not ok:
                    raise ValueError(f"StageA whole_genome_reader toolcall invalid: {'; '.join(errs)}")

                console.print("🧬 [bold cyan]Stage A routed to whole_genome_reader[/bold cyan]")
                from .external_tools import WholeGenomeReader
                reader = WholeGenomeReader(self.neo4j_processor)
                spatial_results = await reader.read_full_genomic_context(question)

                if spatial_results and 'genomic_data' in spatial_results:
                    context = GenomicContext(
                        structured_data=spatial_results['genomic_data'],
                        semantic_data=[],
                        metadata={'analysis_type': 'SPATIAL_GENOMIC', 'tool_used': 'whole_genome_reader'},
                        query_time=0.0,
                        compressed_context=""
                    )
                    formatted_context = self._format_spatial_context(context)
                    return await self._synthesize_answer(
                        question,
                        formatted_context,
                        query_type="SPATIAL_GENOMIC",
                        analysis_type="spatial_genomic",
                    )
                else:
                    return {
                        "question": question,
                        "answer": "No spatial genomic data retrieved.",
                        "confidence": "low",
                        "citations": "",
                        "query_metadata": {"analysis_type": "SPATIAL_GENOMIC", "tool_used": "whole_genome_reader"}
                    }
            # Stage B: database_query via templates
            if router_decision.tool == "database_query":
                params = router_decision.params or {}
                template = params.get("template")
                slots = params.get("slots", {})
                if template:
                    try:
                        self.tracer.emit("router.db_template.start", {"template": template})
                    except Exception:
                        pass
                    # Execute template safely via processor
                    db_result = await self.neo4j_processor.execute_named_template(template, slots)
                    # Convert to GenomicContext and synthesize
                    context = GenomicContext(
                        structured_data=db_result.results,
                        semantic_data=[],
                        metadata={
                            'analysis_type': 'FUNCTIONAL_ANNOTATION',
                            'tool_used': 'database_query',
                            'template': template,
                            'result_count': db_result.metadata.get('result_count', 0)
                        },
                        query_time=db_result.execution_time,
                        compressed_context=""
                    )
                    formatted_context = self._format_context(context)
                    return await self._synthesize_answer(
                        question,
                        formatted_context,
                        query_type=f"template:{template}",
                        analysis_type="functional_annotation",
                    )
        except Exception as e:
            logger.error(f"Stage A/B routing failed or not applicable: {e}")

        # If router suggested something else, log suggestion for tracing
        try:
            if 'router_decision' in locals() and router_decision and router_decision.tool != "whole_genome_reader":
                console.print(f"🧭 [dim]Router suggests: {router_decision.tool}[/dim]")
        except Exception:
            pass
        
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
        
        # Step 1.6: Stage A handled spatial routing already; proceed with standard flow
        
        if classification is None:
            logger.warning("Model allocation failed for classification, falling back to default")
            # Ensure there's a default LM configured for fallback
            if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
                logger.warning("No default LM configured, setting up fallback")
                fallback_lm = dspy.LM(model="openai/gpt-4.1-mini", temperature=0.0, max_tokens=8000)
                dspy.settings.configure(lm=fallback_lm)
            classification = self._run("query_classification", QueryClassifier, question=question)
        
        console.print(f"📊 Query type: {classification.query_type}")
        console.print(f"💭 Reasoning: {classification.reasoning}")
        
        # Step 2: INTELLIGENT GENOME SELECTION - Use LLM to analyze genome selection intent
        genome_filter_required = False
        target_genome = ""
        task_context = "Global query across all genomes"
        
        try:
            # Use unified genome selector for LLM-based analysis  
            llm_selector = self.genome_selector
            
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
            retrieval_plan = self._run("context_preparation", ContextRetriever,
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
        scoped_query, scope_metadata = self.genome_selector.enforce_genome_scope(question, validated_query)
        
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
                compressed_context, compression_stats = compressor.compress_context(all_results, target_size=25000)
                
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
            
            answer_result = self._run("biological_interpretation", GenomicAnswerer,
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
            # Handle envelope format
            if isinstance(result, dict):
                display = result.get("display_text") or ""
            else:
                display = str(result)
            logger.info(f"Literature search completed: {len(display)} characters")
            return display
            
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
            if isinstance(result, dict):
                success = bool(result.get("success"))
                display = result.get("display_text") or result.get("output") or ""
                if success:
                    logger.info("Code interpreter execution completed successfully")
                    return display
                else:
                    logger.warning(f"Code interpreter execution failed: {result.get('message', result.get('error', 'Unknown error'))}")
                    return None
            # Fallback
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
        """Execute unified agent workflow with dynamic tool chaining."""
        console.print("🤖 [bold]Using unified agent execution path[/bold]")
        console.print("🔗 [dim]Agent will dynamically chain tools based on discoveries[/dim]")
        
        try:
            # Import unified agent executor
            from .agent_executor import UnifiedAgentExecutor
            
            # Create and execute unified agent
            agent = UnifiedAgentExecutor(self, note_keeper=self.note_keeper)
            
            if selected_genome:
                console.print(f"🧬 [cyan]Agent will target genome:[/cyan] {selected_genome}")
            
            # Execute agent workflow
            agent_result = await agent.execute_agent_workflow(question, selected_genome)
            
            if not agent_result.success:
                console.print("⚠️ [yellow]Agent execution failed[/yellow]")
                console.print("🔄 [dim]Falling back to traditional mode[/dim]")
                return await self._execute_traditional_query(question)
            
            console.print(f"✅ [green]Agent completed {agent_result.total_steps} steps[/green]")
            console.print(f"🛠️ Tools used: {', '.join(agent_result.tools_used)}")
            console.print(f"⏱️ Total time: {agent_result.total_execution_time:.1f}s")
            
            # Convert agent result to expected format
            return {
                "question": question,
                "answer": agent_result.final_answer,
                "confidence": agent_result.confidence,
                "citations": agent_result.citations,
                "query_metadata": {
                    "execution_mode": "unified_agent",
                    "total_steps": agent_result.total_steps,
                    "tools_used": agent_result.tools_used,
                    "execution_time": agent_result.total_execution_time,
                    "note_taking_enabled": self.note_keeper is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}")
            console.print(f"⚠️ [yellow]Agent execution error: {str(e)}[/yellow]")
            console.print("🔄 [dim]Falling back to traditional mode[/dim]")
            return await self._execute_traditional_query(question)
    
    
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
                    formatted_context = str(result.results)
                    
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
    
    def _format_spatial_context(self, context: GenomicContext) -> str:
        """Format spatial genomic context for prophage/operon analysis."""
        formatted_parts = []
        
        # Add header for spatial genomic data
        if context.structured_data:
            formatted_parts.append(f"=== SPATIAL GENOMIC DATA ({len(context.structured_data)} genomic regions) ===")
            formatted_parts.append("Full genomic context with gene coordinates, annotations, and spatial organization:")
            formatted_parts.append("")
            
            # Format each genomic region with spatial information
            for i, region in enumerate(context.structured_data):
                formatted_parts.append(f"GENOMIC REGION {i+1}:")
                formatted_parts.append(str(region))
                formatted_parts.append("")
            
        # Add analysis metadata
        if context.metadata:
            formatted_parts.append("=== ANALYSIS METADATA ===")
            for key, value in context.metadata.items():
                formatted_parts.append(f"{key}: {value}")
        
        return "\\n".join(formatted_parts)
    
    async def _synthesize_answer(self, question: str, formatted_context: str, query_type: str, analysis_type: str) -> Dict[str, Any]:
        """Synthesize answer from formatted context using appropriate model allocation."""
        try:
            # Use model allocation for biological interpretation
            def answerer_call(module):
                return module(
                    question=question,
                    context=formatted_context,
                    analysis_type=analysis_type
                )
            
            from .dspy_signatures import GenomicAnswerer
            answer_result = self.model_allocator.create_context_managed_call(
                task_name="biological_interpretation",  # Maps to COMPLEX = o3
                signature_class=GenomicAnswerer,
                module_call_func=answerer_call
            )
            
            # Fallback if model allocation fails
            if answer_result is None:
                logger.warning("Model allocation failed for answer generation, falling back to default")
                if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
                    logger.warning("No default LM configured, setting up fallback")
                    fallback_lm = dspy.LM(model="openai/gpt-4.1-mini", temperature=0.0, max_tokens=8000)
                    dspy.settings.configure(lm=fallback_lm)
                
                answer_result = self._run("biological_interpretation", GenomicAnswerer,
                    question=question,
                    context=formatted_context,
                    analysis_type=analysis_type
                )
            
            # Return structured response
            return {
                "question": question,
                "answer": answer_result.answer,
                "confidence": answer_result.confidence,
                "citations": answer_result.citations,
                "query_metadata": {
                    "query_type": query_type,
                    "analysis_type": analysis_type,
                    "search_strategy": "spatial_genomic_tool",
                    "context_size": len(formatted_context),
                    "tool_used": "whole_genome_reader"
                }
            }
            
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return {
                "question": question,
                "answer": f"I encountered an error while analyzing the spatial genomic data: {str(e)}",
                "confidence": "low",
                "citations": "",
                "error": str(e),
                "query_metadata": {
                    "query_type": query_type,
                    "analysis_type": analysis_type,
                    "error": "synthesis_failed"
                }
            }
    
    
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
